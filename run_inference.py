"""
Custom SmolVLM2-256M Inference Script
======================================
These model files use relative imports (from ...utils import ...) because
they are designed to live INSIDE the transformers package directory.

Strategy: install transformers, then overwrite the smolvlm files inside
the installed package with our custom versions. This way all relative
imports resolve correctly and our changes are reflected.

Run on Colab:
    !pip install transformers accelerate Pillow num2words -q
    !python run_inference.py
"""

import os
import sys
import shutil
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image",      default="statue_of_liberty.jpg",  help="Path to input image")
parser.add_argument("--prompt",     default="Describe this image.",    help="Text prompt")
parser.add_argument("--focus",      default="0.5,0.5",                 help="Focus point x,y in [0,1]")
parser.add_argument("--max-tokens", default=100,   type=int,           help="Max new tokens (default: 100)")
parser.add_argument("--dtype",      default="bfloat16",
                    choices=["bfloat16", "float16", "float32", "int8"], help="Model dtype (default: bfloat16)")
parser.add_argument("--budget",     default=0,     type=int,
                    help="Total visual token budget for decoder. 0 = no limit (default: 0)")
parser.add_argument("--focus-pct",  default=50.0,  type=float,
                    help="Percent of budget allocated to focus-point partition (default: 50.0)")
args = parser.parse_args()

# ── Step 1: Find where transformers is installed ──────────────────────────────
import transformers
transformers_root = os.path.dirname(transformers.__file__)
smolvlm_dir = os.path.join(transformers_root, "models", "smolvlm")
print(f"[setup] transformers installed at: {transformers_root}")
print(f"[setup] SmolVLM module directory : {smolvlm_dir}")

if not os.path.isdir(smolvlm_dir):
    raise RuntimeError(
        f"SmolVLM not found in transformers at:\n  {smolvlm_dir}\n"
        "Run: pip install --upgrade transformers"
    )

# ── Step 2: Overwrite the installed files with our custom versions ─────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_TO_PATCH = [
    "modeling_smolvlm.py",
    "image_processing_smolvlm.py",
    "processing_smolvlm.py",
    "configuration_smolvlm.py",
]

print("\n[setup] Patching transformers with custom files...")
for fname in FILES_TO_PATCH:
    src = os.path.join(THIS_DIR, fname)
    dst = os.path.join(smolvlm_dir, fname)
    shutil.copy2(src, dst)
    print(f"  copied {fname} → {dst}")

pycache_dir = os.path.join(smolvlm_dir, "__pycache__")
if os.path.isdir(pycache_dir):
    shutil.rmtree(pycache_dir)
    print(f"  cleared __pycache__ at {pycache_dir}")

# ── Step 3: Force-reload modules ─────────────────────────────────────────────
mods_to_reload = [k for k in sys.modules if "smolvlm" in k.lower()]
for mod in mods_to_reload:
    del sys.modules[mod]
import transformers.models.smolvlm

print("\n[setup] Modules reloaded. Your custom code is now active.\n")

# ── Step 3b: Run focus-partitioning tests ─────────────────────────────────────
from transformers.models.smolvlm.image_processing_smolvlm import run_focus_partitioning_tests
run_focus_partitioning_tests()

# ── Step 4: Load model & processor ───────────────────────────────────────────
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Instruct"
MAX_NEW_TOKENS = args.max_tokens

print(f"[inference] Loading model from: {MODEL_ID}")
print(f"[inference] Device: {DEVICE}\n")

processor = AutoProcessor.from_pretrained(MODEL_ID)

if args.dtype == "int8":
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise RuntimeError("int8 requires bitsandbytes: pip install bitsandbytes -q")
    print("[inference] Loading in 8-bit (bitsandbytes)...")
    model = SmolVLMForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        _attn_implementation="eager",
        device_map="auto",
    )
else:
    TORCH_DTYPE = {"bfloat16": torch.bfloat16,
                   "float16":  torch.float16,
                   "float32":  torch.float32}[args.dtype]
    model = SmolVLMForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        _attn_implementation="eager",
    ).to(DEVICE)

IMAGE_PATH = args.image
PROMPT     = args.prompt

if not os.path.exists(IMAGE_PATH):
    print(f"[warning] Image not found: {IMAGE_PATH}")
    print("[warning] Creating a dummy RGB image for testing...")
    image = Image.new("RGB", (224, 224), color=(128, 64, 32))
else:
    image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ],
    }
]
prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TOKEN BUDGET  (applied at encoder output, BEFORE pixel-shuffle + MLP)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Why at encoder stage?
#  ─────────────────────
#  Encoder output: (N, 1024, 768)   ← budget decision happens here
#  pixel-shuffle merges 4×4 spatial patches → (N, 64, 768×16) → Linear → (N, 64, 576)
#
#  For focus inference we pool encoder tokens to hit the target decoder budget,
#  then GROUP consecutive encoder tokens (instead of spatial pixel-shuffle)
#  and apply only the linear modality_projection.  This produces the same
#  (1, dec_tokens, text_hidden_dim) shape that inputs_merger expects.
#
#  Normal inference: untouched — full grid partitioning as original model.
# ══════════════════════════════════════════════════════════════════════════════

def _pool1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Adaptive average pool a (seq_len, D) tensor → (target_len, D)."""
    if x.shape[0] == target_len:
        return x
    return F.adaptive_avg_pool1d(
        x.T.unsqueeze(0),   # (1, D, seq_len)
        target_len
    ).squeeze(0).T          # (target_len, D)


def _budget_encoder_tokens(
    enc_hs: torch.Tensor,
    total_budget_dec: int,
    focus_pct: float = 50.0,
    is_focus: bool = False,
    scale_factor: int = 4,
) -> tuple:
    """
    Pool encoder-output tokens to match a decoder token budget, then group
    them for the flat modality_projection (skips spatial pixel-shuffle).

    Inputs
    ------
    enc_hs          : (N, seq_len, enc_D)   vision encoder output per partition
    total_budget_dec: target tokens entering the LLM decoder
    focus_pct       : % of decoder budget for partition-0 (focus crop)
    is_focus        : True → N=2, partition-0=local crop, partition-1=global
    scale_factor    : SmolVLM scale_factor (4 for 256M); scale2=scale_factor²

    Output
    ------
    grouped   : (1, actual_dec, enc_D × scale2)  ready for modality_projection
    actual_dec: int — decoder token count (≤ total_budget_dec)

    How grouping replaces pixel-shuffle
    ─────────────────────────────────────
    pixel_shuffle expects a perfect-square seq and reorganises it spatially.
    After pooling, spatial structure is gone, so we instead reshape every
    scale2 consecutive encoder tokens into one wider vector and let the
    existing Linear(enc_D×scale2 → text_hidden_dim) project to LM space.
    Same weight, same output dim, no spatial constraint.
    """
    N, seq_len, enc_D = enc_hs.shape
    scale2 = scale_factor * scale_factor
    enc_budget = total_budget_dec * scale2  # encoder tokens equiv. to decoder budget

    if N * seq_len <= enc_budget:
        # Already within budget — flatten and group without pooling
        actual_dec = (N * seq_len) // scale2
        flat_enc = enc_hs.reshape(1, actual_dec * scale2, enc_D)
    elif is_focus and N == 2:
        focus_dec  = max(1, round(total_budget_dec * focus_pct / 100))
        global_dec = max(1, total_budget_dec - focus_dec)
        focus_dec  = max(1, total_budget_dec - global_dec)  # re-clamp: sum == total_budget_dec
        focus_part  = _pool1d(enc_hs[0], focus_dec  * scale2)
        global_part = _pool1d(enc_hs[1], global_dec * scale2)
        flat_enc = torch.cat([focus_part, global_part], dim=0).unsqueeze(0)
        actual_dec = focus_dec + global_dec
    else:
        T_dec = max(1, total_budget_dec // N)
        parts = [_pool1d(enc_hs[i], T_dec * scale2) for i in range(N)]
        flat_enc = torch.cat(parts, dim=0).unsqueeze(0)
        actual_dec = N * T_dec

    # (1, actual_dec * scale2, enc_D) → (1, actual_dec, enc_D * scale2)
    grouped = flat_enc.reshape(1, actual_dec, enc_D * scale2)
    return grouped, actual_dec


def _run_encoder(pixel_values: torch.Tensor, pixel_attention_mask) -> torch.Tensor:
    """
    Run the preprocessing and vision encoder only (no connector).
    Mirrors SmolVLMModel.get_image_features up to last_hidden_state.
    Returns (N, seq_len, enc_D).
    """
    m = model.model
    batch_size, num_images, num_channels, height, width = pixel_values.shape
    pv = pixel_values.view(batch_size * num_images, num_channels, height, width)
    pv = pv.to(dtype=m.dtype)

    nb_values = pv.shape[1:].numel()
    real_inds = (pv == 0.0).sum(dim=(-1, -2, -3)) != nb_values
    real_inds[0] |= ~torch.any(real_inds)
    pv = pv[real_inds].contiguous()

    if pixel_attention_mask is None:
        pam = torch.ones(
            [pv.shape[0], pv.shape[2], pv.shape[3]], dtype=torch.bool, device=pv.device
        )
    else:
        pam = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
        pam = pam[real_inds].contiguous()

    patch_sz = m.config.vision_config.patch_size
    subgrid = pam.unfold(1, patch_sz, patch_sz).unfold(2, patch_sz, patch_sz)
    patch_mask = (subgrid.sum(dim=(-1, -2)) > 0).bool()

    out = m.vision_model(pixel_values=pv, patch_attention_mask=patch_mask, return_dict=True)
    return out.last_hidden_state  # (N, seq_len, enc_D)


def build_focus_budgeted_inputs(raw_inputs, prompt_text, total_budget, focus_pct):
    """
    Focus-inference budget pipeline:
      1. Run vision encoder only → enc_hs (N, 1024, 768)
      2. Pool encoder tokens to budget, group for flat projection
         → grouped (1, dec_tokens, 768×scale2)
      3. Apply modality_projection (Linear) only — skip spatial pixel-shuffle
         → image_hs (1, dec_tokens, text_hidden_dim)
      4. Rebuild text with image_seq_len = dec_tokens
      5. Return inputs dict with image_hidden_states replacing pixel_values
    """
    pv  = raw_inputs["pixel_values"].to(DEVICE)
    pam = raw_inputs.get("pixel_attention_mask")
    if pam is not None:
        pam = pam.to(DEVICE)

    with torch.no_grad():
        enc_hs = _run_encoder(pv, pam)                      # (N, 1024, 768)

    scale_factor = model.model.connector.scale_factor
    grouped, dec_tokens = _budget_encoder_tokens(
        enc_hs, total_budget, focus_pct=focus_pct, is_focus=True, scale_factor=scale_factor
    )

    with torch.no_grad():
        image_hs = model.model.connector.modality_projection(grouped.to(dtype=model.model.dtype))

    print(f"[budget] encoder {tuple(enc_hs.shape)} → "
          f"grouped {tuple(grouped.shape)} → "
          f"decoder tokens: {dec_tokens} "
          f"(budget={total_budget}, focus_pct={focus_pct:.0f}%)")

    orig_seq_len = processor.image_seq_len
    processor.image_seq_len = dec_tokens
    new_prompt = processor.expand_text_with_image_tokens(
        [prompt_text], image_rows=[[0]], image_cols=[[0]]
    )[0]
    processor.image_seq_len = orig_seq_len

    text_inputs = processor.tokenizer(new_prompt, return_tensors="pt")

    return {
        "image_hidden_states": image_hs.to(DEVICE),
        **{k: v.to(DEVICE) for k, v in text_inputs.items()},
    }, dec_tokens


# ── Token budget tests ────────────────────────────────────────────────────────
def run_token_budget_tests():
    print("\n" + "=" * 60)
    print("Running token-budget tests  (at encoder output stage)")
    print("=" * 60)

    # SmolVLM2-256M encoder: (N, 1024, 768), scale_factor=4 → scale2=16
    ENC_D   = 768
    SEQ_ENC = 1024
    SF      = 4
    SCALE2  = SF * SF   # 16
    PROJ_D  = ENC_D * SCALE2  # 12288 — width seen by modality_projection

    # ── T1: within budget → flatten+group, no pooling ────────────────────────
    hs = torch.rand(2, SEQ_ENC, ENC_D)     # (2, 1024, 768), total_enc = 2048
    # budget=128 dec → enc_budget=128*16=2048 == total_enc → within budget
    out, total = _budget_encoder_tokens(hs, total_budget_dec=128, is_focus=True, scale_factor=SF)
    assert out.shape == (1, 128, PROJ_D), f"T1 FAIL: {out.shape}"
    assert total == 128
    print(f"  T1 PASS  within-budget → grouped {tuple(out.shape)}, dec_tokens={total}")

    # ── T2: focus path, equal split (focus_pct=50), budget=64 ────────────────
    hs = torch.rand(2, SEQ_ENC, ENC_D)
    out, total = _budget_encoder_tokens(hs, total_budget_dec=64, focus_pct=50.0, is_focus=True, scale_factor=SF)
    assert out.shape == (1, 64, PROJ_D), f"T2 FAIL shape: {out.shape}"
    assert total == 64,                  f"T2 FAIL total: {total}"
    print(f"  T2 PASS  (2,1024,768) + budget=64 + focus_pct=50 → {tuple(out.shape)}, dec_tokens={total}")

    # ── T3: focus path, 70% to focus, 30% to global, budget=32 ──────────────
    hs = torch.rand(2, SEQ_ENC, ENC_D)
    budget = 32
    out, total = _budget_encoder_tokens(hs, total_budget_dec=budget, focus_pct=70.0, is_focus=True, scale_factor=SF)
    focus_dec  = max(1, round(budget * 70 / 100))   # 22
    global_dec = max(1, budget - focus_dec)          # 10
    assert total == focus_dec + global_dec,          f"T3 FAIL total: {total}"
    assert out.shape == (1, total, PROJ_D),          f"T3 FAIL shape: {out.shape}"
    print(f"  T3 PASS  focus_pct=70 → focus={focus_dec} global={global_dec} total={total}")

    # ── T4: extreme focus_pct=100 (global gets min 1 dec token) ─────────────
    hs = torch.rand(2, SEQ_ENC, ENC_D)
    out, total = _budget_encoder_tokens(hs, total_budget_dec=32, focus_pct=100.0, is_focus=True, scale_factor=SF)
    # global_dec = max(1, 0) = 1, focus_dec = max(1, 32-1) = 31 → total = 32
    assert out.shape[1] == 32, f"T4 FAIL shape: {out.shape}"
    assert total == 32,        f"T4 FAIL total: {total}"
    assert out.shape[2] == PROJ_D, f"T4 FAIL proj_dim: {out.shape}"
    print(f"  T4 PASS  focus_pct=100 → focus=31 global=1 total={total}, shape {tuple(out.shape)}")

    # ── T5: last dim always == enc_D × scale2 (= modality_projection input) ──
    for b in [16, 32, 64]:
        hs = torch.rand(2, SEQ_ENC, ENC_D)
        out, _ = _budget_encoder_tokens(hs, total_budget_dec=b, focus_pct=50.0, is_focus=True, scale_factor=SF)
        assert out.shape[2] == PROJ_D, f"T5 FAIL b={b}: last_dim={out.shape[2]} expected {PROJ_D}"
    print(f"  T5 PASS  grouped last dim == {PROJ_D} (= enc_D×scale2) for budgets [16,32,64]")

    # ── T6: shape[1] (dec_tokens) always == actual_dec ───────────────────────
    for b in [16, 32, 48, 64]:
        hs = torch.rand(2, SEQ_ENC, ENC_D)
        out, total = _budget_encoder_tokens(hs, total_budget_dec=b, focus_pct=60.0, is_focus=True, scale_factor=SF)
        assert out.shape[1] == total, f"T6 FAIL b={b}: shape={out.shape} total={total}"
    print(f"  T6 PASS  shape[1] == dec_tokens for budgets [16,32,48,64]")

    # ── T7: dtype preserved ──────────────────────────────────────────────────
    hs = torch.rand(2, SEQ_ENC, ENC_D).to(torch.float16)
    out, _ = _budget_encoder_tokens(hs, total_budget_dec=32, is_focus=True, scale_factor=SF)
    assert out.dtype == torch.float16, f"T7 FAIL dtype: {out.dtype}"
    print(f"  T7 PASS  output dtype preserved ({out.dtype})")

    # ── T8: avg-pool contraction (no values outside input range) ─────────────
    hs = torch.rand(2, SEQ_ENC, ENC_D)
    out, _ = _budget_encoder_tokens(hs, total_budget_dec=32, focus_pct=50, is_focus=True, scale_factor=SF)
    # The pooled part of grouped came from avg_pool, so its range ⊆ [hs.min, hs.max]
    assert out.min() >= hs.min() - 1e-5 and out.max() <= hs.max() + 1e-5, \
        f"T8 FAIL: grouped values outside input encoder range"
    print(f"  T8 PASS  grouped values within encoder output range (avg-pool contraction)")

    print(f"\nAll 8 token-budget tests PASSED  (encoder-stage budget, scale_factor={SF})")
    print("=" * 60)


run_token_budget_tests()


# ══════════════════════════════════════════════════════════════════════════════
#  EVAL HELPER
# ══════════════════════════════════════════════════════════════════════════════

class _FirstTokenTimer(StoppingCriteria):
    def __init__(self):
        self.t = None
    def __call__(self, input_ids, scores, **kwargs):
        if self.t is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.t = time.perf_counter()
        return False


def run_eval(label, inputs):
    """
    Run generate() and print:
      1. Visual tokens
      2. TTFT (ms)
      3. Throughput (tok/s)
      4. Peak GPU memory (MB)
    """
    pv = inputs.get("pixel_values")
    hs = inputs.get("image_hidden_states")

    if pv is not None:
        num_visual_tokens = pv.shape[1] * model.model.image_seq_len
        vis_label = f"{pv.shape[1]} partition(s) × {model.model.image_seq_len} seq_len"
    elif hs is not None:
        num_visual_tokens = hs.shape[1]
        vis_label = f"budgeted flat stream"
    else:
        num_visual_tokens = 0
        vis_label = "none"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    timer = _FirstTokenTimer()
    t_start = time.perf_counter()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            stopping_criteria=StoppingCriteriaList([timer]),
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_time  = t_end - t_start
    ttft_ms     = (timer.t - t_start) * 1000 if timer.t else float("nan")
    input_len   = inputs["input_ids"].shape[1]
    new_tokens  = generated_ids.shape[1] - input_len
    throughput  = new_tokens / total_time
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    print(f"\n{'='*60}")
    print(f"  EVAL — {label}")
    print(f"{'='*60}")
    print(f"  1. Visual tokens      {num_visual_tokens:>6}  ({vis_label})")
    print(f"  2. TTFT               {ttft_ms:>7.1f} ms")
    print(f"  3. Throughput         {throughput:>7.1f} tok/s  ({new_tokens} tokens in {total_time:.2f}s)")
    print(f"  4. Peak GPU memory    {peak_mem_mb:>7.1f} MB")
    print(f"{'='*60}")

    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"\n[output]\n{output}")
    return output


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Normal inference (full grid partitioning — no token budget)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  NORMAL INFERENCE  (full grid partitioning, no budget)")
print("─"*60)

inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(DEVICE)
run_eval("Normal", inputs)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Focus-point inference (local crop + global = 2 partitions)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  FOCUS-POINT INFERENCE  (local crop + global = 2 tiles)")
print("─"*60)

FOCUS_POINT = tuple(float(v) for v in args.focus.split(","))

raw_inputs = processor.image_processor.preprocess(
    [image], return_tensors="pt", focus_point=FOCUS_POINT
)
print(f"[focus] pixel_values shape: {tuple(raw_inputs['pixel_values'].shape)}")

# Expand text to match 2-partition layout
focus_prompt_text = processor.expand_text_with_image_tokens(
    [prompt_text], image_rows=[[1]], image_cols=[[1]]
)[0]
text_inputs = processor.tokenizer(focus_prompt_text, return_tensors="pt")
focus_inputs = {**raw_inputs, **text_inputs}
focus_inputs = {k: v.to(DEVICE) for k, v in focus_inputs.items()}

if args.budget > 0:
    focus_inputs, _ = build_focus_budgeted_inputs(
        focus_inputs, prompt_text,
        total_budget=args.budget, focus_pct=args.focus_pct,
    )

run_eval(
    "Focus-point" + (f"  [budget={args.budget}, focus_pct={args.focus_pct:.0f}%]"
                     if args.budget > 0 else ""),
    focus_inputs,
)
