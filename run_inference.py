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
#  TOKEN BUDGET
# ══════════════════════════════════════════════════════════════════════════════

def _pool1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Adaptive average pool a (seq_len, D) tensor → (target_len, D)."""
    if x.shape[0] == target_len:
        return x
    # adaptive_avg_pool1d expects (batch=1, channels=D, length=seq_len)
    return F.adaptive_avg_pool1d(
        x.T.unsqueeze(0),   # (1, D, seq_len)
        target_len
    ).squeeze(0).T          # (target_len, D)


def apply_token_budget(
    image_hidden_states: torch.Tensor,
    total_budget: int,
    focus_pct: float = 50.0,
    is_focus: bool = False,
) -> tuple:
    """
    Pool visual tokens to fit within total_budget before the LLM decoder.

    Inputs
    ------
    image_hidden_states : (N, seq_len, D)   connector output, one row per partition
    total_budget        : max total visual tokens allowed into the decoder
    focus_pct           : % of budget for the focus partition (is_focus=True only)
    is_focus            : True  → N=2, partition-0=local crop, partition-1=global

    Output
    ------
    budgeted     : (1, actual_total, D)   flat token stream ready for inputs_merger
    actual_total : int  — number of tokens in budgeted  (≤ total_budget)

    Why (1, actual_total, D)?
    ─────────────────────────
    inputs_merger requires all partitions to share the same patch_size.
    Flattening to a single pseudo-partition avoids that constraint and lets
    focus_pct allocate different token counts to each original partition.
    """
    N, seq_len, D = image_hidden_states.shape
    current_total = N * seq_len

    if current_total <= total_budget:
        # Already within budget — just flatten into a single stream
        return image_hidden_states.reshape(1, current_total, D), current_total

    if is_focus and N == 2:
        focus_T  = max(1, round(total_budget * focus_pct / 100))
        global_T = max(1, total_budget - focus_T)
        focus_T  = max(1, total_budget - global_T)  # re-clamp so focus_T+global_T == total_budget
        focus_part  = _pool1d(image_hidden_states[0], focus_T)   # (focus_T, D)
        global_part = _pool1d(image_hidden_states[1], global_T)  # (global_T, D)
        budgeted = torch.cat([focus_part, global_part], dim=0).unsqueeze(0)
        actual_total = focus_T + global_T
    else:
        T = max(1, total_budget // N)
        parts = [_pool1d(image_hidden_states[i], T) for i in range(N)]
        budgeted = torch.cat(parts, dim=0).unsqueeze(0)
        actual_total = N * T

    return budgeted, actual_total


def build_budgeted_inputs(raw_inputs, prompt_text, is_focus, total_budget, focus_pct):
    """
    1. Run vision encoder on pixel_values → image_hidden_states (N, seq_len, D)
    2. Apply token_budget → (1, actual_total, D)
    3. Rebuild text with image_seq_len = actual_total (single-image prompt)
    4. Return new inputs dict with image_hidden_states replacing pixel_values

    The rebuilt text uses _prompt_single_image so the text contains exactly
    actual_total <image> tokens in one contiguous block, matching the flat
    (1, actual_total, D) hidden-state tensor.
    """
    pv  = raw_inputs["pixel_values"].to(DEVICE)
    pam = raw_inputs.get("pixel_attention_mask")
    if pam is not None:
        pam = pam.to(DEVICE)

    with torch.no_grad():
        image_hs = model.model.get_image_features(pv, pam).pooler_output  # (N, seq_len, D)

    budgeted_hs, actual_total = apply_token_budget(
        image_hs, total_budget, focus_pct=focus_pct, is_focus=is_focus
    )

    print(f"[budget] {image_hs.shape[0]}×{image_hs.shape[1]}="
          f"{image_hs.shape[0]*image_hs.shape[1]} tokens → "
          f"{actual_total} tokens  "
          f"(budget={total_budget}"
          + (f", focus_pct={focus_pct:.0f}%" if is_focus else "") + ")")

    # Temporarily set processor.image_seq_len so expand_text_with_image_tokens
    # generates exactly actual_total <image> tokens for a single-image prompt
    orig_seq_len = processor.image_seq_len
    processor.image_seq_len = actual_total
    new_prompt = processor.expand_text_with_image_tokens(
        [prompt_text], image_rows=[[0]], image_cols=[[0]]
    )[0]
    processor.image_seq_len = orig_seq_len  # always restore

    text_inputs = processor.tokenizer(new_prompt, return_tensors="pt")

    return {
        "image_hidden_states": budgeted_hs.to(DEVICE),
        **{k: v.to(DEVICE) for k, v in text_inputs.items()},
    }, actual_total


# ── Token budget tests ────────────────────────────────────────────────────────
def run_token_budget_tests():
    print("\n" + "=" * 60)
    print("Running token-budget tests")
    print("=" * 60)

    D = 576  # hidden dim for SmolVLM2-256M

    # ── T1: no pooling needed (within budget) ────────────────────────────────
    hs = torch.rand(2, 32, D)
    out, total = apply_token_budget(hs, total_budget=128, is_focus=True)
    assert out.shape == (1, 64, D), f"T1 FAIL: {out.shape}"
    assert total == 64
    print(f"  T1 PASS  within-budget → shape {tuple(out.shape)}, total={total}")

    # ── T2: focus path, equal split (focus_pct=50) ───────────────────────────
    hs = torch.rand(2, 64, D)          # (2, 64, 576) — your target shape
    out, total = apply_token_budget(hs, total_budget=32, focus_pct=50.0, is_focus=True)
    assert out.shape == (1, 32, D),   f"T2 FAIL shape: {out.shape}"
    assert total == 32,               f"T2 FAIL total: {total}"
    print(f"  T2 PASS  (2,64,576) + budget=32 + focus_pct=50 → {tuple(out.shape)}, total={total}")

    # ── T3: focus path, 70% to focus, 30% to global ──────────────────────────
    hs = torch.rand(2, 64, D)
    budget = 32
    out, total = apply_token_budget(hs, total_budget=budget, focus_pct=70.0, is_focus=True)
    focus_T  = max(1, round(budget * 70 / 100))   # 22
    global_T = max(1, budget - focus_T)            # 10
    assert total == focus_T + global_T,            f"T3 FAIL total: {total}"
    assert out.shape == (1, total, D),             f"T3 FAIL shape: {out.shape}"
    print(f"  T3 PASS  focus_pct=70 → focus={focus_T} global={global_T} total={total}")

    # ── T4: focus path, all tokens to focus (focus_pct=100) ──────────────────
    hs = torch.rand(2, 64, D)
    out, total = apply_token_budget(hs, total_budget=32, focus_pct=100.0, is_focus=True)
    # global_T = max(1, 0) = 1, then focus_T = max(1, 32-1) = 31 → total = 32
    assert out.shape[1] == 32,    f"T4 FAIL shape: {out.shape}"
    assert total == 32,           f"T4 FAIL total: {total}"
    print(f"  T4 PASS  focus_pct=100 → focus=31 global=1 total={total}, shape {tuple(out.shape)}")

    # ── T5: normal (non-focus) path, equal split across 4 partitions ─────────
    hs = torch.rand(4, 64, D)   # 4 partitions, 64 tokens each = 256 total
    out, total = apply_token_budget(hs, total_budget=128, is_focus=False)
    T = 128 // 4                 # 32 per partition
    assert out.shape == (1, 4 * T, D), f"T5 FAIL shape: {out.shape}"
    assert total == 4 * T,             f"T5 FAIL total: {total}"
    print(f"  T5 PASS  4 partitions + budget=128 → {tuple(out.shape)}, total={total}")

    # ── T6: output dim-1 always == actual_total ───────────────────────────────
    for budget in [16, 32, 48, 64]:
        hs = torch.rand(2, 64, D)
        out, total = apply_token_budget(hs, total_budget=budget, focus_pct=60.0, is_focus=True)
        assert out.shape[1] == total, f"T6 FAIL budget={budget}: shape={out.shape} total={total}"
    print(f"  T6 PASS  shape[1] == actual_total for budgets [16,32,48,64]")

    # ── T7: dtype & device preserved ─────────────────────────────────────────
    hs = torch.rand(2, 64, D).to(torch.float16)
    out, _ = apply_token_budget(hs, total_budget=32, is_focus=True)
    assert out.dtype == torch.float16, f"T7 FAIL dtype: {out.dtype}"
    print(f"  T7 PASS  output dtype preserved ({out.dtype})")

    # ── T8: pooling is a contraction (no value out of [min, max] range) ──────
    hs = torch.rand(2, 64, D)
    out, _ = apply_token_budget(hs, total_budget=32, focus_pct=50, is_focus=True)
    assert out.min() >= hs.min() - 1e-5 and out.max() <= hs.max() + 1e-5, \
        f"T8 FAIL: pooled values outside input range"
    print(f"  T8 PASS  pooled values within input range (avg-pool contraction)")

    print(f"\nAll 8 token-budget tests PASSED")
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
#  STEP 5 — Normal inference (full grid partitioning)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  NORMAL INFERENCE  (full grid partitioning)")
print("─"*60)

inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(DEVICE)

if args.budget > 0:
    inputs, _ = build_budgeted_inputs(
        inputs, prompt_text, is_focus=False,
        total_budget=args.budget, focus_pct=args.focus_pct,
    )

run_eval("Normal" + (f"  [budget={args.budget}]" if args.budget > 0 else ""), inputs)


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
    focus_inputs, _ = build_budgeted_inputs(
        focus_inputs, prompt_text, is_focus=True,
        total_budget=args.budget, focus_pct=args.focus_pct,
    )

run_eval(
    "Focus-point" + (f"  [budget={args.budget}, focus_pct={args.focus_pct:.0f}%]"
                     if args.budget > 0 else ""),
    focus_inputs,
)
