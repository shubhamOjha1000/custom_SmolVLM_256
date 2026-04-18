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
parser.add_argument("--crop-pct",    default=25.0,  type=float,
                    help="Local crop size as %% of original image area (default: 25.0)")
parser.add_argument("--focus-only",  action="store_true",
                    help="Pass only the focus crop partition to the encoder, skip global image (encoder shape: (1,1024,768))")
parser.add_argument("--show-partitions", action="store_true",
                    help="Display the two focus partitions (local crop + global) as images")
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
#  PARTITION VISUALISER
# ══════════════════════════════════════════════════════════════════════════════

def show_partitions(image, focus_point):
    """
    Save the two focus partitions as PNG files to /content/.
    Re-runs preprocess with do_normalize=False to get raw [0,1] pixels.
    """
    import numpy as np

    vis = processor.image_processor.preprocess(
        [image], return_tensors="pt", focus_point=focus_point,
        focus_crop_pct=args.crop_pct, do_normalize=False
    )
    pv = vis["pixel_values"]   # (1, 2, C, H, W), values in [0, 1]

    labels = ["partition0_focus_crop", "partition1_global"]
    for i, label in enumerate(labels):
        t = pv[0, i].float().cpu()                                        # (C, H, W)
        img_np = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")      # (H, W, C)
        save_path = f"/content/{label}.png"
        Image.fromarray(img_np).save(save_path)
        print(f"[partitions] saved → {save_path}")


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
run_eval("Normal", inputs)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Focus-point inference (local crop + global = 2 partitions)
# ══════════════════════════════════════════════════════════════════════════════
focus_mode = "focus-only" if args.focus_only else "focus+global"
print("\n" + "─"*60)
print(f"  FOCUS-POINT INFERENCE  ({focus_mode})")
print("─"*60)

FOCUS_POINT = tuple(float(v) for v in args.focus.split(","))

raw_inputs = processor.image_processor.preprocess(
    [image], return_tensors="pt", focus_point=FOCUS_POINT, focus_crop_pct=args.crop_pct
)
print(f"[focus] pixel_values shape: {tuple(raw_inputs['pixel_values'].shape)}")

if args.show_partitions:
    show_partitions(image, FOCUS_POINT)

if args.focus_only:
    # Keep only partition 0 (focus crop), drop partition 1 (global)
    raw_inputs["pixel_values"] = raw_inputs["pixel_values"][:, :1]   # (1,1,3,512,512)
    if "pixel_attention_mask" in raw_inputs:
        raw_inputs["pixel_attention_mask"] = raw_inputs["pixel_attention_mask"][:, :1]
    print(f"[focus-only] pixel_values trimmed to: {tuple(raw_inputs['pixel_values'].shape)}")
    # 1 partition → no sub-grid rows/cols
    focus_prompt_text = processor.expand_text_with_image_tokens(
        [prompt_text], image_rows=[[0]], image_cols=[[0]]
    )[0]
else:
    # Both partitions: focus crop (sub-tile 1×1) + global
    focus_prompt_text = processor.expand_text_with_image_tokens(
        [prompt_text], image_rows=[[1]], image_cols=[[1]]
    )[0]

text_inputs = processor.tokenizer(focus_prompt_text, return_tensors="pt")
focus_inputs = {**raw_inputs, **text_inputs}
focus_inputs = {k: v.to(DEVICE) for k, v in focus_inputs.items()}

run_eval(f"Focus-point [{focus_mode}]", focus_inputs)
