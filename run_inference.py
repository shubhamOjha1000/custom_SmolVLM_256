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
import importlib
import time

# ── Step 1: Find where transformers is installed ──────────────────────────────
import transformers
transformers_root = os.path.dirname(transformers.__file__)
smolvlm_dir = os.path.join(transformers_root, "models", "smolvlm")
print(f"[setup] transformers installed at: {transformers_root}")
print(f"[setup] SmolVLM module directory : {smolvlm_dir}")

# Guard: smolvlm_dir must exist — requires transformers >= 4.48
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

# Clear __pycache__ so Python doesn't load stale .pyc bytecode instead of our
# freshly copied .py files — without this, print statements and code changes
# are silently ignored.
pycache_dir = os.path.join(smolvlm_dir, "__pycache__")
if os.path.isdir(pycache_dir):
    shutil.rmtree(pycache_dir)
    print(f"  cleared __pycache__ at {pycache_dir}")

# ── Step 3: Force-reload transformers so Python picks up the patched files ────
# Remove all cached smolvlm modules
mods_to_reload = [k for k in sys.modules if "smolvlm" in k.lower()]
for mod in mods_to_reload:
    del sys.modules[mod]

# Re-import the submodule — now reads from our patched .py files
import transformers.models.smolvlm  # triggers re-import after cache clear

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
MAX_NEW_TOKENS = 100

print(f"[inference] Loading model from: {MODEL_ID}")
print(f"[inference] Device: {DEVICE}\n")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = SmolVLMForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    _attn_implementation="eager",
).to(DEVICE)

# ── Replace this with your actual image path ──────────────────────────────────
IMAGE_PATH = "/content/drive/MyDrive/Test_imgage_folder/img1.png"   # <-- change this
PROMPT = "What is the person holding"

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


# ── Eval helper ───────────────────────────────────────────────────────────────
class _FirstTokenTimer(StoppingCriteria):
    """Records wall-clock time the moment the first new token is produced."""
    def __init__(self):
        self.t = None
    def __call__(self, input_ids, scores, **kwargs):
        if self.t is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.t = time.perf_counter()
        return False  # never stops generation early


def run_eval(label, inputs):
    """
    Run model.generate() and print four metrics:
      1. Visual tokens  — num_partitions × image_seq_len
      2. TTFT           — ms from generate() start to first new token
      3. Throughput     — new tokens / total generation time (tok/s)
      4. Peak GPU mem   — MB allocated during generation
    Returns decoded output string.
    """
    pv = inputs.get("pixel_values")
    num_partitions  = pv.shape[1] if pv is not None else 0
    image_seq_len   = model.model.image_seq_len
    num_visual_tokens = num_partitions * image_seq_len

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

    total_time   = t_end - t_start
    ttft_ms      = (timer.t - t_start) * 1000 if timer.t else float("nan")
    input_len    = inputs["input_ids"].shape[1]
    new_tokens   = generated_ids.shape[1] - input_len
    throughput   = new_tokens / total_time
    peak_mem_mb  = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    print(f"\n{'='*56}")
    print(f"  EVAL — {label}")
    print(f"{'='*56}")
    print(f"  1. Visual tokens       {num_visual_tokens:>6}  "
          f"({num_partitions} partition(s) × {image_seq_len} seq_len)")
    print(f"  2. TTFT                {ttft_ms:>6.1f} ms")
    print(f"  3. Throughput          {throughput:>6.1f} tok/s  "
          f"({new_tokens} tokens in {total_time:.2f}s)")
    print(f"  4. Peak GPU memory     {peak_mem_mb:>6.1f} MB")
    print(f"{'='*56}")

    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"\n[output]\n{output}")
    return output


# ── Step 5: Normal inference (full grid partitioning) ─────────────────────────
print("\n" + "─"*56)
print("  NORMAL INFERENCE  (full grid partitioning)")
print("─"*56)
inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(DEVICE)
run_eval("Normal (grid partitioning)", inputs)


# ── Step 6: Focus-point inference (2 partitions: local crop + global) ─────────
print("\n" + "─"*56)
print("  FOCUS-POINT INFERENCE  (local crop + global = 2 tiles)")
print("─"*56)

FOCUS_POINT = (0.5, 0.5)   # normalised (x, y) in [0,1]  — change as needed

raw_inputs = processor.image_processor.preprocess(
    [image], return_tensors="pt", focus_point=FOCUS_POINT
)
print(f"[focus] pixel_values shape: {tuple(raw_inputs['pixel_values'].shape)}")

# Expand text to match 2-partition layout (1×1 local crop + global)
focus_prompt_text = processor.expand_text_with_image_tokens(
    [prompt_text], image_rows=[[1]], image_cols=[[1]]
)[0]
text_inputs = processor.tokenizer(focus_prompt_text, return_tensors="pt")
focus_inputs = {**raw_inputs, **text_inputs}
focus_inputs = {k: v.to(DEVICE) for k, v in focus_inputs.items()}

run_eval("Focus-point (local crop + global)", focus_inputs)
