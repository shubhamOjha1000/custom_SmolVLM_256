"""
Custom SmolVLM2-256M Inference Script
======================================
These model files use relative imports (from ...utils import ...) because
they are designed to live INSIDE the transformers package directory.

Strategy: install transformers, then overwrite the smolvlm files inside
the installed package with our custom versions. This way all relative
imports resolve correctly and our changes are reflected.

Run on Colab:
    !pip install transformers accelerate Pillow -q
    !python run_inference.py
"""

import os
import sys
import shutil
import importlib

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

# ── Step 4: Normal HuggingFace inference ──────────────────────────────────────
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Instruct"

print(f"[inference] Loading model from: {MODEL_ID}")
print(f"[inference] Device: {DEVICE}\n")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = SmolVLMForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    _attn_implementation="eager",
).to(DEVICE)

# ── Replace this with your actual image path ──────────────────────────────────
IMAGE_PATH = "statue_of_liberty.jpg"   # <-- change this
PROMPT = "Describe this image."

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
inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(DEVICE)

print("[inference] Running generation...")
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=100)

output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"\n[output]\n{output}")
