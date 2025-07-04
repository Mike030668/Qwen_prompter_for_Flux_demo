
"""
1) берём текст пользователя → Qwen-prompter → JSON-prompt
2) генерируем 8 изображений Flux
3) сохраняем как grid `samples.png`
"""
#!/usr/bin/env python
# --- stub for accelerate.clear_device_cache (works everywhere) ----------
import importlib; m = importlib.import_module("accelerate.utils.memory")
if not hasattr(m, "clear_device_cache"):
    setattr(m, "clear_device_cache", lambda *a, **k: None)
# ────────────────────────────────────────────────────────────────────────

from ranger_generation.prompter.qwen_prompter import generate_structured_prompt
from ranger_generation.generator.flux_runner import generate_flux
...


from ranger_generation.prompter.qwen_prompter import generate_structured_prompt
from ranger_generation.generator.flux_runner import generate_flux
from PIL import Image
import sys, math

user_text = " ".join(sys.argv[1:]) or input("User prompt → ")

# ── step 1: structured prompt ────────────────────────────────────────────
prompt_json = generate_structured_prompt(user_text)

# OPTIONAL: drop Qwen from GPU
import torch, gc
del generate_structured_prompt.__globals__["model"]
torch.cuda.empty_cache(); gc.collect()

# ── step 2: Flux generation (8 seeds) ────────────────────────────────────
seeds   = [1234 + i for i in range(8)]
samples = generate_flux(prompt_json, seeds)

# ── step 3: save a simple 2×4 grid ───────────────────────────────────────
w, h = samples[0].size
grid  = Image.new("RGB", (w*4, h*2))
for idx, img in enumerate(samples):
    grid.paste(img, ((idx%4)*w, (idx//4)*h))
grid.save("samples.png")
print("Saved → samples.png")
