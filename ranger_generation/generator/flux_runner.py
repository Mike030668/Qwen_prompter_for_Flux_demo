"""
Unified wrapper for Black-Forest FLUX models.
• loads model to CPU once
• moves to GPU only for inference
• adapts batch-size to free VRAM
"""
from __future__ import annotations
import torch, gc
from pathlib import Path
from typing import List, Dict
from diffusers import FluxPipeline, DPMSolverMultistepScheduler

# ─── choose a checkpoint that fits your GPU ──────────────────────────────
#FLUX_ID = "black-forest-labs/FLUX.1-schnell"   # dev ≈14 GB, schnell ≈8 GB, tiny ≈5 GB
FLUX_ID = "Freepik/flux.1-lite-8B-alpha"
DTYPE   = torch.float16                        # bf16 also works

# ─── one-time loading on import (weights stay on CPU) ────────────────────
pipe = FluxPipeline.from_pretrained(
    FLUX_ID,
    torch_dtype=DTYPE,
    #device_map="auto",                         # автоматический CPU-offload
    scheduler=DPMSolverMultistepScheduler.from_pretrained(FLUX_ID, subfolder="scheduler")
)
pipe.enable_model_cpu_offload()          # ↓ keeps GPU clean
pipe.enable_xformers_memory_efficient_attention()

def _gpu_free_mb() -> int:
    free, _ = torch.cuda.mem_get_info()
    return free // 2**20

def generate_flux(prompt: Dict, seeds: List[int]):
    """
    prompt = {positive, negative, params{width,height,cfg,steps}}
    seeds  = [int,…]  — any length, will be split into chunks
    returns list[PIL.Image]
    """
    images, chunk_start = [], 0
    est_per_image = 1200 if "schnell" in FLUX_ID else 1800  # MB heuristic
    while chunk_start < len(seeds):
        batch_sz = max(1, min(4, _gpu_free_mb() // est_per_image))
        batch = seeds[chunk_start:chunk_start + batch_sz]
        gens  = [torch.Generator(device="cuda").manual_seed(s) for s in batch]

        pipe.to("cuda")
        imgs = pipe(
            prompt=prompt["positive"],
            negative_prompt=prompt["negative"],
            width = prompt["params"]["width"],
            height= prompt["params"]["height"],
            guidance_scale = prompt["params"]["cfg"],
            num_inference_steps = prompt["params"]["steps"],
            generator = gens,
        ).images
        pipe.to("cpu"); torch.cuda.empty_cache(); gc.collect()

        images.extend(imgs)
        chunk_start += batch_sz
    return images
