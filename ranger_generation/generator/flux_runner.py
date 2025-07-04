# ranger_generation/generator/flux_runner.py
from __future__ import annotations
import random
from typing import List, Optional
from pathlib import Path
import torch
from diffusers import FluxPipeline, DPMSolverMultistepScheduler
from PIL import Image

# ─── Настройки модели ────────────────────────────────────────────────────────
# ─── путь к уже скачанной модели ─────────────────────────────────────────
LOCAL_FLUX_CACHE = (
    Path.home()
    / ".cache" / "huggingface" / "hub"
    / "models--Freepik--flux.1-lite-8B-alpha"
)
# FLUX_ID = "Freepik/flux.1-lite-8B-alpha"
FLUX_ID = str(LOCAL_FLUX_CACHE) if LOCAL_FLUX_CACHE.exists() else "Freepik/flux.1-lite-8B-alpha"
 
DTYPE   = torch.float16  # bf16 тоже можно

# ─── Загрузка пайплайна один раз при импорте (веса остаются на CPU) ──────────
pipe = FluxPipeline.from_pretrained(
    FLUX_ID,
    torch_dtype=DTYPE,
    local_files_only=True,
    device_map="balanced",  # auto не поддерживается, используем balanced
    scheduler=DPMSolverMultistepScheduler.from_pretrained(
        FLUX_ID,
        subfolder="scheduler"
    )
)

# Переключаем на offload → модель выгружается на CPU, на GPU попадают только нужные блоки
pipe.reset_device_map()
pipe.enable_model_cpu_offload()

# Опционально включаем xformers-ускорение, если есть
try:
    pipe.enable_xformers_memory_efficient_attention()
except ModuleNotFoundError:
    # xformers не установлен — просто пропускаем
    pass


def generate_flux(
    prompt: str,
    num_images: int = 1,
    width: int = 512,
    height: int = 512,
    steps: int = 30,
    scale: float = 7.5,
    seed: Optional[int] = None,
) -> List[Image.Image]:
    """
    Генерирует `num_images` картинок по `prompt`.
    Если seed не указан — используется случайный для каждой.
    Возвращает список PIL.Image.
    """
    results: List[Image.Image] = []
    for i in range(num_images):
        # каждый раз новый генератор, чтобы получить разные картинки
        gen_seed = seed if seed is not None else random.randrange(2**32)
        generator = torch.Generator(device="cuda").manual_seed(gen_seed)

        output = pipe(
            prompt,
            num_images_per_prompt=1,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=generator,
        )
        results.append(output.images[0])

        # очистка GPU-кэша между запросами (опционально)
        torch.cuda.empty_cache()

    return results
