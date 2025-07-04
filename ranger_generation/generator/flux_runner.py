import torch
from pathlib import Path
from typing import List, Union, Dict

# путь к локальному кэшу, если вы уже скачали через HF CLI
_FLUX_CACHE = (
    Path.home() / ".cache" / "huggingface" / "hub"
    / "models--Freepik--flux.1-lite-8B-alpha"
)
FLUX_ID = str(_FLUX_CACHE) if _FLUX_CACHE.exists() else "Freepik/flux.1-lite-8B-alpha"
DTYPE = torch.float16

_pipe = None

def _get_pipe():
    global _pipe
    if _pipe is None:
        from diffusers import FluxPipeline

        # Загружаем с автоматическим сбалансированным offload'ом
        _pipe = FluxPipeline.from_pretrained(
            FLUX_ID,
            torch_dtype=DTYPE,
            device_map="balanced",      # balanced — лучший выбор для CPU↔GPU offload
            local_files_only=True,      # берем из ~/.cache, не качаем заново
        )
        # разбиваем attention на чанки чтобы снизить пиковую нагрузку
        _pipe.enable_attention_slicing()

    return _pipe

def generate_flux(
    prompt: Union[Dict, str],
    seeds: List[int],
    num_images: int = 8,
    width: int = 1024,
    height: int = 1536,
    steps: int = 50,
    scale: float = 7.5,
) -> List:
    """
    Генерируем по одному изображению для каждого seed.
    prompt может быть строкой или JSON с ключом "positive".
    """
    pipe = _get_pipe()

    # извлекаем строку
    if isinstance(prompt, dict):
        positive = prompt.get("positive", "")
    else:
        positive = prompt

    images = []
    for seed in seeds[:num_images]:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        out = pipe(
            positive,
            num_inference_steps=steps,
            guidance_scale=scale,
            width=width,
            height=height,
            generator=gen,
            output_type="pil",
        )
        images.append(out.images[0])
        torch.cuda.empty_cache()

    return images
