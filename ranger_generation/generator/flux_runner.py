import os
import torch
from pathlib import Path
from typing import List, Dict

# где лежат локальные веса FLUX (если вы их закачали вручную)
_FLUX_CACHE = (
    Path.home() / ".cache" / "huggingface" / "hub"
    / "models--Freepik--flux.1-lite-8B-alpha"
)
FLUX_ID = str(_FLUX_CACHE) if _FLUX_CACHE.exists() else "Freepik/flux.1-lite-8B-alpha"
DTYPE   = torch.float16

# необязательная переменная окружения с путём до вашего LoRA-адаптера
LORA_PATH = os.getenv("FLUX_LORA_PATH", None)

_pipe = None

def _get_pipe():
    global _pipe
    if _pipe is None:
        from diffusers import FluxPipeline, DPMSolverMultistepScheduler

        # загружаем scheduler из кэша
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            FLUX_ID, subfolder="scheduler", local_files_only=True
        )

        # создаём пайплайн с offload-стратегией balanced
        _pipe = FluxPipeline.from_pretrained(
            FLUX_ID,
            torch_dtype=DTYPE,
            scheduler=scheduler,
            device_map="balanced",         # <=== balanced offload
            local_files_only=True,
        ).to("cuda")

        # если указали LoRA-вес, подгружаем его
        if LORA_PATH:
            _pipe.load_lora_weights(LORA_PATH, adapter_name="default")

    return _pipe


def generate_flux(
    prompt: Dict | str,
    seeds: List[int],
    num_images: int = 8,
    width: int      = 1024,
    height: int     = 1536,
    steps: int      = 50,
    scale: float    = 7.5,
) -> List:
    """
    Генерируем картинки по одному seed'у за раз, выгружаем память между запусками.
    """
    pipe = _get_pipe()

    if isinstance(prompt, dict):
        positive = prompt["positive"]
        negative = prompt["negative"]
    else:
        positive = prompt
        negative = None

    images = []
    for seed in seeds[:num_images]:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        out = pipe(
            positive,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=scale,
            width=width,
            height=height,
            generator=gen,
        )
        images.append(out.images[0])

        # чистим кеш между картинками, чтобы не “убить” GPU-память
        torch.cuda.empty_cache()

    return images
