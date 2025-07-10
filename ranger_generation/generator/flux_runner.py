# ranger_generation/generator/flux_runner.py

import os
import torch
from diffusers import FluxPipeline
from typing import List, Optional
from PIL import Image

# читаем из окружения
HF_TOKEN   = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
sub_name_model = "dev" # "schnell"
FLUX_ID    = os.getenv("FLUX_MODEL_ID", f"black-forest-labs/FLUX.1-{sub_name_model}")

LOCAL_ONLY = os.getenv("FLUX_LOCAL_ONLY", "false").lower() in ("1","true","yes")
DTYPE      = torch.bfloat16 if sub_name_model in FLUX_ID.lower() else torch.float16

_PIPE: Optional[FluxPipeline] = None

def _get_pipe() -> FluxPipeline:
    global _PIPE
    if _PIPE is None:
        init_kwargs = {
            "torch_dtype":      DTYPE,
            "local_files_only": LOCAL_ONLY,
            **({"use_auth_token": HF_TOKEN} if HF_TOKEN and not LOCAL_ONLY else {}),
        }
        _PIPE = FluxPipeline.from_pretrained(FLUX_ID, **init_kwargs)
        _PIPE.enable_model_cpu_offload()
    return _PIPE

def generate_flux(
    positive_prompt: str,
    seeds: List[int],
    num_images: int = 1,
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    scale: float = 7.5,
    negative_prompt: Optional[str] = None,
    true_cfg_scale: float = 1.5,  # >1.0 → жёсткое true-CFG, =1.0 → soft-CFG
) -> List[Image.Image]:
    """
    Запускает FluxPipeline с поддержкой T5 и CLIP, 
    передаёт positive и optional negative prompts.
    """
    pipe = _get_pipe()
    images: List[Image.Image] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for idx, seed in enumerate(seeds[:num_images]):
        gen = torch.Generator(device=device).manual_seed(seed)

        # Базовые аргументы
        call_kwargs = {
            "prompt":              positive_prompt,
            "prompt_2":            positive_prompt,
            "guidance_scale":      scale,
            "true_cfg_scale":      true_cfg_scale,
            "num_inference_steps": steps,
            "height":              height,
            "width":               width,
            "generator":           gen,
            # num_images_per_prompt по умолчанию =1
        }

        # Если есть negative-подсказка — передаём её
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt
            # опционально, можно явно передать и для T5
            call_kwargs["negative_prompt_2"] = negative_prompt

        # Запуск
        out = pipe(**call_kwargs)
        images.append(out.images[0])
        # Очистка VRAM
        torch.cuda.empty_cache()

    return images

def make_grid(images: List[Image.Image]) -> Image.Image:
    """Lay out images in an (almost) square grid."""
    import math
    cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for i, img in enumerate(images):
        x = (i % cols) * w
        y = (i // cols) * h
        grid.paste(img, (x, y))
    return grid