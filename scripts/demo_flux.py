#!/usr/bin/env python3
import os
import argparse
import math
import torch
from PIL import Image
from diffusers import FluxPipeline

# get sub_name used model Flux
#from ranger_generation.generator.flux_runner import SUM_NAME
SUM_NAME = "dev"
def parse_args():
    ap = argparse.ArgumentParser(
        description="Demo: from user prompt → Flux generation"
    )
    ap.add_argument(
        "prompt", nargs="*", help="User prompt (если не передан, будет запрошен интерактивно)"
    )
    ap.add_argument(
        "--num_images", "-n", type=int, default=1,
        help="Сколько изображений сгенерировать"
    )
    ap.add_argument(
        "--width", "-W", type=int, default=512,
        help="Ширина (px)"
    )
    ap.add_argument(
        "--height", "-H", type=int, default=512,
        help="Высота (px)"
    )
    ap.add_argument(
        "--steps", "-s", type=int, default=4,
        help="Шагов инференса"
    )
    ap.add_argument(
        "--scale", "-g", type=float, default=7.5,
        help="Guidance scale"
    )
    ap.add_argument("--seed", type=int, default=1234, help="Base seed")
    
    ap.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated list of seeds; overrides --seed"
    )
    ap.add_argument(
        "--output", "-o", default="samples.png",
        help="Имя выходного файла (PNG)"
    )
    return ap.parse_args()

# Read from env
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
FLUX_ID   = os.getenv("FLUX_MODEL_ID", f"black-forest-labs/FLUX.1-{SUM_NAME}")
LOCAL_ONLY= os.getenv("FLUX_LOCAL_ONLY", "false").lower() in ("1","true","yes")
DTYPE     = torch.bfloat16 if SUM_NAME in FLUX_ID.lower() else torch.float16


_PIPE: FluxPipeline | None = None
def _get_pipe() -> FluxPipeline:
    global _PIPE
    if _PIPE is None:
        init_kwargs = {
            "torch_dtype": DTYPE,
            "local_files_only": LOCAL_ONLY,
            **({"use_auth_token": HF_TOKEN} if HF_TOKEN and not LOCAL_ONLY else {})
        }
        _PIPE = FluxPipeline.from_pretrained(FLUX_ID, **init_kwargs)
        _PIPE.enable_model_cpu_offload()
    return _PIPE

def make_grid(images: list[Image.Image]) -> Image.Image:
    """Lay out images in an (almost) square grid."""
    count = len(images)
    cols  = int(math.ceil(math.sqrt(count)))
    rows  = int(math.ceil(count / cols))
    w, h  = images[0].size
    grid  = Image.new("RGB", (cols * w, rows * h))
    for idx, img in enumerate(images):
        x = (idx % cols) * w
        y = (idx // cols) * h
        grid.paste(img, (x, y))
    return grid

def main():
    args = parse_args()

    # build seeds list
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        if len(seeds) < args.num_images:
            raise ValueError(f"You passed {len(seeds)} seeds but asked for {args.num_images} images.")
    else:
        seeds = [args.seed + i for i in range(args.num_images)]

    # assemble prompt
    if args.prompt:
        prompt = " ".join(args.prompt)
    else:
        prompt = input("Enter prompt: ").strip()
        if not prompt:
            print("Prompt must not be empty.")
            return

    pipe = _get_pipe()

    images = []
    for seed in seeds[: args.num_images]:
        # create a torch.Generator with this seed
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device).manual_seed(seed)

        out = pipe(
            prompt,
            guidance_scale=args.scale,
            num_inference_steps=args.steps,
            width=args.width,
            height=args.height,
            generator=gen
        )
        images.append(out.images[0])
        # free up GPU between runs
        torch.cuda.empty_cache()

    # save
    if len(images) > 1:
        grid = make_grid(images)
        grid.save(args.output)
        print(f"Saved grid → {args.output}")
    else:
        images[0].save(args.output)
        print(f"Saved image → {args.output}")

if __name__ == "__main__":
    main()
