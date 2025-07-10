#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from IPython.display import display, Image as IPyImage

from ranger_generation.generator.flux_runner import generate_flux, make_grid
from ranger_generation.prompter.qwen_prompter import generate_structured_prompt
from ranger_generation.prompter.builder import build_flux_prompts

def parse_args():
    ap = argparse.ArgumentParser(
        description="Demo: [Qwen → JSON] → [Flux generation]"
    )
    ap.add_argument("prompt", nargs="*", help="User prompt")
    ap.add_argument("--num_images","-n", type=int, default=1,
                    help="How many images to generate")
    ap.add_argument("--width","-W", type=int, default=512, help="Width in px")
    ap.add_argument("--height","-H", type=int, default=512, help="Height in px")
    ap.add_argument("--steps","-s", type=int, default=4, help="Inference steps")
    ap.add_argument("--scale","-g", type=float, default=7.5, help="Guidance scale")
    ap.add_argument("--seed", type=int, default=1234, help="Base seed")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seed list; overrides --seed")
    ap.add_argument("--use_qwen", action="store_true",
                    help="Preprocess prompt through Qwen→JSON")
    ap.add_argument("--use_negative", action="store_true",
                    help="Use Qwen’s negative prompt (implies --use_qwen)")
    ap.add_argument("--output","-o", default="out_qwen_flux.png",
                    help="Output filename (PNG)")
    return ap.parse_args()

def main():
    args = parse_args()

    # build seeds list
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        if len(seeds) < args.num_images:
            raise ValueError(f"Provided {len(seeds)} seeds but asked for {args.num_images} images")
    else:
        seeds = [args.seed + i for i in range(args.num_images)]

    # assemble raw prompt
    raw = " ".join(args.prompt) if args.prompt else input("Enter prompt: ").strip()
    if not raw:
        raise SystemExit("Prompt must not be empty")

    # force use_qwen if negative requested
    if args.use_negative and not args.use_qwen:
        print("[!] --use_negative implies --use_qwen; enabling it")
        args.use_qwen = True

    if args.use_qwen:
        print("→ querying Qwen for structured JSON …")
        struct = generate_structured_prompt(raw)
        # struct now has 'positive','negative', etc.
        positive, negative = build_flux_prompts(struct)
        
        # show the JSON for debug
        print("Qwen JSON:\n", json.dumps(struct, ensure_ascii=False, indent=2))
    else:
        positive = raw
        negative = None

    # if they didn’t request negative, drop it
    if not args.use_negative:
        negative = None

    # run Flux
    print(f"→ running Flux with positive:\n   {positive!r}")
    if negative:
        print(f"                 negative:\n   {negative!r}")

    images = generate_flux(
        positive_prompt=positive,
        negative_prompt=negative,
        seeds=seeds,
        num_images=args.num_images,
        width=args.width,
        height=args.height,
        steps=args.steps,
        scale=args.scale,
    )

    # save
    from PIL import Image as PILImage
    if len(images) > 1:
        grid = make_grid(images)
        grid.save(args.output)
    else:
        images[0].save(args.output)

    print(f"✅ Saved → {args.output}")

    # show in notebook
    try:
        display(IPyImage(args.output))
    except Exception:
        pass

if __name__ == "__main__":
    main()
