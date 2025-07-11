#!/usr/bin/env python3
"""
Batch image generation script supporting Qwen preprocessing and negative prompts.
Saves images and a single JSON prompts file to the specified output directory.
"""
import json
import argparse
from pathlib import Path
from ranger_generation.generator.flux_runner import generate_flux
from ranger_generation.prompter.qwen_prompter import generate_structured_prompt
from ranger_generation.prompter.builder import build_flux_prompts

def parse_args():
    ap = argparse.ArgumentParser(
        description="Batch generate images with optional Qwen preprocessing"
    )
    ap.add_argument("prompt", nargs="*", help="User prompt text")
    ap.add_argument("--num_images", "-n", type=int, default=1,
                    help="Number of images to generate (ignored if --seeds provided)")
    ap.add_argument("--width", "-W", type=int, default=512, help="Image width")
    ap.add_argument("--height", "-H", type=int, default=512, help="Image height")
    ap.add_argument("--steps", "-s", type=int, default=20, help="Inference steps")
    ap.add_argument("--scale", "-g", type=float, default=7.5, help="Guidance scale")
    ap.add_argument("--seed", type=int, default=42, help="Base seed")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated list of seeds (overrides --seed and --num_images)")
    ap.add_argument("--use_qwen", action="store_true",
                    help="Preprocess prompt via Qwen → JSON")
    ap.add_argument("--use_negative", action="store_true",
                    help="Include Qwen-generated negative prompt (implies --use_qwen)")
    ap.add_argument("--output_dir", "-o", default="out_batch",
                    help="Directory to save images and prompt JSON")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # assemble raw prompt
    raw = " ".join(args.prompt) if args.prompt else input("Enter prompt: ").strip()
    if not raw:
        raise SystemExit("Prompt must not be empty")

    # ensure use_qwen if negative requested
    if args.use_negative and not args.use_qwen:
        print("[!] --use_negative implies --use_qwen; enabling it")
        args.use_qwen = True

    # build seeds list
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
        if len(seeds) != args.num_images:
            print(f"[!] --seeds list length {len(seeds)} != num_images {args.num_images}; using seeds list and overriding num_images.")
        args.num_images = len(seeds)
    else:
        seeds = [args.seed + i for i in range(args.num_images)]

    # Qwen preprocessing
    if args.use_qwen:
        print("→ querying Qwen for structured JSON…")
        struct = generate_structured_prompt(raw)
        positive, negative = build_flux_prompts(struct)
        print("Qwen JSON:\n", json.dumps(struct, ensure_ascii=False, indent=2))
    else:
        positive = raw
        negative = None
    if not args.use_negative:
        negative = None

    # prepare output directory
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # save single prompts JSON for this batch
    prompt_data = {
        "prompt": positive,
        "prompt2": positive,
        "negative": negative or "",
        "negative2": negative or ""
    }
    json_path = outdir / "prompts.json"
    json_path.write_text(json.dumps(prompt_data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"✅ Saved prompts JSON to {json_path}")

    # generate images
    print(f"→ Generating {args.num_images} images with seeds: {seeds}")
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

    # save image files
    for seed, img in zip(seeds, images):
        img_path = outdir / f"{seed}.png"
        img.save(img_path)
        print(f"✅ Saved image for seed {seed}")

    print(f"All outputs saved to {outdir.resolve()}")
