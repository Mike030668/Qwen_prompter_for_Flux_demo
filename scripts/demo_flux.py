#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1) из текста делаем JSON-промпт Qwen
2) генерируем N изображений Flux (+LoRA, если указали FLUX_LORA_PATH)
3) сохраняем одиночную картинку или grid
"""

import importlib
# stub для accelerate.clear_device_cache — работает в любой среде
m = importlib.import_module("accelerate.utils.memory")
if not hasattr(m, "clear_device_cache"):
    setattr(m, "clear_device_cache", lambda *a, **k: None)

import sys, math, gc, torch, argparse
from pathlib import Path
from PIL import Image
from ranger_generation.prompter.qwen_prompter import generate_structured_prompt
from ranger_generation.generator.flux_runner import generate_flux


def parse_args():
    ap = argparse.ArgumentParser(
        description="Demo: текст → JSON → Flux (+LoRA) → картинка"
    )
    ap.add_argument("prompt", nargs="*", help="Ваш текст (если не передан, спросим через input)")
    ap.add_argument("-n", "--num_images", type=int, default=1, help="сколько сгенерировать")
    ap.add_argument("-W", "--width",      type=int, default=512, help="ширина")
    ap.add_argument("-H", "--height",     type=int, default=512, help="высота")
    ap.add_argument("-s", "--steps",      type=int, default=30, help="число шагов")
    ap.add_argument("-g", "--scale",      type=float, default=7.5, help="guidance scale")
    ap.add_argument("--seed",             type=int, default=1234, help="стартовый seed")
    ap.add_argument("-o", "--output",     type=str, default="samples.png", help="выходной файл")
    return ap.parse_args()


def main():
    args = parse_args()

    # 1) сформировать текст
    if args.prompt:
        user_text = " ".join(args.prompt)
    else:
        user_text = input("User prompt → ").strip()

    # 2) сделать JSON-промпт через Qwen
    prompt_json = generate_structured_prompt(user_text)

    # выгружаем Qwen-модель
    del generate_structured_prompt.__globals__["model"]
    torch.cuda.empty_cache()
    gc.collect()

    # 3) готовим список seeds и генерируем
    seeds = [args.seed + i for i in range(args.num_images)]
    samples = generate_flux(
        prompt_json["positive"],
        seeds,
        num_images=args.num_images,
        width=args.width,
        height=args.height,
        steps=args.steps,
        scale=args.scale,
    )

    # 4) сохраняем: одиночку или grid
    if args.num_images == 1:
        samples[0].save(args.output)
        print(f"Saved → {args.output}")
    else:
        cols = min(4, args.num_images)
        rows = math.ceil(args.num_images / cols)
        w, h = samples[0].size
        grid = Image.new("RGB", (w * cols, h * rows))
        for idx, img in enumerate(samples):
            grid.paste(img, ((idx % cols) * w, (idx // cols) * h))
        grid.save(args.output)
        print(f"Saved grid → {args.output}")


if __name__ == "__main__":
    main()
