#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo: user prompt → Qwen → Flux → сохранение (1 картинка или grid).
"""

# stub для accelerate.clear_device_cache
import importlib
m = importlib.import_module("accelerate.utils.memory")
if not hasattr(m, "clear_device_cache"):
    setattr(m, "clear_device_cache", lambda *a, **k: None)

import sys
import math
import torch
import gc
import argparse
from PIL import Image

from ranger_generation.prompter.qwen_prompter import generate_structured_prompt
from ranger_generation.generator.flux_runner import generate_flux

def parse_args():
    ap = argparse.ArgumentParser(
        description="Demo: user prompt → structured JSON → Flux generation"
    )
    ap.add_argument("prompt", nargs="*", help="Текстовый запрос")
    ap.add_argument("-n", "--num_images", type=int, default=1, help="Сколько картинок")
    ap.add_argument("-W", "--width",      type=int, default=512, help="Ширина")
    ap.add_argument("-H", "--height",     type=int, default=512, help="Высота")
    ap.add_argument("-s", "--steps",      type=int, default=30, help="Шаги инференса")
    ap.add_argument("-g", "--scale",      type=float, default=7.5, help="Guidance scale")
    ap.add_argument("--seed",             type=int, default=1234, help="Стартовый seed")
    ap.add_argument("-o", "--output",     default="samples.png", help="Выходной файл")
    return ap.parse_args()

def main():
    args = parse_args()

    # формируем user_text
    if args.prompt:
        user_text = " ".join(args.prompt)
    else:
        user_text = input("User prompt → ").strip()

    # 1) Qwen → JSON
    prompt_json = generate_structured_prompt(user_text)

    # разгружаем Qwen-модель
    del generate_structured_prompt.__globals__["model"]
    torch.cuda.empty_cache()
    gc.collect()

    # 2) Flux генерация
    seeds = [args.seed + i for i in range(args.num_images)]
    samples = generate_flux(
        prompt_json,
        seeds,
        num_images=args.num_images,
        width=args.width,
        height=args.height,
        steps=args.steps,
        scale=args.scale,
    )

    # 3) сохранение
    if args.num_images == 1:
        samples[0].save(args.output)
        print(f"Saved → {args.output}")
    else:
        cols = min(4, args.num_images)
        rows = math.ceil(args.num_images / cols)
        w, h = samples[0].size
        grid = Image.new("RGB", (w * cols, h * rows))
        for idx, img in enumerate(samples):
            x = (idx % cols) * w
            y = (idx // cols) * h
            grid.paste(img, (x, y))
        grid.save(args.output)
        print(f"Saved grid → {args.output}")

if __name__ == "__main__":
    main()
