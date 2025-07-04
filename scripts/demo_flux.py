#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1) берём текст пользователя → Qwen-prompter → JSON-prompt
2) генерируем N изображений Flux
3) сохраняем в виде grid или одиночную картинку
"""

# --- stub for accelerate.clear_device_cache (works everywhere) ----------
import importlib
m = importlib.import_module("accelerate.utils.memory")
if not hasattr(m, "clear_device_cache"):
    setattr(m, "clear_device_cache", lambda *a, **k: None)
# ------------------------------------------------------------------------

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
        description="Demo: from user prompt → structured JSON → Flux generation"
    )
    ap.add_argument(
        "prompt", nargs="*", help="User prompt (если не передан, будет запросено через input)"
    )
    ap.add_argument(
        "--num_images", "-n", type=int, default=1,
        help="Сколько изображений сгенерировать (по умолчанию 1)"
    )
    ap.add_argument(
        "--width", "-W", type=int, default=512,
        help="Ширина изображения (по умолчанию 512)"
    )
    ap.add_argument(
        "--height", "-H", type=int, default=512,
        help="Высота изображения (по умолчанию 512)"
    )
    ap.add_argument(
        "--steps", "-s", type=int, default=30,
        help="Число шагов инференса (по умолчанию 30)"
    )
    ap.add_argument(
        "--scale", "-g", type=float, default=7.5,
        help="Guidance scale (по умолчанию 7.5)"
    )
    ap.add_argument(
        "--seed", type=int, default=None,
        help="Начальное зерно для генерации (по умолчанию случайное)"
    )
    ap.add_argument(
        "--output", "-o", default="samples.png",
        help="Имя выходного файла (png или jpg)"
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # собираем пользовательский текст
    if args.prompt:
        user_text = " ".join(args.prompt)
    else:
        user_text = input("User prompt → ").strip()

    # ── step 1: structured prompt JSON
    prompt_json = generate_structured_prompt(user_text)

    # OPTIONAL: разгружаем Qwen с GPU
    del generate_structured_prompt.__globals__["model"]
    torch.cuda.empty_cache()
    gc.collect()

    # ── step 2: Flux generation
    # используем только positive-подстроку
    pos_prompt = prompt_json["positive"]
    samples = generate_flux(
        prompt=pos_prompt,
        num_images=args.num_images,
        width=args.width,
        height=args.height,
        steps=args.steps,
        scale=args.scale,
        seed=args.seed,
    )

    # ── step 3: сохраняем результаты
    # если 1 картинка — просто сохраняем одиночный файл
    if args.num_images == 1:
        samples[0].save(args.output)
        print(f"Saved → {args.output}")
    else:
        # grid: динамически подбираем строки/столбцы
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
