# Range\_Gen\_Image

Набор скриптов и модулей для управления генерацией изображений с помощью Qwen-VL → JSON → Flux (с поддержкой negative prompt, двойного CFG, CLIP+T5).

---

## 📋 Содержание

* [Требования](#-требования)
* [Установка](#-установка)

  * [Локально](#локально)
  * [RunPod](#runpod)
* [Аутентификация HF](#-аутентификация-hf)
* [Примеры использования](#-примеры-использования)
* [Структура проекта](#-структура-проекта)
* [Дальнейшее развитие](#-дальнейшее-развитие)

---

## 🔧 Требования

* Python **3.8–3.11**
* CUDA **11.8** (рекомендуется для GPU-ускорения)
* Git

---

## ⚙️ Установка

### Локально

```bash
git clone https://github.com/Mike030668/ranger_generation.git
cd ranger_generation
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
.\.venv\Scripts\activate         # Windows
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### RunPod

```bash
git clone https://github.com/Mike030668/ranger_generation.git
cd ranger_generation
conda create -n flux_gen python=3.11 -y
conda activate flux_gen
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

> **Важно:** если используется CUDA не 11.8, замените URL в `requirements.txt` на соответствующий.

---

## 🔒 Аутентификация HF

```bash
huggingface-cli login
```

---

## 🚀 Примеры использования

1. **Prompt with Qwen**

   ```bash
   python -m scripts.demo_qwen \
     --prompt "A front-view portrait of a vintage car in cinematic color grading during golden hour" \
     --output outputs/qwen_prompt.json
   ```

   *Выводит JSON-структуру с полями ****************`positive`****************, ****************`negative`**************** и деталями.*

2. **Чистый Flux**

   ```bash
   python -m scripts.demo_flux \
     --prompt "A front-view portrait of a vintage car in cinematic color grading during golden hour" \
     --num_images 4 \
     --width 512 --height 512 \
     --steps 20 \
     --scale 3.0 \
     --seeds 42 100 256 \
     --output outputs/flux_only
   ```

3. **Qwen → Flux**

   ```bash
   python -m scripts.demo_qwen_flux \
     --prompt "A front-view portrait of a vintage car in cinematic color grading during golden hou" \
     --use_qwen \
     --no_negative \
     --num_images 4 \
     --width 512 --height 512 \
     --steps 20 \
     --scale 3.0 \
     --seeds 42 100 256 \
     --output outputs/qwen_flux
   ```

4. **Qwen → Flux (+ negative prompt)**

   ```bash
   python -m scripts.demo_qwen_flux \
     --prompt "A front-view portrait of a vintage car in cinematic color grading during golden hou" \
     --use_qwen \
     --use_negative \
     --num_images 4 \
     --width 512 --height 512 \
     --steps 20 \
     --scale 3.0 \
     --seeds 42 100 256 \
     --output outputs/qwen_flux_neg
   ```

> Во 2–4 вариантах флаг `--seeds` позволяет передавать список семян для параллельной генерации.

---

## 📂 Структура проекта

```
ranger_generation/
├── ranger_generation/       # Пакет с основными модулями
│   ├── prompter/            # Парсер и сборщик JSON-промтов
│   └── generator/           # FluxRunner с CLIP+T5, CFG, negative
├── scripts/
│   ├── demo_qwen.py         # Вывод JSON через Qwen
│   ├── demo_flux.py         # Запуск чистого Flux
│   └── demo_qwen_flux.py    # Запуск Qwen → Flux (+ negative)
├── tests/                   # (в разработке) автотесты
├── rules/                   # YAML-файлы правил (common, lighting, objects…)
├── requirements.txt         # Все зависимости, включая PyTorch+CUDA
├── setup.py                 # Editable-модуль
└── README.md                # Вы читаете :)
```

---

## 🛠 Дальнейшее развитие

* Сбор и аналитика метрик качества без/с Qwen и negative
* Интеграция SigLIP-2 для ранжирования
* REST/GRPC-обёртка (FastAPI) для продакшена
* Покрытие тестами парсера правил, сборщика промтов и FluxRunner

---
