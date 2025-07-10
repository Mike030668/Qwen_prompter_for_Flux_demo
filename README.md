
# ranger_generation

End-to-end pipeline:

1. **Qwen-Prompter** 👉 generates structured JSON (positive / negative / params)
2. **Flux 1** 👉 text-to-image (multiple seeds, dynamic batching)
3. **SigLIP-2** 👉 scores & ranks images (coming next)

---

## 1. Quick install (CUDA 12.1 GPU)

```bash
python -m venv .venv && source .venv/bin/activate          # or use conda/mamba
uv pip install --upgrade pip                                # optional but fast
uv pip install -r requirements.txt
````

> • For inference on CPU only, drop the `+cu121` wheels and install plain
> `torch==2.3.0 torchvision==0.18.0` (much slower).
> • Tested on Ubuntu 22.04, Py 3.10.18, RTX 4090 + driver 576, CUDA 12.9.

---

## 2. Hugging Face authentication

Flux checkpoints are gated.
Create a **read** token on [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and run:

```bash
huggingface-cli login
# or
export HUGGINGFACE_HUB_TOKEN=hf_********************************
```

---

## 3. First-run demo

```bash
python scripts/demo_prompt.py "Хочу Сберкота анфас..."
python scripts/demo_flux.py   "Хочу Сберкота анфас..."
xdg-open samples.png          # (or open in your OS)
```

The first call downloads ≈ 10 GB of Flux weights; afterwards each run is instant.

---

## 4. Using LoRA / QLoRA adapters with Flux 1

```bash
# ① Place adapter folder inside ./checkpoints or any path
export LORA_PATH=/path/to/my_cinematic_lora

# ② Enable LoRA loading in flux_runner.py
from diffusers.loaders import FluxLoraLoaderMixin
pipe = FluxPipeline.from_pretrained(
    FLUX_ID,
    torch_dtype=torch.float16,
    lora_loader=FluxLoraLoaderMixin.from_pretrained(LORA_PATH),
    ...
)
```

*If you need to **train** a new adapter:*

```bash
# example: DreamBooth-style fine-tune
accelerate launch train_lora.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.1-schnell \
  --instance_data_dir ./my_dataset --output_dir ./my_lora \
  --resolution 1024 --train_text_encoder --mixed_precision fp16
```

---

## 5. Known issues & patches

| Symptom                               | Fix                                                                                                                                                     |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ImportError: clear_device_cache`     | Patched at runtime: the first lines of `ranger_generation/prompter/qwen_prompter.py` add a stub to `accelerate.utils.memory` for any accelerate ≥ 0.25. |
| `xFormers can't load C++/CUDA`        | Rebuild xformers against your Torch (see above) *or* ignore (pipeline works without memory-efficient attention).                                        |
| AutoGPTQ “CUDA kernels not installed” | Optional speed-up; compile with `python -m auto_gptq.cuda_setup install` if desired.                                                                    |

---

*Happy generating!*

```


## 6 Как подключить ваш LoRA-адаптер
Подготовьте или обучите LoRA-веса (например, через ai-toolkit и train_lora_flux_24gb.yaml).

Положите my_lora.safetensors куда угодно на диск.

Перед запуском демо-скрипта установите переменную окружения:

bash
Копировать
Редактировать
export FLUX_LORA_PATH=/путь/к/my_lora.safetensors
Запускайте:

bash
Копировать
Редактировать
python scripts/demo_flux.py "Ваш запрос" \
  --num_images 4 --width 512 --height 512 --steps 15 --scale 7.5 \
  --output my_out.png
— в результате LoRA-адаптер автоматически загрузится в FluxPipeline.


---

### Next steps

* When SigLIP-2 ranking is ready, add its runtime dependency (`siglip2` or custom scorer) and update the README.  
* Consider wrapping `scripts/demo_flux.py` into a FastAPI endpoint for easy integration.

Let me know if you’d like any tweaks!
```
