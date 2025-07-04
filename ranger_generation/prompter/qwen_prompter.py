# ranger_generation/prompter/qwen_prompter.py
# ─────────────────────────────────────────────────────────────
"""
Qwen-Prompter: builds structured prompt JSON.
"""
from __future__ import annotations


# ── ONE-LINE PATCH: add stub to existing module ──────────────────────────
import importlib; m = importlib.import_module("accelerate.utils.memory")
if not hasattr(m, "clear_device_cache"):
    setattr(m, "clear_device_cache", lambda *a, **k: None)
# ───────────────────────────────────────────────────────────────────────────

import json, pathlib, re
from typing import Dict
import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# ① объявляем модель ⬇︎ ДО всяких проверок
#_MODEL_ID = "Qwen/Qwen2-7B-Instruct"   # HF repo
_MODEL_ID = "Qwen/Qwen2-7B-Instruct-GPTQ-Int4"   # HF repo (4-bit)

# дальше идут импорты, зависящие от _MODEL_ID
from transformers import AutoTokenizer
if "GPTQ" in _MODEL_ID.upper():
    from auto_gptq import AutoGPTQForCausalLM
else:
    from transformers import AutoModelForCausalLM
from .parser import detect_rules          # реализуете в parser.py
from .builder import compose_prompt_json   # реализуете в builder.py


_HF_HOME  = pathlib.Path.home() / ".cache" / "huggingface"

# 1. инициализируем модель один раз при первом импорте
tokenizer = AutoTokenizer.from_pretrained(
    _MODEL_ID, trust_remote_code=True
)
# ===== инициализация модели =====
# model = AutoModelForCausalLM.from_pretrained(
#     _MODEL_ID,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     trust_remote_code=True
# )
if "GPTQ" in _MODEL_ID.upper():
    model = AutoGPTQForCausalLM.from_quantized(
        _MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True     # веса уже в .safetensors
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

def generate_structured_prompt(user_text: str) -> Dict:
    """
    Возвращает JSON c полями positive / negative / params,
    собранный по rules/ + «мозги» Qwen2-7B-Instruct.
    """
    # 2. детектируем, какие rules срабатывают
    matched_rules = detect_rules(user_text)
    # 3. готовим system-инструкцию для Qwen
    system_prompt = (
        "You are PromptCraft, a helpful assistant that converts user's request "
        "plus rule hints into a structured JSON with keys positive, negative "
        "and params. Output *ONLY* valid JSON.\n"
        f"Rule hints:\n{json.dumps(matched_rules, ensure_ascii=False, indent=2)}"
    )
    # 4. вызываем модель
    input_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user",   "content": user_text}],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,        # ← ключевое слово!
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.95
        )
    raw = tokenizer.decode(output[0], skip_special_tokens=True)
    # 5. вытаскиваем JSON – первый {...} в ответе
    match = re.search(r"\{.*?\}", raw, re.S)
    if not match:
        raise ValueError("Qwen did not return JSON:\n" + raw)
    prompt_json = json.loads(match.group(0))

    return compose_prompt_json(prompt_json, matched_rules)
