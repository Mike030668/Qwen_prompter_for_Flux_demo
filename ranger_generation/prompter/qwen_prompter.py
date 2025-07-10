from __future__ import annotations
import json, re, os, tempfile
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from .parser import detect_rules
from .builder import compose_prompt_json

# --- модель и процессор VL с ограничением пиков ---
_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    _MODEL_ID,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    trust_remote_code=True,
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    _MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

def _resize_image(path: str, min_dim=64, max_dim=256) -> str:
    """Открывает картинку, ресайзит под диапазон и сохраняет в temp-файл."""
    img = Image.open(path)
    w, h = img.size
    # масштабируем вверх, если слишком мало, вниз — если слишком много
    scale = max(min_dim / w if w < min_dim else 1.0,
                min_dim / h if h < min_dim else 1.0,
                min(max_dim / w if w > max_dim else 1.0,
                    max_dim / h if h > max_dim else 1.0))
    if scale != 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(path).suffix)
    img.save(tmp.name)
    return tmp.name

def generate_structured_prompt(user_text: str, max_per_rule: int = 8) -> dict:
    matched = detect_rules(user_text)
    # 1) собираем визуальные подсказки
    vis_paths: list[str] = []
    RULES_ROOT = Path(__file__).resolve().parents[2] / "rules"
    for rule in matched:
        rule_dir = RULES_ROOT / rule["id"]
        # сначала good, потом bad, но не более max_per_rule
        for kind in ("good", "bad"):
            rel = rule.get("examples", {}).get(kind)
            if not rel:
                continue
            imgs = sorted((rule_dir/rel).iterdir(),
                          key=lambda p: p.name)[: max_per_rule//2]
            for img_path in imgs:
                if img_path.suffix.lower() in (".jpg","jpeg","png","bmp","gif"):
                    resized = _resize_image(str(img_path))
                    vis_paths.append(resized)



    system = """
    You are PromptCraft. Your job is to take the user’s textual (and optional visual) description plus the given rule-hints, and produce *only* one JSON object (no commentary, no markdown) with exactly these keys (in English):
    action    – any verb or pose (e.g. “standing”, “jumping”, “walking”)  
    subject   – main object of the scene  
    scene     – where and how it appears (location, framing, motion)  
    style     – artistic style or resolution  
    lighting  – type and direction of light  
    mood      – emotional tone or atmosphere  
    positive  – a single English prompt string ready for the generator, **preserving the original human phrasing first**, then style, lighting, mood  
    negative  – a single English negative-prompt string ready for the generator  

    **All values must be in English.** Respond *ONLY* with valid JSON, nothing else.

    Example for “Sbercat mascot standing center frame under backlit warm sunlight, photorealistic style”:

    ```json
    {
    "action":   "standing",
    "subject":  "Sbercat mascot",
    "scene":    "center frame under backlit warm sunlight",
    "style":    "photorealistic style",
    "lighting": "backlit warm sunlight",
    "mood":     "warm and inviting",
    "positive": "Sbercat mascot standing center frame under backlit warm sunlight, photorealistic style, warm and inviting mood",
    "negative": "lowres, blurry, watermark, unrealistic colors"
    }

    ⚠️ Respond *only in English*.  
    All keys **and values** must be in English.  
    Output *ONLY* valid JSON (no commentary, no markdown, no extra fields).
    """.strip()
                   

    # 1) Формируем чат
    chat = [{"role":"system","content":system}]
    multimodal = []
    for p in vis_paths:
        multimodal.append({"type":"image","image":p})
    multimodal.append({"type":"text","text": user_text+" (please respond in English only)"})
    chat.append({"role":"user","content": multimodal})

    # 3) токенизация, генерация
    text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    vision_inputs, video_inputs = process_vision_info(chat)
    inputs = processor(
        text=[text],
        images=vision_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=512)

    # 4) чистим temp-файлы
    for p in vis_paths:
        try: os.remove(p)
        except: pass

    raw = tokenizer.batch_decode(
        generated[:, inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    )[0]
    m = re.search(r"\{.*?\}", raw, re.S)
    if not m:
        raise ValueError("Qwen did not return JSON:\n"+raw)
    user_json = json.loads(m.group(0))
    return compose_prompt_json(user_json, matched)
