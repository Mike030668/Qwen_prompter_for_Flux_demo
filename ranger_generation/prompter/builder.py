# ranger_generation/prompter/builder.py
from pathlib import Path
import yaml

_RULES_ROOT = Path(__file__).resolve().parents[2] / "rules"

def _load_rule_data(rule_id: str) -> dict:
    cat, name = rule_id.split("/", 1)
    rule_file = _RULES_ROOT / cat / name / f"{name}.yaml"
    with open(rule_file, encoding="utf-8") as f:
        return yaml.safe_load(f)

def compose_prompt_json(user_json: dict, matched_rules: list[dict]) -> dict:
    out: dict = {}
    for k in ("subject", "scene", "style", "lighting", "mood", "positive", "negative"):
        v = user_json.get(k)
        if v is None:
            continue
        out[k] = str(v)

    for rule in matched_rules:
        rd = _load_rule_data(rule["id"])
        for extra in rd.get("positive_boost", []):
            out["positive"] = out.get("positive", "") + f", {extra}"
        for extra in rd.get("negative_boost", []):
            out["negative"] = out.get("negative", "") + f", {extra}"
        for fld, vals in rd.get("field_boost", {}).items():
            if not vals:
                continue
            joined = ", ".join(vals)
            out[fld] = out.get(fld, "") + f", {joined}"

    for k in ("subject", "scene", "style", "lighting", "mood", "positive", "negative"):
        if k not in out:
            continue
        parts = [p.strip() for p in out[k].split(",") if p.strip()]
        out[k] = ", ".join(dict.fromkeys(parts))

    return out


def build_flux_prompts(j: dict) -> tuple[str,str]:
    # Попробуем извлечь действие:
    action = j.get("action","").strip()
    scene  = j.get("scene","").strip()
    # 1) Positive:
    if j.get("positive"):
        p = j["positive"].strip()
    else:
        parts = []
        # если есть действие, вставляем его первым
        if action:
            parts.append(action)
        # далее subject
        if j.get("subject"):
            parts.append(j["subject"].strip())
        # потом именно где–как (scene)
        if scene:
            parts.append(scene)
        # стиль, свет, настроение
        for key in ("style","lighting"):
            if j.get(key):
                parts.append(j[key].strip())
        if j.get("mood"):
            parts.append(f"{j['mood'].strip()} mood")
        p = ", ".join(parts)

    # 2) Negative:
    n = j.get("negative","").strip()

    return p, n

