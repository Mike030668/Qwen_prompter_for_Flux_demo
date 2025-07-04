from pathlib import Path
import yaml, itertools

_RULES_ROOT = Path(__file__).resolve().parents[2] / "rules"

def _load_rule(rule_rel_path):
    with open(_RULES_ROOT / rule_rel_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compose_prompt_json(qwen_json, matched_rules):
    out = {
        "positive": qwen_json.get("positive", ""),
        "negative": qwen_json.get("negative", "lowres, blurry, watermark"),
        "params":   qwen_json.get("params", {"width":1024,"height":1536,"cfg":7.5,"steps":4})
    }

    for rule in matched_rules:
        data = _load_rule(rule["id"])
        out["positive"] += ", " + ", ".join(data.get("positive_boost", []))
        out["negative"] += ", " + ", ".join(data.get("negative_boost", []))
        out["params"].update(data.get("params", {}))     # rule > common

    # чистим двойные запятые и пробелы
    out["positive"] = ", ".join(dict.fromkeys(
        s.strip() for s in out["positive"].split(",") if s.strip()))
    out["negative"] = ", ".join(dict.fromkeys(
        s.strip() for s in out["negative"].split(",") if s.strip()))

    return out
