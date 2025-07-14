# ranger_generation/prompter/parser.py

import logging
from pathlib import Path
import yaml

log = logging.getLogger(__name__)

# поправил путь к корню правил:
_RULES_ROOT = Path(__file__).resolve().parents[3] / "ranger_generation" / "rules"

def detect_rules(user_text: str) -> list[dict]:
    text = user_text.lower()
    matches: list[dict] = []

    log.info(f"Looking for rules in `{_RULES_ROOT}`")
    if not _RULES_ROOT.exists():
        log.error(f"Rules directory not found: {_RULES_ROOT}")
        return matches

    for cat in sorted(_RULES_ROOT.iterdir()):
        if not cat.is_dir(): 
            continue
        for rule_dir in sorted(cat.iterdir()):
            if not rule_dir.is_dir(): 
                continue
            rule_file = rule_dir / f"{rule_dir.name}.yaml"
            if not rule_file.exists():
                continue

            rule = yaml.safe_load(rule_file.open(encoding="utf-8"))

            # объединяем patterns и synonyms
            candidates = []
            candidates += rule.get("patterns", [])
            candidates += rule.get("synonyms", [])
            # если ни того, ни другого — пропускаем
            if not candidates:
                continue

            # ищем первое совпадение
            for token in candidates:
                if not isinstance(token, str):
                    continue
                if token.lower() in text:
                    # соберём картинки из examples
                    imgs: list[str] = []
                    for kind in ("good", "bad"):
                        rel = rule.get("examples", {}).get(kind)
                        if not rel:
                            continue
                        dirp = rule_dir / rel
                        if not dirp.is_dir():
                            continue
                        for img in sorted(dirp.iterdir()):
                            if img.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".gif"):
                                imgs.append(str(img.resolve()))

                    log.info(f"  ↳ matched `{cat.name}/{rule_dir.name}`, images: {len(imgs)} files")
                    matches.append({
                        "id": f"{cat.name}/{rule_dir.name}",
                        "images": imgs
                    })
                    break  # больше не ищем внутри этого rule_dir

    if not matches:
        log.info("  (no rules matched)")
    return matches

