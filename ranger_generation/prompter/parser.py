# ranger_generation/prompter/parser.py  (или detect_rules.py)
import re
from pathlib import Path
from typing import List, Dict

# --- ВСТАВЛЯЕМ СЮДА ---
_SYNONYM_INDEX = {
    # objects
    r"\bсберкот\b|\bsbercat\b": "objects/sbercat/sbercat.yaml",

    # viewpoint
    r"\bанфас\b|\bfront view\b|\bhead-?on\b": "viewpoint/front_view/front_view.yaml",

    # lighting
    r"\bзакат\w*|\bgolden hour\b": "lighting/golden_hour/golden_hour.yaml",

    # style
    r"\bкинематограф\w*|\bкиношн\w*|\bcinematic\b": "style/cinematic_photo/cinematic_photo.yaml",
}

def detect_rules(user_text: str) -> List[Dict]:
    """
    Возвращает список словарей вида {"id": "path/to/rule.yaml"}
    для всех сработавших regexp.
    """
    hits = []
    for pattern, path in _SYNONYM_INDEX.items():
        if re.search(pattern, user_text, re.I):
            hits.append({"id": path})
    return hits
