ranger_generation/
    ├─ prompter/
    │       ├─ qwen_prompter.py
    ├─ scripts/
    │       ├─ validate_rules.py
    │       ├─ demo_prompt.py
    └─ rules/
        ├─ objects/                # контролируемые объекты (люди, маскоты, бренды…)
        │   ├─ sbercat/
        │   │   ├─ sbercat.yaml
        │   │   └─ images/
        │   │       ├─ good/
        │   │       │   ├─ 1.png
        │   │       │   └─ 2.png
        │   │       └─ bad/
        │   │           ├─ 1.png
        │   │           └─ 2.png
        │   └─ …/
        │
        ├─ viewpoint/              # ракурсы / камера
        │   ├─ front_view/
        │   │   ├─ front_view.yaml
        │   │   └─ images/
        │   │       ├─ good/
        │   │       │   ├─ 1.png
        │   │       │   ├─ 2.jpg
        │   │       │   └─ ....
        │   │       └─ bad/
        │   │           ├─ 1.png
        │   │           ├─ 2.jpg
        │   │           └─ ....
        │   └─ top_down/…
        │
        ├─ lighting/               # освещение
        │   ├─ golden_hour/
        │   │   ├─ golden_hour.yaml
        │   │   └─ images/
        │   │       ├─ good/
        │   │       │   ├─ 1.png
        │   │       │   ├─ 2.jpg
        │   │       │   └─ ....
        │   │       └─ bad/
        │   │           ├─ 1.png
        │   │           ├─ 2.jpg
        │   │           └─ ....
        │   └─ studio_softbox/…
        │
        ├─ style/                  # жанр / визуальный стиль
        │   ├─ cinematic_photo/
        │   │   ├─ cinematic_photo.yaml
        │   │   └─ images/
        │   │       ├─ good/
        │   │       │   ├─ 1.png
        │   │       │   ├─ 2.jpg
        │   │       │   └─ ....
        │   │       └─ bad/
        │   │           ├─ 1.png
        │   │           ├─ 2.jpg
        │   │           └─ ....
        │   └─ cyberpunk/…
        │
        └─ common.yaml             # глобальные дефолты и чёрный список
