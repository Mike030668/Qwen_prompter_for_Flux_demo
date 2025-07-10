import yaml, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1] / "ranger_generation/rules"

def check_yaml(path):
    data = yaml.safe_load(path.read_text())
    required = {"id", "category", "positive_boost", "negative_boost", "examples"}
    missing = required - data.keys()
    if missing:
        print(f"[ERROR] {path}: missing keys {missing}")
    for k in ("good", "bad"):
        p = (path.parent / data["examples"][k]).resolve()
        if not p.exists():
            print(f"[ERROR] {path}: folder {p} not found")

def walk_rules():
    for yml in ROOT.rglob("*.yaml"):
        if yml.name == "common.yaml":
            continue
        check_yaml(yml)

if __name__ == "__main__":
    walk_rules()
    print("Validation finished")
