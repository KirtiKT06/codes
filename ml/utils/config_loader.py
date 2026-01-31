import yaml
from pathlib import Path


def load_config(name: str):
    cfg_path = Path("ml/configs") / name
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)
