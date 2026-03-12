from functools import lru_cache
import json
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.json"


@lru_cache(maxsize=1)
def get_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
