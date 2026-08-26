"""Settings loaded from defaults.yaml."""

from functools import lru_cache
from pathlib import Path

import yaml

DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")


@lru_cache(maxsize=1)
def load_defaults() -> dict:
    """Load and cache the contents of defaults.yaml."""
    return yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))


def export_decimal_places() -> int:
    """Number of decimal places used by the CSV export."""
    return int(load_defaults()["export"]["decimal_places"])


def store_format() -> str:
    return str(load_defaults()["storage"]["format"])
