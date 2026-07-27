from pathlib import Path
from typing import Any

from smc_validator.config import load_config as _load_yaml_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return _load_yaml_config(path)
