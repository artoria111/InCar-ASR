"""Load the single source of truth for the reproducible inference baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "baseline.json"


def load_baseline_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        config = json.load(source)

    required_paths = (
        ("model", "name"),
        ("model", "archive_url"),
        ("model", "archive_sha256"),
        ("model", "model_file"),
        ("model", "tokens_file"),
        ("frontend", "sample_rate"),
        ("frontend", "feature_dim"),
        ("decoder", "method"),
    )
    for section, key in required_paths:
        if section not in config or key not in config[section]:
            raise ValueError(f"Missing baseline configuration: {section}.{key}")
    return config
