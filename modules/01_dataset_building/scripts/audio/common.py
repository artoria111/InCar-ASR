"""Shared helpers for the in-car audio data pipeline."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping

import numpy as np
import yaml


AUDIO_SUFFIXES = {".wav", ".flac"}
MODULE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = MODULE_ROOT / "configs" / "audio.yaml"

RAW_METADATA_FIELDS = [
    "recording_id",
    "file_path",
    "category",
    "recorded_at",
    "duration_seconds",
    "sample_rate",
    "channels",
    "device",
    "vehicle_state",
    "speed_kmh",
    "road_surface",
    "weather",
    "window_state",
    "microphone_position",
    "location_type",
    "source_type",
    "source_url",
    "license",
    "contains_speech",
    "rms_dbfs",
    "peak_dbfs",
    "clipped_ratio",
    "notes",
]

PROCESSED_METADATA_FIELDS = [
    "segment_id",
    "file_path",
    "source_file",
    "source_recording_id",
    "category",
    "split",
    "segment_index",
    "start_seconds",
    "duration_seconds",
    "sample_rate",
    "channels",
    "rms_dbfs",
    "peak_dbfs",
    "clipped_ratio",
]


def load_config(path: Path) -> Dict[str, Any]:
    """Load and minimally validate a YAML configuration file."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    required_sections = {"audio", "collection", "preprocessing", "validation"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(sorted(missing))}")

    sample_rate = int(config["audio"].get("sample_rate", 0))
    channels = int(config["audio"].get("channels", 0))
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("audio.sample_rate and audio.channels must be positive")

    categories = config["collection"].get("categories", [])
    if not isinstance(categories, list) or len(categories) < 10:
        raise ValueError("collection.categories must contain at least 10 categories")
    return config


def resolve_module_path(value: str | Path) -> Path:
    """Resolve a configured data path relative to the dataset module."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else MODULE_ROOT / path


def iter_audio_files(root: Path) -> Iterator[Path]:
    """Yield supported audio files below *root* in deterministic order."""
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            yield path


def safe_slug(value: str) -> str:
    """Convert user-provided labels to a safe filename component."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-_")
    return slug.lower() or "unknown"


def dbfs(amplitude: float) -> float:
    """Convert a linear full-scale amplitude to dBFS."""
    if amplitude <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(amplitude)


def audio_metrics(samples: np.ndarray) -> Dict[str, float]:
    """Compute simple quality metrics for float audio in [-1, 1]."""
    data = np.asarray(samples, dtype=np.float64)
    if data.size == 0:
        return {
            "rms_dbfs": float("-inf"),
            "peak_dbfs": float("-inf"),
            "clipped_ratio": 0.0,
            "dc_offset": 0.0,
        }

    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(data))))
    return {
        "rms_dbfs": dbfs(rms),
        "peak_dbfs": dbfs(peak),
        "clipped_ratio": float(np.mean(np.abs(data) >= 0.999)),
        "dc_offset": float(np.mean(data)),
    }


def display_number(value: float, digits: int = 3) -> str:
    """Render finite audio metrics consistently for CSV output."""
    if not math.isfinite(value):
        return "-inf"
    return f"{value:.{digits}f}"


def append_csv_row(path: Path, fields: Iterable[str], row: Mapping[str, Any]) -> None:
    """Append one row to a CSV file, creating the header when necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    field_list = list(fields)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in field_list})


def read_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    """Index raw metadata by normalized path and filename."""
    if not path.is_file():
        return {}
    index: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            file_path = row.get("file_path", "").strip()
            if not file_path:
                continue
            index[file_path.replace("\\", "/")] = row
            index.setdefault(Path(file_path).name, row)
    return index


def portable_path(path: Path, base: Path | None = None) -> str:
    """Prefer a repository-relative POSIX path when possible."""
    resolved = path.resolve()
    root = (base or Path.cwd()).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()
