"""Portable JSONL manifest and character-error-rate helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Iterator


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield item


def write_jsonl(path: Path, items: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for item in items:
            output.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_audio_path(
    source: str,
    manifest_path: Path,
    data_root: Path | None = None,
) -> Path:
    source_path = Path(source)
    candidates: list[Path] = []

    if source_path.is_absolute():
        candidates.append(source_path)
    else:
        candidates.append(manifest_path.parent / source_path)
        if data_root is not None:
            candidates.append(data_root / source_path)

    configured_root = os.environ.get("INCAR_ASR_DATA_ROOT")
    if configured_root:
        configured_path = Path(configured_root)
        candidates.append(
            configured_path / (source_path.name if source_path.is_absolute() else source_path)
        )

    # Legacy manifests contain Windows absolute paths. A configured data root
    # can still resolve them by filename without hard-coding a contributor drive.
    if data_root is not None and (":" in source or "\\" in source):
        candidates.append(data_root / PureWindowsPath(source).name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    attempted = ", ".join(str(path) for path in candidates) or source
    raise FileNotFoundError(f"Audio not found for {source!r}; tried: {attempted}")


def normalize_transcript(text: str) -> str:
    return "".join(text.split())


def edit_counts(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Return (edit distance, reference characters, hypothesis characters)."""
    reference = normalize_transcript(reference)
    hypothesis = normalize_transcript(hypothesis)
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, hyp_char in enumerate(hypothesis, start=1):
            substitution = previous[column - 1] + (ref_char != hyp_char)
            deletion = previous[column] + 1
            insertion = current[column - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1], len(reference), len(hypothesis)


def character_error_rate(reference: str, hypothesis: str) -> float:
    edits, reference_length, hypothesis_length = edit_counts(reference, hypothesis)
    if reference_length == 0:
        return float(hypothesis_length > 0)
    return edits / reference_length
