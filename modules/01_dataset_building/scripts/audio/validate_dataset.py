#!/usr/bin/env python3
"""Validate audio format, signal quality, metadata, and class coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import soundfile as sf

from common import (
    DEFAULT_CONFIG_PATH,
    audio_metrics,
    iter_audio_files,
    load_config,
    portable_path,
    resolve_module_path,
)


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    file_path: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check an in-car noise dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input-dir", type=Path, help="Dataset root to validate")
    parser.add_argument("--metadata", type=Path, help="Processed metadata CSV")
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    parser.add_argument(
        "--fail-on-warning", action="store_true", help="Return a non-zero code for warnings"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def inspect_audio_file(
    path: Path, expected_rate: int, expected_channels: int, validation: Mapping[str, Any]
) -> Tuple[Dict[str, Any], List[Issue]]:
    issues: List[Issue] = []
    display_path = portable_path(path)
    try:
        info = sf.info(path)
        samples, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    except (OSError, RuntimeError, sf.LibsndfileError) as exc:
        return {}, [Issue("error", "unreadable", display_path, str(exc))]

    channels = int(samples.shape[1])
    duration = float(info.frames / info.samplerate) if info.samplerate else 0.0
    metrics = audio_metrics(samples)
    if int(sample_rate) != expected_rate:
        issues.append(
            Issue(
                "error",
                "sample_rate",
                display_path,
                f"expected {expected_rate} Hz, found {sample_rate} Hz",
            )
        )
    if channels != expected_channels:
        issues.append(
            Issue(
                "error",
                "channels",
                display_path,
                f"expected {expected_channels} channel(s), found {channels}",
            )
        )
    if duration < float(validation["minimum_duration_seconds"]):
        issues.append(
            Issue(
                "error",
                "too_short",
                display_path,
                f"duration is only {duration:.3f}s",
            )
        )
    if metrics["clipped_ratio"] > float(validation["maximum_clipped_ratio"]):
        issues.append(
            Issue(
                "warning",
                "clipping",
                display_path,
                f"{metrics['clipped_ratio']:.3%} of samples are clipped",
            )
        )
    if metrics["rms_dbfs"] < float(validation["minimum_rms_dbfs"]):
        issues.append(
            Issue(
                "warning",
                "low_level",
                display_path,
                f"RMS is {metrics['rms_dbfs']:.1f} dBFS",
            )
        )
    if abs(metrics["dc_offset"]) > float(validation["maximum_dc_offset"]):
        issues.append(
            Issue(
                "warning",
                "dc_offset",
                display_path,
                f"mean amplitude is {metrics['dc_offset']:.4f}",
            )
        )

    stats = {
        "file_path": display_path,
        "duration_seconds": round(duration, 6),
        "sample_rate": int(sample_rate),
        "channels": channels,
        "rms_dbfs": finite_or_none(metrics["rms_dbfs"]),
        "peak_dbfs": finite_or_none(metrics["peak_dbfs"]),
        "clipped_ratio": metrics["clipped_ratio"],
        "dc_offset": metrics["dc_offset"],
    }
    return stats, issues


def read_processed_metadata(path: Path) -> Tuple[List[Dict[str, str]], List[Issue]]:
    if not path.is_file():
        return [], [Issue("warning", "metadata_missing", portable_path(path), "file not found")]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"file_path", "source_recording_id", "category", "split"}
    columns = set(rows[0]) if rows else set()
    missing = sorted(required.difference(columns))
    issues = []
    if missing:
        issues.append(
            Issue(
                "error",
                "metadata_columns",
                portable_path(path),
                f"missing columns: {', '.join(missing)}",
            )
        )
    return rows, issues


def infer_category(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    parts = relative.parts
    if len(parts) >= 3 and parts[0] in {"train", "validation", "val", "test"}:
        return parts[1]
    return parts[0] if len(parts) > 1 else "unknown"


def validate_dataset(
    root: Path, metadata_path: Path, config: Mapping[str, Any]
) -> Dict[str, Any]:
    expected_rate = int(config["audio"]["sample_rate"])
    expected_channels = int(config["audio"]["channels"])
    validation = config["validation"]
    files = list(iter_audio_files(root))
    issues: List[Issue] = []
    stats: List[Dict[str, Any]] = []
    categories: Counter[str] = Counter()
    hashes: Dict[str, List[str]] = defaultdict(list)

    if not files:
        issues.append(Issue("error", "empty_dataset", portable_path(root), "no audio files found"))
    for path in files:
        file_stats, file_issues = inspect_audio_file(
            path, expected_rate, expected_channels, validation
        )
        stats.append(file_stats)
        issues.extend(file_issues)
        categories[infer_category(path, root)] += 1
        hashes[sha256_file(path)].append(portable_path(path))

    for duplicate_paths in hashes.values():
        if len(duplicate_paths) > 1:
            issues.append(
                Issue(
                    "warning",
                    "duplicate_audio",
                    duplicate_paths[0],
                    f"identical content also appears in {', '.join(duplicate_paths[1:])}",
                )
            )

    metadata_rows, metadata_issues = read_processed_metadata(metadata_path)
    issues.extend(metadata_issues)
    file_paths = {portable_path(path) for path in files}
    metadata_paths = {row.get("file_path", "").replace("\\", "/") for row in metadata_rows}
    for missing_path in sorted(file_paths.difference(metadata_paths)):
        issues.append(Issue("warning", "metadata_row_missing", missing_path, "no metadata row"))
    for stale_path in sorted(metadata_paths.difference(file_paths) - {""}):
        issues.append(Issue("warning", "audio_file_missing", stale_path, "metadata points to no file"))

    source_splits: Dict[str, set[str]] = defaultdict(set)
    for row in metadata_rows:
        source_splits[row.get("source_recording_id", "")].add(row.get("split", ""))
    for source_id, splits in source_splits.items():
        nonempty_splits = {split for split in splits if split}
        if source_id and len(nonempty_splits) > 1:
            issues.append(
                Issue(
                    "error",
                    "split_leakage",
                    source_id,
                    f"source recording appears in multiple splits: {sorted(nonempty_splits)}",
                )
            )

    required_category_count = int(validation["minimum_category_count"])
    if len(categories) < required_category_count:
        issues.append(
            Issue(
                "warning",
                "category_coverage",
                portable_path(root),
                f"found {len(categories)} categories; target is at least {required_category_count}",
            )
        )

    total_duration = sum(item.get("duration_seconds", 0.0) for item in stats)
    counts = Counter(issue.severity for issue in issues)
    return {
        "summary": {
            "file_count": len(files),
            "total_duration_seconds": round(total_duration, 3),
            "category_count": len(categories),
            "categories": dict(sorted(categories.items())),
            "errors": counts["error"],
            "warnings": counts["warning"],
        },
        "issues": [asdict(issue) for issue in issues],
        "files": stats,
    }


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        root = (
            args.input_dir
            or resolve_module_path(config["preprocessing"]["output_dir"])
        ).resolve()
        metadata_path = args.metadata or root / "metadata.csv"
        report = validate_dataset(root, metadata_path, config)
        summary = report["summary"]
        print(
            f"Checked {summary['file_count']} files, "
            f"{summary['total_duration_seconds']:.1f}s, "
            f"{summary['category_count']} categories"
        )
        print(f"Errors: {summary['errors']}; warnings: {summary['warnings']}")
        for issue in report["issues"]:
            print(
                f"[{issue['severity'].upper()}] {issue['code']}: "
                f"{issue['file_path']} - {issue['message']}"
            )

        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report.with_suffix(args.report.suffix + ".tmp")
            temporary.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            temporary.replace(args.report)
            print(f"JSON report: {args.report}")

        if summary["errors"]:
            return 1
        if args.fail_on_warning and summary["warnings"]:
            return 2
        return 0
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
