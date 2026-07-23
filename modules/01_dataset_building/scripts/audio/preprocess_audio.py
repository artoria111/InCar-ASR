#!/usr/bin/env python3
"""Standardize and segment raw in-car noise recordings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

from common import (
    DEFAULT_CONFIG_PATH,
    PROCESSED_METADATA_FIELDS,
    audio_metrics,
    display_number,
    iter_audio_files,
    load_config,
    portable_path,
    read_metadata,
    resolve_module_path,
    safe_slug,
)


@dataclass(frozen=True)
class Segment:
    index: int
    start_seconds: float
    samples: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw noise recordings into standardized ASR augmentation clips."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input-dir", type=Path, help="Override raw audio directory")
    parser.add_argument("--output-dir", type=Path, help="Override processed audio directory")
    parser.add_argument("--source-metadata", type=Path, help="Override raw metadata CSV")
    parser.add_argument("--output-metadata", type=Path, help="Override processed metadata CSV")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing audio clips")
    return parser.parse_args()


def to_mono(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        return data
    if data.ndim != 2:
        raise ValueError(f"Expected one- or two-dimensional audio, got {data.shape}")
    return np.mean(data, axis=1, dtype=np.float32)


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    divisor = math.gcd(source_rate, target_rate)
    result = resample_poly(samples, target_rate // divisor, source_rate // divisor)
    return np.asarray(result, dtype=np.float32)


def highpass_filter(samples: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return np.asarray(samples, dtype=np.float32)
    nyquist = sample_rate / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError("High-pass cutoff must be below the Nyquist frequency")
    sos = butter(4, cutoff_hz / nyquist, btype="highpass", output="sos")
    return np.asarray(sosfilt(sos, samples), dtype=np.float32)


def trim_silence(samples: np.ndarray, threshold_dbfs: float) -> np.ndarray:
    if samples.size == 0:
        return samples
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    active = np.flatnonzero(np.abs(samples) >= threshold)
    if active.size == 0:
        return np.empty(0, dtype=np.float32)
    return np.asarray(samples[active[0] : active[-1] + 1], dtype=np.float32)


def normalize_rms(
    samples: np.ndarray, target_dbfs: float, peak_limit_dbfs: float = -1.0
) -> np.ndarray:
    if samples.size == 0:
        return samples
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms <= 1e-12:
        return np.asarray(samples, dtype=np.float32)
    target_rms = 10.0 ** (target_dbfs / 20.0)
    scaled = np.asarray(samples * (target_rms / rms), dtype=np.float32)
    peak = float(np.max(np.abs(scaled)))
    peak_limit = 10.0 ** (peak_limit_dbfs / 20.0)
    if peak > peak_limit:
        scaled *= peak_limit / peak
    return scaled


def segment_audio(
    samples: np.ndarray,
    sample_rate: int,
    segment_seconds: float,
    hop_seconds: float,
    pad_last: bool,
    min_tail_seconds: float,
) -> Iterator[Segment]:
    segment_length = int(round(segment_seconds * sample_rate))
    hop_length = int(round(hop_seconds * sample_rate))
    min_tail_length = int(round(min_tail_seconds * sample_rate))
    if segment_length <= 0 or hop_length <= 0:
        raise ValueError("Segment and hop durations must be positive")

    index = 0
    start = 0
    while start < samples.size:
        chunk = np.asarray(samples[start : start + segment_length], dtype=np.float32)
        if chunk.size < segment_length:
            if chunk.size < min_tail_length:
                break
            if pad_last:
                chunk = np.pad(chunk, (0, segment_length - chunk.size))
        yield Segment(index=index, start_seconds=start / sample_rate, samples=chunk)
        index += 1
        if start + segment_length >= samples.size:
            break
        start += hop_length


def choose_split(key: str, ratios: Mapping[str, float]) -> str:
    ordered = [(name, float(ratio)) for name, ratio in ratios.items()]
    total = sum(ratio for _, ratio in ordered)
    if total <= 0:
        raise ValueError("dataset_splits must have a positive total")
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) / float(16**16)
    cumulative = 0.0
    for name, ratio in ordered:
        cumulative += ratio / total
        if value < cumulative:
            return name
    return ordered[-1][0]


def find_source_metadata(
    source: Path, metadata_index: Mapping[str, Dict[str, str]]
) -> Dict[str, str]:
    candidates = [portable_path(source), source.as_posix(), source.name]
    for candidate in candidates:
        if candidate in metadata_index:
            return metadata_index[candidate]
    return {}


def process_file(
    source: Path,
    input_root: Path,
    output_root: Path,
    metadata_index: Mapping[str, Dict[str, str]],
    config: Mapping[str, Any],
    overwrite: bool,
) -> List[Dict[str, Any]]:
    audio_config = config["audio"]
    preprocessing = config["preprocessing"]
    target_rate = int(audio_config["sample_rate"])

    samples, source_rate = sf.read(source, always_2d=True, dtype="float32")
    mono = to_mono(samples)
    if bool(preprocessing.get("remove_dc_offset", True)) and mono.size:
        mono = mono - float(np.mean(mono))
    mono = resample_audio(mono, int(source_rate), target_rate)
    mono = highpass_filter(mono, target_rate, float(preprocessing.get("highpass_hz", 0)))
    if bool(preprocessing.get("trim_silence", False)):
        mono = trim_silence(mono, float(preprocessing["silence_threshold_dbfs"]))
    if bool(preprocessing.get("normalize_rms", False)):
        mono = normalize_rms(
            mono,
            float(preprocessing["target_rms_dbfs"]),
            float(preprocessing["peak_limit_dbfs"]),
        )
    if mono.size == 0:
        raise ValueError("audio is empty after preprocessing")

    relative = source.relative_to(input_root)
    source_row = find_source_metadata(source, metadata_index)
    category = source_row.get("category") or (relative.parts[0] if len(relative.parts) > 1 else "unknown")
    category = safe_slug(category)
    source_recording_id = source_row.get("recording_id") or source.stem
    split = choose_split(source_recording_id, preprocessing["dataset_splits"])
    rows: List[Dict[str, Any]] = []

    segments = segment_audio(
        mono,
        target_rate,
        float(preprocessing["segment_seconds"]),
        float(preprocessing["hop_seconds"]),
        bool(preprocessing["pad_last_segment"]),
        float(preprocessing["minimum_tail_seconds"]),
    )
    for segment in segments:
        segment_id = f"{safe_slug(source_recording_id)}-{segment.index:04d}"
        destination = output_root / split / category / f"{segment_id}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not destination.exists():
            temporary = destination.with_suffix(".tmp.wav")
            sf.write(temporary, segment.samples, target_rate, subtype=str(audio_config["subtype"]))
            temporary.replace(destination)

        metrics = audio_metrics(segment.samples)
        rows.append(
            {
                "segment_id": segment_id,
                "file_path": portable_path(destination),
                "source_file": portable_path(source),
                "source_recording_id": source_recording_id,
                "category": category,
                "split": split,
                "segment_index": segment.index,
                "start_seconds": f"{segment.start_seconds:.3f}",
                "duration_seconds": f"{segment.samples.size / target_rate:.3f}",
                "sample_rate": target_rate,
                "channels": 1,
                "rms_dbfs": display_number(metrics["rms_dbfs"]),
                "peak_dbfs": display_number(metrics["peak_dbfs"]),
                "clipped_ratio": display_number(metrics["clipped_ratio"], 6),
            }
        )
    if not rows:
        raise ValueError("audio is shorter than minimum_tail_seconds")
    return rows


def write_processed_metadata(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROCESSED_METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        collection = config["collection"]
        preprocessing = config["preprocessing"]
        input_root = (
            args.input_dir or resolve_module_path(collection["raw_audio_dir"])
        ).resolve()
        output_root = (
            args.output_dir or resolve_module_path(preprocessing["output_dir"])
        ).resolve()
        source_metadata = args.source_metadata or resolve_module_path(
            collection["metadata_file"]
        )
        output_metadata = args.output_metadata or output_root / "metadata.csv"

        files = list(iter_audio_files(input_root))
        if not files:
            raise ValueError(f"No WAV or FLAC files found below {input_root}")

        metadata_index = read_metadata(source_metadata)
        rows: List[Dict[str, Any]] = []
        failures: List[Tuple[Path, str]] = []
        for source in files:
            try:
                rows.extend(
                    process_file(
                        source,
                        input_root,
                        output_root,
                        metadata_index,
                        config,
                        args.overwrite,
                    )
                )
            except (OSError, RuntimeError, ValueError, sf.LibsndfileError) as exc:
                failures.append((source, str(exc)))

        if not rows:
            details = "; ".join(f"{path.name}: {reason}" for path, reason in failures[:3])
            raise RuntimeError(f"No files were processed. {details}")
        write_processed_metadata(output_metadata, rows)

        print(f"Processed {len(files) - len(failures)}/{len(files)} source files")
        print(f"Created or indexed {len(rows)} segments in {output_root}")
        print(f"Metadata: {output_metadata}")
        for path, reason in failures:
            print(f"Warning: skipped {path}: {reason}", file=sys.stderr)
        return 0 if not failures else 2
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
