#!/usr/bin/env python3
"""Record a labelled in-car noise sample and append its metadata."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf

from common import (
    DEFAULT_CONFIG_PATH,
    RAW_METADATA_FIELDS,
    append_csv_row,
    audio_metrics,
    display_number,
    load_config,
    portable_path,
    resolve_module_path,
    safe_slug,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record one labelled car-noise sample as PCM WAV."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--category", help="Noise category defined in the config")
    parser.add_argument("--duration", type=float, help="Recording duration in seconds")
    parser.add_argument("--device", help="Input device index or name")
    parser.add_argument("--output-dir", type=Path, help="Override raw audio directory")
    parser.add_argument("--metadata", type=Path, help="Override metadata CSV path")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--vehicle-state", default="unknown")
    parser.add_argument("--speed-kmh", type=float)
    parser.add_argument("--road-surface", default="unknown")
    parser.add_argument("--weather", default="unknown")
    parser.add_argument("--window-state", default="unknown")
    parser.add_argument("--microphone-position", default="unknown")
    parser.add_argument("--location-type", default="unknown")
    parser.add_argument("--contains-speech", choices=("yes", "no", "unknown"), default="no")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def import_sounddevice():
    try:
        import sounddevice as sd  # type: ignore
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "sounddevice is unavailable. Install requirements and ensure PortAudio is present."
        ) from exc
    return sd


def resolve_device(sd: Any, device_value: str | None) -> Any:
    if device_value is None:
        return None
    try:
        return int(device_value)
    except ValueError:
        return device_value


def collect(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    audio_config: Dict[str, Any] = config["audio"]
    collection_config: Dict[str, Any] = config["collection"]

    if not args.category:
        raise ValueError("--category is required unless --list-devices is used")
    categories = set(collection_config["categories"])
    if args.category not in categories:
        raise ValueError(
            f"Unknown category '{args.category}'. Choose one of: {', '.join(sorted(categories))}"
        )

    duration = (
        float(collection_config["default_duration_seconds"])
        if args.duration is None
        else args.duration
    )
    if duration <= 0:
        raise ValueError("Recording duration must be positive")

    sample_rate = int(audio_config["sample_rate"])
    channels = int(audio_config["channels"])
    output_root = args.output_dir or resolve_module_path(collection_config["raw_audio_dir"])
    metadata_path = args.metadata or resolve_module_path(collection_config["metadata_file"])
    category_slug = safe_slug(args.category)
    category_dir = output_root / category_slug
    category_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().astimezone()
    recording_id = f"{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    final_path = category_dir / f"{category_slug}_{recording_id}.wav"
    temporary_path = final_path.with_suffix(".tmp.wav")

    sd = import_sounddevice()
    device = resolve_device(sd, args.device)
    try:
        device_info = sd.query_devices(device, "input")
        frames = int(round(duration * sample_rate))
        print(
            f"Recording {duration:.1f}s at {sample_rate} Hz, {channels} channel(s) "
            f"from {device_info['name']}..."
        )
        samples = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=device,
        )
        sd.wait()
    except Exception as exc:
        raise RuntimeError(f"Audio capture failed: {exc}") from exc

    samples = np.asarray(samples, dtype=np.float32)
    sf.write(temporary_path, samples, sample_rate, subtype=str(audio_config["subtype"]))
    temporary_path.replace(final_path)

    metrics = audio_metrics(samples)
    row = {
        "recording_id": recording_id,
        "file_path": portable_path(final_path),
        "category": args.category,
        "recorded_at": now.isoformat(timespec="seconds"),
        "duration_seconds": f"{duration:.3f}",
        "sample_rate": sample_rate,
        "channels": channels,
        "device": device_info["name"],
        "vehicle_state": args.vehicle_state,
        "speed_kmh": "" if args.speed_kmh is None else args.speed_kmh,
        "road_surface": args.road_surface,
        "weather": args.weather,
        "window_state": args.window_state,
        "microphone_position": args.microphone_position,
        "location_type": args.location_type,
        "source_type": "self_recorded",
        "source_url": "",
        "license": "project_internal",
        "contains_speech": args.contains_speech,
        "rms_dbfs": display_number(metrics["rms_dbfs"]),
        "peak_dbfs": display_number(metrics["peak_dbfs"]),
        "clipped_ratio": display_number(metrics["clipped_ratio"], 6),
        "notes": args.notes,
    }
    append_csv_row(metadata_path, RAW_METADATA_FIELDS, row)

    print(f"Saved: {final_path}")
    print(
        "Quality: "
        f"RMS {display_number(metrics['rms_dbfs'], 1)} dBFS, "
        f"peak {display_number(metrics['peak_dbfs'], 1)} dBFS, "
        f"clipped {metrics['clipped_ratio']:.3%}"
    )
    if metrics["clipped_ratio"] > 0.001:
        print("Warning: clipping detected; reduce microphone gain and record again.")
    return final_path


def main() -> int:
    args = parse_args()
    try:
        sd = import_sounddevice()
        if args.list_devices:
            print(sd.query_devices())
            return 0
        collect(args)
        return 0
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
