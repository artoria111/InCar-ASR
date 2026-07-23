#!/usr/bin/env python3
"""Transcribe a mono PCM16 WAV with the reproducible CPU Paraformer baseline."""

from __future__ import annotations

import argparse
import json
import os
import time
import wave
from pathlib import Path
from typing import Any

try:
    from scripts.baseline_config import load_baseline_config
except ModuleNotFoundError:
    from baseline_config import load_baseline_config

BASELINE_CONFIG = load_baseline_config()
MODEL_NAME = BASELINE_CONFIG["model"]["name"]


def read_pcm16_wave(path: Path):
    import numpy as np

    try:
        with wave.open(str(path), "rb") as audio_file:
            channels = audio_file.getnchannels()
            sample_width = audio_file.getsampwidth()
            sample_rate = audio_file.getframerate()
            frames = audio_file.readframes(audio_file.getnframes())
    except (EOFError, wave.Error) as error:
        raise ValueError(f"{path} is not a readable WAV file: {error}") from error

    if channels != 1:
        raise ValueError(f"{path} must be mono; found {channels} channels")
    if sample_width != 2:
        raise ValueError(
            f"{path} must contain signed 16-bit PCM samples; "
            f"found {sample_width * 8}-bit samples"
        )

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        raise ValueError(f"{path} contains no audio samples")
    return samples, sample_rate


def create_recognizer(model_dir: Path, num_threads: int):
    try:
        import sherpa_onnx
    except ImportError as error:
        raise RuntimeError(
            "sherpa-onnx is not installed. Run ./scripts/run_demo.sh or "
            "install requirements-demo.txt."
        ) from error

    model_path = model_dir / BASELINE_CONFIG["model"]["model_file"]
    tokens_path = model_dir / BASELINE_CONFIG["model"]["tokens_file"]
    missing = [path for path in (model_path, tokens_path) if not path.is_file()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing model resources: {missing_list}. "
            "Run scripts/download_demo_model.py."
        )

    return sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=str(model_path),
        tokens=str(tokens_path),
        num_threads=num_threads,
        sample_rate=BASELINE_CONFIG["frontend"]["sample_rate"],
        feature_dim=BASELINE_CONFIG["frontend"]["feature_dim"],
        decoding_method=BASELINE_CONFIG["decoder"]["method"],
        provider="cpu",
    )


def decode_samples(
    samples,
    sample_rate: int,
    recognizer,
    *,
    audio_name: str = "<memory>",
    num_threads: int | None = None,
) -> dict[str, Any]:
    """Decode float32 mono samples with an already-created recognizer."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if len(samples) == 0:
        raise ValueError("audio contains no samples")

    duration_seconds = len(samples) / sample_rate
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)

    start = time.perf_counter()
    recognizer.decode_stream(stream)
    elapsed_seconds = time.perf_counter() - start
    result = stream.result

    return {
        "audio": audio_name,
        "model": MODEL_NAME,
        "text": result.text.strip(),
        "duration_seconds": round(duration_seconds, 4),
        "elapsed_seconds": round(elapsed_seconds, 4),
        "rtf": round(elapsed_seconds / duration_seconds, 4),
        "sample_rate": sample_rate,
        "num_threads": num_threads,
    }


def transcribe(audio_path: Path, model_dir: Path, num_threads: int) -> dict[str, Any]:
    samples, sample_rate = read_pcm16_wave(audio_path)
    recognizer = create_recognizer(model_dir, num_threads)
    return decode_samples(
        samples,
        sample_rate,
        recognizer,
        audio_name=str(audio_path),
        num_threads=num_threads,
    )


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    default_model_dir = repository_root / "models" / MODEL_NAME
    default_audio = default_model_dir / "test_wavs" / "0.wav"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        nargs="?",
        type=Path,
        default=default_audio,
        help="Mono PCM16 WAV file; defaults to the model's Chinese test WAV",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help="Extracted Paraformer model directory",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="CPU inference threads",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")

    result = transcribe(args.audio, args.model_dir, args.threads)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("InCar-ASR reproducible CPU baseline")
    print(f"Audio:   {result['audio']}")
    print(f"Text:    {result['text']}")
    print(f"Latency: {result['elapsed_seconds']:.3f}s")
    print(f"RTF:     {result['rtf']:.4f}")


if __name__ == "__main__":
    main()
