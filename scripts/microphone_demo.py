#!/usr/bin/env python3
"""Live microphone endpointing demo for the offline Paraformer baseline."""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

try:
    from scripts.baseline_config import load_baseline_config
    from scripts.transcribe import create_recognizer, decode_samples, read_pcm16_wave
except ModuleNotFoundError:
    from baseline_config import load_baseline_config
    from transcribe import create_recognizer, decode_samples, read_pcm16_wave

CONFIG = load_baseline_config()
MODEL_NAME = CONFIG["model"]["name"]


def publish_result(url: str | None, result: dict) -> None:
    if not url:
        return
    payload = json.dumps(result, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/api/results",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        # Dashboard is normally localhost; bypass corporate/system HTTP proxies.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=3) as response:
            if response.status >= 300:
                raise RuntimeError(f"dashboard returned HTTP {response.status}")
    except Exception as error:
        print(f"[dashboard] publish failed: {error}", file=sys.stderr)


def run_simulation(args: argparse.Namespace, recognizer) -> None:
    samples, sample_rate = read_pcm16_wave(args.simulate)
    result = decode_samples(
        samples,
        sample_rate,
        recognizer,
        audio_name=str(args.simulate),
        num_threads=args.threads,
    )
    result["source"] = "simulation"
    result["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(json.dumps(result, ensure_ascii=False))
    publish_result(args.dashboard, result)


def run_microphone(args: argparse.Namespace, recognizer) -> None:
    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "sounddevice is required for microphone input; "
            "install requirements-microphone.txt"
        ) from error

    sample_rate = CONFIG["frontend"]["sample_rate"]
    block_samples = int(sample_rate * args.block_ms / 1000)
    min_blocks = max(1, int(args.min_speech_ms / args.block_ms))
    end_silence_blocks = max(1, int(args.end_silence_ms / args.block_ms))
    maximum_blocks = max(1, int(args.max_utterance_seconds * 1000 / args.block_ms))
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, timing, status):
        del frames, timing
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    print("Listening. Speak naturally; press Ctrl+C to stop.")
    active: list[np.ndarray] = []
    speech_blocks = 0
    trailing_silence = 0
    noise_rms = args.energy_threshold / 3.0

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block_samples,
        callback=callback,
    ):
        while True:
            block = audio_queue.get()
            rms = float(np.sqrt(np.mean(block * block) + 1e-12))
            threshold = max(args.energy_threshold, noise_rms * args.noise_ratio)
            is_speech = rms >= threshold

            if not active and not is_speech:
                noise_rms = 0.95 * noise_rms + 0.05 * rms
                continue

            active.append(block)
            if is_speech:
                speech_blocks += 1
                trailing_silence = 0
            else:
                trailing_silence += 1

            reached_endpoint = (
                speech_blocks >= min_blocks
                and trailing_silence >= end_silence_blocks
            )
            reached_limit = len(active) >= maximum_blocks
            if not reached_endpoint and not reached_limit:
                continue

            samples = np.concatenate(active)
            result = decode_samples(
                samples,
                sample_rate,
                recognizer,
                audio_name="<microphone>",
                num_threads=args.threads,
            )
            result["source"] = "microphone"
            result["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            print(json.dumps(result, ensure_ascii=False))
            publish_result(args.dashboard, result)
            active.clear()
            speech_blocks = 0
            trailing_silence = 0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=root / "models" / MODEL_NAME
    )
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--dashboard", help="Dashboard base URL, e.g. http://127.0.0.1:8765")
    parser.add_argument("--simulate", type=Path, help="Decode a WAV once without microphone hardware")
    parser.add_argument("--block-ms", type=int, default=30)
    parser.add_argument("--min-speech-ms", type=int, default=240)
    parser.add_argument("--end-silence-ms", type=int, default=720)
    parser.add_argument("--max-utterance-seconds", type=float, default=20.0)
    parser.add_argument("--energy-threshold", type=float, default=0.008)
    parser.add_argument("--noise-ratio", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    recognizer = create_recognizer(args.model_dir, args.threads)
    if args.simulate:
        run_simulation(args, recognizer)
        return
    try:
        run_microphone(args, recognizer)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
