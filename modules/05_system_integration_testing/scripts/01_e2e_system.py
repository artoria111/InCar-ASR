#!/usr/bin/env python3
"""Portable system entry point for file, batch, and microphone validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.baseline_config import load_baseline_config
from scripts.manifest_utils import read_jsonl, resolve_audio_path
from scripts.transcribe import create_recognizer, decode_samples, read_pcm16_wave

CONFIG = load_baseline_config()
MODEL_NAME = CONFIG["model"]["name"]


def run_file(path: Path, recognizer, threads: int) -> dict:
    samples, sample_rate = read_pcm16_wave(path)
    return decode_samples(
        samples,
        sample_rate,
        recognizer,
        audio_name=str(path),
        num_threads=threads,
    )


def run_batch(
    manifest: Path,
    data_root: Path,
    recognizer,
    threads: int,
    output: Path | None,
) -> None:
    results = []
    failures = []
    for index, sample in enumerate(read_jsonl(manifest)):
        key = str(sample.get("key", index))
        try:
            path = resolve_audio_path(
                str(sample["source"]), manifest, data_root
            )
            result = run_file(path, recognizer, threads)
            result["key"] = key
            result["reference"] = str(sample.get("target", ""))
            results.append(result)
        except Exception as error:
            failures.append(
                {"key": key, "error": f"{type(error).__name__}: {error}"}
            )
    report = {
        "model": MODEL_NAME,
        "evaluated": len(results),
        "failed": len(failures),
        "results": results,
        "errors": failures,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not results:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("file", "batch", "mic"), required=True)
    parser.add_argument("--input", type=Path, help="WAV or JSONL manifest")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPOSITORY_ROOT / "models" / MODEL_NAME,
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dashboard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "mic":
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "microphone_demo.py"),
            "--model-dir",
            str(args.model_dir),
            "--threads",
            str(args.threads),
        ]
        if args.dashboard:
            command.extend(["--dashboard", args.dashboard])
        raise SystemExit(subprocess.run(command, check=False).returncode)

    if not args.input:
        raise SystemExit("--input is required for file and batch modes")
    recognizer = create_recognizer(args.model_dir, args.threads)
    if args.mode == "file":
        print(
            json.dumps(
                run_file(args.input, recognizer, args.threads),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        if not args.data_root:
            raise SystemExit("--data-root is required for batch mode")
        run_batch(
            args.input,
            args.data_root,
            recognizer,
            args.threads,
            args.output,
        )


if __name__ == "__main__":
    main()
