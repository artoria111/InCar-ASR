#!/usr/bin/env python3
"""Create a portable ASR JSONL manifest from WAV files and a transcript map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.manifest_utils import write_jsonl


def read_transcripts(path: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected '<utterance-id> <transcript>'"
                )
            key, text = parts
            if key in transcripts:
                raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
            transcripts[key] = text
    return transcripts


def build_manifest(
    audio_root: Path,
    transcripts: dict[str, str],
    domain: str,
) -> list[dict[str, str]]:
    audio_root = audio_root.resolve()
    wav_by_key = {path.stem: path for path in audio_root.rglob("*.wav")}
    missing_audio = sorted(set(transcripts) - set(wav_by_key))
    if missing_audio:
        preview = ", ".join(missing_audio[:5])
        raise FileNotFoundError(
            f"{len(missing_audio)} transcripts have no matching WAV; first: {preview}"
        )

    return [
        {
            "key": key,
            "source": wav_by_key[key].relative_to(audio_root).as_posix(),
            "target": transcripts[key],
            "domain": domain,
        }
        for key in sorted(transcripts)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument(
        "--transcripts",
        type=Path,
        required=True,
        help="UTF-8 text file containing '<utterance-id> <transcript>' per line",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", default="unknown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transcripts = read_transcripts(args.transcripts)
    manifest = build_manifest(args.audio_root, transcripts, args.domain)
    write_jsonl(args.output, manifest)
    print(f"Wrote {len(manifest)} samples to {args.output}")
    print(f"Resolve relative sources with --data-root {args.audio_root.resolve()}")


if __name__ == "__main__":
    main()
