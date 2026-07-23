#!/usr/bin/env python3
"""Build a portable padded NumPy feature cache from an ASR JSONL manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.baseline_config import load_baseline_config
from scripts.frontend import extract_lfr_features, read_wave_float32
from scripts.manifest_utils import normalize_transcript, read_jsonl, resolve_audio_path


def load_tokens(path: Path) -> tuple[dict[str, int], int]:
    token_to_id: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip()
            if not line:
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2 and parts[1].isdigit():
                token, token_id = parts[0], int(parts[1])
            else:
                token, token_id = line, line_number - 1
            token_to_id[token] = token_id
    unknown_id = token_to_id.get("<unk>", token_to_id.get("<UNK>", 1))
    return token_to_id, unknown_id


def tokenize(text: str, token_to_id: dict[str, int], unknown_id: int) -> list[int]:
    # The baseline vocabulary contains Chinese characters and BPE pieces. The
    # cache builder intentionally uses character tokens for Chinese command data.
    return [
        token_to_id.get(character, unknown_id)
        for character in normalize_transcript(text)
    ]


def build_cache(
    manifest: Path,
    data_root: Path,
    tokens_path: Path,
    output: Path,
) -> dict[str, int]:
    config = load_baseline_config()
    token_to_id, unknown_id = load_tokens(tokens_path)
    features = []
    token_ids = []
    keys = []

    for index, sample in enumerate(read_jsonl(manifest)):
        audio_path = resolve_audio_path(str(sample["source"]), manifest, data_root)
        samples, sample_rate = read_wave_float32(audio_path)
        sample_features = extract_lfr_features(samples, sample_rate, config)
        sample_tokens = tokenize(str(sample["target"]), token_to_id, unknown_id)
        if not sample_tokens:
            raise ValueError(f"Sample {sample.get('key', index)!r} has an empty target")
        features.append(sample_features)
        token_ids.append(np.asarray(sample_tokens, dtype=np.int64))
        keys.append(str(sample.get("key", index)))

    if not features:
        raise ValueError(f"No samples found in {manifest}")

    max_frames = max(item.shape[0] for item in features)
    feature_width = features[0].shape[1]
    max_tokens = max(len(item) for item in token_ids)
    padded_features = np.zeros(
        (len(features), max_frames, feature_width),
        dtype=np.float32,
    )
    padded_tokens = np.zeros((len(features), max_tokens), dtype=np.int64)
    feature_lengths = np.zeros(len(features), dtype=np.int64)
    token_lengths = np.zeros(len(features), dtype=np.int64)
    for index, (sample_features, sample_tokens) in enumerate(
        zip(features, token_ids)
    ):
        padded_features[index, : len(sample_features)] = sample_features
        padded_tokens[index, : len(sample_tokens)] = sample_tokens
        feature_lengths[index] = len(sample_features)
        token_lengths[index] = len(sample_tokens)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=padded_features,
        feature_lengths=feature_lengths,
        token_ids=padded_tokens,
        token_lengths=token_lengths,
        keys=np.asarray(keys),
    )
    return {
        "samples": len(features),
        "max_frames": max_frames,
        "feature_width": feature_width,
        "max_tokens": max_tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--tokens",
        type=Path,
        required=True,
        help="Token vocabulary belonging to the exact model being trained",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_cache(args.manifest, args.data_root, args.tokens, args.output)
    print(f"Cache: {args.output}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
