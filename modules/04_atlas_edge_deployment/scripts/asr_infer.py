#!/usr/bin/env python3
"""Run a fixed-shape full Paraformer ONNX model with the shared frontend."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.baseline_config import load_baseline_config
from scripts.frontend import extract_lfr_features, read_wave_float32


def load_tokens(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [str(token) for token in value]
        if isinstance(value, dict):
            if all(str(index).isdigit() for index in value):
                entries = {int(index): str(token) for index, token in value.items()}
            elif all(isinstance(index, int) for index in value.values()):
                entries = {int(index): str(token) for token, index in value.items()}
            else:
                raise ValueError("tokens JSON dict must map ids to tokens or tokens to ids")
            tokens = [""] * (max(entries, default=-1) + 1)
            for index, token in entries.items():
                tokens[index] = token
            return tokens
        raise ValueError("tokens JSON must be a list or id-to-token object")

    entries: dict[int, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines()
    ):
        line = raw_line.rstrip()
        if not line:
            continue
        parts = line.rsplit(maxsplit=1)
        if len(parts) == 2 and parts[1].isdigit():
            token, token_id = parts[0], int(parts[1])
        else:
            token, token_id = line, line_number
        entries[token_id] = token
    tokens = [""] * (max(entries, default=-1) + 1)
    for token_id, token in entries.items():
        tokens[token_id] = token
    return tokens


def session_inputs(session, features: np.ndarray, feature_length: int) -> dict:
    inputs = {}
    for index, descriptor in enumerate(session.get_inputs()):
        if index == 0:
            value = features[np.newaxis].astype(np.float32)
        elif "len" in descriptor.name.lower():
            value = np.asarray([feature_length], dtype=np.int64)
        else:
            raise ValueError(f"unsupported extra ONNX input: {descriptor.name}")
        if "int32" in descriptor.type:
            value = value.astype(np.int32)
        inputs[descriptor.name] = value
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument(
        "--fixed-frames",
        type=int,
        default=300,
        help="Pad/crop LFR input to the ATC fixed frame count",
    )
    args = parser.parse_args()

    import onnxruntime as ort

    samples, sample_rate = read_wave_float32(args.wav)
    features = extract_lfr_features(
        samples, sample_rate, load_baseline_config()
    )
    original_frames = min(len(features), args.fixed_frames)
    fixed = np.zeros((args.fixed_frames, features.shape[1]), dtype=np.float32)
    fixed[:original_frames] = features[:original_frames]

    session = ort.InferenceSession(
        str(args.model), providers=["CPUExecutionProvider"]
    )
    start = time.perf_counter()
    outputs = session.run(
        None, session_inputs(session, fixed, original_frames)
    )
    elapsed = time.perf_counter() - start

    logits = np.asarray(outputs[0])[0]
    token_count = (
        int(np.asarray(outputs[1]).reshape(-1)[0])
        if len(outputs) > 1
        else logits.shape[0]
    )
    tokens = load_tokens(args.tokens)
    special_ids = {0, 2, 3}
    text = "".join(
        tokens[token_id]
        for token_id in np.argmax(logits[:token_count], axis=-1)
        if token_id not in special_ids and token_id < len(tokens)
    )
    duration = len(samples) / sample_rate
    print(
        json.dumps(
            {
                "audio": str(args.wav),
                "model": str(args.model),
                "text": text,
                "input_frames": original_frames,
                "token_count": token_count,
                "elapsed_seconds": elapsed,
                "rtf": elapsed / duration,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
