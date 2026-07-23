#!/usr/bin/env python3
"""Compare reference and candidate ONNX outputs on the same cached features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def create_inputs(session, features: np.ndarray, feature_length: int) -> dict:
    inputs = {}
    for index, descriptor in enumerate(session.get_inputs()):
        name = descriptor.name
        element_type = descriptor.type
        if index == 0:
            value = features[np.newaxis].astype(np.float32)
        elif "len" in name.lower():
            value = np.asarray([feature_length], dtype=np.int32)
        else:
            raise ValueError(
                f"Cannot infer value for ONNX input {name!r}; "
                "only feature and length inputs are supported"
            )
        if "int64" in element_type:
            value = value.astype(np.int64)
        inputs[name] = value
    return inputs


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "passed": False,
        }
    reference = reference.astype(np.float64).reshape(-1)
    candidate = candidate.astype(np.float64).reshape(-1)
    difference = np.abs(reference - candidate)
    denominator = np.linalg.norm(reference) * np.linalg.norm(candidate)
    cosine = float(np.dot(reference, candidate) / max(denominator, 1e-12))
    return {
        "shape_match": True,
        "cosine_similarity": cosine,
        "max_absolute_diff": float(difference.max(initial=0.0)),
        "mean_absolute_diff": float(difference.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0, help="First cache index")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--cosine-threshold", type=float, default=0.999)
    parser.add_argument("--max-diff-threshold", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import onnxruntime as ort

    cache = np.load(args.cache, allow_pickle=False)
    reference_session = ort.InferenceSession(
        str(args.reference),
        providers=["CPUExecutionProvider"],
    )
    candidate_session = ort.InferenceSession(
        str(args.candidate),
        providers=["CPUExecutionProvider"],
    )
    if args.sample_index < 0 or args.sample_index >= len(cache["feature_lengths"]):
        raise IndexError("sample index is outside the cache")
    stop = min(
        len(cache["feature_lengths"]),
        args.sample_index + max(1, args.max_samples),
    )
    samples = []
    for index in range(args.sample_index, stop):
        feature_length = int(cache["feature_lengths"][index])
        features = cache["features"][index, :feature_length]
        reference_outputs = reference_session.run(
            None,
            create_inputs(reference_session, features, feature_length),
        )
        candidate_outputs = candidate_session.run(
            None,
            create_inputs(candidate_session, features, feature_length),
        )
        if len(reference_outputs) != len(candidate_outputs):
            raise RuntimeError(
                f"Output count mismatch: {len(reference_outputs)} != "
                f"{len(candidate_outputs)}"
            )

        comparisons = [
            compare_arrays(reference, candidate)
            for reference, candidate in zip(reference_outputs, candidate_outputs)
        ]
        for comparison in comparisons:
            comparison["passed"] = bool(
                comparison.get("shape_match")
                and comparison["cosine_similarity"] >= args.cosine_threshold
                and comparison["max_absolute_diff"] <= args.max_diff_threshold
            )
        samples.append(
            {
                "sample_index": index,
                "feature_length": feature_length,
                "outputs": comparisons,
                "passed": all(item["passed"] for item in comparisons),
            }
        )

    report = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "cache": str(args.cache),
        "sample_range": [args.sample_index, stop],
        "thresholds": {
            "cosine_similarity": args.cosine_threshold,
            "max_absolute_diff": args.max_diff_threshold,
        },
        "samples": samples,
        "passed": all(item["passed"] for item in samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
