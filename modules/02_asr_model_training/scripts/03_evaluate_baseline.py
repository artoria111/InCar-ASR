#!/usr/bin/env python3
"""Evaluate the reproducible Paraformer baseline and emit JSON/Markdown reports."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.manifest_utils import edit_counts, read_jsonl, resolve_audio_path
from scripts.transcribe import MODEL_NAME, transcribe


def evaluate(
    manifest_path: Path,
    data_root: Path,
    model_dir: Path,
    threads: int,
    max_samples: int,
) -> dict:
    details = []
    errors = []
    total_edits = 0
    total_reference_characters = 0

    for index, sample in enumerate(read_jsonl(manifest_path)):
        if max_samples > 0 and index >= max_samples:
            break
        key = str(sample.get("key", index))
        try:
            audio_path = resolve_audio_path(
                str(sample["source"]),
                manifest_path,
                data_root,
            )
            inference = transcribe(audio_path, model_dir, threads)
            hypothesis = inference["text"]
            reference = str(sample["target"])
            edits, reference_length, _ = edit_counts(reference, hypothesis)
            total_edits += edits
            total_reference_characters += reference_length
            details.append(
                {
                    **inference,
                    "key": key,
                    "reference": reference,
                    "edits": edits,
                    "reference_characters": reference_length,
                    "cer": edits / max(1, reference_length),
                }
            )
        except Exception as error:
            errors.append({"key": key, "error": f"{type(error).__name__}: {error}"})

    if not details:
        raise RuntimeError(
            f"No samples were evaluated successfully; first errors: {errors[:3]}"
        )

    rtfs = [item["rtf"] for item in details]
    latencies = [item["elapsed_seconds"] * 1000 for item in details]
    bad_cases = sorted(details, key=lambda item: item["cer"], reverse=True)[:20]
    return {
        "model": MODEL_NAME,
        "manifest": str(manifest_path),
        "summary": {
            "evaluated": len(details),
            "failed": len(errors),
            "cer": total_edits / max(1, total_reference_characters),
            "total_edits": total_edits,
            "reference_characters": total_reference_characters,
            "mean_rtf": statistics.fmean(rtfs),
            "mean_latency_ms": statistics.fmean(latencies),
            "p95_latency_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        },
        "bad_cases": bad_cases,
        "errors": errors,
        "details": details,
    }


def markdown_report(result: dict) -> str:
    summary = result["summary"]
    lines = [
        "# ASR baseline evaluation",
        "",
        f"- Model: `{result['model']}`",
        f"- Manifest: `{result['manifest']}`",
        f"- Evaluated: {summary['evaluated']}",
        f"- Failed: {summary['failed']}",
        f"- Corpus CER: {summary['cer']:.2%}",
        f"- Mean RTF: {summary['mean_rtf']:.4f}",
        f"- Mean latency: {summary['mean_latency_ms']:.1f} ms",
        f"- P95 latency: {summary['p95_latency_ms']:.1f} ms",
        "",
        "## Worst cases",
        "",
        "| Key | Reference | Hypothesis | CER |",
        "| --- | --- | --- | ---: |",
    ]
    for item in result["bad_cases"][:10]:
        reference = item["reference"].replace("|", "\\|")
        hypothesis = item["text"].replace("|", "\\|")
        lines.append(
            f"| {item['key']} | {reference} | {hypothesis} | {item['cer']:.2%} |"
        )
    if result["errors"]:
        lines.extend(
            [
                "",
                "## Errors",
                "",
                *[
                    f"- `{item['key']}`: {item['error']}"
                    for item in result["errors"][:20]
                ],
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPOSITORY_ROOT / "models" / MODEL_NAME,
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "baseline-evaluation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(
        args.manifest,
        args.data_root,
        args.model_dir,
        args.threads,
        args.max_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.output}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
