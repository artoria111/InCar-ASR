#!/usr/bin/env python3
"""Create a machine-readable report from a real Atlas smoke-test log."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=False, text=True, capture_output=True, timeout=10
        ).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def extract_metrics(log: str) -> dict:
    patterns = {
        "latency_ms": r"(?:Latency|推理耗时)\s*[:=]\s*([0-9.]+)\s*ms",
        "rtf": r"(?:RTF|Real.?Time Factor)\s*[:=]\s*([0-9.]+)",
        "peak_memory_mb": r"(?:Peak Memory|峰值内存)\s*[:=]\s*([0-9.]+)\s*MB",
    }
    output = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, log, re.IGNORECASE)
        output[key] = float(match.group(1)) if match else None
    text_match = re.search(
        r"=== Recognition Result ===\s*(.*?)\s*=+", log, re.DOTALL
    )
    output["text"] = text_match.group(1).strip() if text_match else None
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--acl-status", type=int, required=True)
    parser.add_argument("--asr-status", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-text")
    args = parser.parse_args()

    log = args.log.read_text(encoding="utf-8", errors="replace")
    metrics = extract_metrics(log)
    text_match = (
        metrics["text"] == args.expected_text
        if args.expected_text is not None
        else None
    )
    process_passed = args.acl_status == 0 and args.asr_status == 0
    passed = process_passed and text_match is not False
    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verified_on_device": process_passed,
        "status": "passed" if passed else "failed",
        "exit_codes": {"acl_hello": args.acl_status, "asr": args.asr_status},
        "host": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "kernel": platform.release(),
            "npu_smi": command_output(["npu-smi", "info"]),
        },
        "artifacts": {
            "model": {"path": str(args.model), "sha256": sha256(args.model)},
            "tokens": {"path": str(args.tokens), "sha256": sha256(args.tokens)},
            "wav": {"path": str(args.wav), "sha256": sha256(args.wav)},
        },
        "metrics": metrics,
        "text_parity": {
            "expected": args.expected_text,
            "passed": text_match,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
