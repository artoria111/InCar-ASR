#!/usr/bin/env python3
"""Write checksums and conversion metadata for an ONNX/OM artifact pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atc_version() -> str | None:
    try:
        result = subprocess.run(
            ["atc", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return (result.stdout or result.stderr).strip() or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--om", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--soc-version", required=True)
    parser.add_argument("--input-shape", required=True)
    parser.add_argument("--atc-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.platform(),
        "atc_version": atc_version(),
        "mode": args.mode,
        "soc_version": args.soc_version,
        "input_shape": args.input_shape,
        "onnx": {
            "path": str(args.onnx.resolve()),
            "bytes": args.onnx.stat().st_size,
            "sha256": sha256(args.onnx),
        },
        "om": {
            "path": str(args.om.resolve()),
            "bytes": args.om.stat().st_size,
            "sha256": sha256(args.om),
        },
        "atc_log": str(args.atc_log.resolve()),
        "verified_on_device": False,
    }
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
