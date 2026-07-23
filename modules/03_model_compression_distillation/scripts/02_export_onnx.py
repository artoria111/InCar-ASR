#!/usr/bin/env python3
"""Export a local or hub FunASR Paraformer model through the official API."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def load_checkpoint(model, checkpoint_path: Path) -> None:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not exactly match the model: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="FunASR hub ID or local model directory")
    parser.add_argument("--checkpoint", type=Path, help="Optional fine-tuned checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--opset", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from funasr import AutoModel

    model = AutoModel(model=args.model, device="cpu", disable_update=True)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)

    exported = model.export(
        type="onnx",
        quantize=args.quantize,
        opset_version=args.opset,
        device="cpu",
    )
    exported_path = Path(exported)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if exported_path.is_dir() and exported_path.resolve() != args.output_dir.resolve():
        for item in exported_path.iterdir():
            destination = args.output_dir / item.name
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)
    elif exported_path.is_file():
        destination = args.output_dir / exported_path.name
        if exported_path.resolve() != destination.resolve():
            shutil.copy2(exported_path, destination)
    if not any(args.output_dir.glob("*.onnx")):
        raise RuntimeError(
            f"FunASR export produced no ONNX file in {args.output_dir}"
        )

    manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_model": args.model,
        "checkpoint": (
            {
                "path": str(args.checkpoint.resolve()),
                "sha256": sha256(args.checkpoint),
            }
            if args.checkpoint
            else None
        ),
        "quantized": args.quantize,
        "opset": args.opset,
        "funasr_export_result": str(exported_path),
        "output_dir": str(args.output_dir.resolve()),
        "artifacts": [
            {
                "path": str(path.relative_to(args.output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(args.output_dir.rglob("*"))
            if path.is_file() and path.name != "export-manifest.json"
        ],
        "parity_status": "pending",
    }
    (args.output_dir / "export-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
