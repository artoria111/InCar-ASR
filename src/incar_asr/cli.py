"""Command-line entry point for code and device teams."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from .audio import read_wav
from .backends import MockBackend, OnnxBackend, SubprocessBackend
from .commands import CommandMatcher, load_catalog, write_catalog
from .contracts import ModelContract
from .evaluation import load_manifest, run_evaluation
from .device_info import collect_device_info
from .streaming import EnergyVAD, VADConfig, simulate_stream


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="incar-asr",
        description="In-car ASR reference, evaluation, and device handoff toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    contract = subparsers.add_parser("validate-contract")
    contract.add_argument("--contract", type=Path, required=True)
    contract.add_argument("--require-artifacts", action="store_true")

    catalog = subparsers.add_parser("build-catalog")
    catalog.add_argument("--output", type=Path, required=True)

    manifest = subparsers.add_parser("validate-manifest")
    manifest.add_argument("--manifest", type=Path, required=True)
    manifest.add_argument("--allow-missing-audio", action="store_true")

    device_info = subparsers.add_parser("collect-device-info")
    device_info.add_argument("--output", type=Path, required=True)
    device_info.add_argument("--artifact", type=Path, action="append", default=[])

    infer = subparsers.add_parser("infer")
    _add_backend_arguments(infer)
    infer.add_argument("--audio", type=Path, required=True)
    infer.add_argument("--catalog", type=Path)

    evaluate = subparsers.add_parser("evaluate")
    _add_backend_arguments(evaluate)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--catalog", type=Path)

    stream = subparsers.add_parser("stream")
    _add_backend_arguments(stream)
    stream.add_argument("--audio", type=Path, required=True)
    stream.add_argument("--chunk-ms", type=int, default=40)
    stream.add_argument("--catalog", type=Path)
    return parser


def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-contract":
            contract = ModelContract.load(args.contract)
            contract.validate(require_artifacts=args.require_artifacts)
            print(json.dumps(contract.to_dict(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "build-catalog":
            write_catalog(args.output)
            print(f"Command catalog written: {args.output}")
            return 0
        if args.command == "validate-manifest":
            samples = load_manifest(
                args.manifest, require_audio=not args.allow_missing_audio
            )
            print(json.dumps({"samples": len(samples), "status": "ok"}))
            return 0
        if args.command == "collect-device-info":
            payload = collect_device_info(args.output, args.artifact)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        backend = _create_backend(args)
        catalog = load_catalog(args.catalog) if args.catalog else None
        matcher = CommandMatcher(catalog)
        if args.command == "infer":
            audio = read_wav(args.audio, target_rate=backend.sample_rate)
            started = time.perf_counter()
            result = backend.recognize(audio.samples, audio.sample_rate)
            match = matcher.match(result.text)
            payload = {
                "text": result.text,
                "corrected_text": match.corrected_text,
                "intent": match.intent,
                "slots": match.slots,
                "command_confidence": match.confidence,
                "command_rejected": match.rejected,
                "inference_ms": result.inference_ms,
                "total_ms": (time.perf_counter() - started) * 1000.0,
                "backend": result.backend,
            }
            print(json.dumps(payload, ensure_ascii=False))
            return 0
        if args.command == "evaluate":
            summary = run_evaluation(
                args.manifest, backend, args.output, matcher=matcher
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if args.command == "stream":
            audio = read_wav(args.audio, target_rate=backend.sample_rate)
            vad = EnergyVAD(VADConfig(sample_rate=backend.sample_rate))
            segments = simulate_stream(audio.samples, vad, chunk_ms=args.chunk_ms)
            payload = []
            for index, segment in enumerate(segments, start=1):
                result = backend.recognize(
                    segment.samples,
                    backend.sample_rate,
                    context={"sample_id": f"segment-{index}"},
                )
                match = matcher.match(result.text)
                payload.append(
                    {
                        "segment": index,
                        "start_ms": segment.start_sample * 1000.0 / backend.sample_rate,
                        "end_ms": segment.end_sample * 1000.0 / backend.sample_rate,
                        "reason": segment.reason,
                        "text": result.text,
                        "corrected_text": match.corrected_text,
                        "intent": match.intent,
                        "slots": match.slots,
                    }
                )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        raise ValueError(f"unsupported command: {args.command}")
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend", choices=("mock", "onnx", "device"), required=True
    )
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--mock-text", default="")
    parser.add_argument("--mock-map", type=Path)
    parser.add_argument(
        "--device-command",
        help="Quoted command containing {wav}; device program must print JSON",
    )
    parser.add_argument(
        "--device-command-json",
        help="JSON array form of the device command; safer for paths with spaces",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)


def _create_backend(args: argparse.Namespace):
    if args.backend == "onnx":
        if not args.contract:
            raise ValueError("--contract is required for the ONNX backend")
        return OnnxBackend(ModelContract.load(args.contract))
    if args.backend == "device":
        if bool(args.device_command) == bool(args.device_command_json):
            raise ValueError(
                "set exactly one of --device-command or --device-command-json"
            )
        if args.device_command_json:
            command = json.loads(args.device_command_json)
            if not isinstance(command, list) or not all(
                isinstance(item, str) for item in command
            ):
                raise ValueError("--device-command-json must be a JSON string array")
        else:
            command = shlex.split(args.device_command)
        return SubprocessBackend(command, sample_rate=args.sample_rate)
    hypotheses = {}
    if args.mock_map:
        with args.mock_map.open("r", encoding="utf-8") as handle:
            hypotheses = json.load(handle)
    return MockBackend(
        text=args.mock_text,
        hypotheses=hypotheses,
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
