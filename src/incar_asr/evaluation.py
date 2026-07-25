"""Manifest validation and on-the-fly noise robustness evaluation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .audio import mix_at_snr, read_wav, resample_audio
from .backends import ASRBackend
from .commands import CommandMatcher
from .metrics import edit_counts, summarize_results
from .report import write_report_bundle


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    clean_audio: Path
    transcript: str
    intent: Optional[str]
    slots: Dict[str, Any]
    speaker_id: str
    split: str
    noise_audio: Optional[Path] = None
    noise_type: Optional[str] = None
    snr_db: Optional[float] = None
    seed: int = 0


def load_manifest(path: Path, require_audio: bool = True) -> List[EvaluationSample]:
    manifest = path.expanduser().resolve()
    root = manifest.parent
    samples: List[EvaluationSample] = []
    identifiers = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest}:{line_number}: invalid JSON: {exc}") from exc
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"{manifest}:{line_number}: sample_id is required")
            if sample_id in identifiers:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            identifiers.add(sample_id)
            clean_audio = _resolve_audio(root, row.get("clean_audio"))
            noise_value = row.get("noise_audio")
            noise_audio = _resolve_audio(root, noise_value) if noise_value else None
            if require_audio and not clean_audio.is_file():
                raise FileNotFoundError(f"clean audio not found: {clean_audio}")
            if require_audio and noise_audio is not None and not noise_audio.is_file():
                raise FileNotFoundError(f"noise audio not found: {noise_audio}")
            snr_value = row.get("snr_db")
            if noise_audio is not None and snr_value is None:
                raise ValueError(f"{sample_id}: snr_db is required when noise_audio is set")
            samples.append(
                EvaluationSample(
                    sample_id=sample_id,
                    clean_audio=clean_audio,
                    transcript=str(row.get("transcript", "")).strip(),
                    intent=_optional_text(row.get("intent")),
                    slots=dict(row.get("slots") or {}),
                    speaker_id=str(row.get("speaker_id", "")).strip(),
                    split=str(row.get("split", "test")).strip(),
                    noise_audio=noise_audio,
                    noise_type=_optional_text(row.get("noise_type")),
                    snr_db=float(snr_value) if snr_value is not None else None,
                    seed=int(row.get("seed", 0)),
                )
            )
    if not samples:
        raise ValueError(f"manifest contains no samples: {manifest}")
    validate_split_leakage(samples)
    return samples


def validate_split_leakage(samples: Sequence[EvaluationSample]) -> None:
    speakers: Dict[str, set] = {}
    sources: Dict[str, set] = {}
    for sample in samples:
        if sample.speaker_id:
            speakers.setdefault(sample.speaker_id, set()).add(sample.split)
        sources.setdefault(str(sample.clean_audio), set()).add(sample.split)
    leaked_speakers = sorted(key for key, splits in speakers.items() if len(splits) > 1)
    leaked_sources = sorted(key for key, splits in sources.items() if len(splits) > 1)
    if leaked_speakers:
        raise ValueError(
            "speaker leakage across splits: " + ", ".join(leaked_speakers[:10])
        )
    if leaked_sources:
        raise ValueError(
            "source recording leakage across splits: " + ", ".join(leaked_sources[:10])
        )


def run_evaluation(
    manifest_path: Path,
    backend: ASRBackend,
    output_dir: Path,
    matcher: Optional[CommandMatcher] = None,
    peak_limit: float = 0.99,
) -> Dict[str, Any]:
    samples = load_manifest(manifest_path, require_audio=True)
    command_matcher = matcher or CommandMatcher()
    results: List[Dict[str, Any]] = []

    for sample in samples:
        started = time.perf_counter()
        try:
            clean = read_wav(sample.clean_audio, target_rate=backend.sample_rate)
            audio = clean.samples
            achieved_snr = None
            if sample.noise_audio is not None:
                noise = read_wav(sample.noise_audio, target_rate=backend.sample_rate)
                mixed = mix_at_snr(
                    clean.samples,
                    noise.samples,
                    float(sample.snr_db),
                    seed=sample.seed,
                    peak_limit=peak_limit,
                )
                audio = mixed.samples
                achieved_snr = mixed.achieved_snr_db
            backend_result = backend.recognize(
                audio,
                backend.sample_rate,
                context={"sample_id": sample.sample_id},
            )
            match = command_matcher.match(backend_result.text)
            total_ms = (time.perf_counter() - started) * 1000.0
            duration_ms = audio.size * 1000.0 / backend.sample_rate
            raw_edits = edit_counts(sample.transcript, backend_result.text)
            corrected_edits = edit_counts(sample.transcript, match.corrected_text)
            results.append(
                {
                    "sample_id": sample.sample_id,
                    "backend": backend_result.backend,
                    "reference": sample.transcript,
                    "hypothesis": backend_result.text,
                    "corrected_text": match.corrected_text,
                    "expected_intent": sample.intent,
                    "predicted_intent": match.intent,
                    "expected_slots": sample.slots,
                    "predicted_slots": match.slots,
                    "command_confidence": match.confidence,
                    "command_margin": match.margin,
                    "command_rejected": match.rejected,
                    "noise_type": sample.noise_type,
                    "requested_snr_db": sample.snr_db,
                    "achieved_snr_db": achieved_snr,
                    "audio_duration_ms": duration_ms,
                    "inference_ms": backend_result.inference_ms,
                    "total_ms": total_ms,
                    "rtf": total_ms / max(duration_ms, 1e-9),
                    "raw_cer": raw_edits.cer,
                    "corrected_cer": corrected_edits.cer,
                    "backend_details": backend_result.raw,
                    "error": None,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "sample_id": sample.sample_id,
                    "reference": sample.transcript,
                    "noise_type": sample.noise_type,
                    "requested_snr_db": sample.snr_db,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary = summarize_results(results)
    write_report_bundle(output_dir, results, summary)
    return summary


def _resolve_audio(root: Path, value: Any) -> Path:
    if not value:
        raise ValueError("clean_audio path is required")
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
