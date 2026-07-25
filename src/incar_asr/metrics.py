"""Evidence-based ASR, command, intent, slot, latency, and grouping metrics."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def cer(self) -> float:
        if self.reference_length == 0:
            return float(self.insertions > 0)
        return self.errors / self.reference_length


def normalize_transcript(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()]+", "", str(text).lower())


def edit_counts(reference: str, hypothesis: str) -> EditCounts:
    ref = list(normalize_transcript(reference))
    hyp = list(normalize_transcript(hypothesis))
    rows: List[List[Tuple[int, int, int, int]]] = [
        [(index, 0, index, 0) for index in range(len(hyp) + 1)]
    ]
    for ref_index in range(1, len(ref) + 1):
        row: List[Tuple[int, int, int, int]] = [(ref_index, 0, 0, ref_index)]
        for hyp_index in range(1, len(hyp) + 1):
            if ref[ref_index - 1] == hyp[hyp_index - 1]:
                previous = rows[ref_index - 1][hyp_index - 1]
                row.append((previous[0], previous[1], previous[2], previous[3]))
                continue
            substitution = rows[ref_index - 1][hyp_index - 1]
            deletion = rows[ref_index - 1][hyp_index]
            insertion = row[hyp_index - 1]
            candidates = [
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
                (deletion[0] + 1, deletion[1], deletion[2], deletion[3] + 1),
                (insertion[0] + 1, insertion[1], insertion[2] + 1, insertion[3]),
            ]
            row.append(min(candidates, key=lambda item: item[0]))
        rows.append(row)
    _, substitutions, insertions, deletions = rows[-1][-1]
    return EditCounts(substitutions, deletions, insertions, len(ref))


def summarize_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid = [item for item in results if not item.get("error")]
    if not valid:
        raise ValueError("evaluation produced zero valid samples")
    raw_counts = [edit_counts(item["reference"], item["hypothesis"]) for item in valid]
    corrected_counts = [
        edit_counts(item["reference"], item["corrected_text"]) for item in valid
    ]
    latencies = [float(item["total_ms"]) for item in valid]
    rtfs = [float(item["rtf"]) for item in valid]
    exact_raw = sum(
        normalize_transcript(item["reference"])
        == normalize_transcript(item["hypothesis"])
        for item in valid
    )
    exact_corrected = sum(
        normalize_transcript(item["reference"])
        == normalize_transcript(item["corrected_text"])
        for item in valid
    )
    intent_rows = [item for item in valid if item.get("expected_intent")]
    intent_correct = sum(
        item.get("predicted_intent") == item.get("expected_intent")
        for item in intent_rows
    )
    slot_counts = _slot_counts(valid)
    summary: Dict[str, Any] = {
        "sample_count": len(valid),
        "failed_count": len(results) - len(valid),
        "raw": {
            "corpus_cer": _corpus_cer(raw_counts),
            "mean_utterance_cer": _mean([item.cer for item in raw_counts]),
            "exact_match_accuracy": exact_raw / len(valid),
        },
        "command_aware": {
            "corpus_cer": _corpus_cer(corrected_counts),
            "mean_utterance_cer": _mean([item.cer for item in corrected_counts]),
            "exact_match_accuracy": exact_corrected / len(valid),
            "rejection_rate": sum(bool(item.get("command_rejected")) for item in valid)
            / len(valid),
        },
        "intent_accuracy": (
            intent_correct / len(intent_rows) if intent_rows else None
        ),
        "slot_micro_f1": _f1(*slot_counts) if any(slot_counts) else None,
        "latency_ms": {
            "mean": _mean(latencies),
            "p50": percentile(latencies, 50),
            "p90": percentile(latencies, 90),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
        },
        "rtf": {
            "mean": _mean(rtfs),
            "p95": percentile(rtfs, 95),
        },
        "groups": {
            "noise_type": _group_results(valid, "noise_type"),
            "snr_db": _group_results(valid, "requested_snr_db"),
        },
    }
    return summary


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _group_results(
    results: Sequence[Mapping[str, Any]], key: str
) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in results:
        value = item.get(key)
        label = "clean" if value is None or value == "" else str(value)
        groups[label].append(item)
    output: Dict[str, Dict[str, float]] = {}
    for label, items in sorted(groups.items()):
        counts = [edit_counts(item["reference"], item["hypothesis"]) for item in items]
        corrected = [
            edit_counts(item["reference"], item["corrected_text"]) for item in items
        ]
        output[label] = {
            "samples": len(items),
            "raw_cer": _corpus_cer(counts),
            "corrected_cer": _corpus_cer(corrected),
            "exact_match_accuracy": sum(
                normalize_transcript(item["reference"])
                == normalize_transcript(item["corrected_text"])
                for item in items
            )
            / len(items),
            "mean_rtf": _mean([float(item["rtf"]) for item in items]),
        }
    return output


def _slot_counts(results: Sequence[Mapping[str, Any]]) -> Tuple[int, int, int]:
    true_positive = false_positive = false_negative = 0
    for item in results:
        expected = {
            (str(key), str(value))
            for key, value in dict(item.get("expected_slots") or {}).items()
        }
        predicted = {
            (str(key), str(value))
            for key, value in dict(item.get("predicted_slots") or {}).items()
        }
        true_positive += len(expected & predicted)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)
    return true_positive, false_positive, false_negative


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def _corpus_cer(items: Iterable[EditCounts]) -> float:
    items = list(items)
    errors = sum(item.errors for item in items)
    reference_length = sum(item.reference_length for item in items)
    return errors / reference_length if reference_length else float(errors > 0)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
