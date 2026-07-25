"""Generate traceable JSONL, JSON, and Markdown reports from real results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_report_bundle(
    output_dir: Path,
    results: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "results.jsonl", results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_markdown(summary, results), encoding="utf-8"
    )


def render_markdown(
    summary: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> str:
    raw = summary["raw"]
    corrected = summary["command_aware"]
    latency = summary["latency_ms"]
    rtf = summary["rtf"]
    lines = [
        "# InCar-ASR 可复现实验报告",
        "",
        "> 本报告由 `results.jsonl` 自动生成，不包含随机或手工填写的性能数据。",
        "",
        "## 总体结果",
        "",
        "| 指标 | 原始 ASR | 命令增强后 |",
        "|---|---:|---:|",
        f"| Corpus CER | {raw['corpus_cer']:.2%} | {corrected['corpus_cer']:.2%} |",
        (
            f"| 指令完全匹配率 | {raw['exact_match_accuracy']:.2%} | "
            f"{corrected['exact_match_accuracy']:.2%} |"
        ),
        f"| 平均句级 CER | {raw['mean_utterance_cer']:.2%} | {corrected['mean_utterance_cer']:.2%} |",
        "",
        f"- 有效样本：{summary['sample_count']}",
        f"- 失败样本：{summary['failed_count']}",
        f"- 意图准确率：{_optional_percent(summary.get('intent_accuracy'))}",
        f"- 槽位 Micro-F1：{_optional_percent(summary.get('slot_micro_f1'))}",
        f"- 命令拒绝率：{corrected['rejection_rate']:.2%}",
        "",
        "## 性能",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 平均延迟 | {latency['mean']:.2f} ms |",
        f"| P50 延迟 | {latency['p50']:.2f} ms |",
        f"| P95 延迟 | {latency['p95']:.2f} ms |",
        f"| P99 延迟 | {latency['p99']:.2f} ms |",
        f"| 平均 RTF | {rtf['mean']:.4f} |",
        f"| P95 RTF | {rtf['p95']:.4f} |",
        "",
    ]
    for group_name, title in (("noise_type", "按噪声类型"), ("snr_db", "按 SNR")):
        lines.extend(
            [
                f"## {title}",
                "",
                "| 分组 | 样本数 | 原始 CER | 增强后 CER | 指令准确率 | 平均 RTF |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for label, values in summary["groups"][group_name].items():
            lines.append(
                f"| {label} | {values['samples']} | {values['raw_cer']:.2%} | "
                f"{values['corrected_cer']:.2%} | "
                f"{values['exact_match_accuracy']:.2%} | {values['mean_rtf']:.4f} |"
            )
        lines.append("")

    bad_cases = sorted(
        (item for item in results if not item.get("error")),
        key=lambda item: float(item.get("raw_cer", 0.0)),
        reverse=True,
    )[:20]
    lines.extend(
        [
            "## Bad Cases",
            "",
            "| Sample | 标注 | 原始识别 | 命令增强 | 原始 CER |",
            "|---|---|---|---|---:|",
        ]
    )
    for item in bad_cases:
        lines.append(
            f"| {item['sample_id']} | {_cell(item['reference'])} | "
            f"{_cell(item['hypothesis'])} | {_cell(item['corrected_text'])} | "
            f"{float(item['raw_cer']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## 可追溯性",
            "",
            "- 每个样本的原始识别、纠错结果、耗时、噪声与 SNR 均位于 `results.jsonl`。",
            "- 汇总计算结果位于 `summary.json`。",
            "- 如果有效样本数为 0，评测程序会失败，不会生成“通过”结果。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _optional_percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2%}"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
