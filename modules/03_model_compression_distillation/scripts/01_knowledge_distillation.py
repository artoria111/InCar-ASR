#!/usr/bin/env python3
"""Distill a FunASR Paraformer teacher into a student using cached features."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class FeatureCache(Dataset):
    def __init__(self, path: Path):
        cache = np.load(path, allow_pickle=False)
        self.features = cache["features"]
        self.feature_lengths = cache["feature_lengths"]
        self.token_ids = cache["token_ids"]
        self.token_lengths = cache["token_lengths"]

    def __len__(self) -> int:
        return len(self.feature_lengths)

    def __getitem__(self, index: int):
        feature_length = int(self.feature_lengths[index])
        token_length = int(self.token_lengths[index])
        return (
            torch.from_numpy(self.features[index, :feature_length].copy()),
            torch.from_numpy(self.token_ids[index, :token_length].copy()),
        )


def collate(batch):
    features, targets = zip(*batch)
    feature_lengths = torch.tensor([len(item) for item in features])
    target_lengths = torch.tensor([len(item) for item in targets])
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True)
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)
    return padded_features, feature_lengths, padded_targets, target_lengths


def encoder_output(model, features, feature_lengths):
    encoded = model.encode(features, feature_lengths)
    hidden, lengths = encoded[0], encoded[1]
    if isinstance(hidden, tuple):
        hidden = hidden[0]
    return hidden, lengths


def ctc_logits(model, hidden):
    if not hasattr(model, "ctc") or not hasattr(model.ctc, "ctc_lo"):
        raise RuntimeError("model does not expose the expected FunASR CTC head")
    return model.ctc.ctc_lo(hidden)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True, help="FunASR hub ID or local path")
    parser.add_argument("--student", required=True, help="FunASR hub ID or local path")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--ctc-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--hidden-weight", type=float, default=0.3)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch size must be positive")

    from funasr import AutoModel

    device = torch.device(args.device)
    teacher_wrapper = AutoModel(model=args.teacher, device="cpu", disable_update=True)
    student_wrapper = AutoModel(model=args.student, device="cpu", disable_update=True)
    teacher = teacher_wrapper.model.to(device).eval()
    student = student_wrapper.model.to(device).train()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    dataset = FeatureCache(args.train_cache)
    if not len(dataset):
        raise RuntimeError("training cache is empty")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )

    probe_features, probe_lengths, _, _ = next(iter(loader))
    probe_features = probe_features.to(device)
    probe_lengths = probe_lengths.to(device)
    with torch.no_grad():
        teacher_hidden, _ = encoder_output(teacher, probe_features, probe_lengths)
        student_hidden, _ = encoder_output(student, probe_features, probe_lengths)
        teacher_vocab = ctc_logits(teacher, teacher_hidden).shape[-1]
        student_vocab = ctc_logits(student, student_hidden).shape[-1]

    hidden_adapter: nn.Module
    if student_hidden.shape[-1] == teacher_hidden.shape[-1]:
        hidden_adapter = nn.Identity()
    else:
        hidden_adapter = nn.Linear(
            student_hidden.shape[-1], teacher_hidden.shape[-1], bias=False
        ).to(device)

    trainable = list(student.parameters()) + list(hidden_adapter.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=1e-5
    )
    supervised_ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        totals = {"loss": 0.0, "ctc": 0.0, "kl": 0.0, "hidden": 0.0}
        batches = 0
        for features, feature_lengths, targets, target_lengths in loader:
            features = features.to(device)
            feature_lengths = feature_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            if int(targets.max()) >= student_vocab:
                raise RuntimeError(
                    "cache token IDs do not belong to the student vocabulary"
                )

            with torch.no_grad():
                teacher_hidden, teacher_lengths = encoder_output(
                    teacher, features, feature_lengths
                )
                teacher_scores = ctc_logits(teacher, teacher_hidden)

            student_hidden, student_lengths = encoder_output(
                student, features, feature_lengths
            )
            student_scores = ctc_logits(student, student_hidden)
            ctc_loss = supervised_ctc(
                F.log_softmax(student_scores, dim=-1).transpose(0, 1),
                targets,
                student_lengths,
                target_lengths,
            )

            shared_time = min(student_scores.shape[1], teacher_scores.shape[1])
            projected = hidden_adapter(student_hidden[:, :shared_time])
            hidden_loss = F.mse_loss(
                projected, teacher_hidden[:, :shared_time].detach()
            )

            if student_vocab == teacher_vocab:
                temperature = args.temperature
                kl_loss = F.kl_div(
                    F.log_softmax(
                        student_scores[:, :shared_time] / temperature, dim=-1
                    ),
                    F.softmax(
                        teacher_scores[:, :shared_time] / temperature, dim=-1
                    ),
                    reduction="batchmean",
                ) * temperature**2
            else:
                kl_loss = student_scores.new_zeros(())

            loss = (
                args.ctc_weight * ctc_loss
                + args.hidden_weight * hidden_loss
                + args.kl_weight * kl_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()

            for name, value in (
                ("loss", loss),
                ("ctc", ctc_loss),
                ("kl", kl_loss),
                ("hidden", hidden_loss),
            ):
                totals[name] += float(value.detach())
            batches += 1

        epoch_result = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "student-distilled.pt"
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "hidden_adapter_state_dict": hidden_adapter.state_dict(),
            "teacher": args.teacher,
            "student": args.student,
            "student_vocab": student_vocab,
            "teacher_vocab": teacher_vocab,
        },
        checkpoint_path,
    )
    report = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "training_completed",
        "teacher": args.teacher,
        "student": args.student,
        "cache": {
            "path": str(args.train_cache.resolve()),
            "sha256": sha256(args.train_cache),
            "samples": len(dataset),
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
        },
        "configuration": vars(args) | {
            "train_cache": str(args.train_cache),
            "output_dir": str(args.output_dir),
        },
        "vocabulary": {
            "teacher": teacher_vocab,
            "student": student_vocab,
            "logit_distillation_enabled": teacher_vocab == student_vocab,
        },
        "history": history,
        "evaluation_status": "pending",
    }
    (args.output_dir / "distillation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Checkpoint: {checkpoint_path}")
    print("Training is complete; CER/ONNX parity remain separate acceptance gates.")


if __name__ == "__main__":
    main()
