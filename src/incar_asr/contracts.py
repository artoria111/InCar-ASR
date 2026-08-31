"""Strict model I/O contract shared by CPU and device backends."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass(frozen=True)
class FrontendConfig:
    sample_rate: int = 16000
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    fft_size: int = 512
    mel_bins: int = 80
    preemphasis: float = 0.97
    spectrum: str = "power"
    lfr_m: int = 1
    lfr_n: int = 1
    lfr_padding: str = "edge"
    cmvn_means: List[float] = field(default_factory=list)
    cmvn_scales: List[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrontendConfig":
        return cls(
            sample_rate=int(data.get("sample_rate", 16000)),
            frame_length_ms=float(data.get("frame_length_ms", 25)),
            frame_shift_ms=float(data.get("frame_shift_ms", 10)),
            fft_size=int(data.get("fft_size", 512)),
            mel_bins=int(data.get("mel_bins", 80)),
            preemphasis=float(data.get("preemphasis", 0.97)),
            spectrum=str(data.get("spectrum", "power")),
            lfr_m=int(data.get("lfr_m", 1)),
            lfr_n=int(data.get("lfr_n", 1)),
            lfr_padding=str(data.get("lfr_padding", "edge")),
            cmvn_means=[float(value) for value in data.get("cmvn_means", [])],
            cmvn_scales=[float(value) for value in data.get("cmvn_scales", [])],
        )

    @property
    def output_dim(self) -> int:
        return self.mel_bins * self.lfr_m

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("frontend.sample_rate must be positive")
        if self.frame_length_ms <= 0 or self.frame_shift_ms <= 0:
            raise ValueError("frontend frame durations must be positive")
        if self.fft_size <= 0 or self.mel_bins <= 0:
            raise ValueError("frontend fft_size and mel_bins must be positive")
        if self.spectrum not in {"power", "magnitude"}:
            raise ValueError("frontend.spectrum must be 'power' or 'magnitude'")
        if self.lfr_m <= 0 or self.lfr_n <= 0:
            raise ValueError("frontend LFR parameters must be positive")
        if self.lfr_padding not in {"edge", "zero"}:
            raise ValueError("frontend.lfr_padding must be 'edge' or 'zero'")
        if bool(self.cmvn_means) != bool(self.cmvn_scales):
            raise ValueError("CMVN means and scales must either both be set or both be empty")
        if self.cmvn_means and len(self.cmvn_means) != self.output_dim:
            raise ValueError(
                f"CMVN dimension {len(self.cmvn_means)} does not match frontend output "
                f"dimension {self.output_dim}"
            )
        if self.cmvn_scales and len(self.cmvn_scales) != self.output_dim:
            raise ValueError("CMVN scales do not match frontend output dimension")


@dataclass(frozen=True)
class ModelContract:
    version: int
    model_name: str
    backend: str
    model_path: Path
    tokens_path: Path
    sample_rate: int
    features_name: str
    lengths_name: Optional[str]
    input_shape: List[int]
    input_dtype: str
    logits_name: str
    output_lengths_name: Optional[str]
    output_dtype: str
    decoder_type: str
    blank_id: int
    special_token_ids: List[int]
    frontend: FrontendConfig
    source_path: Path

    @classmethod
    def load(cls, path: Union[Path, str]) -> "ModelContract":
        source_path = Path(path).expanduser().resolve()
        with source_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        root = source_path.parent
        input_config = data.get("input", {})
        output_config = data.get("output", {})
        decoder_config = data.get("decoder", {})
        frontend = FrontendConfig.from_dict(data.get("frontend", {}))
        contract = cls(
            version=int(data.get("version", 0)),
            model_name=str(data.get("model_name", "")).strip(),
            backend=str(data.get("backend", "")).strip(),
            model_path=_resolve(root, data.get("model_path", "")),
            tokens_path=_resolve(root, data.get("tokens_path", "")),
            sample_rate=int(data.get("sample_rate", frontend.sample_rate)),
            features_name=str(input_config.get("features_name", "")).strip(),
            lengths_name=_optional_text(input_config.get("lengths_name")),
            input_shape=[int(value) for value in input_config.get("shape", [])],
            input_dtype=str(input_config.get("dtype", "float32")),
            logits_name=str(output_config.get("logits_name", "")).strip(),
            output_lengths_name=_optional_text(output_config.get("lengths_name")),
            output_dtype=str(output_config.get("dtype", "float32")),
            decoder_type=str(decoder_config.get("type", "")).strip(),
            blank_id=int(decoder_config.get("blank_id", 0)),
            special_token_ids=[
                int(value) for value in decoder_config.get("special_token_ids", [])
            ],
            frontend=frontend,
            source_path=source_path,
        )
        contract.validate()
        return contract

    def validate(self, require_artifacts: bool = False) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported model contract version: {self.version}")
        if not self.model_name:
            raise ValueError("model_name is required")
        if not self.features_name or not self.logits_name:
            raise ValueError("input.features_name and output.logits_name are required")
        if self.sample_rate != self.frontend.sample_rate:
            raise ValueError("contract and frontend sample rates disagree")
        if self.decoder_type not in {"ctc", "paraformer"}:
            raise ValueError("decoder.type must be 'ctc' or 'paraformer'")
        if self.input_dtype not in {"float32", "float16"}:
            raise ValueError("only float32 and float16 model inputs are supported")
        if self.output_dtype not in {"float32", "float16"}:
            raise ValueError("only float32 and float16 model outputs are supported")
        if len(self.input_shape) != 3:
            raise ValueError("input.shape must be a three-dimensional [B,T,D] shape")
        if self.input_shape[0] not in {-1, 1}:
            raise ValueError("only batch size 1 or a dynamic batch is supported")
        if self.blank_id < 0 or any(value < 0 for value in self.special_token_ids):
            raise ValueError("decoder token ids must be non-negative")
        self.frontend.validate()
        expected_dim = self.frontend.output_dim
        if self.input_shape[-1] not in {-1, expected_dim}:
            raise ValueError(
                f"model input dimension {self.input_shape[-1]} does not match "
                f"frontend output dimension {expected_dim}"
            )
        if require_artifacts:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"model artifact not found: {self.model_path}")
            if not self.tokens_path.is_file():
                raise FileNotFoundError(f"token file not found: {self.tokens_path}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "model_name": self.model_name,
            "backend": self.backend,
            "model_path": str(self.model_path),
            "tokens_path": str(self.tokens_path),
            "sample_rate": self.sample_rate,
            "input_dim": self.frontend.output_dim,
            "decoder_type": self.decoder_type,
            "features_name": self.features_name,
            "lengths_name": self.lengths_name,
            "logits_name": self.logits_name,
            "output_lengths_name": self.output_lengths_name,
        }


def _resolve(root: Path, value: Any) -> Path:
    text = str(value or "").strip()
    path = Path(text).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
