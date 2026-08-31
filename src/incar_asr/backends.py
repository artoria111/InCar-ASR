"""Interchangeable inference backends for local, CI, and Atlas execution."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

import numpy as np

from .audio import extract_features, write_wav
from .contracts import ModelContract


@dataclass(frozen=True)
class BackendResult:
    text: str
    inference_ms: float
    backend: str
    raw: Dict[str, Any] = field(default_factory=dict)


class ASRBackend(Protocol):
    name: str
    sample_rate: int

    def recognize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        context: Optional[Mapping[str, Any]] = None,
    ) -> BackendResult:
        ...


class MockBackend:
    """Deterministic backend used by unit tests and pipeline smoke tests."""

    name = "mock"

    def __init__(
        self,
        text: str = "",
        hypotheses: Optional[Mapping[str, str]] = None,
        sample_rate: int = 16000,
    ):
        self.text = text
        self.hypotheses = dict(hypotheses or {})
        self.sample_rate = sample_rate

    def recognize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        context: Optional[Mapping[str, Any]] = None,
    ) -> BackendResult:
        sample_id = str((context or {}).get("sample_id", ""))
        text = self.hypotheses.get(sample_id, self.text)
        return BackendResult(
            text=text,
            inference_ms=0.0,
            backend=self.name,
            raw={"sample_count": int(np.asarray(audio).size)},
        )


class OnnxBackend:
    """ONNX Runtime reference backend driven entirely by ``ModelContract``."""

    name = "onnxruntime"

    def __init__(self, contract: ModelContract, providers: Optional[List[str]] = None):
        contract.validate(require_artifacts=True)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed; run `python -m pip install -e '.[onnx]'`"
            ) from exc
        self.contract = contract
        self.sample_rate = contract.sample_rate
        self.tokens = load_tokens(contract.tokens_path)
        self.session = ort.InferenceSession(
            str(contract.model_path),
            providers=providers or ["CPUExecutionProvider"],
        )
        self._validate_session_io()

    def recognize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        context: Optional[Mapping[str, Any]] = None,
    ) -> BackendResult:
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"backend expects {self.sample_rate} Hz but received {sample_rate} Hz"
            )
        feature_result = extract_features(audio, self.contract.frontend)
        features = feature_result.features
        if self.contract.input_dtype == "float16":
            features = features.astype(np.float16)
        inputs: Dict[str, np.ndarray] = {
            self.contract.features_name: features[np.newaxis, :, :]
        }
        if self.contract.lengths_name:
            inputs[self.contract.lengths_name] = np.asarray(
                [features.shape[0]], dtype=np.int32
            )

        output_names = [self.contract.logits_name]
        if self.contract.output_lengths_name:
            output_names.append(self.contract.output_lengths_name)
        start = time.perf_counter()
        outputs = self.session.run(output_names, inputs)
        inference_ms = (time.perf_counter() - start) * 1000.0
        logits = np.asarray(outputs[0])
        if logits.ndim == 3:
            logits = logits[0]
        if logits.ndim != 2:
            raise RuntimeError(f"expected [T,V] logits, received shape {logits.shape}")
        token_count = logits.shape[0]
        if self.contract.output_lengths_name:
            token_count = min(token_count, int(np.asarray(outputs[1]).reshape(-1)[0]))
        token_ids = np.argmax(logits[:token_count], axis=-1).astype(int).tolist()
        if self.contract.decoder_type == "ctc":
            token_ids = collapse_ctc(token_ids, self.contract.blank_id)
        text = tokens_to_text(
            token_ids, self.tokens, set(self.contract.special_token_ids)
        )
        return BackendResult(
            text=text,
            inference_ms=inference_ms,
            backend=self.name,
            raw={
                "fbank_frames": feature_result.fbank_frames,
                "feature_frames": feature_result.lfr_frames,
                "token_count": len(token_ids),
            },
        )

    def _validate_session_io(self) -> None:
        inputs = {item.name: item for item in self.session.get_inputs()}
        outputs = {item.name: item for item in self.session.get_outputs()}
        input_names = set(inputs)
        output_names = set(outputs)
        required_inputs = {self.contract.features_name}
        if self.contract.lengths_name:
            required_inputs.add(self.contract.lengths_name)
        required_outputs = {self.contract.logits_name}
        if self.contract.output_lengths_name:
            required_outputs.add(self.contract.output_lengths_name)
        if not required_inputs.issubset(input_names):
            raise ValueError(
                f"contract inputs {sorted(required_inputs)} are not present in "
                f"model inputs {sorted(input_names)}"
            )
        if not required_outputs.issubset(output_names):
            raise ValueError(
                f"contract outputs {sorted(required_outputs)} are not present in "
                f"model outputs {sorted(output_names)}"
            )
        feature_input = inputs[self.contract.features_name]
        expected_input_type = f"tensor({self.contract.input_dtype.replace('32', '')})"
        if feature_input.type != expected_input_type:
            raise ValueError(
                f"contract input dtype {self.contract.input_dtype} does not match "
                f"ONNX type {feature_input.type}"
            )
        if len(feature_input.shape) != 3:
            raise ValueError(
                f"ONNX feature input must be [B,T,D], got {feature_input.shape}"
            )
        model_dimension = feature_input.shape[-1]
        if isinstance(model_dimension, int) and model_dimension != self.contract.frontend.output_dim:
            raise ValueError(
                f"ONNX input dimension {model_dimension} does not match frontend "
                f"dimension {self.contract.frontend.output_dim}"
            )
        if self.contract.lengths_name:
            length_input = inputs[self.contract.lengths_name]
            if length_input.type not in ("tensor(int64)", "tensor(int32)"):
                raise ValueError("length input must use int64 or int32")
        logits_output = outputs[self.contract.logits_name]
        expected_output_type = f"tensor({self.contract.output_dtype.replace('32', '')})"
        if logits_output.type != expected_output_type:
            raise ValueError(
                f"contract output dtype {self.contract.output_dtype} does not match "
                f"ONNX type {logits_output.type}"
            )
        if self.contract.output_lengths_name:
            length_output = outputs[self.contract.output_lengths_name]
            if length_output.type not in ("tensor(int64)", "tensor(int32)"):
                raise ValueError("output length tensor must use int64 or int32")


class SubprocessBackend:
    """Adapter for a device CLI that accepts a WAV path and prints one JSON object."""

    name = "ascend-cli"

    def __init__(
        self,
        command: Sequence[str],
        sample_rate: int = 16000,
        timeout_seconds: float = 120.0,
    ):
        if not command:
            raise ValueError("device command cannot be empty")
        if not any("{wav}" in item for item in command):
            raise ValueError("device command must contain a {wav} placeholder")
        self.command = list(command)
        self.sample_rate = sample_rate
        self.timeout_seconds = timeout_seconds

    def recognize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        context: Optional[Mapping[str, Any]] = None,
    ) -> BackendResult:
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"device backend expects {self.sample_rate} Hz, got {sample_rate} Hz"
            )
        with tempfile.TemporaryDirectory(prefix="incar-asr-") as temporary:
            wav_path = Path(temporary) / "input.wav"
            write_wav(wav_path, audio, sample_rate)
            command = [item.replace("{wav}", str(wav_path)) for item in self.command]
            start = time.perf_counter()
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            wall_ms = (time.perf_counter() - start) * 1000.0
        if completed.returncode != 0:
            raise RuntimeError(
                f"device command failed ({completed.returncode}): "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        payload = _last_json_object(completed.stdout)
        if "text" not in payload:
            raise RuntimeError("device JSON output must contain a 'text' field")
        return BackendResult(
            text=str(payload["text"]),
            inference_ms=float(payload.get("inference_ms", wall_ms)),
            backend=self.name,
            raw={**payload, "device_wall_ms": wall_ms},
        )


def load_tokens(path: Path) -> Dict[int, str]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return {index: str(token) for index, token in enumerate(data)}
        if isinstance(data, dict):
            if all(str(key).lstrip("-").isdigit() for key in data):
                return {int(key): str(value) for key, value in data.items()}
            return {int(value): str(key) for key, value in data.items()}
        raise ValueError("token JSON must be a list or object")
    with path.open("r", encoding="utf-8") as handle:
        return {index: line.rstrip("\r\n") for index, line in enumerate(handle)}


def collapse_ctc(token_ids: Sequence[int], blank_id: int) -> List[int]:
    output: List[int] = []
    previous = blank_id
    for token_id in token_ids:
        if token_id != blank_id and token_id != previous:
            output.append(int(token_id))
        previous = int(token_id)
    return output


def tokens_to_text(
    token_ids: Sequence[int], tokens: Mapping[int, str], special_ids: set
) -> str:
    pieces: List[str] = []
    for token_id in token_ids:
        if token_id in special_ids:
            continue
        token = tokens.get(int(token_id), "")
        if token in {"<blank>", "<unk>", "<sos>", "<eos>"}:
            continue
        if token.startswith("▁"):
            token = (" " if pieces else "") + token[1:]
        pieces.append(token)
    return "".join(pieces).strip()


def _last_json_object(output: str) -> Dict[str, Any]:
    for line in reversed(output.splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("device command did not print a JSON object")
