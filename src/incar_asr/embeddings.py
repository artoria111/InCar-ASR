"""Semantic intent matching via ONNX sentence embeddings.

Uses bge-small-zh-v1.5 INT8 ONNX model (~23 MB) for CPU inference.
At ~20 ms per embedding on a single Cortex-A55 core, this fits within
the ~500 ms end-to-end latency budget for in-car voice control.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "bge-small-zh"
_SNAPSHOT_DIR: Optional[Path] = None


def _find_snapshot_dir() -> Path:
    """Locate the downloaded HuggingFace snapshot directory."""
    global _SNAPSHOT_DIR
    if _SNAPSHOT_DIR is not None:
        return _SNAPSHOT_DIR

    candidates = list((_MODEL_DIR / "models--Xenova--bge-small-zh-v1.5" / "snapshots").glob("*"))
    if not candidates:
        # Try the simple flat layout (post-copy)
        onnx_path = _MODEL_DIR / "model_int8.onnx"
        if onnx_path.exists():
            _SNAPSHOT_DIR = _MODEL_DIR
            return _SNAPSHOT_DIR
        raise FileNotFoundError(
            f"No bge-small-zh model snapshot found under {_MODEL_DIR}. "
            "Run: python -m incar_asr.download_model"
        )
    _SNAPSHOT_DIR = candidates[0]
    return _SNAPSHOT_DIR


class EmbeddingMatcher:
    """Semantic command matcher using bge-small-zh-v1.5 ONNX embeddings.

    Usage::

        matcher = EmbeddingMatcher()
        matcher.index_catalog(["打开空调", "关闭车窗", ...])
        best_text, score = matcher.match("好热啊")
    """

    def __init__(self, model_dir: Optional[str] = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if model_dir:
            snapshot = Path(model_dir)
        else:
            snapshot = _find_snapshot_dir()

        onnx_path = snapshot / "onnx" / "model_int8.onnx"
        if not onnx_path.exists():
            onnx_path = snapshot / "model_int8.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        tokenizer_path = snapshot / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        self._session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Pre-computed catalog
        self._catalog_texts: List[str] = []
        self._catalog_embeddings: Optional[np.ndarray] = None  # [N, 512]

        # Warm up with a dummy inference
        self.encode("测试")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_catalog(self, texts: Sequence[str]) -> None:
        """Pre-compute and store embeddings for all catalog commands.

        Call this once after construction, or after changing the catalog.
        """
        self._catalog_texts = list(texts)
        if not texts:
            self._catalog_embeddings = None
            return
        embeddings = []
        for text in texts:
            embeddings.append(self.encode(text))
        self._catalog_embeddings = np.stack(embeddings, axis=0).astype(np.float32)

    def match(self, text: str) -> tuple:
        """Return (best_matching_text, cosine_similarity) for `text`.

        Returns (\"\", 0.0) if the catalog has not been indexed yet.
        """
        if self._catalog_embeddings is None or len(self._catalog_texts) == 0:
            return ("", 0.0)
        query = self.encode(text)
        similarities = np.dot(self._catalog_embeddings, query)
        best_idx = int(np.argmax(similarities))
        return (self._catalog_texts[best_idx], float(similarities[best_idx]))

    def all_similarities(self, text: str) -> np.ndarray:
        """Return cosine similarity of `text` against all indexed catalog entries.

        Returns a [N] float32 array, one score per catalog text.
        """
        if self._catalog_embeddings is None:
            return np.array([], dtype=np.float32)
        query = self.encode(text)
        return np.dot(self._catalog_embeddings, query).astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text to a 512-dim L2-normalized embedding."""
        encoding = self._tokenizer.encode(text)
        # Pad or truncate to a reasonable length (model supports up to 512)
        max_len = 512
        input_ids = encoding.ids[:max_len]
        attn_mask = encoding.attention_mask[:max_len] if encoding.attention_mask else [1] * len(input_ids)
        type_ids = encoding.type_ids[:max_len] if encoding.type_ids else [0] * len(input_ids)

        # Pad to max_len or keep as-is (ONNX dynamic batch/seq)
        seq_len = len(input_ids)
        pad_len = max_len - seq_len
        if pad_len > 0:
            input_ids += [0] * pad_len
            attn_mask += [0] * pad_len
            type_ids += [0] * pad_len

        onnx_inputs = {
            "input_ids": np.array([input_ids], dtype=np.int64),
            "attention_mask": np.array([attn_mask], dtype=np.int64),
            "token_type_ids": np.array([type_ids], dtype=np.int64),
        }

        outputs = self._session.run(None, onnx_inputs)
        hidden = np.array(outputs[0])  # [1, seq_len, 512]

        # Mean pooling over non-padded tokens
        mask = np.array(attn_mask, dtype=np.float32).reshape(1, -1, 1)
        masked = hidden * mask
        summed = masked.sum(axis=1)  # [1, 512]
        counts = mask.sum(axis=1).clip(min=1e-9)  # [1, 1]
        mean_pooled = summed / counts  # [1, 512]

        # L2 normalize
        norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        normalized = mean_pooled / norm.clip(min=1e-9)
        return normalized.astype(np.float32).reshape(512)


# ------------------------------------------------------------------
# Singleton for reuse across the process
# ------------------------------------------------------------------
_global_matcher: Optional[EmbeddingMatcher] = None


def get_embedding_matcher() -> EmbeddingMatcher:
    """Return a process-wide singleton EmbeddingMatcher."""
    global _global_matcher
    if _global_matcher is None:
        _global_matcher = EmbeddingMatcher()
    return _global_matcher
