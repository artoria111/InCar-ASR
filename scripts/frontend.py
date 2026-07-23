"""Portable NumPy frontend used for training caches and Atlas parity tests."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from scripts.baseline_config import load_baseline_config


def read_wave_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1:
            raise ValueError(f"{path} must be mono")
        if source.getsampwidth() != 2:
            raise ValueError(f"{path} must be PCM16")
        sample_rate = source.getframerate()
        payload = source.readframes(source.getnframes())
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    if not samples.size:
        raise ValueError(f"{path} is empty")
    return samples, sample_rate


def resample_linear(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    output_length = max(1, round(len(samples) * target_rate / source_rate))
    source_positions = np.linspace(0, len(samples) - 1, num=len(samples))
    target_positions = np.linspace(0, len(samples) - 1, num=output_length)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _mel_filterbank(sample_rate: int, fft_size: int, feature_dim: int) -> np.ndarray:
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_points = np.linspace(
        hz_to_mel(0.0),
        hz_to_mel(sample_rate / 2.0),
        feature_dim + 2,
    )
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, fft_size // 2)
    filters = np.zeros((feature_dim, fft_size // 2 + 1), dtype=np.float32)
    for index in range(feature_dim):
        left, center, right = bins[index : index + 3]
        if center > left:
            filters[index, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)
        if right > center:
            filters[index, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)
    return filters


def extract_lfr_features(
    samples: np.ndarray,
    sample_rate: int,
    config: dict | None = None,
) -> np.ndarray:
    config = config or load_baseline_config()
    frontend = config["frontend"]
    target_rate = int(frontend["sample_rate"])
    samples = resample_linear(samples, sample_rate, target_rate)

    frame_length = target_rate * int(frontend["frame_length_ms"]) // 1000
    frame_shift = target_rate * int(frontend["frame_shift_ms"]) // 1000
    fft_size = int(frontend["fft_size"])
    feature_dim = int(frontend["feature_dim"])
    preemphasis = float(frontend["preemphasis"])

    emphasized = np.empty_like(samples, dtype=np.float32)
    emphasized[0] = samples[0]
    emphasized[1:] = samples[1:] - preemphasis * samples[:-1]
    if len(emphasized) < frame_length:
        emphasized = np.pad(emphasized, (0, frame_length - len(emphasized)))

    frame_count = 1 + (len(emphasized) - frame_length) // frame_shift
    window = np.hamming(frame_length).astype(np.float32)
    frames = np.stack(
        [
            emphasized[index * frame_shift : index * frame_shift + frame_length]
            * window
            for index in range(frame_count)
        ]
    )
    magnitude = np.abs(np.fft.rfft(frames, n=fft_size, axis=1)).astype(np.float32)
    filters = _mel_filterbank(target_rate, fft_size, feature_dim)
    fbank = np.log(np.maximum(magnitude @ filters.T, 1e-10))

    lfr_m = int(frontend["lfr_m"])
    lfr_n = int(frontend["lfr_n"])
    if len(fbank) < lfr_m:
        fbank = np.pad(fbank, ((0, lfr_m - len(fbank)), (0, 0)))
    lfr = np.stack(
        [
            fbank[start : start + lfr_m].reshape(-1)
            for start in range(0, len(fbank) - lfr_m + 1, lfr_n)
        ]
    ).astype(np.float32)

    if frontend["normalization"] == "utterance_global":
        mean = float(lfr.mean())
        standard_deviation = float(lfr.std())
        lfr = (lfr - mean) / max(standard_deviation, 1e-8)
    return lfr.astype(np.float32)
