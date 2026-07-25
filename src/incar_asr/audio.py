"""Portable audio I/O, feature extraction, LFR, CMVN, and SNR mixing."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .contracts import FrontendConfig


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return float(self.samples.size / self.sample_rate)


@dataclass(frozen=True)
class FeatureResult:
    features: np.ndarray
    fbank_frames: int
    lfr_frames: int


@dataclass(frozen=True)
class MixResult:
    samples: np.ndarray
    requested_snr_db: float
    achieved_snr_db: float
    noise_gain: float
    peak_scale: float


def read_wav(path: Union[Path, str], target_rate: Optional[int] = None) -> AudioData:
    """Read PCM WAV as mono float32 and optionally resample it."""
    source = Path(path)
    with wave.open(str(source), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        compression = handle.getcomptype()
        raw = handle.readframes(frame_count)
    if compression != "NONE":
        raise ValueError(f"compressed WAV is unsupported: {source}")
    samples = _pcm_to_float(raw, sample_width)
    if channels <= 0 or samples.size % channels:
        raise ValueError(f"invalid channel layout in WAV: {source}")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    if target_rate and sample_rate != target_rate:
        samples = resample_audio(samples, sample_rate, target_rate)
        sample_rate = target_rate
    return AudioData(np.asarray(samples, dtype=np.float32), int(sample_rate))


def write_wav(
    path: Union[Path, str], samples: np.ndarray, sample_rate: int = 16000
) -> None:
    """Write mono PCM16 WAV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(samples, dtype=np.float32)
    data = np.clip(data, -1.0, 1.0)
    pcm = np.round(data * 32767.0).astype("<i2")
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Deterministic dependency-free linear resampling."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    data = np.asarray(samples, dtype=np.float32)
    if source_rate == target_rate or data.size == 0:
        return data.copy()
    output_size = max(1, int(round(data.size * target_rate / source_rate)))
    source_positions = np.arange(data.size, dtype=np.float64)
    target_positions = np.arange(output_size, dtype=np.float64) * source_rate / target_rate
    target_positions = np.minimum(target_positions, max(0, data.size - 1))
    return np.interp(target_positions, source_positions, data).astype(np.float32)


def extract_features(samples: np.ndarray, config: FrontendConfig) -> FeatureResult:
    """Extract log Mel filterbank features, LFR stack, and optional CMVN."""
    config.validate()
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("cannot extract features from empty audio")

    if config.preemphasis:
        emphasized = np.empty_like(audio)
        emphasized[0] = audio[0]
        emphasized[1:] = audio[1:] - config.preemphasis * audio[:-1]
    else:
        emphasized = audio

    frame_length = int(round(config.sample_rate * config.frame_length_ms / 1000.0))
    frame_shift = int(round(config.sample_rate * config.frame_shift_ms / 1000.0))
    if audio.size < frame_length:
        emphasized = np.pad(emphasized, (0, frame_length - audio.size))
    frame_count = 1 + (emphasized.size - frame_length) // frame_shift
    indices = (
        np.arange(frame_length, dtype=np.int64)[None, :]
        + np.arange(frame_count, dtype=np.int64)[:, None] * frame_shift
    )
    frames = emphasized[indices]
    window = np.hamming(frame_length).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, n=config.fft_size, axis=1)
    magnitude = np.abs(spectrum).astype(np.float32)
    spectral_values = magnitude**2 if config.spectrum == "power" else magnitude
    filters = build_mel_filterbank(
        config.sample_rate, config.fft_size, config.mel_bins
    )
    mel_energy = spectral_values @ filters.T
    fbank = np.log(np.maximum(mel_energy, np.finfo(np.float32).eps)).astype(np.float32)
    lfr = apply_lfr(fbank, config.lfr_m, config.lfr_n, config.lfr_padding)
    if config.cmvn_means:
        means = np.asarray(config.cmvn_means, dtype=np.float32)
        scales = np.asarray(config.cmvn_scales, dtype=np.float32)
        lfr = (lfr + means) * scales
    return FeatureResult(
        features=np.ascontiguousarray(lfr, dtype=np.float32),
        fbank_frames=int(fbank.shape[0]),
        lfr_frames=int(lfr.shape[0]),
    )


def build_mel_filterbank(
    sample_rate: int, fft_size: int, mel_bins: int
) -> np.ndarray:
    """Build a triangular HTK-style Mel filterbank."""
    fft_bins = fft_size // 2 + 1
    mel_low = _hz_to_mel(0.0)
    mel_high = _hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(mel_low, mel_high, mel_bins + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, fft_bins - 1)
    filters = np.zeros((mel_bins, fft_bins), dtype=np.float32)
    for index in range(mel_bins):
        left, center, right = bin_points[index : index + 3]
        if center > left:
            filters[index, left:center] = (
                np.arange(left, center) - left
            ) / float(center - left)
        if right > center:
            filters[index, center:right] = (
                right - np.arange(center, right)
            ) / float(right - center)
    return filters


def apply_lfr(
    fbank: np.ndarray, lfr_m: int, lfr_n: int, padding: str = "edge"
) -> np.ndarray:
    """Stack ``lfr_m`` frames every ``lfr_n`` frames with deterministic padding."""
    features = np.asarray(fbank, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("FBank input must be a non-empty [T,D] matrix")
    if lfr_m == 1 and lfr_n == 1:
        return features.copy()
    left = (lfr_m - 1) // 2
    pad_mode = "edge" if padding == "edge" else "constant"
    padded = np.pad(features, ((left, lfr_m, ), (0, 0)), mode=pad_mode)
    output_frames = int(math.ceil(features.shape[0] / lfr_n))
    output = np.empty((output_frames, features.shape[1] * lfr_m), dtype=np.float32)
    for index in range(output_frames):
        start = index * lfr_n
        window = padded[start : start + lfr_m]
        if window.shape[0] < lfr_m:
            window = np.pad(
                window,
                ((0, lfr_m - window.shape[0]), (0, 0)),
                mode=pad_mode,
            )
        output[index] = window.reshape(-1)
    return output


def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    seed: int = 0,
    peak_limit: float = 0.99,
) -> MixResult:
    """Mix clean speech and noise at a measured SNR without writing an audio file."""
    clean_data = np.asarray(clean, dtype=np.float32).reshape(-1)
    noise_data = np.asarray(noise, dtype=np.float32).reshape(-1)
    if clean_data.size == 0:
        raise ValueError("clean audio is empty")
    if noise_data.size == 0:
        raise ValueError("noise audio is empty")
    clean_rms = _rms(clean_data)
    noise_rms = _rms(noise_data)
    if clean_rms <= 1e-8:
        raise ValueError("clean audio has no measurable signal")
    if noise_rms <= 1e-8:
        raise ValueError("noise audio has no measurable signal")

    rng = np.random.default_rng(seed)
    if noise_data.size < clean_data.size:
        repeats = int(math.ceil(clean_data.size / noise_data.size))
        noise_data = np.tile(noise_data, repeats)
    max_start = noise_data.size - clean_data.size
    start = int(rng.integers(0, max_start + 1)) if max_start else 0
    aligned_noise = noise_data[start : start + clean_data.size]
    aligned_noise = aligned_noise - float(np.mean(aligned_noise))
    noise_rms = _rms(aligned_noise)
    desired_noise_rms = clean_rms / (10.0 ** (float(snr_db) / 20.0))
    noise_gain = desired_noise_rms / max(noise_rms, 1e-12)
    scaled_noise = aligned_noise * noise_gain
    mixed = clean_data + scaled_noise
    peak = float(np.max(np.abs(mixed)))
    peak_scale = min(1.0, peak_limit / peak) if peak > 0 else 1.0
    mixed = mixed * peak_scale
    achieved = 20.0 * math.log10(
        max(_rms(clean_data * peak_scale), 1e-12)
        / max(_rms(scaled_noise * peak_scale), 1e-12)
    )
    return MixResult(
        samples=np.asarray(mixed, dtype=np.float32),
        requested_snr_db=float(snr_db),
        achieved_snr_db=float(achieved),
        noise_gain=float(noise_gain),
        peak_scale=float(peak_scale),
    )


def _pcm_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            data[:, 0].astype(np.int32)
            | (data[:, 1].astype(np.int32) << 8)
            | (data[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")


def _rms(samples: np.ndarray) -> float:
    data = np.asarray(samples, dtype=np.float64)
    return float(np.sqrt(np.mean(data * data))) if data.size else 0.0


def _hz_to_mel(hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)
