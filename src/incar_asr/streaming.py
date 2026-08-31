"""Stateful energy VAD and WAV-based streaming simulation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    start_frames: int = 3
    end_frames: int = 15
    pre_roll_ms: int = 200
    min_utterance_ms: int = 400
    max_utterance_ms: int = 15000
    threshold_ratio: float = 3.0
    minimum_rms: float = 0.008
    noise_alpha: float = 0.95


@dataclass(frozen=True)
class SpeechSegment:
    samples: np.ndarray
    start_sample: int
    end_sample: int
    reason: str


class EnergyVAD:
    """Streaming VAD with pre-roll, adaptive noise floor, and hangover."""

    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self.frame_samples = int(
            self.config.sample_rate * self.config.frame_ms / 1000
        )
        self.pre_roll_frames = max(
            1, int(round(self.config.pre_roll_ms / self.config.frame_ms))
        )
        self.reset()

    def reset(self) -> None:
        self.pending = np.empty(0, dtype=np.float32)
        self.pre_roll: Deque[np.ndarray] = deque(maxlen=self.pre_roll_frames)
        self.utterance: List[np.ndarray] = []
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.position = 0
        self.start_sample = 0
        self.noise_floor = self.config.minimum_rms / max(
            self.config.threshold_ratio, 1.0
        )

    def feed(self, chunk: np.ndarray) -> List[SpeechSegment]:
        data = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if data.size:
            self.pending = np.concatenate((self.pending, data))
        events: List[SpeechSegment] = []
        while self.pending.size >= self.frame_samples:
            frame = self.pending[: self.frame_samples]
            self.pending = self.pending[self.frame_samples :]
            event = self._process_frame(frame)
            if event is not None:
                events.append(event)
        return events

    def flush(self) -> List[SpeechSegment]:
        events: List[SpeechSegment] = []
        if self.pending.size:
            frame = np.pad(
                self.pending, (0, self.frame_samples - self.pending.size)
            ).astype(np.float32)
            self.pending = np.empty(0, dtype=np.float32)
            event = self._process_frame(frame)
            if event is not None:
                events.append(event)
        if self.in_speech and self.utterance:
            event = self._finish("flush")
            if event is not None:
                events.append(event)
        return events

    def _process_frame(self, frame: np.ndarray) -> Optional[SpeechSegment]:
        frame_start = self.position
        self.position += self.frame_samples
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        threshold = max(
            self.config.minimum_rms,
            self.noise_floor * self.config.threshold_ratio,
        )
        speech = rms >= threshold

        if not self.in_speech:
            self.pre_roll.append(frame.copy())
            if speech:
                self.speech_run += 1
            else:
                self.speech_run = 0
                self.noise_floor = (
                    self.config.noise_alpha * self.noise_floor
                    + (1.0 - self.config.noise_alpha) * rms
                )
            if self.speech_run >= self.config.start_frames:
                self.in_speech = True
                self.silence_run = 0
                self.utterance = list(self.pre_roll)
                self.start_sample = max(
                    0, self.position - len(self.utterance) * self.frame_samples
                )
                self.pre_roll.clear()
            return None

        self.utterance.append(frame.copy())
        if speech:
            self.silence_run = 0
        else:
            self.silence_run += 1
        utterance_samples = len(self.utterance) * self.frame_samples
        max_samples = int(
            self.config.max_utterance_ms * self.config.sample_rate / 1000
        )
        if utterance_samples >= max_samples:
            return self._finish("max_duration")
        if self.silence_run >= self.config.end_frames:
            return self._finish("silence")
        return None

    def _finish(self, reason: str) -> Optional[SpeechSegment]:
        audio = (
            np.concatenate(self.utterance)
            if self.utterance
            else np.empty(0, dtype=np.float32)
        )
        end_sample = self.position
        start_sample = self.start_sample
        minimum = int(
            self.config.min_utterance_ms * self.config.sample_rate / 1000
        )
        self.in_speech = False
        self.utterance = []
        self.speech_run = 0
        self.silence_run = 0
        self.pre_roll.clear()
        if audio.size < minimum:
            return None
        return SpeechSegment(audio, start_sample, end_sample, reason)


def simulate_stream(
    audio: np.ndarray,
    vad: Optional[EnergyVAD] = None,
    chunk_ms: int = 40,
) -> List[SpeechSegment]:
    detector = vad or EnergyVAD()
    chunk_samples = int(detector.config.sample_rate * chunk_ms / 1000)
    events: List[SpeechSegment] = []
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    for start in range(0, samples.size, chunk_samples):
        events.extend(detector.feed(samples[start : start + chunk_samples]))
    events.extend(detector.flush())
    return events
