from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "audio"
sys.path.insert(0, str(SCRIPT_DIR))

from common import audio_metrics, load_config, safe_slug  # noqa: E402
import collect_audio as collect_module  # noqa: E402
from preprocess_audio import (  # noqa: E402
    choose_split,
    process_file,
    resample_audio,
    segment_audio,
    to_mono,
)
from validate_dataset import inspect_audio_file  # noqa: E402


def test_repository_config_has_at_least_ten_categories() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "audio.yaml"
    config = load_config(config_path)
    assert config["audio"]["sample_rate"] == 16000
    assert len(config["collection"]["categories"]) >= 10


def test_mono_resample_and_metrics() -> None:
    source_rate = 8000
    time = np.arange(source_rate, dtype=np.float32) / source_rate
    tone = 0.2 * np.sin(2 * np.pi * 440 * time)
    stereo = np.column_stack((tone, tone))

    mono = to_mono(stereo)
    resampled = resample_audio(mono, source_rate, 16000)
    metrics = audio_metrics(resampled)

    assert mono.shape == (source_rate,)
    assert abs(resampled.size - 16000) <= 1
    assert -20 < metrics["rms_dbfs"] < -15
    assert metrics["clipped_ratio"] == 0


def test_segment_audio_pads_only_valid_tail() -> None:
    samples = np.arange(25, dtype=np.float32)
    segments = list(
        segment_audio(
            samples,
            sample_rate=1,
            segment_seconds=10,
            hop_seconds=10,
            pad_last=True,
            min_tail_seconds=5,
        )
    )
    assert [segment.start_seconds for segment in segments] == [0, 10, 20]
    assert all(segment.samples.size == 10 for segment in segments)
    assert np.all(segments[-1].samples[5:] == 0)


def test_split_is_deterministic_per_source_recording() -> None:
    ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
    assert choose_split("recording-123", ratios) == choose_split("recording-123", ratios)
    assert choose_split("recording-123", ratios) in ratios


def test_process_and_validate_audio(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    source = input_root / "engine_idle" / "sample.wav"
    source.parent.mkdir(parents=True)
    rate = 8000
    time = np.arange(rate * 3, dtype=np.float32) / rate
    tone = 0.1 * np.sin(2 * np.pi * 220 * time)
    sf.write(source, np.column_stack((tone, tone)), rate)

    config = {
        "audio": {"sample_rate": 16000, "channels": 1, "subtype": "PCM_16"},
        "preprocessing": {
            "remove_dc_offset": True,
            "highpass_hz": 0,
            "trim_silence": False,
            "normalize_rms": False,
            "segment_seconds": 2,
            "hop_seconds": 2,
            "pad_last_segment": True,
            "minimum_tail_seconds": 1,
            "dataset_splits": {"train": 1.0},
        },
    }
    rows = process_file(source, input_root, output_root, {}, config, overwrite=False)

    assert len(rows) == 2
    output_files = list(output_root.rglob("*.wav"))
    assert len(output_files) == 2
    stats, issues = inspect_audio_file(
        output_files[0],
        expected_rate=16000,
        expected_channels=1,
        validation={
            "minimum_duration_seconds": 1,
            "maximum_clipped_ratio": 0.001,
            "minimum_rms_dbfs": -55,
            "maximum_dc_offset": 0.02,
        },
    )
    assert stats["sample_rate"] == 16000
    assert not [issue for issue in issues if issue.severity == "error"]


def test_safe_slug() -> None:
    assert safe_slug("Road Asphalt") == "road-asphalt"


def test_collection_writes_audio_and_metadata(tmp_path: Path, monkeypatch) -> None:
    class FakeSoundDevice:
        @staticmethod
        def query_devices(device, kind):
            assert kind == "input"
            return {"name": "test microphone"}

        @staticmethod
        def rec(frames, samplerate, channels, dtype, device):
            assert dtype == "float32"
            return np.full((frames, channels), 0.05, dtype=np.float32)

        @staticmethod
        def wait():
            return None

    config_path = tmp_path / "audio.yaml"
    config = {
        "audio": {"sample_rate": 16000, "channels": 1, "subtype": "PCM_16"},
        "collection": {
            "raw_audio_dir": str(tmp_path / "raw"),
            "metadata_file": str(tmp_path / "metadata.csv"),
            "default_duration_seconds": 1,
            "categories": [f"category_{index}" for index in range(10)],
        },
        "preprocessing": {},
        "validation": {},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(collect_module, "import_sounddevice", lambda: FakeSoundDevice())
    args = argparse.Namespace(
        config=config_path,
        category="category_0",
        duration=0.1,
        device=None,
        output_dir=None,
        metadata=None,
        vehicle_state="parked",
        speed_kmh=0,
        road_surface="asphalt",
        weather="dry",
        window_state="closed",
        microphone_position="center_console",
        location_type="test",
        contains_speech="no",
        notes="synthetic test",
    )

    output_path = collect_module.collect(args)

    assert output_path.is_file()
    info = sf.info(output_path)
    assert info.samplerate == 16000
    assert info.channels == 1
    with (tmp_path / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["category"] == "category_0"
    assert rows[0]["device"] == "test microphone"
    assert rows[0]["rms_dbfs"]
