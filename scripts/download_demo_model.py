#!/usr/bin/env python3
"""Download and verify the small offline Paraformer model used by the demo."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

try:
    from scripts.baseline_config import load_baseline_config
except ModuleNotFoundError:
    from baseline_config import load_baseline_config

BASELINE_CONFIG = load_baseline_config()
MODEL_NAME = BASELINE_CONFIG["model"]["name"]
MODEL_URL = BASELINE_CONFIG["model"]["archive_url"]
MODEL_ARCHIVE_SHA256 = BASELINE_CONFIG["model"]["archive_sha256"]
REQUIRED_FILES = (
    BASELINE_CONFIG["model"]["model_file"],
    BASELINE_CONFIG["model"]["tokens_file"],
    "test_wavs/0.wav",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_is_complete(model_dir: Path) -> bool:
    return all((model_dir / relative_path).is_file() for relative_path in REQUIRED_FILES)


def _safe_members(archive: tarfile.TarFile, destination: Path):
    destination = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if destination not in member_path.parents and member_path != destination:
            raise ValueError(f"Unsafe archive path: {member.name}")
        yield member


def extract_archive(archive_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:bz2") as archive:
        archive.extractall(
            destination,
            members=_safe_members(archive, destination),
            filter="data",
        )

    model_dir = destination / MODEL_NAME
    if not model_is_complete(model_dir):
        missing = [
            relative_path
            for relative_path in REQUIRED_FILES
            if not (model_dir / relative_path).is_file()
        ]
        raise RuntimeError(f"Extracted model is incomplete; missing: {', '.join(missing)}")
    return model_dir


def download_model(
    destination: Path,
    force: bool = False,
    archive: Path | None = None,
) -> Path:
    model_dir = destination / MODEL_NAME
    if model_is_complete(model_dir) and not force:
        print(f"Model ready: {model_dir}")
        return model_dir

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="incar-asr-model-") as temp_dir:
        if archive is None:
            archive_path = Path(temp_dir) / f"{MODEL_NAME}.tar.bz2"
            print(f"Downloading {MODEL_URL}")
            with urllib.request.urlopen(MODEL_URL) as response:
                with archive_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
        else:
            archive_path = archive.resolve()
            if not archive_path.is_file():
                raise FileNotFoundError(f"Model archive does not exist: {archive_path}")
            print(f"Using local archive: {archive_path}")

        actual_sha256 = sha256_file(archive_path)
        if actual_sha256 != MODEL_ARCHIVE_SHA256:
            raise RuntimeError(
                "Model checksum mismatch: "
                f"expected {MODEL_ARCHIVE_SHA256}, got {actual_sha256}"
            )

        if model_dir.exists():
            shutil.rmtree(model_dir)
        extracted_dir = extract_archive(archive_path, destination)

    print(f"Verified SHA-256: {MODEL_ARCHIVE_SHA256}")
    print(f"Model ready: {extracted_dir}")
    return extracted_dir


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=repository_root / "models",
        help="Directory in which the model directory will be created",
    )
    parser.add_argument("--force", action="store_true", help="Download again")
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an already-downloaded archive instead of accessing the network",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_model(args.destination, force=args.force, archive=args.archive)


if __name__ == "__main__":
    main()
