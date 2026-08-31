"""Collect reproducible device and artifact metadata without exposing secrets."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def collect_device_info(
    output: Path,
    artifacts: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "tools": {
            "npu_smi": _run_version(["npu-smi", "info"]),
            "atc": _run_version(["atc", "--version"]),
            "cmake": _run_version(["cmake", "--version"]),
            "compiler": _run_version(["c++", "--version"]),
        },
        "artifacts": [],
    }
    for value in artifacts or []:
        path = value.expanduser().resolve()
        item: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            item.update({"bytes": path.stat().st_size, "sha256": _sha256(path)})
        payload["artifacts"].append(item)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _run_version(command: Sequence[str]) -> Dict[str, Any]:
    if not command or shutil.which(command[0]) is None:
        return {"available": False, "command": list(command), "output": ""}
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        combined = (completed.stdout or completed.stderr).strip()
        return {
            "available": True,
            "command": list(command),
            "returncode": completed.returncode,
            "output": "\n".join(combined.splitlines()[:80]),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": True,
            "command": list(command),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
