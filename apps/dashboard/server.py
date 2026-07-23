#!/usr/bin/env python3
"""Dependency-free local dashboard for ASR result and performance history."""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

STATIC_DIR = Path(__file__).resolve().parent
HISTORY: deque[dict[str, Any]] = deque(maxlen=200)
HISTORY_LOCK = threading.Lock()


def add_result(payload: dict[str, Any]) -> dict[str, Any]:
    required = ("text", "duration_seconds", "elapsed_seconds", "rtf")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("missing fields: " + ", ".join(missing))
    clean = {
        "id": int(time.time_ns()),
        "created_at": payload.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%S%z")),
        "source": str(payload.get("source", "unknown")),
        "text": str(payload["text"]),
        "audio": str(payload.get("audio", "")),
        "model": str(payload.get("model", "")),
        "duration_seconds": float(payload["duration_seconds"]),
        "elapsed_seconds": float(payload["elapsed_seconds"]),
        "rtf": float(payload["rtf"]),
    }
    with HISTORY_LOCK:
        HISTORY.appendleft(clean)
    return clean


def snapshot() -> list[dict[str, Any]]:
    with HISTORY_LOCK:
        return list(HISTORY)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "InCarASRDashboard/1.0"

    def _json(self, status: int, body: Any) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        if path == "/api/history":
            self._json(HTTPStatus.OK, {"results": snapshot()})
            return
        if path in ("/", "/index.html"):
            data = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/results":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 1_000_000:
                raise ValueError("request body must be between 1 byte and 1 MB")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            result = add_result(payload)
        except (ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._json(HTTPStatus.CREATED, result)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[dashboard] {self.address_string()} {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"InCar-ASR dashboard: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
