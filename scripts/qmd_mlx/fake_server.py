#!/usr/bin/env python3
"""
fake_server.py — Lightweight disposable fake MLX embedding server for offline testing & rehearsals.

Features:
- Dual-listener architecture: Primary inference port + dedicated loopback control port.
- Exact protocol compliance: /health, /ready, /descriptor, /memory, /stats, /embed, /embed-bin.
- Deterministic synthetic embeddings: Hash-seeded float32 unit-normalized vectors.
- Controllable failure modes: Optional hang simulation (--hang-after-requests N, --hang-on-embed).
- Zero MLX/PyTorch/weight dependencies: 100% pure Python + numpy standard.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import math
import os
import signal
import socket
import socketserver
import struct
import sys
import threading
import time
from typing import Any, Optional
import numpy as np


def generate_deterministic_embedding(text: str, dims: int = 2560) -> np.ndarray:
    """Generates a deterministic unit-normalized float32 embedding vector from text hash."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], byteorder="big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dims).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


class FakeMLXControlHandler(http.server.BaseHTTPRequestHandler):
    """Handles /health, /ready, /descriptor on the dedicated control port."""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        server: FakeMLXControlServer = self.server  # type: ignore
        if self.path in ("/health", "/health/"):
            uptime_s = round(max(0.0, time.time() - getattr(server, "start_time", time.time())), 2)
            payload = {
                "status": "ok",
                "ready": server.ready,
                "state": "ready" if server.ready else "starting",
                "dims": server.dims,
                "model": server.model_name,
                "instance_token": server.instance_token,
                "pid": os.getpid(),
                "uptime_s": uptime_s,
                "worker_alive": True,
                "worker_idle": server.in_flight == 0,
                "completed_sequence": server.request_count,
                "active_job_age_s": None,
                "queue_depth": 0,
                "in_flight_requests": server.in_flight,
                "worker_progress": {
                    "in_flight": server.in_flight,
                    "queue_depth": 0,
                    "completed_requests": server.request_count,
                    "last_active_duration_s": 0.0,
                },
                "descriptor": {
                    "version": 1,
                    "backend": "mlx_fake",
                    "model": server.model_name,
                    "nativeDimensions": server.dims,
                    "outputDimensions": server.dims,
                    "normalized": True,
                    "maxSequenceLength": 2048,
                    "pooling": "mean",
                },
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/ready", "/ready/"):
            status_code = 200 if server.ready else 503
            body = json.dumps({"ready": server.ready}).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/descriptor", "/descriptor/"):
            body = json.dumps({
                "version": 1,
                "backend": "mlx_fake",
                "model": server.model_name,
                "nativeDimensions": server.dims,
                "outputDimensions": server.dims,
                "normalized": True,
                "maxSequenceLength": 2048,
                "pooling": "mean",
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


class FakeMLXInferenceHandler(http.server.BaseHTTPRequestHandler):
    """Handles inference and data queries on the primary inference port."""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        server: FakeMLXInferenceServer = self.server  # type: ignore
        if self.path in ("/health", "/health/"):
            uptime_s = round(max(0.0, time.time() - server.start_time), 2)
            payload = {
                "status": "ok",
                "ready": server.ready,
                "state": "ready" if server.ready else "starting",
                "dims": server.dims,
                "model": server.model_name,
                "instance_token": server.instance_token,
                "pid": os.getpid(),
                "uptime_s": uptime_s,
                "worker_alive": True,
                "worker_idle": server.in_flight == 0,
                "completed_sequence": server.request_count,
                "active_job_age_s": None,
                "queue_depth": 0,
                "in_flight_requests": server.in_flight,
                "worker_progress": {
                    "in_flight": server.in_flight,
                    "queue_depth": 0,
                    "completed_requests": server.request_count,
                    "last_active_duration_s": 0.0,
                },
                "descriptor": {
                    "version": 1,
                    "backend": "mlx_fake",
                    "model": server.model_name,
                    "nativeDimensions": server.dims,
                    "outputDimensions": server.dims,
                    "normalized": True,
                    "maxSequenceLength": 2048,
                    "pooling": "mean",
                },
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/ready", "/ready/"):
            status_code = 200 if server.ready else 503
            body = json.dumps({"ready": server.ready}).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/descriptor", "/descriptor/"):
            body = json.dumps({
                "version": 1,
                "backend": "mlx_fake",
                "model": server.model_name,
                "nativeDimensions": server.dims,
                "outputDimensions": server.dims,
                "normalized": True,
                "maxSequenceLength": 2048,
                "pooling": "mean",
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/memory", "/memory/"):
            body = json.dumps({"active_mb": 42.0, "peak_mb": 64.0, "model_mb": 0.0}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/stats", "/stats/"):
            body = json.dumps({
                "total_requests": server.request_count,
                "avg_ms": 1.5,
                "compiled_shapes": 1,
                "uptime_sec": int(time.time() - server.start_time),
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        server: FakeMLXInferenceServer = self.server  # type: ignore

        # Check hang simulation trigger
        if server.hang_on_embed or (
            server.hang_after_requests > 0 and server.request_count >= server.hang_after_requests
        ):
            # Hang deliberately to simulate stuck worker
            while True:
                time.sleep(1.0)

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        if self.path in ("/embed", "/embed/"):
            try:
                data = json.loads(raw_body.decode("utf-8"))
            except Exception:
                self.send_response(400)
                self.end_headers()
                return

            texts = data.get("texts", [])
            if not isinstance(texts, list):
                self.send_response(400)
                self.end_headers()
                return

            server.request_count += 1
            dims = data.get("dims", server.dims) or server.dims

            embeddings = []
            for t in texts:
                emb = generate_deterministic_embedding(str(t), dims=dims)
                embeddings.append(emb.tolist())

            resp_payload = {
                "embeddings": embeddings,
                "model": server.model_name,
                "dims": dims,
                "count": len(embeddings),
            }
            body = json.dumps(resp_payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path in ("/embed-bin", "/embed-bin/"):
            try:
                data = json.loads(raw_body.decode("utf-8"))
            except Exception:
                self.send_response(400)
                self.end_headers()
                return

            texts = data.get("texts", [])
            server.request_count += 1
            dims = data.get("dims", server.dims) or server.dims

            count = len(texts)
            arr = np.empty((count, dims), dtype=np.float32)
            for i, t in enumerate(texts):
                arr[i] = generate_deterministic_embedding(str(t), dims=dims)

            header = struct.pack("<ii", count, dims)
            body = header + arr.tobytes()

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


class FakeMLXControlServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass,
        instance_token: str,
        model_name: str,
        dims: int,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.instance_token = instance_token
        self.model_name = model_name
        self.dims = dims
        self.ready = True
        self.in_flight = 0
        self.request_count = 0
        self.start_time = time.time()


class FakeMLXInferenceServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass,
        instance_token: str,
        model_name: str,
        dims: int,
        hang_on_embed: bool = False,
        hang_after_requests: int = 0,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.instance_token = instance_token
        self.model_name = model_name
        self.dims = dims
        self.ready = True
        self.in_flight = 0
        self.request_count = 0
        self.start_time = time.time()
        self.hang_on_embed = hang_on_embed
        self.hang_after_requests = hang_after_requests


def run_fake_server(
    port: int = 8797,
    control_port: int = 8798,
    host: str = "127.0.0.1",
    model_name: str = "fake-model-qwen3-4b",
    dims: int = 2560,
    instance_token: Optional[str] = None,
    hang_on_embed: bool = False,
    hang_after_requests: int = 0,
) -> None:
    """Runs the fake server with dual HTTP listeners until SIGINT or SIGTERM."""
    token = instance_token or os.environ.get("MLX_INSTANCE_TOKEN", "fake-token-12345")

    control_server = FakeMLXControlServer(
        (host, control_port),
        FakeMLXControlHandler,
        instance_token=token,
        model_name=model_name,
        dims=dims,
    )

    inference_server = FakeMLXInferenceServer(
        (host, port),
        FakeMLXInferenceHandler,
        instance_token=token,
        model_name=model_name,
        dims=dims,
        hang_on_embed=hang_on_embed,
        hang_after_requests=hang_after_requests,
    )

    t_ctrl = threading.Thread(target=control_server.serve_forever, daemon=True, name="FakeMLX-Control")
    t_inf = threading.Thread(target=inference_server.serve_forever, daemon=True, name="FakeMLX-Inference")

    t_ctrl.start()
    t_inf.start()

    print(f"[fake-server] Started inference on http://{host}:{port}, control on http://{host}:{control_port} (token={token}, dims={dims})")
    sys.stdout.flush()

    stop_event = threading.Event()

    def _sig_handler(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        print("[fake-server] Shutting down...")
        inference_server.shutdown()
        inference_server.server_close()
        control_server.shutdown()
        control_server.server_close()
        t_inf.join(timeout=2.0)
        t_ctrl.join(timeout=2.0)
        print("[fake-server] Clean exit.")


def main():
    parser = argparse.ArgumentParser(description="Disposable Fake MLX Embedding Server")
    parser.add_argument("--port", type=int, default=8797, help="Inference port")
    parser.add_argument("--control-port", type=int, default=8798, help="Control port")
    parser.add_argument("--host", default="127.0.0.1", help="Loopback host")
    parser.add_argument("--model", default="fake-model-qwen3-4b", help="Model name")
    parser.add_argument("--dims", type=int, default=2560, help="Embedding dimension")
    parser.add_argument("--instance-token", default=None, help="Instance token")
    parser.add_argument("--hang-on-embed", action="store_true", help="Simulate a stuck worker on /embed")
    parser.add_argument("--hang-after-requests", type=int, default=0, help="Simulate a stuck worker after N requests")

    args = parser.parse_args()
    run_fake_server(
        port=args.port,
        control_port=args.control_port,
        host=args.host,
        model_name=args.model,
        dims=args.dims,
        instance_token=args.instance_token,
        hang_on_embed=args.hang_on_embed,
        hang_after_requests=args.hang_after_requests,
    )


if __name__ == "__main__":
    main()
