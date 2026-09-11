"""
server.py — Bounded HTTP Server with Dedicated Control Listener, Single GPU Owner, and Health Checks
"""

from __future__ import annotations

import http.server
import json
import math
import os
import socketserver
import sys
import threading
import time
import uuid
from typing import Any, Optional
import numpy as np

from .executor import GPUExecutor
from .model_manager import ModelResidencyManager, ModelState
from .protocol import (
    DeadlineExceededError,
    InvalidInputError,
    MLXServerError,
    ModelUnavailableError,
    OutOfMemoryError,
    OverloadedError,
    ProtocolError,
    RequestCancelledError,
    UnsupportedModelError,
    WorkerUnavailableError,
    decode_binary_embeddings,
    encode_binary_embeddings,
    parse_timeout,
    validate_embed_request,
    validate_generate_request,
    validate_rerank_request,
)
from .runtime import MLXEmbeddingRuntime


class ServerState:
    STARTING = "starting"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    STOPPING = "stopping"


def get_health_payload(srv: ThreadedMLXServer) -> dict[str, Any]:
    """Generates a complete, typed, finite JSON health dictionary from server and worker state."""
    has_any_stage = bool(
        srv.runtime is not None
        or srv.rerank_adapter is not None
        or srv.generate_adapter is not None
    )
    is_ready = bool(
        srv.state == ServerState.READY
        and has_any_stage
        and srv.executor is not None
        and srv.executor.is_alive()
    )
    desc = None
    if is_ready:
        if srv.runtime and hasattr(srv.runtime, "get_descriptor"):
            try:
                d = srv.runtime.get_descriptor()
                if isinstance(d, dict):
                    desc = d
            except Exception:
                pass
        elif srv.rerank_adapter and hasattr(srv.rerank_adapter, "get_descriptor"):
            try:
                d = srv.rerank_adapter.get_descriptor()
                if isinstance(d, dict):
                    desc = d
            except Exception:
                pass
        elif srv.generate_adapter and hasattr(srv.generate_adapter, "get_descriptor"):
            try:
                d = srv.generate_adapter.get_descriptor()
                if isinstance(d, dict):
                    desc = d
            except Exception:
                pass

    worker_progress = {}
    if srv.executor and hasattr(srv.executor, "get_worker_progress"):
        try:
            wp = srv.executor.get_worker_progress()
            if isinstance(wp, dict):
                worker_progress = wp
        except Exception:
            pass

    queue_depth = 0
    if "queue_depth" in worker_progress and isinstance(worker_progress["queue_depth"], int):
        queue_depth = worker_progress["queue_depth"]
    elif srv.executor and hasattr(srv.executor, "get_queue_depth"):
        try:
            qd = srv.executor.get_queue_depth()
            if isinstance(qd, int):
                queue_depth = qd
        except Exception:
            pass

    is_overloaded = False
    if srv.executor and hasattr(srv.executor, "is_overloaded"):
        try:
            ov = srv.executor.is_overloaded()
            if isinstance(ov, bool):
                is_overloaded = ov
        except Exception:
            pass

    in_flight = 0
    if srv.runtime and hasattr(srv.runtime, "in_flight_requests"):
        try:
            ifr = srv.runtime.in_flight_requests
            if isinstance(ifr, int):
                in_flight = ifr
        except Exception:
            pass

    worker_alive = False
    if "worker_alive" in worker_progress and isinstance(worker_progress["worker_alive"], bool):
        worker_alive = worker_progress["worker_alive"]
    elif srv.executor is not None and hasattr(srv.executor, "is_alive"):
        try:
            worker_alive = bool(srv.executor.is_alive())
        except Exception:
            pass

    worker_idle = bool(worker_progress.get("is_idle", True)) if "is_idle" in worker_progress else True

    active_job_age = worker_progress.get("active_job_age_s")
    if not (
        isinstance(active_job_age, (int, float))
        and not isinstance(active_job_age, bool)
        and math.isfinite(active_job_age)
        and active_job_age >= 0
    ):
        active_job_age = None

    active_job_desc = worker_progress.get("active_job_description")
    if not isinstance(active_job_desc, str):
        active_job_desc = None

    completed_seq = worker_progress.get("completed_sequence", 0)
    if not isinstance(completed_seq, int) or isinstance(completed_seq, bool) or completed_seq < 0:
        completed_seq = 0

    status_str = (
        "ok"
        if is_ready
        else ("starting" if srv.state in (ServerState.STARTING, ServerState.LOADING) else "degraded")
    )

    model_name = getattr(srv.runtime, "model_name", None) if srv.runtime else None
    if not isinstance(model_name, str):
        model_name = None

    native_dims = getattr(srv.runtime, "native_dims", None) if (srv.runtime and is_ready) else None
    if not isinstance(native_dims, int) or isinstance(native_dims, bool):
        native_dims = None

    rerank_name = getattr(srv.rerank_adapter, "model_name", None) if srv.rerank_adapter else None
    if not isinstance(rerank_name, str):
        rerank_name = None

    gen_name = getattr(srv.generate_adapter, "model_name", None) if srv.generate_adapter else None
    if not isinstance(gen_name, str):
        gen_name = None

    uptime_s = round(max(0.0, time.monotonic() - srv.start_time_monotonic), 2)

    return {
        "status": status_str,
        "state": srv.state,
        "pid": srv.server_pid,
        "instance_token": srv.instance_token,
        "uptime_s": uptime_s,
        "model": model_name,
        "dims": native_dims,
        "ready": is_ready,
        "queue_depth": queue_depth,
        "in_flight_requests": in_flight,
        "overloaded": is_overloaded,
        "worker_alive": worker_alive,
        "worker_idle": worker_idle,
        "active_job_age_s": active_job_age,
        "active_job_description": active_job_desc,
        "completed_sequence": completed_seq,
        "descriptor": desc,
        "rerank_model": rerank_name,
        "generate_model": gen_name,
        "error": srv.state_error,
    }


class ThreadedMLXServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        max_threads: int = 128,
        max_concurrent_inference: int = 32,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.state = ServerState.STARTING
        self.state_error: Optional[str] = None
        self.bind_host: str = server_address[0]
        # Use MLX_INSTANCE_TOKEN from env if present, else fresh uuid
        self.instance_token: str = os.getenv("MLX_INSTANCE_TOKEN") or uuid.uuid4().hex
        self.server_pid: int = os.getpid()
        self.start_time_monotonic: float = time.monotonic()
        self.start_time_epoch: float = time.time()
        self.max_body_bytes: int = 10 * 1024 * 1024  # 10 MB limit
        self.max_concurrent_inference = max_concurrent_inference
        self.thread_limiter = threading.BoundedSemaphore(max_threads)
        self.inference_limiter = threading.BoundedSemaphore(max_concurrent_inference)
        self.executor: Optional[GPUExecutor] = None
        self.model_manager: Optional[ModelResidencyManager] = None
        self.runtime: Optional[MLXEmbeddingRuntime] = None
        self.rerank_adapter: Optional[Any] = None
        self.generate_adapter: Optional[Any] = None
        self.control_server: Optional[MLXControlServer] = None
        self.control_port: Optional[int] = None
        self.t_http: Optional[threading.Thread] = None
        self.t_ctrl: Optional[threading.Thread] = None
        self._serving_event = threading.Event()
        self._stop_event = threading.Event()
        self._stopped_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._stop_lock = self._lifecycle_lock
        self._before_serve_hook: Optional[Any] = None
        self._active_sockets: set[socket.socket] = set()
        self._sockets_lock = threading.Lock()

    def get_request(self):
        sock, addr = super().get_request()
        with self._sockets_lock:
            self._active_sockets.add(sock)
        return sock, addr

    def _close_all_active_sockets(self):
        with self._sockets_lock:
            for sock in list(self._active_sockets):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    sock.close()
                except Exception:
                    pass
            self._active_sockets.clear()

    def stop(self, timeout: float = 10.0):
        """
        Coordinates graceful server shutdown lifecycle:
        1. Sets STOPPING state and sets stop event under lifecycle lock.
        2. Closes all owned active client connections on both inference and control listeners.
        3. Stops control server if attached and joins control serving thread.
        4. Stops runtime admission to reject new requests and wake waiting admission leases.
        5. Explicitly shuts down shared server.executor (cancelling queued work and joining worker).
        6. If worker thread has terminated, safely unloads all resident models and sets _stopped_event.
           If worker thread is still alive (e.g. blocked job beyond join budget), defers unload and _stopped_event,
           truthfully reporting pending state without unloading live kernel.
        """
        with self._lifecycle_lock:
            self.state = ServerState.STOPPING
            self._stop_event.set()

            # Close active client connections to unblock any pending handlers immediately
            self._close_all_active_sockets()

            if self.control_server is not None:
                try:
                    self.control_server._close_all_active_sockets()
                    self.control_server.stop()
                    self.control_server.server_close()
                except Exception:
                    pass

            if self.t_ctrl is not None and self.t_ctrl.is_alive():
                try:
                    self.t_ctrl.join(timeout=1.0)
                except Exception:
                    pass

            # Stop runtime admission first
            if self.runtime:
                try:
                    self.runtime.stop_admission()
                except Exception:
                    pass

            # Explicitly shut down shared executor regardless of runtime presence
            if self.executor:
                try:
                    self.executor.shutdown(timeout=timeout)
                except Exception:
                    pass

            # If executor worker thread is still running (e.g. noncooperative forward beyond join budget),
            # do NOT unload models or falsely claim stopped.
            if self.executor and self.executor.is_worker_alive():
                print("[mlx-server] Worker thread still active after join timeout; cleanup pending.", file=sys.stderr)
                return

            # Safe to unload models ONLY AFTER the execution owner thread has completely exited
            if self.model_manager:
                try:
                    self.model_manager.unload_all()
                except Exception:
                    pass
            elif self.runtime:
                try:
                    if hasattr(self.runtime, "adapter") and not (self.executor and self.executor.is_worker_alive()):
                        self.runtime.adapter.unload()
                except Exception:
                    pass

            self._stopped_event.set()

    def serve_forever(self, poll_interval: float = 0.2):
        """
        Lifecycle-safe managed request dispatching loop.
        Uses a short poll timeout (handle_request) and repeatedly inspects _stop_event,
        eliminating races between stop() and BaseServer.shutdown() / serve_forever() handshakes.
        """
        self.timeout = poll_interval
        self._serving_event.set()
        try:
            while not self._stop_event.is_set():
                try:
                    self.handle_request()
                except (KeyboardInterrupt, OSError, ValueError):
                    break
        finally:
            self._serving_event.clear()

    def shutdown(self):
        self.stop()

    def server_close(self):
        self.stop()
        self._close_all_active_sockets()
        try:
            super().server_close()
        except Exception:
            pass

    def process_request(self, request, client_address):
        if not self.thread_limiter.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 429 Too Many Requests\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Connection: close\r\n\r\n"
                    b"{\"error\": \"Server overloaded with concurrent connections\", \"type\": \"overloaded\"}"
                )
                request.close()
            except Exception:
                pass
            return
        super().process_request(request, client_address)

    def close_request(self, request):
        with self._sockets_lock:
            self._active_sockets.discard(request)
        try:
            super().close_request(request)
        finally:
            try:
                self.thread_limiter.release()
            except ValueError:
                pass


class _BoundedDeadlineRfile:
    """
    Stream wrapper for control listener that enforces an absolute monotonic wall-clock deadline
    and maximum byte budget across all reads for a single HTTP request/header.
    """
    def __init__(self, raw_socket: socket.socket, deadline: float, max_bytes: int = 16384):
        self._sock = raw_socket
        self._deadline = deadline
        self._max_bytes = max_bytes
        self._bytes_read = 0
        self._buf = bytearray()

    def _check_deadline(self):
        rem = self._deadline - time.monotonic()
        if rem <= 0:
            raise socket.timeout("Control request absolute deadline exceeded")
        self._sock.settimeout(max(0.001, rem))

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size > 0:
            if self._buf:
                chunk = bytes(self._buf[:size])
                self._buf = self._buf[size:]
                return chunk
            self._check_deadline()
            to_read = min(size, 4096)
            data = self._sock.recv(to_read)
            self._bytes_read += len(data)
            if self._bytes_read > self._max_bytes:
                raise ValueError("Control request exceeded maximum byte budget")
            return data
        chunks = [bytes(self._buf)] if self._buf else []
        self._buf.clear()
        while True:
            self._check_deadline()
            chunk = self._sock.recv(4096)
            if not chunk:
                break
            self._bytes_read += len(chunk)
            if self._bytes_read > self._max_bytes:
                raise ValueError("Control request exceeded maximum byte budget")
            chunks.append(chunk)
        return b"".join(chunks)

    def readline(self, limit: int = -1) -> bytes:
        while True:
            nl = self._buf.find(b"\n")
            if nl != -1:
                line = bytes(self._buf[: nl + 1])
                self._buf = self._buf[nl + 1 :]
                if 0 < limit < len(line):
                    excess = line[limit:]
                    self._buf = bytearray(excess) + self._buf
                    line = line[:limit]
                return line
            if 0 < limit <= len(self._buf):
                line = bytes(self._buf[:limit])
                self._buf = self._buf[limit:]
                return line
            self._check_deadline()
            chunk = self._sock.recv(4096)
            if not chunk:
                line = bytes(self._buf)
                self._buf.clear()
                return line
            self._bytes_read += len(chunk)
            if self._bytes_read > self._max_bytes:
                raise ValueError("Control request exceeded maximum byte budget")
            self._buf.extend(chunk)

    def close(self):
        pass


class MLXControlServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """
    Dedicated loopback HTTP control listener providing isolated control-plane capacity.
    Operates on a separate port from the inference listener so health checks and diagnostics
    never block or time out during inference socket saturation.
    """
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        mlx_server: ThreadedMLXServer,
        max_threads: int = 16,
        request_timeout_s: float = 5.0,
        max_header_bytes: int = 16384,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.mlx_server = mlx_server
        self.bind_host: str = server_address[0]
        self.request_timeout_s = request_timeout_s
        self.max_header_bytes = max_header_bytes
        self.thread_limiter = threading.BoundedSemaphore(max_threads)
        self._serving_event = threading.Event()
        self._stop_event = threading.Event()
        self._stopped_event = threading.Event()
        self._stop_lock = threading.Lock()
        self._active_sockets: set[socket.socket] = set()
        self._sockets_lock = threading.Lock()

    def get_request(self):
        sock, addr = super().get_request()
        with self._sockets_lock:
            self._active_sockets.add(sock)
        return sock, addr

    def _close_all_active_sockets(self):
        with self._sockets_lock:
            for sock in list(self._active_sockets):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    sock.close()
                except Exception:
                    pass
            self._active_sockets.clear()

    def stop(self):
        with self._stop_lock:
            self._stop_event.set()
            self._stopped_event.set()

    def shutdown(self):
        self.stop()

    def server_close(self):
        self.stop()
        self._close_all_active_sockets()
        try:
            super().server_close()
        except Exception:
            pass

    def serve_forever(self, poll_interval: float = 0.2):
        self.timeout = poll_interval
        self._serving_event.set()
        try:
            while not self._stop_event.is_set():
                try:
                    self.handle_request()
                except (KeyboardInterrupt, OSError, ValueError):
                    break
        finally:
            self._serving_event.clear()

    def process_request(self, request, client_address):
        if not self.thread_limiter.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 429 Too Many Requests\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Connection: close\r\n\r\n"
                    b"{\"error\": \"Control server overloaded\", \"type\": \"overloaded\"}"
                )
                request.close()
            except Exception:
                pass
            return
        super().process_request(request, client_address)

    def close_request(self, request):
        with self._sockets_lock:
            self._active_sockets.discard(request)
        try:
            super().close_request(request)
        finally:
            try:
                self.thread_limiter.release()
            except ValueError:
                pass


class MLXControlRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    Request handler for the dedicated loopback control listener.
    Serves /health, /ready, /descriptor, /memory, /stats.
    Rejects all POST / inference endpoints with 405 Method Not Allowed.
    Enforces absolute wall-clock request/header deadlines and byte bounds.
    Unconditionally closes connection after every response (no keepalive).
    """
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.close_connection = True
        try:
            timeout_s = getattr(self.control_server, "request_timeout_s", 5.0)
            max_bytes = getattr(self.control_server, "max_header_bytes", 16384)
            deadline = time.monotonic() + timeout_s
            self.rfile = _BoundedDeadlineRfile(self.connection, deadline, max_bytes=max_bytes)  # type: ignore
        except Exception:
            pass

    @property
    def control_server(self) -> MLXControlServer:
        return self.server  # type: ignore

    @property
    def mlx_server(self) -> ThreadedMLXServer:
        return self.control_server.mlx_server

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data: dict, status: int = 200, close_connection: bool = True):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Unconditionally close control connection (no keepalive)
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _validate_origin_and_referer(self) -> bool:
        """Validates Host, Origin, and Referer headers to protect against DNS rebinding & CSRF."""
        allowed_hosts = {"127.0.0.1", "localhost", self.control_server.bind_host, ""}
        host = self.headers.get("Host", "").split(":")[0]
        if host not in allowed_hosts:
            return False

        origin = self.headers.get("Origin")
        if origin:
            from urllib.parse import urlparse
            parsed = urlparse(origin)
            orig_host = (parsed.hostname or "").lower()
            if orig_host not in allowed_hosts:
                return False

        referer = self.headers.get("Referer")
        if referer:
            from urllib.parse import urlparse
            parsed = urlparse(referer)
            ref_host = (parsed.hostname or "").lower()
            if ref_host not in allowed_hosts:
                return False

        return True

    def do_GET(self):
        if not self._validate_origin_and_referer():
            self._send_json({"error": "Forbidden: Invalid Host, Origin, or Referer header"}, 403, close_connection=True)
            return

        path = self.path.split("?")[0].rstrip("/")
        srv = self.mlx_server

        if path == "/health":
            self._send_json(get_health_payload(srv), 200)
        elif path == "/ready":
            has_any_stage = bool(
                srv.runtime is not None
                or srv.rerank_adapter is not None
                or srv.generate_adapter is not None
            )
            is_ready = bool(
                srv.state == ServerState.READY
                and has_any_stage
                and srv.executor is not None
                and srv.executor.is_alive()
            )
            if is_ready:
                self._send_json({"ready": True}, 200)
            else:
                self._send_json({"ready": False, "state": srv.state, "error": srv.state_error}, 503)
        elif path == "/descriptor":
            if srv.state != ServerState.READY:
                self._send_json({"error": "Server not ready"}, 503)
            elif srv.runtime:
                d = srv.runtime.get_descriptor()
                if srv.rerank_adapter:
                    d["rerank"] = srv.rerank_adapter.get_descriptor()
                if srv.generate_adapter:
                    d["generate"] = srv.generate_adapter.get_descriptor()
                self._send_json(d, 200)
            elif srv.rerank_adapter:
                d = srv.rerank_adapter.get_descriptor()
                if srv.generate_adapter:
                    d["generate"] = srv.generate_adapter.get_descriptor()
                self._send_json(d, 200)
            elif srv.generate_adapter:
                d = srv.generate_adapter.get_descriptor()
                self._send_json(d, 200)
            else:
                self._send_json({"error": "No model stages loaded"}, 503)
        elif path == "/memory":
            if srv.runtime:
                self._send_json(srv.runtime.get_memory_info(), 200)
            else:
                active_mb = 0.0
                try:
                    import mlx.core as mx
                    if hasattr(mx, "get_active_memory"):
                        active_mb = mx.get_active_memory() / (1024 * 1024)
                    elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                        active_mb = mx.metal.get_active_memory() / (1024 * 1024)
                except Exception:
                    pass
                model_mb = 0.0
                if srv.rerank_adapter and hasattr(srv.rerank_adapter, "model_memory_mb"):
                    model_mb += srv.rerank_adapter.model_memory_mb
                if srv.generate_adapter and hasattr(srv.generate_adapter, "model_memory_mb"):
                    model_mb += srv.generate_adapter.model_memory_mb
                self._send_json({
                    "active_mb": round(active_mb, 1),
                    "peak_mb": round(max(active_mb, model_mb), 1),
                    "model_mb": round(model_mb, 1),
                }, 200)
        elif path == "/stats":
            if srv.runtime:
                self._send_json(srv.runtime.get_stats_info(), 200)
            elif srv.rerank_adapter:
                avg_ms = round(srv.rerank_adapter.total_latency_ms / max(1, srv.rerank_adapter.total_requests), 2)
                self._send_json({
                    "total_requests": srv.rerank_adapter.total_requests,
                    "total_pairs_scored": srv.rerank_adapter.total_pairs_scored,
                    "avg_ms": avg_ms,
                    "stage": "rerank",
                }, 200)
            elif srv.generate_adapter:
                self._send_json(srv.generate_adapter.get_stats_info(), 200)
            else:
                self._send_json({"total_requests": 0, "avg_ms": 0.0, "compiled_shapes": 0, "uptime_sec": 0}, 200)
        else:
            self._send_json({"error": f"Not found: {self.path}"}, 404)

    def do_POST(self):
        # Dedicated control listener NEVER serves inference requests
        self._send_json(
            {
                "error": "Inference requests are disabled on control port; use dedicated inference port",
                "type": "method_not_allowed",
            },
            405,
            close_connection=True,
        )

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionError, OSError):
            pass


class MLXHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    Request handler for the primary inference HTTP server.
    Handles inference endpoints (/embed, /embed-bin, /tokenize, /rerank, /generate)
    as well as diagnostic GET endpoints for backward compatibility.
    """
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        try:
            self.connection.settimeout(15.0)
        except Exception:
            pass

    @property
    def mlx_server(self) -> ThreadedMLXServer:
        return self.server  # type: ignore

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data: dict, status: int = 200, close_connection: bool = False):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if close_connection or self.close_connection:
            self.send_header("Connection", "close")
            self.close_connection = True
        else:
            self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, data: bytes, status: int = 200, close_connection: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        if close_connection or self.close_connection:
            self.send_header("Connection", "close")
            self.close_connection = True
        else:
            self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(data)

    def _validate_origin_and_referer(self) -> bool:
        """Validates Host, Origin, and Referer headers to protect against DNS rebinding & CSRF."""
        allowed_hosts = {"127.0.0.1", "localhost", self.mlx_server.bind_host, ""}
        host = self.headers.get("Host", "").split(":")[0]
        if host not in allowed_hosts:
            return False

        origin = self.headers.get("Origin")
        if origin:
            from urllib.parse import urlparse
            parsed = urlparse(origin)
            orig_host = (parsed.hostname or "").lower()
            if orig_host not in allowed_hosts:
                return False

        referer = self.headers.get("Referer")
        if referer:
            from urllib.parse import urlparse
            parsed = urlparse(referer)
            ref_host = (parsed.hostname or "").lower()
            if ref_host not in allowed_hosts:
                return False

        return True

    def _handle_exception(self, exc: Exception):
        """Maps typed exceptions to appropriate HTTP error status codes."""
        if isinstance(exc, MLXServerError):
            self._send_json({"error": str(exc), "type": getattr(exc, "error_type", "error")}, exc.status_code)
        elif isinstance(exc, TimeoutError):
            self._send_json({"error": f"Deadline exceeded: {exc}", "type": "deadline_exceeded"}, 504)
        else:
            self._send_json({"error": f"Internal server error: {exc}", "type": "server_error"}, 500)

    def do_GET(self):
        if not self._validate_origin_and_referer():
            self._send_json({"error": "Forbidden: Invalid Host, Origin, or Referer header"}, 403, close_connection=True)
            return

        path = self.path.split("?")[0].rstrip("/")
        srv = self.mlx_server

        if path == "/health":
            self._send_json(get_health_payload(srv), 200)
        elif path == "/ready":
            has_any_stage = bool(
                srv.runtime is not None
                or srv.rerank_adapter is not None
                or srv.generate_adapter is not None
            )
            is_ready = bool(
                srv.state == ServerState.READY
                and has_any_stage
                and srv.executor is not None
                and srv.executor.is_alive()
            )
            if is_ready:
                self._send_json({"ready": True}, 200)
            else:
                self._send_json({"ready": False, "state": srv.state, "error": srv.state_error}, 503)
        elif path == "/descriptor":
            if srv.state != ServerState.READY:
                self._send_json({"error": "Server not ready"}, 503)
            elif srv.runtime:
                d = srv.runtime.get_descriptor()
                if srv.rerank_adapter:
                    d["rerank"] = srv.rerank_adapter.get_descriptor()
                if srv.generate_adapter:
                    d["generate"] = srv.generate_adapter.get_descriptor()
                self._send_json(d, 200)
            elif srv.rerank_adapter:
                d = srv.rerank_adapter.get_descriptor()
                if srv.generate_adapter:
                    d["generate"] = srv.generate_adapter.get_descriptor()
                self._send_json(d, 200)
            elif srv.generate_adapter:
                d = srv.generate_adapter.get_descriptor()
                self._send_json(d, 200)
            else:
                self._send_json({"error": "No model stages loaded"}, 503)
        elif path == "/memory":
            if srv.runtime:
                self._send_json(srv.runtime.get_memory_info(), 200)
            else:
                active_mb = 0.0
                try:
                    import mlx.core as mx
                    if hasattr(mx, "get_active_memory"):
                        active_mb = mx.get_active_memory() / (1024 * 1024)
                    elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                        active_mb = mx.metal.get_active_memory() / (1024 * 1024)
                except Exception:
                    pass
                model_mb = 0.0
                if srv.rerank_adapter and hasattr(srv.rerank_adapter, "model_memory_mb"):
                    model_mb += srv.rerank_adapter.model_memory_mb
                if srv.generate_adapter and hasattr(srv.generate_adapter, "model_memory_mb"):
                    model_mb += srv.generate_adapter.model_memory_mb
                self._send_json({
                    "active_mb": round(active_mb, 1),
                    "peak_mb": round(max(active_mb, model_mb), 1),
                    "model_mb": round(model_mb, 1),
                }, 200)
        elif path == "/stats":
            if srv.runtime:
                self._send_json(srv.runtime.get_stats_info(), 200)
            elif srv.rerank_adapter:
                avg_ms = round(srv.rerank_adapter.total_latency_ms / max(1, srv.rerank_adapter.total_requests), 2)
                self._send_json({
                    "total_requests": srv.rerank_adapter.total_requests,
                    "total_pairs_scored": srv.rerank_adapter.total_pairs_scored,
                    "avg_ms": avg_ms,
                    "stage": "rerank",
                }, 200)
            elif srv.generate_adapter:
                self._send_json(srv.generate_adapter.get_stats_info(), 200)
            else:
                self._send_json({"total_requests": 0, "avg_ms": 0.0, "compiled_shapes": 0, "uptime_sec": 0}, 200)
        else:
            self._send_json({"error": f"Not found: {self.path}"}, 404)

    def do_POST(self):
        if not self._validate_origin_and_referer():
            self._send_json({"error": "Forbidden: Invalid Host, Origin, or Referer header"}, 403, close_connection=True)
            return

        srv = self.mlx_server
        has_any_stage = bool(
            srv.runtime is not None
            or srv.rerank_adapter is not None
            or srv.generate_adapter is not None
        )
        if srv.state != ServerState.READY or not has_any_stage:
            self._send_json({"error": f"Server not ready (current state: {srv.state})"}, 503, close_connection=True)
            return

        path = self.path.split("?")[0].rstrip("/")
        if path not in ("/embed", "/embed-bin", "/tokenize", "/rerank", "/generate"):
            self._send_json({"error": f"Not found: {self.path}"}, 404, close_connection=True)
            return

        # Dedicated control capacity: limit concurrent inference requests so control plane remains reachable
        if not srv.inference_limiter.acquire(blocking=False):
            # Must send Connection: close when rejecting before reading body to prevent HTTP protocol desync
            self._send_json(
                {
                    "error": "Inference capacity saturated: maximum concurrent inference requests reached",
                    "type": "overloaded",
                },
                429,
                close_connection=True,
            )
            return

        try:
            self._handle_post_inference(path)
        finally:
            try:
                srv.inference_limiter.release()
            except ValueError:
                pass

    def _handle_post_inference(self, path: str):
        srv = self.mlx_server

        # Check content length
        try:
            cl = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length header"}, 400, close_connection=True)
            return

        if cl <= 0:
            self._send_json({"error": "Request body cannot be empty"}, 400, close_connection=True)
            return
        if cl > srv.max_body_bytes:
            self._send_json({"error": f"Request body exceeds {srv.max_body_bytes} bytes"}, 413, close_connection=True)
            return

        # Content-Type check
        ct = self.headers.get("Content-Type", "")
        ct_main = ct.split(";")[0].strip().lower()
        if ct_main != "application/json":
            self._send_json({"error": f"Invalid Content-Type: expected application/json, got '{ct}'"}, 415, close_connection=True)
            return

        # Read body
        try:
            body_bytes = self.rfile.read(cl)
            if len(body_bytes) != cl:
                self._send_json({"error": "Incomplete request body read"}, 400, close_connection=True)
                return
            payload = json.loads(body_bytes)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON in request body"}, 400, close_connection=True)
            return
        except Exception as e:
            self._send_json({"error": f"Failed to read request body: {e}"}, 400, close_connection=True)
            return

        timeout_header = self.headers.get("X-Request-Timeout")

        # Handle tokenize endpoint
        if path == "/tokenize":
            if srv.runtime is None:
                self._send_json({"error": "Embedding / tokenize adapter not configured on this server"}, 501)
                return
            try:
                if not isinstance(payload, dict):
                    raise InvalidInputError("Payload must be a JSON object")
                texts = payload.get("texts", [])
                if not isinstance(texts, list):
                    raise InvalidInputError("'texts' must be a list of strings")
                if len(texts) > 512:
                    raise InvalidInputError(f"'texts' exceeds per-request limit of 512 (got {len(texts)})")
                for i, t in enumerate(texts):
                    if not isinstance(t, str):
                        raise InvalidInputError(f"texts[{i}] is not a string")
                    if len(t) > 256 * 1024:
                        raise InvalidInputError(f"texts[{i}] exceeds 256KB")
                req_timeout = parse_timeout(payload, timeout_header, default_timeout=120.0)
                deadline = time.monotonic() + req_timeout
                cancel_event = threading.Event()
                tokens = srv.runtime.tokenize(texts, deadline=deadline, cancel_event=cancel_event)
                counts = [len(t) for t in tokens]
                self._send_json({"tokens": tokens, "counts": counts}, 200)
            except Exception as exc:
                self._handle_exception(exc)
            return

        # Rerank endpoint
        if path == "/rerank":
            if srv.rerank_adapter is None:
                self._send_json({"error": "Rerank adapter not configured on this server"}, 501)
                return
            try:
                query, documents = validate_rerank_request(payload)
                req_timeout = parse_timeout(payload, timeout_header, default_timeout=120.0)
                deadline = time.monotonic() + req_timeout
                cancel_event = threading.Event()
                scores = srv.rerank_adapter.score_pairs(
                    query,
                    documents,
                    timeout_s=req_timeout,
                    cancel_event=cancel_event,
                    deadline=deadline,
                )
                self._send_json({
                    "scores": scores,
                    "model": srv.rerank_adapter.model_name,
                    "count": len(scores),
                }, 200)
            except Exception as exc:
                self._handle_exception(exc)
            return

        # Generate endpoint
        if path == "/generate":
            if srv.generate_adapter is None:
                self._send_json({"error": "Generate adapter not configured on this server"}, 501)
                return
            try:
                prompt, max_tokens, temperature = validate_generate_request(payload)
                req_timeout = parse_timeout(payload, timeout_header, default_timeout=120.0)
                deadline = time.monotonic() + req_timeout
                cancel_event = threading.Event()
                text = srv.generate_adapter.submit_generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=req_timeout,
                    cancel_event=cancel_event,
                    deadline=deadline,
                )
                self._send_json({
                    "text": text,
                    "model": srv.generate_adapter.model_name,
                }, 200)
            except Exception as exc:
                self._handle_exception(exc)
            return

        # Embed endpoint (/embed and /embed-bin)
        if srv.runtime is None:
            self._send_json({"error": "Embedding adapter not configured on this server"}, 501)
            return

        try:
            texts, requested_dims, is_query = validate_embed_request(payload)
            # Use validated request timeout with 300.0s default — NEVER override to 60s
            req_timeout = parse_timeout(payload, timeout_header, default_timeout=300.0)
            deadline = time.monotonic() + req_timeout
            cancel_event = threading.Event()

            embeddings_arr = srv.runtime.submit_embed(
                texts=texts,
                requested_dims=requested_dims,
                is_query=is_query,
                timeout=req_timeout,
                cancel_event=cancel_event,
                deadline=deadline,
            )

            if path == "/embed-bin":
                count, dims = embeddings_arr.shape
                bin_data = encode_binary_embeddings(embeddings_arr, count, dims)
                self._send_binary(bin_data, 200)
            else:
                if not np.all(np.isfinite(embeddings_arr)):
                    raise ProtocolError("Embeddings contain NaN or Infinite values")
                self._send_json({
                    "embeddings": embeddings_arr.tolist(),
                    "model": srv.runtime.model_name,
                    "dims": embeddings_arr.shape[1],
                }, 200)
        except Exception as exc:
            self._handle_exception(exc)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionError, OSError):
            pass


def start_server(
    model_name: Optional[str] = "mlx-community/nomic-embed-text-v1.5",
    port: int = 8787,
    bind_host: str = "127.0.0.1",
    control_port: Optional[int] = None,
    quantization: str = "bf16",
    dtype_str: str = "float32",
    max_length: int = 2048,
    preload: bool = True,
    warmup: bool = True,
    max_batch_tokens: int = 0,
    rerank_model: Optional[str] = None,
    generate_model: Optional[str] = None,
    max_threads: int = 128,
    max_concurrent_inference: int = 32,
    control_max_threads: int = 16,
    control_request_timeout_s: float = 5.0,
    control_max_header_bytes: int = 16384,
    _before_serve_hook: Optional[Any] = None,
) -> tuple[ThreadedMLXServer, threading.Thread]:
    """
    Initializes unified executor, residency manager, adapters, and starts both:
    1. Primary inference HTTP server on (bind_host, port).
    2. Dedicated loopback control HTTP server on (127.0.0.1, control_port).
    Serves HTTP /health immediately in STARTING/LOADING state on both listeners
    so health probes do not hang or fail during cold model init.
    """
    server = ThreadedMLXServer(
        (bind_host, port),
        MLXHTTPRequestHandler,
        max_threads=max_threads,
        max_concurrent_inference=max_concurrent_inference,
    )
    server._before_serve_hook = _before_serve_hook
    server.state = ServerState.STARTING

    # Determine control port
    if control_port is None:
        effective_control_port = 0 if port == 0 else (port + 1)
    else:
        effective_control_port = control_port

    control_server = MLXControlServer(
        ("127.0.0.1", effective_control_port),
        MLXControlRequestHandler,
        mlx_server=server,
        max_threads=control_max_threads,
        request_timeout_s=control_request_timeout_s,
        max_header_bytes=control_max_header_bytes,
    )
    server.control_server = control_server
    server.control_port = control_server.server_address[1]

    # Dedicated serving threads for both listeners
    t_http = threading.Thread(target=server.serve_forever, daemon=True, name="MLX-HTTP-Serve")
    t_ctrl = threading.Thread(target=control_server.serve_forever, daemon=True, name="MLX-Control-Serve")
    server.t_http = t_http
    server.t_ctrl = t_ctrl
    t_http.start()
    t_ctrl.start()

    def _init_and_coordinate():
        try:
            with server._lifecycle_lock:
                if server._stop_event.is_set():
                    return

                executor = GPUExecutor()
                server.executor = executor

                model_manager = ModelResidencyManager(executor)
                server.model_manager = model_manager

                server.state = ServerState.LOADING if preload else ServerState.READY

            has_embed = bool(model_name and model_name.strip().lower() not in ("", "none", "null", "false"))
            has_rerank = bool(rerank_model and rerank_model.strip().lower() not in ("", "none", "null", "false"))
            has_generate = bool(generate_model and generate_model.strip().lower() not in ("", "none", "null", "false"))

            if not (has_embed or has_rerank or has_generate):
                raise ValueError("No model configured (specify model_name, rerank_model, or generate_model)")

            if has_embed:
                runtime = MLXEmbeddingRuntime(
                    model_name=model_name,
                    quantization=quantization,
                    dtype_str=dtype_str,
                    max_length=max_length,
                    max_batch_tokens=max_batch_tokens,
                    executor=executor,
                    model_manager=model_manager,
                    lazy_load=not preload,
                )

                with server._lifecycle_lock:
                    if server._stop_event.is_set():
                        return
                    server.runtime = runtime

                if preload and warmup:
                    if server._stop_event.is_set():
                        return
                    runtime.warmup()

            if has_rerank:
                if server._stop_event.is_set():
                    return
                from .rerank import MLXRerankAdapter
                rerank_adapter = MLXRerankAdapter(
                    model_name=rerank_model,
                    max_length=2048,
                    lazy_load=not preload,
                    executor=executor,
                    model_manager=model_manager,
                )
                model_manager.register_adapter("rerank", rerank_adapter)
                if preload:
                    model_manager.ensure_loaded("rerank")
                    if warmup:
                        rerank_adapter.warmup()
                with server._lifecycle_lock:
                    if server._stop_event.is_set():
                        return
                    server.rerank_adapter = rerank_adapter
                print(f"[mlx-server] Rerank adapter ready: {rerank_model}")

            if has_generate:
                if server._stop_event.is_set():
                    return
                from .generate import MLXGenerateAdapter
                generate_adapter = MLXGenerateAdapter(
                    model_name=generate_model,
                    lazy_load=not preload,
                    executor=executor,
                    model_manager=model_manager,
                )
                model_manager.register_adapter("generate", generate_adapter)
                if preload:
                    model_manager.ensure_loaded("generate")
                    if warmup:
                        generate_adapter.warmup()
                with server._lifecycle_lock:
                    if server._stop_event.is_set():
                        return
                    server.generate_adapter = generate_adapter
                print(f"[mlx-server] Generate adapter ready: {generate_model}")

            with server._lifecycle_lock:
                if server._stop_event.is_set():
                    return
                server.state = ServerState.READY
                primary = model_name if has_embed else (rerank_model if has_rerank else generate_model)
                print(f"[mlx-server] Server ready (primary='{primary}') on http://{bind_host}:{port} (control: http://127.0.0.1:{server.control_port})")

        except Exception as e:
            with server._lifecycle_lock:
                if not server._stop_event.is_set():
                    server.state = ServerState.FAILED
                    server.state_error = str(e)
                    print(f"[mlx-server] Model loading failed: {e}", file=sys.stderr)

        if server._before_serve_hook:
            try:
                server._before_serve_hook()
            except Exception:
                pass

        # Keep coordinator alive until stopped
        while not server._stop_event.is_set():
            time.sleep(0.05)

        t_http.join(timeout=2.0)
        t_ctrl.join(timeout=2.0)
        server.stop()

    t = threading.Thread(target=_init_and_coordinate, daemon=True, name="MLX-Init-Coordinator")
    t.start()
    return server, t
