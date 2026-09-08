"""
server.py — Bounded HTTP Server with Lifecycle Management, Single GPU Owner, and Health Checks
"""

from __future__ import annotations

import http.server
import json
import socketserver
import sys
import threading
import time
from typing import Any, Optional

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


class ThreadedMLXServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, max_threads: int = 64):
        super().__init__(server_address, RequestHandlerClass)
        self.state = ServerState.STARTING
        self.state_error: Optional[str] = None
        self.bind_host: str = server_address[0]
        self.max_body_bytes: int = 10 * 1024 * 1024  # 10 MB limit
        self.thread_limiter = threading.BoundedSemaphore(max_threads)
        self.executor: Optional[GPUExecutor] = None
        self.model_manager: Optional[ModelResidencyManager] = None
        self.runtime: Optional[MLXEmbeddingRuntime] = None
        self.rerank_adapter: Optional[Any] = None
        self.generate_adapter: Optional[Any] = None

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
        try:
            super().close_request(request)
        finally:
            try:
                self.thread_limiter.release()
            except ValueError:
                pass


class MLXHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        try:
            self.connection.settimeout(30.0)
        except Exception:
            pass

    @property
    def mlx_server(self) -> ThreadedMLXServer:
        return self.server  # type: ignore

    def log_message(self, fmt, *args):
        # Silence default logging to avoid cluttering stdout
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, data: bytes, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
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
            self._send_json({"error": "Forbidden: Invalid Host, Origin, or Referer header"}, 403)
            return

        path = self.path.split("?")[0].rstrip("/")
        srv = self.mlx_server

        if path == "/health":
            is_ready = (
                srv.state == ServerState.READY
                and srv.runtime is not None
                and srv.executor is not None
                and srv.executor.is_alive()
            )
            desc = srv.runtime.get_descriptor() if is_ready else None
            self._send_json({
                "status": "ok" if is_ready else "degraded",
                "state": srv.state,
                "model": srv.runtime.model_name if srv.runtime else None,
                "dims": srv.runtime.native_dims if srv.runtime and is_ready else None,
                "ready": is_ready,
                "descriptor": desc,
                "rerank_model": srv.rerank_adapter.model_name if srv.rerank_adapter else None,
                "generate_model": srv.generate_adapter.model_name if srv.generate_adapter else None,
                "error": srv.state_error,
            })
        elif path == "/ready":
            is_ready = (
                srv.state == ServerState.READY
                and srv.runtime is not None
                and srv.executor is not None
                and srv.executor.is_alive()
            )
            if is_ready:
                self._send_json({"ready": True}, 200)
            else:
                self._send_json({"ready": False, "state": srv.state, "error": srv.state_error}, 503)
        elif path == "/descriptor":
            if srv.state != ServerState.READY or not srv.runtime:
                self._send_json({"error": "Server not ready"}, 503)
            else:
                d = srv.runtime.get_descriptor()
                if srv.rerank_adapter:
                    d["rerank"] = srv.rerank_adapter.get_descriptor()
                if srv.generate_adapter:
                    d["generate"] = srv.generate_adapter.get_descriptor()
                self._send_json(d, 200)
        elif path == "/memory":
            if not srv.runtime:
                self._send_json({"active_mb": 0.0, "peak_mb": 0.0, "model_mb": 0.0}, 200)
            else:
                self._send_json(srv.runtime.get_memory_info(), 200)
        elif path == "/stats":
            if not srv.runtime:
                self._send_json({"total_requests": 0, "avg_ms": 0.0, "compiled_shapes": 0, "uptime_sec": 0}, 200)
            else:
                self._send_json(srv.runtime.get_stats_info(), 200)
        else:
            self._send_json({"error": f"Not found: {self.path}"}, 404)

    def do_POST(self):
        if not self._validate_origin_and_referer():
            self._send_json({"error": "Forbidden: Invalid Host, Origin, or Referer header"}, 403)
            return

        srv = self.mlx_server
        if srv.state != ServerState.READY or not srv.runtime:
            self._send_json({"error": f"Server not ready (current state: {srv.state})"}, 503)
            return

        path = self.path.split("?")[0].rstrip("/")
        if path not in ("/embed", "/embed-bin", "/tokenize", "/rerank", "/generate"):
            self._send_json({"error": f"Not found: {self.path}"}, 404)
            return

        # Check content length
        try:
            cl = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length header"}, 400)
            return

        if cl <= 0:
            self._send_json({"error": "Request body cannot be empty"}, 400)
            return
        if cl > srv.max_body_bytes:
            self._send_json({"error": f"Request body exceeds {srv.max_body_bytes} bytes"}, 413)
            return

        # Content-Type check
        ct = self.headers.get("Content-Type", "")
        ct_main = ct.split(";")[0].strip().lower()
        if ct_main != "application/json":
            self._send_json({"error": f"Invalid Content-Type: expected application/json, got '{ct}'"}, 415)
            return

        # Read body
        try:
            body_bytes = self.rfile.read(cl)
            if len(body_bytes) != cl:
                self._send_json({"error": "Incomplete request body read"}, 400)
                return
            payload = json.loads(body_bytes)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON in request body"}, 400)
            return
        except Exception as e:
            self._send_json({"error": f"Failed to read request body: {e}"}, 400)
            return

        timeout_header = self.headers.get("X-Request-Timeout")

        # Handle tokenize endpoint
        if path == "/tokenize":
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
                tokens = srv.runtime.tokenize(texts)
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
    model_name: str,
    port: int = 8787,
    bind_host: str = "127.0.0.1",
    quantization: str = "bf16",
    dtype_str: str = "float32",
    max_length: int = 2048,
    preload: bool = True,
    warmup: bool = True,
    max_batch_tokens: int = 0,
    rerank_model: Optional[str] = None,
    generate_model: Optional[str] = None,
) -> tuple[ThreadedMLXServer, threading.Thread]:
    """
    Initializes unified executor, residency manager, adapters, and starts HTTP server.
    """
    server = ThreadedMLXServer((bind_host, port), MLXHTTPRequestHandler)
    server.state = ServerState.STARTING

    def _init_and_serve():
        try:
            executor = GPUExecutor()
            server.executor = executor

            model_manager = ModelResidencyManager(executor)
            server.model_manager = model_manager

            server.state = ServerState.LOADING if preload else ServerState.READY

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
            server.runtime = runtime

            if preload and warmup:
                runtime.warmup()

            if rerank_model:
                from .rerank import MLXRerankAdapter
                rerank_adapter = MLXRerankAdapter(
                    model_name=rerank_model,
                    max_length=2048,
                    lazy_load=True,
                    executor=executor,
                    model_manager=model_manager,
                )
                model_manager.register_adapter("rerank", rerank_adapter)
                if preload:
                    model_manager.ensure_loaded("rerank")
                server.rerank_adapter = rerank_adapter
                print(f"[mlx-server] Rerank adapter ready: {rerank_model}")

            if generate_model:
                from .generate import MLXGenerateAdapter
                generate_adapter = MLXGenerateAdapter(
                    model_name=generate_model,
                    lazy_load=True,
                    executor=executor,
                    model_manager=model_manager,
                )
                model_manager.register_adapter("generate", generate_adapter)
                if preload:
                    model_manager.ensure_loaded("generate")
                server.generate_adapter = generate_adapter
                print(f"[mlx-server] Generate adapter ready: {generate_model}")

            server.state = ServerState.READY
            print(f"[mlx-server] Model '{model_name}' ready on http://{bind_host}:{port}")
        except Exception as e:
            server.state = ServerState.FAILED
            server.state_error = str(e)
            print(f"[mlx-server] Model loading failed: {e}", file=sys.stderr)

        try:
            server.serve_forever()
        except (KeyboardInterrupt, OSError):
            pass
        finally:
            server.state = ServerState.STOPPING
            if server.model_manager:
                try:
                    server.model_manager.unload_all()
                except Exception:
                    pass
            elif server.runtime:
                try:
                    server.runtime.shutdown()
                except Exception:
                    pass
            if server.executor:
                server.executor.shutdown()

    t = threading.Thread(target=_init_and_serve, daemon=True)
    t.start()
    return server, t
