"""
server.py — Bounded HTTP Server with Lifecycle Management, Admission Deadlines, and Health Checks
"""

import http.server
import json
import socketserver
import sys
import threading
import time
from typing import Any, Optional

from .protocol import (
    encode_binary_embeddings,
    validate_embed_request,
    ProtocolError,
)
from .runtime import MLXEmbeddingRuntime, MLXRuntimeError


def _validate_rerank_request(payload: dict) -> tuple[str, list[str]]:
    """Validates a /rerank request body. Returns (query, documents)."""
    query = payload.get("query")
    documents = payload.get("documents")
    if not isinstance(query, str) or not query.strip():
        raise ProtocolError("'query' must be a non-empty string")
    if not isinstance(documents, list) or len(documents) == 0:
        raise ProtocolError("'documents' must be a non-empty list of strings")
    if len(documents) > 128:
        raise ProtocolError(f"'documents' exceeds per-request limit of 128 (got {len(documents)})")
    for i, doc in enumerate(documents):
        if not isinstance(doc, str):
            raise ProtocolError(f"document at index {i} is not a string")
        if len(doc) > 256 * 1024:
            raise ProtocolError(f"document at index {i} exceeds 256KB")
    return query, documents


def _validate_generate_request(payload: dict) -> tuple[str, int, float]:
    """Validates a /generate request body. Returns (prompt, max_tokens, temperature)."""
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ProtocolError("'prompt' must be a non-empty string")
    if len(prompt) > 128 * 1024:
        raise ProtocolError("'prompt' exceeds 128KB")
    max_tokens = payload.get("max_tokens", 600)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0 or max_tokens > 4096:
        raise ProtocolError(f"'max_tokens' must be an integer in (0, 4096], got {max_tokens}")
    temperature = payload.get("temperature", 0.0)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not (0.0 <= float(temperature) <= 2.0):
        raise ProtocolError(f"'temperature' must be a number in [0.0, 2.0], got {temperature}")
    return prompt, max_tokens, float(temperature)


class ServerState:
    STARTING = "starting"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    STOPPING = "stopping"


class MLXHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Injected by server
    runtime: Optional[MLXEmbeddingRuntime] = None
    rerank_adapter: Optional[Any] = None
    generate_adapter: Optional[Any] = None
    state: str = ServerState.STARTING
    state_error: Optional[str] = None
    max_body_bytes: int = 10 * 1024 * 1024  # 10 MB limit
    bind_host: str = "127.0.0.1"

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

    def _validate_host(self) -> bool:
        """Validates host header to protect against DNS rebinding / CSRF."""
        host = self.headers.get("Host", "").split(":")[0]
        if host in ("127.0.0.1", "localhost", self.bind_host, ""):
            return True
        return False

    def do_GET(self):
        if not self._validate_host():
            self._send_json({"error": "Forbidden: Invalid Host header"}, 403)
            return

        path = self.path.split("?")[0].rstrip("/")

        if path == "/health":
            desc = self.runtime.get_descriptor() if self.runtime and self.state == ServerState.READY else None
            self._send_json({
                "status": "ok" if self.state == ServerState.READY else "degraded",
                "state": self.state,
                "model": self.runtime.model_name if self.runtime else None,
                "dims": self.runtime.native_dims if self.runtime else None,
                "ready": self.state == ServerState.READY,
                "descriptor": desc,
                "rerank_model": self.rerank_adapter.model_name if self.rerank_adapter else None,
                "generate_model": self.generate_adapter.model_name if self.generate_adapter else None,
                "error": self.state_error,
            })
        elif path == "/ready":
            if self.state == ServerState.READY:
                self._send_json({"ready": True}, 200)
            else:
                self._send_json({"ready": False, "state": self.state, "error": self.state_error}, 503)
        elif path == "/descriptor":
            if self.state != ServerState.READY or not self.runtime:
                self._send_json({"error": "Server not ready"}, 503)
            else:
                d = self.runtime.get_descriptor()
                if self.rerank_adapter:
                    d["rerank"] = self.rerank_adapter.get_descriptor()
                if self.generate_adapter:
                    d["generate"] = self.generate_adapter.get_descriptor()
                self._send_json(d, 200)
        elif path == "/memory":
            if not self.runtime:
                self._send_json({"active_mb": 0.0, "peak_mb": 0.0, "model_mb": 0.0}, 200)
            else:
                self._send_json(self.runtime.get_memory_info(), 200)
        elif path == "/stats":
            if not self.runtime:
                self._send_json({"total_requests": 0, "avg_ms": 0.0, "compiled_shapes": 0, "uptime_sec": 0}, 200)
            else:
                self._send_json(self.runtime.get_stats_info(), 200)
        else:
            self._send_json({"error": f"Not found: {self.path}"}, 404)

    def do_POST(self):
        if not self._validate_host():
            self._send_json({"error": "Forbidden: Invalid Host header"}, 403)
            return

        if self.state != ServerState.READY or not self.runtime:
            self._send_json({"error": f"Server not ready (current state: {self.state})"}, 503)
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
        if cl > self.max_body_bytes:
            self._send_json({"error": f"Request body exceeds {self.max_body_bytes} bytes"}, 413)
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

        # Handle tokenize endpoint
        if path == "/tokenize":
            texts = payload.get("texts", [])
            if not isinstance(texts, list):
                self._send_json({"error": "'texts' must be a list of strings"}, 400)
                return
            tokens = self.runtime.tokenize(texts)
            counts = [len(t) for t in tokens]
            self._send_json({"tokens": tokens, "counts": counts}, 200)
            return

        # Rerank endpoint
        if path == "/rerank":
            if self.rerank_adapter is None:
                self._send_json({"error": "Rerank adapter not configured on this server"}, 501)
                return
            try:
                query, documents = _validate_rerank_request(payload)
            except ProtocolError as pe:
                self._send_json({"error": str(pe)}, 400)
                return
            try:
                scores = self.rerank_adapter.score_pairs(query, documents)
                self._send_json({
                    "scores": scores,
                    "model": self.rerank_adapter.model_name,
                    "count": len(scores),
                }, 200)
            except Exception as exc:
                self._send_json({"error": f"Rerank inference failed: {exc}"}, 500)
            return

        # Generate endpoint
        if path == "/generate":
            if self.generate_adapter is None:
                self._send_json({"error": "Generate adapter not configured on this server"}, 501)
                return
            try:
                prompt, max_tokens, temperature = _validate_generate_request(payload)
            except ProtocolError as pe:
                self._send_json({"error": str(pe)}, 400)
                return
            try:
                text = self.generate_adapter.submit_generate(
                    prompt, max_tokens=max_tokens, temperature=temperature, timeout=120.0
                )
                self._send_json({
                    "text": text,
                    "model": self.generate_adapter.model_name,
                }, 200)
            except Exception as exc:
                self._send_json({"error": f"Generate inference failed: {exc}"}, 500)
            return

        # Validate embed request
        try:
            texts, requested_dims, is_query = validate_embed_request(payload)
        except ProtocolError as pe:
            self._send_json({"error": str(pe)}, 400)
            return

        # Execute embedding through runtime owner queue
        cancel_event = threading.Event()
        try:
            embeddings_arr = self.runtime.submit_embed(
                texts=texts,
                requested_dims=requested_dims,
                is_query=is_query,
                timeout=60.0,
                cancel_event=cancel_event,
            )

            if path == "/embed-bin":
                count, dims = embeddings_arr.shape
                bin_data = encode_binary_embeddings(embeddings_arr, count, dims)
                self._send_binary(bin_data, 200)
            else:
                self._send_json({
                    "embeddings": embeddings_arr.tolist(),
                    "model": self.runtime.model_name,
                    "dims": embeddings_arr.shape[1],
                }, 200)
        except MLXRuntimeError as re:
            self._send_json({"error": str(re)}, 503)
        except Exception as exc:
            self._send_json({"error": f"Embedding inference failed: {exc}"}, 500)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionError, OSError):
            pass


class ThreadedMLXServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


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
    Initializes runtime (+ optional rerank/generate adapters), starts HTTP server.
    """
    MLXHTTPRequestHandler.bind_host = bind_host
    MLXHTTPRequestHandler.state = ServerState.STARTING

    server = ThreadedMLXServer((bind_host, port), MLXHTTPRequestHandler)

    def _init_and_serve():
        if preload:
            MLXHTTPRequestHandler.state = ServerState.LOADING
            try:
                runtime = MLXEmbeddingRuntime(
                    model_name=model_name,
                    quantization=quantization,
                    dtype_str=dtype_str,
                    max_length=max_length,
                    max_batch_tokens=max_batch_tokens,
                )
                if warmup:
                    runtime.warmup()
                MLXHTTPRequestHandler.runtime = runtime

                if rerank_model:
                    from .rerank import MLXRerankAdapter
                    MLXHTTPRequestHandler.rerank_adapter = MLXRerankAdapter(
                        model_name=rerank_model, max_length=2048
                    )
                    print(f"[mlx-server] Rerank adapter ready: {rerank_model}")

                if generate_model:
                    from .generate import MLXGenerateAdapter
                    MLXHTTPRequestHandler.generate_adapter = MLXGenerateAdapter(
                        model_name=generate_model
                    )
                    print(f"[mlx-server] Generate adapter ready: {generate_model}")

                MLXHTTPRequestHandler.state = ServerState.READY
                print(f"[mlx-server] Model '{model_name}' ready on http://{bind_host}:{port}")
            except Exception as e:
                MLXHTTPRequestHandler.state = ServerState.FAILED
                MLXHTTPRequestHandler.state_error = str(e)
                print(f"[mlx-server] Model loading failed: {e}", file=sys.stderr)
        else:
            MLXHTTPRequestHandler.state = ServerState.READY

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if MLXHTTPRequestHandler.runtime:
                MLXHTTPRequestHandler.runtime.shutdown()

    t = threading.Thread(target=_init_and_serve, daemon=True)
    t.start()
    return server, t
