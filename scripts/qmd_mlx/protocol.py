"""
protocol.py — Framing, binary serialization, typed errors, and request/response validation
"""

from enum import Enum
import struct
import numpy as np
from typing import Any, Optional


class WorkClass(str, Enum):
    """Workload classification for priority scheduling and microbatch sizing."""
    INTERACTIVE = "interactive"
    BULK = "bulk"


class MLXRuntimeError(Exception):
    """Base runtime error for all MLX operations."""
    pass


class MLXServerError(MLXRuntimeError):
    """Base exception for MLX server errors with HTTP status mapping."""
    status_code: int = 500
    error_type: str = "server_error"

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class ProtocolError(MLXServerError, ValueError):
    """Raised when wire protocol serialization or payload validation fails."""
    status_code = 400
    error_type = "invalid_protocol"


class InvalidInputError(ProtocolError):
    """Raised on invalid or malformed input parameters."""
    status_code = 400
    error_type = "invalid_input"


class OverloadedError(MLXServerError):
    """Raised when the executor queue is full or server is overwhelmed."""
    status_code = 429
    error_type = "overloaded"


class DeadlineExceededError(MLXServerError, TimeoutError):
    """Raised when request execution exceeds the absolute deadline."""
    status_code = 504
    error_type = "deadline_exceeded"


class RequestCancelledError(MLXServerError):
    """Raised when client aborts/cancels the request before completion."""
    status_code = 400
    error_type = "request_cancelled"


class UnsupportedModelError(MLXServerError):
    """Raised when a requested model architecture or identifier is unsupported."""
    status_code = 400
    error_type = "unsupported_model"


class ModelUnavailableError(MLXServerError):
    """Raised when model loading fails or adapter is not configured."""
    status_code = 503
    error_type = "model_unavailable"


class OutOfMemoryError(MLXServerError):
    """Raised when GPU/Metal memory allocation fails after retry budget."""
    status_code = 503
    error_type = "out_of_memory"


class WorkerUnavailableError(MLXServerError):
    """Raised when the execution owner worker is stopped or crashed."""
    status_code = 503
    error_type = "worker_unavailable"


def parse_timeout(
    payload: Optional[dict] = None,
    header_val: Optional[str] = None,
    default_timeout: float = 300.0,
    min_timeout: float = 0.1,
    max_timeout: float = 3600.0,
) -> float:
    """
    Extracts and validates request timeout in seconds from header or body payload.
    """
    timeout = default_timeout

    if header_val is not None:
        try:
            val = float(header_val)
            if val > 0:
                timeout = val
        except ValueError:
            pass

    if payload and isinstance(payload, dict) and "timeout" in payload:
        val = payload["timeout"]
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if val > 0:
                timeout = float(val)

    return max(min_timeout, min(timeout, max_timeout))


def encode_binary_embeddings(arr: np.ndarray, count: int, dims: int) -> bytes:
    """
    Encodes float32 numpy array into binary wire format:
    Header: [int32 count (4B)][int32 dims (4B)] little-endian (<ii)
    Data: raw float32 bytes
    """
    if count == 0 or dims == 0:
        return struct.pack("<ii", 0, 0)

    if arr.ndim != 2:
        raise ProtocolError(f"Expected 2D array, got ndim={arr.ndim} with shape {arr.shape}")
    if arr.shape[0] != count or arr.shape[1] != dims:
        raise ProtocolError(f"Shape {arr.shape} does not match count={count}, dims={dims}")

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    if not np.all(np.isfinite(arr)):
        raise ProtocolError("Array contains NaN or Infinite float values")

    header = struct.pack("<ii", count, dims)
    return header + arr.tobytes()


def decode_binary_embeddings(buffer: bytes) -> tuple[np.ndarray, int, int]:
    """
    Decodes binary payload into (np.ndarray, count, dims).
    """
    if len(buffer) < 8:
        raise ProtocolError(f"Buffer length {len(buffer)} too short for 8-byte header")

    count, dims = struct.unpack("<ii", buffer[:8])
    if count < 0 or dims < 0:
        raise ProtocolError(f"Invalid count={count} or dims={dims}")

    if count == 0 or dims == 0:
        return np.empty((0, 0), dtype=np.float32), 0, 0

    expected_bytes = 8 + count * dims * 4
    if len(buffer) != expected_bytes:
        raise ProtocolError(
            f"Buffer length {len(buffer)} does not match expected {expected_bytes} for {count}x{dims} float32"
        )

    floats = np.frombuffer(buffer[8:], dtype=np.float32).reshape((count, dims))
    if not np.all(np.isfinite(floats)):
        raise ProtocolError("Decoded array contains NaN or Infinite float values")

    return floats, count, dims


def validate_embed_request(
    payload: Any,
    max_texts: int = 512,
    max_text_len: int = 256 * 1024,
) -> tuple[list[str], Optional[int], bool]:
    """
    Validates JSON request payload for /embed and /embed-bin.
    Returns (texts, dims, is_query).
    """
    if not isinstance(payload, dict):
        raise InvalidInputError("Payload must be a JSON object")

    texts = payload.get("texts")
    if not isinstance(texts, list):
        raise InvalidInputError("'texts' must be a list of strings")
    if len(texts) == 0:
        raise InvalidInputError("'texts' list cannot be empty")
    if len(texts) > max_texts:
        raise InvalidInputError(f"Request exceeds maximum allowed texts ({len(texts)} > {max_texts})")

    cleaned_texts: list[str] = []
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise InvalidInputError(f"Item at index {i} is not a string (type={type(t).__name__})")
        if not t.strip():
            raise InvalidInputError(f"Item at index {i} is empty or whitespace")
        if len(t) > max_text_len:
            raise InvalidInputError(f"Item at index {i} exceeds max length ({len(t)} > {max_text_len})")
        cleaned_texts.append(t)

    dims = payload.get("dims")
    if dims is not None:
        if isinstance(dims, bool) or not isinstance(dims, (int, float)) or int(dims) <= 0:
            raise InvalidInputError(f"'dims' must be a positive integer, got: {dims}")
        dims = int(dims)

    is_query = bool(payload.get("is_query", False))
    return cleaned_texts, dims, is_query


def validate_rerank_request(
    payload: Any,
    max_documents: int = 128,
    max_doc_len: int = 256 * 1024,
) -> tuple[str, list[str]]:
    """Validates a /rerank request body. Returns (query, documents)."""
    if not isinstance(payload, dict):
        raise InvalidInputError("Payload must be a JSON object")

    query = payload.get("query")
    documents = payload.get("documents")
    if not isinstance(query, str) or not query.strip():
        raise InvalidInputError("'query' must be a non-empty string")
    if not isinstance(documents, list) or len(documents) == 0:
        raise InvalidInputError("'documents' must be a non-empty list of strings")
    if len(documents) > max_documents:
        raise InvalidInputError(f"'documents' exceeds per-request limit of {max_documents} (got {len(documents)})")
    for i, doc in enumerate(documents):
        if not isinstance(doc, str):
            raise InvalidInputError(f"document at index {i} is not a string (type={type(doc).__name__})")
        if not doc.strip():
            raise InvalidInputError(f"document at index {i} is empty or whitespace")
        if len(doc) > max_doc_len:
            raise InvalidInputError(f"document at index {i} exceeds max length {max_doc_len} bytes")
    return query, documents


def validate_generate_request(
    payload: Any,
    max_prompt_len: int = 128 * 1024,
) -> tuple[str, int, float]:
    """Validates a /generate request body. Returns (prompt, max_tokens, temperature)."""
    if not isinstance(payload, dict):
        raise InvalidInputError("Payload must be a JSON object")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InvalidInputError("'prompt' must be a non-empty string")
    if len(prompt) > max_prompt_len:
        raise InvalidInputError(f"'prompt' exceeds max length {max_prompt_len} bytes")
    max_tokens = payload.get("max_tokens", 600)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0 or max_tokens > 4096:
        raise InvalidInputError(f"'max_tokens' must be an integer in (0, 4096], got {max_tokens}")
    temperature = payload.get("temperature", 0.0)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not (0.0 <= float(temperature) <= 2.0):
        raise InvalidInputError(f"'temperature' must be a number in [0.0, 2.0], got {temperature}")
    return prompt, max_tokens, float(temperature)
