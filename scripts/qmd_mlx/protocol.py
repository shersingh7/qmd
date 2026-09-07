"""
protocol.py — Framing, binary serialization, and request/response validation
"""

import struct
import numpy as np
from typing import Any, Optional


class ProtocolError(ValueError):
    """Raised when wire protocol serialization or payload validation fails."""
    pass


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

    # Ensure float32 contiguous array
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    # Check finite values
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


def validate_embed_request(payload: Any, max_texts: int = 512, max_text_len: int = 100_000) -> tuple[list[str], Optional[int], bool]:
    """
    Validates JSON request payload for /embed and /embed-bin.
    Returns (texts, dims, is_query).
    """
    if not isinstance(payload, dict):
        raise ProtocolError("Payload must be a JSON object")

    texts = payload.get("texts")
    if not isinstance(texts, list):
        raise ProtocolError("'texts' must be a list of strings")
    if len(texts) == 0:
        raise ProtocolError("'texts' list cannot be empty")
    if len(texts) > max_texts:
        raise ProtocolError(f"Request exceeds maximum allowed texts ({len(texts)} > {max_texts})")

    cleaned_texts: list[str] = []
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise ProtocolError(f"Item at index {i} is not a string (type={type(t).__name__})")
        if len(t) > max_text_len:
            raise ProtocolError(f"Item at index {i} exceeds max length ({len(t)} > {max_text_len})")
        cleaned_texts.append(t)

    dims = payload.get("dims")
    if dims is not None:
        if isinstance(dims, bool) or not isinstance(dims, (int, float)) or int(dims) <= 0:
            raise ProtocolError(f"'dims' must be a positive integer, got: {dims}")
        dims = int(dims)

    is_query = bool(payload.get("is_query", False))
    return cleaned_texts, dims, is_query
