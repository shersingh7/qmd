"""
test_mlx_generate.py — Unit tests for MLX generate adapter (real model, Metal).
"""

import pytest


@pytest.mark.real_model
def test_generate_adapter_loads():
    from scripts.qmd_mlx.generate import MLXGenerateAdapter
    a = MLXGenerateAdapter(model_name="mlx-community/Qwen3-1.7B-4bit")
    assert a.model is not None
    d = a.get_descriptor()
    assert d["kind"] == "generate"
    assert "Qwen3-1.7B-4bit" in d["model"]
    a.shutdown()


@pytest.mark.real_model
def test_generate_determinism_and_bounds():
    from scripts.qmd_mlx.generate import MLXGenerateAdapter
    a = MLXGenerateAdapter(model_name="mlx-community/Qwen3-1.7B-4bit")
    try:
        t1 = a.generate("/no_think Expand this search query: database indexing", max_tokens=24)
        t2 = a.generate("/no_think Expand this search query: database indexing", max_tokens=24)
        assert isinstance(t1, str) and len(t1) > 0
        # Deterministic decoding (temp=0) must produce identical output
        assert t1 == t2
    finally:
        a.shutdown()


def test_generate_token_counting_and_stats():
    from unittest.mock import patch, MagicMock
    from scripts.qmd_mlx.generate import MLXGenerateAdapter

    a = MLXGenerateAdapter(model_name="fake-gen-model", lazy_load=True)
    a.model = MagicMock()
    a.tokenizer = MagicMock()
    a.raw_hf_tokenizer = MagicMock()
    a.raw_hf_tokenizer.encode.return_value = [1, 2, 3]
    a.raw_hf_tokenizer.apply_chat_template.side_effect = Exception("no template")

    class StreamChunk:
        def __init__(self, text):
            self.text = text

    chunks = [StreamChunk(f"word{i} ") for i in range(7)]

    with patch("scripts.qmd_mlx.generate.mlx_lm.stream_generate", return_value=iter(chunks)):
        out = a._generate_sync("prompt text", max_tokens=20, temperature=0.0)
        assert "word0 word1 word2 word3 word4 word5 word6" in out

    assert a.total_tokens_generated == 7
    assert a.total_requests == 1
    stats = a.get_stats_info()
    assert stats["total_tokens_generated"] == 7
    assert stats["total_requests"] == 1
    assert stats["avg_ms"] >= 0.0


def test_generate_rejects_invalid_input():
    from scripts.qmd_mlx.generate import MLXGenerateAdapter, GenerateError
    a = MLXGenerateAdapter(model_name="fake-gen-model", lazy_load=True)
    try:
        with pytest.raises(GenerateError):
            a.submit_generate("")
        with pytest.raises(GenerateError):
            a.submit_generate("valid", max_tokens=0)
        with pytest.raises(GenerateError):
            a.submit_generate("valid", max_tokens=999999)
    finally:
        a.shutdown()


def test_generate_cancellation_mid_decode():
    import threading
    import time
    from unittest.mock import patch, MagicMock
    from scripts.qmd_mlx.generate import MLXGenerateAdapter
    from scripts.qmd_mlx.protocol import RequestCancelledError

    a = MLXGenerateAdapter(model_name="mlx-community/Qwen3-1.7B-4bit", lazy_load=True)
    a.model = MagicMock()
    a.tokenizer = MagicMock()
    a.raw_hf_tokenizer = MagicMock()
    a.raw_hf_tokenizer.encode.return_value = [1, 2, 3]
    a.raw_hf_tokenizer.apply_chat_template.side_effect = Exception("no template")

    cancel_event = threading.Event()

    class StreamResp:
        def __init__(self, text):
            self.text = text

    def slow_stream(*args, **kwargs):
        for i in range(10):
            time.sleep(0.05)
            if i == 2:
                cancel_event.set()
            yield StreamResp(f"tok_{i} ")

    with patch("scripts.qmd_mlx.generate.mlx_lm.stream_generate", side_effect=slow_stream):
        t0 = time.monotonic()
        with pytest.raises(RequestCancelledError):
            a._generate_sync("hello", max_tokens=10, temperature=0.0, cancel_event=cancel_event)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.35


def test_generate_deadline_mid_decode():
    import time
    from unittest.mock import patch, MagicMock
    from scripts.qmd_mlx.generate import MLXGenerateAdapter
    from scripts.qmd_mlx.protocol import DeadlineExceededError

    a = MLXGenerateAdapter(model_name="mlx-community/Qwen3-1.7B-4bit", lazy_load=True)
    a.model = MagicMock()
    a.tokenizer = MagicMock()
    a.raw_hf_tokenizer = MagicMock()
    a.raw_hf_tokenizer.encode.return_value = [1, 2, 3]
    a.raw_hf_tokenizer.apply_chat_template.side_effect = Exception("no template")

    class StreamResp:
        def __init__(self, text):
            self.text = text

    def slow_stream(*args, **kwargs):
        for i in range(10):
            time.sleep(0.05)
            yield StreamResp(f"tok_{i} ")

    with patch("scripts.qmd_mlx.generate.mlx_lm.stream_generate", side_effect=slow_stream):
        deadline = time.monotonic() + 0.12
        with pytest.raises(DeadlineExceededError):
            a._generate_sync("hello", max_tokens=10, temperature=0.0, deadline=deadline)