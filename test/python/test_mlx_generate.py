"""
test_mlx_generate.py — Unit tests for MLX generate adapter (real model, Metal).
"""

import pytest


def test_generate_adapter_loads():
    from scripts.qmd_mlx.generate import MLXGenerateAdapter
    a = MLXGenerateAdapter(model_name="mlx-community/Qwen3-1.7B-4bit")
    assert a.model is not None
    d = a.get_descriptor()
    assert d["kind"] == "generate"
    assert "Qwen3-1.7B-4bit" in d["model"]
    a.shutdown()


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


def test_generate_rejects_invalid_input():
    from scripts.qmd_mlx.generate import MLXGenerateAdapter, GenerateError
    a = MLXGenerateAdapter(model_name="mlx-community/Qwen3-1.7B-4bit", lazy_load=True)
    try:
        with pytest.raises(GenerateError):
            a.submit_generate("")
        with pytest.raises(GenerateError):
            a.submit_generate("valid", max_tokens=0)
        with pytest.raises(GenerateError):
            a.submit_generate("valid", max_tokens=999999)
    finally:
        a.shutdown()