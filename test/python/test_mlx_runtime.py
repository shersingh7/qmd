"""
test_mlx_runtime.py — Tests for MLX embedding runtime, padding invariance, and normalization
"""

import numpy as np
import pytest
from scripts.qmd_mlx.runtime import MLXEmbeddingRuntime, MLXRuntimeError


@pytest.fixture(scope="module")
def mlx_runtime():
    # Use cached small model for real Metal execution
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    runtime = MLXEmbeddingRuntime(
        model_name=model_name,
        quantization="bf16",
        dtype_str="float32",
        max_length=512,
    )
    yield runtime
    runtime.shutdown()


def test_idle_unload_and_transparent_reload(mlx_runtime):
    """Idle policy: weights unload after the deadline, reload transparently,
    and embeddings are IDENTICAL before/after the cycle."""
    rt = mlx_runtime
    rt.idle_unload_s = 1.0  # test-fast deadline
    before = rt.embed_direct(["idle unload verification text"])[0]

    # Force the idle path: wait past the deadline with an empty queue.
    import time as _t
    _t.sleep(1.5)
    rt._work_queue.join()  # let the worker observe idleness
    deadline = _t.time() + 5
    while rt._weights_loaded and _t.time() < deadline:
        _t.sleep(0.2)
    assert not rt._weights_loaded, "weights should unload after idle deadline"

    after = rt.embed_direct(["idle unload verification text"])[0]
    assert rt._weights_loaded, "weights should reload on demand"
    np.testing.assert_allclose(before, after, atol=1e-6)
    rt.idle_unload_s = 0  # disable for the rest of the module


def test_runtime_initialization(mlx_runtime):
    assert mlx_runtime.native_dims == 384
    assert mlx_runtime.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    desc = mlx_runtime.get_descriptor()
    assert desc["backend"] == "mlx"
    assert desc["nativeDimensions"] == 384
    assert desc["outputDimensions"] == 384
    assert desc["normalized"] is True


def test_embedding_normalization_and_finiteness(mlx_runtime):
    texts = ["Fast semantic search on Apple Silicon", "Hybrid retrieval with SQLite and MLX"]
    res = mlx_runtime.embed_direct(texts)

    assert res.shape == (2, 384)
    assert np.all(np.isfinite(res))
    norms = np.linalg.norm(res, axis=-1)
    assert np.allclose(norms, [1.0, 1.0], atol=1e-5)


def test_padding_invariance(mlx_runtime):
    """
    Critical correctness gate: Embedding of text X must be invariant
    to other texts of different lengths in the same batch (no attention leakage across padding).
    """
    text1 = "Short query"
    long_text = (
        "This is an extensive document with numerous tokens designed to test that padding masks "
        "properly isolate shorter sequences in the batch from attention leakage or corruption."
    )

    vec1_alone = mlx_runtime.embed_direct([text1])[0]
    vec1_batched = mlx_runtime.embed_direct([text1, long_text])[0]

    assert np.allclose(vec1_alone, vec1_batched, atol=1e-4)


def test_matryoshka_dimension_reduction(mlx_runtime):
    res_full = mlx_runtime.embed_direct(["test text"])
    assert res_full.shape == (1, 384)

    res_128 = mlx_runtime.embed_direct(["test text"], requested_dims=128)
    assert res_128.shape == (1, 128)
    assert np.all(np.isfinite(res_128))
    norm_128 = np.linalg.norm(res_128, axis=-1)
    assert np.allclose(norm_128, [1.0], atol=1e-5)


def test_tokenize_endpoint(mlx_runtime):
    texts = ["apple silicon", "mlx embedding"]
    tokens = mlx_runtime.tokenize(texts)
    assert len(tokens) == 2
    assert all(isinstance(t, list) for t in tokens)
    assert all(len(t) > 0 for t in tokens)


def test_unsupported_model_raises_error():
    with pytest.raises(MLXRuntimeError):
        MLXEmbeddingRuntime("nonexistent/invalid-model-name-xyz-999")


def test_runtime_deadline_propagation(mlx_runtime):
    import time
    from scripts.qmd_mlx.protocol import DeadlineExceededError
    # Past deadline must fail immediately with DeadlineExceededError
    past_deadline = time.monotonic() - 1.0
    with pytest.raises(DeadlineExceededError):
        mlx_runtime.submit_embed(["quick test"], deadline=past_deadline)


def test_runtime_cancellation_propagation(mlx_runtime):
    import threading
    from scripts.qmd_mlx.protocol import RequestCancelledError
    cancel_event = threading.Event()
    cancel_event.set()
    with pytest.raises(RequestCancelledError):
        mlx_runtime.submit_embed(["quick test"], cancel_event=cancel_event)


def test_runtime_bulk_micro_batch_ordering(mlx_runtime):
    texts = [f"Distinct query string number {i} with some extra padding text" for i in range(12)]
    # Embed in bulk
    res = mlx_runtime.submit_embed(texts, is_query=False)
    assert res.shape == (12, 384)

    # Embed individually and compare
    individual = [mlx_runtime.submit_embed([t], is_query=True)[0] for t in texts]
    for i in range(12):
        assert np.allclose(res[i], individual[i], atol=1e-4)


def test_runtime_mixed_stage_eviction_and_micro_batch_reload():
    """
    Blocker 2: Deterministically test real MLX runtime mixed-stage eviction
    and next embedding micro-batch reload on executor thread.
    """
    from scripts.qmd_mlx.executor import GPUExecutor
    from scripts.qmd_mlx.model_manager import ModelResidencyManager, ModelState

    executor = GPUExecutor(max_queue_size=20)
    # Residency budget of 1500MB (can fit either embed at 1000MB or rerank at 1000MB, but not both)
    manager = ModelResidencyManager(executor, residency_budget_mb=1500.0)

    class ControllableMockAdapter:
        def __init__(self, name: str, memory_mb: float = 1000.0):
            self.model_name = name
            self.model_memory_mb = memory_mb
            self._loaded = False
            self.load_count = 0
            self.unload_count = 0

        def is_loaded(self):
            return self._loaded

        def load(self):
            self._loaded = True
            self.load_count += 1

        def unload(self):
            self._loaded = False
            self.unload_count += 1

    # Create real runtime with sentence-transformers/all-MiniLM-L6-v2
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    runtime = MLXEmbeddingRuntime(
        model_name=model_name,
        quantization="bf16",
        dtype_str="float32",
        max_length=512,
        executor=executor,
        model_manager=manager,
        lazy_load=False,
    )
    # Set embed adapter memory_mb to 1000MB for residency budgeting
    runtime.adapter.model_memory_mb = 1000.0

    # Register a competing stage adapter
    ad_rerank = ControllableMockAdapter("rerank-model", memory_mb=1000.0)
    manager.register_adapter("rerank", ad_rerank)

    # Force multi-micro-batch execution: max_batch_tokens = 10 so each sentence is a separate micro-batch
    runtime.batch_planner.max_batch_tokens = 10

    texts = [
        "First micro-batch sentence for runtime eviction test.",
        "Second micro-batch sentence for runtime eviction test.",
        "Third micro-batch sentence for runtime eviction test.",
    ]

    # Verify initial state
    assert runtime.adapter.is_loaded()
    assert manager.get_state("embed") == ModelState.READY

    # Hook into forward_batch to evict embed stage right after the 1st micro-batch runs
    orig_forward_batch = runtime.adapter.forward_batch
    forward_call_count = [0]

    def hooking_forward_batch(batch, requested_dims=None):
        forward_call_count[0] += 1
        res = orig_forward_batch(batch, requested_dims=requested_dims)
        if forward_call_count[0] == 1:
            # Right after micro-batch 1 runs, load rerank stage on executor to evict embed!
            # Since we are already on the executor thread during forward_batch, call ensure_loaded("rerank")
            manager.ensure_loaded("rerank")
        return res

    runtime.adapter.forward_batch = hooking_forward_batch

    # Run multi-micro-batch embedding request
    res = runtime.submit_embed(texts, is_query=False)

    assert res.shape == (3, 384)
    assert np.all(np.isfinite(res))
    assert forward_call_count[0] == 3  # All 3 micro-batches ran
    # embed was evicted when rerank loaded, then reloaded on micro-batch 2 on the owner thread!
    assert manager.get_state("embed") == ModelState.READY
    assert runtime.adapter.is_loaded()

    runtime.shutdown()
    executor.shutdown()


def test_runtime_shutdown_under_owner_thread():
    """Verify runtime.shutdown serializes model unload on owner thread before executor shutdown."""
    from scripts.qmd_mlx.executor import GPUExecutor
    from scripts.qmd_mlx.model_manager import ModelResidencyManager, ModelState

    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor, residency_budget_mb=4000.0)
    runtime = MLXEmbeddingRuntime(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        executor=executor,
        model_manager=manager,
    )
    assert runtime.adapter.is_loaded()
    assert manager.get_state("embed") == ModelState.READY

    runtime.shutdown()

    assert not runtime.adapter.is_loaded()
    assert manager.get_state("embed") == ModelState.UNLOADED
    executor.shutdown()


def test_descriptor_reports_measured_quantization_and_revision():
    from scripts.qmd_mlx.adapters.embedding import QwenEmbeddingAdapter
    from unittest.mock import patch, MagicMock

    adapter = QwenEmbeddingAdapter(model_name="test-qwen")
    # Before load, if unconfigured, fallback must be 'unknown'
    assert adapter.get_descriptor()["quantization"] == "unknown"
    assert adapter.get_descriptor()["revision"] == "unknown"

    # Mock loaded model with config exposing 4bit quantization and commit hash
    fake_model = MagicMock()
    fake_tok = MagicMock()
    fake_config = {
        "quantization": {"bits": 4, "group_size": 64},
        "_commit_hash": "abc1234def",
        "hidden_size": 1024,
    }

    with patch("scripts.qmd_mlx.adapters.embedding._MLX_AVAILABLE", True), \
         patch("mlx_lm.load", return_value=(fake_model, fake_tok, fake_config)):
        adapter.load()

    desc = adapter.get_descriptor()
    assert desc["quantization"] == "4bit"
    assert desc["revision"] == "abc1234def"


def test_concurrent_tokenize_and_submit_embed_serialized_init():
    import threading
    import time
    from unittest.mock import patch, MagicMock
    from scripts.qmd_mlx.adapters.embedding import QwenEmbeddingAdapter
    from scripts.qmd_mlx.runtime import MLXEmbeddingRuntime
    from scripts.qmd_mlx.executor import GPUExecutor
    from scripts.qmd_mlx.model_manager import ModelResidencyManager

    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor)

    load_count = [0]
    padding_side_sets = [0]

    class MockTokenizer:
        def __init__(self):
            self._padding_side = "right"
            self.pad_token_id = 0

        @property
        def padding_side(self):
            return self._padding_side

        @padding_side.setter
        def padding_side(self, val):
            padding_side_sets[0] += 1
            self._padding_side = val

        def encode(self, text, **kwargs):
            return [1, 2, 3]

        def __call__(self, texts, **kwargs):
            return {
                "input_ids": np.ones((len(texts), 4), dtype=np.int32),
                "attention_mask": np.ones((len(texts), 4), dtype=np.int32),
            }

    fake_tok = MockTokenizer()
    fake_model = MagicMock()
    fake_config = {"hidden_size": 384}

    def slow_load(*args, **kwargs):
        time.sleep(0.08)
        load_count[0] += 1
        return fake_model, fake_tok, fake_config

    with patch("scripts.qmd_mlx.adapters.embedding._MLX_AVAILABLE", True), \
         patch("mlx_lm.load", side_effect=slow_load):
        runtime = MLXEmbeddingRuntime(
            model_name="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
            executor=executor,
            model_manager=manager,
            lazy_load=True,
        )

        # Mock adapter forward_batch
        runtime.adapter.forward_batch = lambda batch, requested_dims=None: np.zeros((len(batch), 384), dtype=np.float32)

        errors = []
        threads = []

        def run_tokenize():
            try:
                toks = runtime.tokenize(["hello world test query"])
                assert len(toks) == 1
            except Exception as e:
                errors.append(e)

        def run_embed():
            try:
                emb = runtime.submit_embed(["hello world test query"])
                assert emb.shape == (1, 384)
            except Exception as e:
                errors.append(e)

        for _ in range(4):
            t1 = threading.Thread(target=run_tokenize)
            t2 = threading.Thread(target=run_embed)
            threads.extend([t1, t2])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert load_count[0] == 1
        assert padding_side_sets[0] == 1

        runtime.shutdown()
        executor.shutdown()



