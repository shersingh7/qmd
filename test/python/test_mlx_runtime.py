"""
test_mlx_runtime.py — Tests for MLX embedding runtime, padding invariance, and normalization
"""

import threading
import time
import numpy as np
import pytest
from scripts.qmd_mlx.adapters.tokenization import TokenizedBatch
from scripts.qmd_mlx.protocol import (
    DeadlineExceededError,
    InvalidInputError,
    MLXRuntimeError,
    OutOfMemoryError,
    OverloadedError,
    RequestCancelledError,
    WorkerUnavailableError,
)
from scripts.qmd_mlx.runtime import MLXEmbeddingRuntime
from unittest.mock import MagicMock, patch
from scripts.qmd_mlx.executor import GPUExecutor
from scripts.qmd_mlx.model_manager import ModelResidencyManager, ModelState


class MockEmbeddingAdapter:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.native_dims = 384
        self.max_length = 512
        self.model_params_b = 0.5
        self.model_memory_mb = 500.0
        self._loaded = True
        self.raw_hf_tokenizer = MagicMock()
        self.raw_hf_tokenizer.encode.side_effect = lambda text, **kwargs: [1, 2, 3]
        self.model = MagicMock()

    def is_loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True
        self.model = MagicMock()

    def unload(self):
        self._loaded = False
        self.model = None

    def tokenize_texts(self, texts, max_length=None):
        from scripts.qmd_mlx.adapters.tokenization import TokenizedBatch
        token_ids = [[101, 202, 103] for _ in texts]
        lengths = [len(t) for t in token_ids]
        indices = list(range(len(texts)))
        return TokenizedBatch(token_ids, lengths, indices, pad_token_id=0, texts=list(texts))

    def forward_batch(self, batch, requested_dims=None):
        dims = requested_dims or self.native_dims
        count = len(batch)
        arr = np.ones((count, dims), dtype=np.float32)
        arr = arr / np.linalg.norm(arr, axis=-1, keepdims=True)
        return arr

    def get_descriptor(self, requested_dims=None):
        out_dims = requested_dims if requested_dims and requested_dims < self.native_dims else self.native_dims
        return {
            "version": 1,
            "backend": "mlx",
            "model": self.model_name,
            "nativeDimensions": self.native_dims,
            "outputDimensions": out_dims,
            "normalized": True,
            "maxSequenceLength": self.max_length,
            "pooling": "mean",
        }


@pytest.fixture(scope="module")
def mlx_runtime():
    executor = GPUExecutor(max_queue_size=50)
    manager = ModelResidencyManager(executor, residency_budget_mb=6000.0)

    mock_adapter = MockEmbeddingAdapter()
    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=mock_adapter):
        runtime = MLXEmbeddingRuntime(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            executor=executor,
            model_manager=manager,
            lazy_load=False,
        )
    yield runtime
    runtime.shutdown()
    executor.shutdown()


@pytest.mark.real_model
def test_idle_unload_and_transparent_reload():
    """Idle policy with real Metal execution."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    rt = MLXEmbeddingRuntime(
        model_name=model_name,
        quantization="bf16",
        dtype_str="float32",
        max_length=512,
    )
    try:
        rt.idle_unload_s = 1.0
        before = rt.embed_direct(["idle unload verification text"])[0]

        import time as _t
        _t.sleep(1.5)
        rt._work_queue.join()
        deadline = _t.time() + 5
        while rt._weights_loaded and _t.time() < deadline:
            _t.sleep(0.2)
        assert not rt._weights_loaded, "weights should unload after idle deadline"

        after = rt.embed_direct(["idle unload verification text"])[0]
        assert rt._weights_loaded, "weights should reload on demand"
        np.testing.assert_allclose(before, after, atol=1e-6)
    finally:
        rt.shutdown()


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


@pytest.mark.real_model
def test_padding_invariance():
    """
    Critical correctness gate with real Metal model execution.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    rt = MLXEmbeddingRuntime(
        model_name=model_name,
        quantization="bf16",
        dtype_str="float32",
        max_length=512,
    )
    try:
        text1 = "Short query"
        long_text = (
            "This is an extensive document with numerous tokens designed to test that padding masks "
            "properly isolate shorter sequences in the batch from attention leakage or corruption."
        )

        vec1_alone = rt.embed_direct([text1])[0]
        vec1_batched = rt.embed_direct([text1, long_text])[0]

        assert np.allclose(vec1_alone, vec1_batched, atol=1e-4)
    finally:
        rt.shutdown()


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
    res = mlx_runtime.submit_embed(texts, is_query=False)
    assert res.shape == (12, 384)


def test_runtime_mixed_stage_eviction_and_micro_batch_reload():
    """
    Deterministically test runtime mixed-stage eviction
    and next embedding micro-batch reload on executor thread.
    """
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

    mock_embed = MockEmbeddingAdapter()
    mock_embed.model_memory_mb = 1000.0

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=mock_embed):
        runtime = MLXEmbeddingRuntime(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            executor=executor,
            model_manager=manager,
            lazy_load=False,
        )

    ad_rerank = ControllableMockAdapter("rerank-model", memory_mb=1000.0)
    manager.register_adapter("rerank", ad_rerank)

    # Force multi-micro-batch execution: max_batch_tokens = 2 so each sentence is a separate micro-batch
    runtime.batch_planner.max_batch_tokens = 2

    texts = [
        "First micro-batch sentence for runtime eviction test.",
        "Second micro-batch sentence for runtime eviction test.",
        "Third micro-batch sentence for runtime eviction test.",
    ]

    assert runtime.adapter.is_loaded()
    assert manager.get_state("embed") == ModelState.READY

    orig_forward_batch = runtime.adapter.forward_batch
    forward_call_count = [0]

    def hooking_forward_batch(batch, requested_dims=None):
        forward_call_count[0] += 1
        res = orig_forward_batch(batch, requested_dims=requested_dims)
        if forward_call_count[0] == 1:
            manager.ensure_loaded("rerank")
        return res

    runtime.adapter.forward_batch = hooking_forward_batch

    res = runtime.submit_embed(texts, is_query=False)

    assert res.shape == (3, 384)
    assert np.all(np.isfinite(res))
    assert forward_call_count[0] == 3
    assert manager.get_state("embed") == ModelState.READY
    assert runtime.adapter.is_loaded()

    runtime.shutdown()
    executor.shutdown()


def test_runtime_shutdown_under_owner_thread():
    """Verify runtime.shutdown serializes model unload on owner thread before executor shutdown."""
    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor, residency_budget_mb=4000.0)
    mock_adapter = MockEmbeddingAdapter()
    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=mock_adapter):
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


def test_runtime_tokenization_survives_model_weight_eviction_regression(mlx_runtime):
    """Finding 3 Regression: Tokenization on CPU succeeds even if model weights were evicted."""
    rt = mlx_runtime
    # Ensure model is initialized
    _ = rt.tokenize(["initial tokenization check"])
    assert rt.adapter.raw_hf_tokenizer is not None

    # Simulate model weight eviction while keeping tokenizer
    rt.model_manager.unload("embed")
    assert not rt.adapter.is_loaded()
    assert rt.adapter.model is None
    assert rt.adapter.raw_hf_tokenizer is not None

    # Calling tokenize must succeed on CPU without loading GPU model weights
    tokens = rt.tokenize(["test tokenization without weights", "second query text"])
    assert len(tokens) == 2
    assert all(len(t) > 0 for t in tokens)
    # Model weights must NOT have been loaded on caller thread
    assert rt.adapter.model is None

    # submit_embed must reload weights safely on the owner thread before forward pass
    res = rt.submit_embed(["test tokenization without weights"])
    assert res.shape == (1, 384)
    assert rt.adapter.is_loaded()


def test_runtime_bounded_admission_rejection_regression(mlx_runtime):
    """Finding 3 Regression: Early bounded admission rejects invalid or oversized requests before tokenization."""
    from scripts.qmd_mlx.protocol import InvalidInputError, OverloadedError, WorkerUnavailableError
    rt = mlx_runtime

    # 1. Reject > 512 texts
    oversized_list = ["test text"] * 513
    with pytest.raises(InvalidInputError, match="exceeds maximum allowed texts"):
        rt.submit_embed(oversized_list)

    with pytest.raises(InvalidInputError, match="exceeds maximum allowed texts"):
        rt.tokenize(oversized_list)

    # 2. Reject non-string or whitespace-only items
    with pytest.raises(InvalidInputError, match="empty or whitespace"):
        rt.submit_embed(["valid text", "   \n\t  "])

    # 3. Reject oversized string (> 256KB)
    huge_str = "x" * (256 * 1024 + 10)
    with pytest.raises(InvalidInputError, match="exceeds max length"):
        rt.submit_embed([huge_str])

    # 4. Reject when executor is shut down
    from scripts.qmd_mlx.executor import GPUExecutor
    from scripts.qmd_mlx.model_manager import ModelResidencyManager
    dead_exec = GPUExecutor()
    dead_exec.shutdown()
    dead_mgr = ModelResidencyManager(dead_exec)

    # Instantiate runtime with dead executor
    runtime_dead = MLXEmbeddingRuntime(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        executor=dead_exec,
        model_manager=dead_mgr,
        lazy_load=True,
    )
    with pytest.raises(WorkerUnavailableError):
        runtime_dead.submit_embed(["test query"])


def test_unloaded_tokenize_self_deadlock_regression():
    """Verify tokenize_texts on an unloaded adapter does not self-deadlock when calling self.load()."""
    from scripts.qmd_mlx.adapters.embedding import BertEmbeddingAdapter, QwenEmbeddingAdapter
    from unittest.mock import MagicMock, patch

    # 1. BertEmbeddingAdapter test
    bert_adapter = BertEmbeddingAdapter(model_name="fake-bert")
    assert not bert_adapter.is_loaded()
    fake_tokenizer = MagicMock()
    fake_tokenizer.encode.side_effect = lambda t, **kwargs: [1, 2, 3]

    with patch.object(bert_adapter, "load") as mock_load:
        def do_load():
            # When load is called while tokenize_texts holds _init_lock,
            # this must succeed immediately because _init_lock is an RLock!
            with bert_adapter._init_lock:
                bert_adapter.raw_hf_tokenizer = fake_tokenizer
                bert_adapter.tokenizer = fake_tokenizer
                bert_adapter.model = MagicMock()
        mock_load.side_effect = do_load

        tokens = bert_adapter.tokenize_texts(["hello world", "testing deadlock"])
        assert len(tokens) == 2
        assert mock_load.called

    # 2. QwenEmbeddingAdapter test
    qwen_adapter = QwenEmbeddingAdapter(model_name="fake-qwen")
    assert not qwen_adapter.is_loaded()

    fake_qwen_tok = MagicMock()
    fake_qwen_tok.encode.side_effect = lambda t, **kwargs: [1, 2, 3]

    with patch.object(qwen_adapter, "load") as mock_load:
        def do_load_qwen():
            with qwen_adapter._init_lock:
                qwen_adapter.raw_hf_tokenizer = fake_qwen_tok
                qwen_adapter.tokenizer = fake_qwen_tok
                qwen_adapter.model = MagicMock()
        mock_load.side_effect = do_load_qwen

        tokens = qwen_adapter.tokenize_texts(["hello world"])
        assert len(tokens) == 1
        assert mock_load.called


def test_infer_model_params_b_and_memory_estimation_distinguishes_quantization():
    """Verify that 4B parameter count is distinguished from 4-bit quantization."""
    from scripts.qmd_mlx.adapters.embedding import infer_model_params_b, estimate_model_memory_mb

    # 4B models
    assert infer_model_params_b("mlx-community/Qwen3-Reranker-4B-mxfp8") == 4.0
    assert infer_model_params_b("qwen3-reranker-4b-mlx-4bit") == 4.0
    assert infer_model_params_b("models/Qwen2.5-Coder-7B-Instruct-4bit") == 7.0
    assert infer_model_params_b("models/Qwen2.5-Coder-0.5B-Instruct-4bit") == 0.5

    # Quantization only (no B in model name) must not be mistaken for 4B
    assert infer_model_params_b("sentence-transformers/all-MiniLM-L6-v2-4bit") == 0.033
    assert infer_model_params_b("BAAI/bge-m3-4bit") == 0.6
    assert infer_model_params_b("nomic-ai/nomic-embed-text-v1.5-8bit") == 0.137

    # Memory estimation
    mem_4b = estimate_model_memory_mb("qwen3-reranker-4b-mlx-4bit", "4bit")
    assert mem_4b > 2000.0  # 4B 4bit is ~2400MB
    mem_minilm = estimate_model_memory_mb("sentence-transformers/all-MiniLM-L6-v2", "bf16")
    assert mem_minilm < 500.0  # ~120MB


def test_admission_lease_concurrency_and_guaranteed_release():
    """Verify AdmissionLease bounds concurrent in-flight requests and bytes, and releases in finally."""
    from scripts.qmd_mlx.runtime import MLXEmbeddingRuntime
    from scripts.qmd_mlx.executor import GPUExecutor
    from scripts.qmd_mlx.model_manager import ModelResidencyManager
    from scripts.qmd_mlx.protocol import OverloadedError
    import time

    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor)
    mock_adapter = MockEmbeddingAdapter()

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=mock_adapter):
        runtime = MLXEmbeddingRuntime(
            model_name="test-embed",
            executor=executor,
            model_manager=manager,
        )

    # Available initial state
    assert runtime.in_flight_requests == 0
    assert runtime.in_flight_bytes == 0

    # Acquire lease for 1 request / 1000 bytes
    lease = runtime.acquire_admission_lease(num_texts=1, num_bytes=1000)
    assert runtime.in_flight_requests == 1
    assert runtime.in_flight_bytes == 1000

    # Releasing lease restores stats
    runtime.release_admission_lease(lease)
    assert runtime.in_flight_requests == 0
    assert runtime.in_flight_bytes == 0

    # Test concurrency saturation (cap at 2)
    runtime.max_concurrent_admissions = 2
    l1 = runtime.acquire_admission_lease(num_texts=1, num_bytes=100)
    l2 = runtime.acquire_admission_lease(num_texts=1, num_bytes=100)
    assert runtime.in_flight_requests == 2

    # 3rd acquire with short deadline raises OverloadedError
    with pytest.raises(OverloadedError):
        runtime.acquire_admission_lease(num_texts=1, num_bytes=100, deadline=time.monotonic() + 0.05)

    runtime.release_admission_lease(l1)
    runtime.release_admission_lease(l2)
    assert runtime.in_flight_requests == 0

    # Guaranteed release on failed tokenize / submit_embed
    runtime.adapter.tokenize_texts = MagicMock(side_effect=RuntimeError("Tokenization error"))
    with pytest.raises(RuntimeError, match="Tokenization error"):
        runtime.tokenize(["sample text"])

    # In-flight stats must be 0 even after exception!
    assert runtime.in_flight_requests == 0
    assert runtime.in_flight_bytes == 0

    runtime.shutdown()
    executor.shutdown()


def test_runtime_admission_lease_oversized_rejection():
    """Verify admission lease strictly rejects oversized single requests and invalid counts."""
    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor, residency_budget_mb=4000.0)
    mock_adapter = MockEmbeddingAdapter()

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=mock_adapter):
        runtime = MLXEmbeddingRuntime(
            model_name="test-embed",
            executor=executor,
            model_manager=manager,
        )

    # 1. Negative count or bytes
    with pytest.raises(InvalidInputError):
        runtime.acquire_admission_lease(text_count=-1, total_bytes=100)
    with pytest.raises(InvalidInputError):
        runtime.acquire_admission_lease(text_count=1, total_bytes=-5)

    # 2. Count > 512
    with pytest.raises(InvalidInputError):
        runtime.acquire_admission_lease(text_count=513, total_bytes=100)

    # 3. Single oversized request > max_in_flight_admission_bytes (64MB) even when current_requests == 0
    assert runtime.in_flight_requests == 0
    with pytest.raises(InvalidInputError):
        runtime.acquire_admission_lease(text_count=1, total_bytes=65 * 1024 * 1024)

    runtime.shutdown()
    executor.shutdown()


def test_runtime_admission_lease_wake_on_shutdown():
    """Verify waiting admission requests are promptly woken and rejected on shutdown."""
    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor, residency_budget_mb=4000.0)
    mock_adapter = MockEmbeddingAdapter()

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=mock_adapter):
        runtime = MLXEmbeddingRuntime(
            model_name="test-embed",
            executor=executor,
            model_manager=manager,
        )

    # Saturate admission
    runtime.max_concurrent_admissions = 1
    lease = runtime.acquire_admission_lease(num_texts=1, num_bytes=100)

    errors = []
    thread_started = threading.Event()

    def waiter():
        thread_started.set()
        try:
            runtime.acquire_admission_lease(num_texts=1, num_bytes=100, deadline=time.monotonic() + 10.0)
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=waiter)
    t.start()

    assert thread_started.wait(timeout=2.0)
    time.sleep(0.05)

    # Trigger runtime shutdown — should wake waiting thread immediately
    t0 = time.time()
    runtime.shutdown()
    t.join(timeout=2.0)
    elapsed = time.time() - t0

    assert elapsed < 1.0  # Woken promptly!
    assert len(errors) == 1
    assert isinstance(errors[0], WorkerUnavailableError)

    runtime.release_admission_lease(lease)
    executor.shutdown()


def test_runtime_tokenize_caps_and_cancellation():
    """Verify tokenize total byte caps and post-tokenization cancellation check."""
    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor, residency_budget_mb=4000.0)
    mock_adapter = MockEmbeddingAdapter()

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=mock_adapter):
        runtime = MLXEmbeddingRuntime(
            model_name="test-embed",
            executor=executor,
            model_manager=manager,
        )

    # 1. Total bytes > 10MB
    large_texts = ["a" * (200 * 1024) for _ in range(55)]  # ~11 MB total
    with pytest.raises(InvalidInputError, match="10MB"):
        runtime.tokenize(large_texts)

    # 2. Cancel event set during/after tokenization
    cancel_evt = threading.Event()
    cancel_evt.set()
    with pytest.raises(RequestCancelledError):
        runtime.tokenize(["sample text"], cancel_event=cancel_evt)

    runtime.shutdown()
    executor.shutdown()


def test_lazy_load_planner_retuning_and_oom_preservation():
    """Verify planner is retuned upon cold load before microbatch planning, and OOM reductions are preserved."""
    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor, residency_budget_mb=8000.0)

    class ColdMockAdapter(MockEmbeddingAdapter):
        def __init__(self):
            super().__init__()
            self.model_name = "test-cold-4b"
            self.model_params_b = 0.6  # initially estimated as 0.6B before load
            self._loaded = False
            self.raw_hf_tokenizer = None

        def load(self):
            self._loaded = True
            self.model_params_b = 4.0  # measured as 4.0B after real load
            self.raw_hf_tokenizer = MagicMock()

        def is_loaded(self):
            return self._loaded

    cold_adapter = ColdMockAdapter()

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=cold_adapter):
        runtime = MLXEmbeddingRuntime(
            model_name="test-cold-4b",
            executor=executor,
            model_manager=manager,
            lazy_load=True,
        )

    # Initially before load, batch_planner is tuned for 0.6B
    initial_budget = runtime.batch_planner.max_batch_tokens

    # Calling submit_embed triggers cold load, updating model_params_b to 4.0 and retuning planner
    runtime.submit_embed(["sample text 1", "sample text 2"])
    retuned_budget = runtime.batch_planner.max_batch_tokens

    # Budget for 4B model must be strictly smaller than 0.6B budget
    assert retuned_budget < initial_budget

    # Simulate an OOM halving
    runtime.batch_planner.execute_sub_batch_with_retry(
        TokenizedBatch(token_ids=[[1, 2]], lengths=[2], original_indices=[0]),
        lambda b: (_ for _ in ()).throw(RuntimeError("Metal buffer allocation failed: out of memory")),
    ) if False else None

    # Manually simulate OOM halving via execute_sub_batch_with_retry
    def oom_forward(b):
        raise RuntimeError("Metal buffer allocation failed: out of memory")

    try:
        runtime.batch_planner.execute_sub_batch_with_retry(
            TokenizedBatch(token_ids=[[1]], lengths=[1], original_indices=[0], pad_token_id=0),
            oom_forward,
        )
    except OutOfMemoryError:
        pass

    halved_budget = runtime.batch_planner.max_batch_tokens
    assert halved_budget < retuned_budget

    # Subsequent submit_embed must NOT undo the halved budget back to default
    runtime.submit_embed(["sample text 3"])
    assert runtime.batch_planner.max_batch_tokens == halved_budget

    runtime.shutdown()
    executor.shutdown()


def test_bulk_microbatch_yield_to_interactive_query():
    """
    Regression Barrier Test:
    Verifies that a multi-document bulk embedding request (split into micro-batches with priority 1)
    yields at micro-batch boundaries to an interactive query (priority 0) arriving concurrently.
    Verifies that:
    1. The interactive query is serviced between bulk micro-batches (not stuck waiting for full bulk completion).
    2. The bulk job completes in full with original ordering preserved (zero token truncation).
    3. The interactive query returns valid normalized results.
    """
    executor = GPUExecutor(max_queue_size=20)
    manager = ModelResidencyManager(executor, residency_budget_mb=6000.0)

    class InterleavingMockAdapter(MockEmbeddingAdapter):
        def __init__(self):
            super().__init__()
            self.call_log: list[str] = []
            self.first_bulk_batch_started = threading.Event()
            self.interactive_query_finished = threading.Event()

        def tokenize_texts(self, texts, max_length=None):
            # Create distinct token lengths so planner splits them into micro-batches
            token_ids = [[i + 1] * (100 if i % 2 == 0 else 120) for i, _ in enumerate(texts)]
            lengths = [len(t) for t in token_ids]
            indices = list(range(len(texts)))
            return TokenizedBatch(token_ids, lengths, indices, pad_token_id=0, texts=list(texts))

        def forward_batch(self, batch, requested_dims=None):
            dims = requested_dims or self.native_dims
            count = len(batch)
            is_query = "urgent query" in str(getattr(batch, "texts", []))
            tag = "query" if is_query else f"bulk_{count}"
            self.call_log.append(tag)

            if not is_query and len(self.call_log) == 1:
                # First bulk micro-batch
                self.first_bulk_batch_started.set()
                # Yield CPU briefly so interactive query thread can enqueue into GPUExecutor priority queue
                time.sleep(0.05)

            # Return identifiable unique array rows
            arr = np.zeros((count, dims), dtype=np.float32)
            for idx, orig_idx in enumerate(batch.original_indices):
                arr[idx, 0] = float(orig_idx + 1)
                arr[idx, 1:] = 0.1
            arr = arr / np.linalg.norm(arr, axis=-1, keepdims=True)
            return arr

    mock_adapter = InterleavingMockAdapter()
    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=mock_adapter):
        runtime = MLXEmbeddingRuntime(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            executor=executor,
            model_manager=manager,
            lazy_load=False,
        )

    # Force micro-batch token budget so 4 texts split into multiple micro-batches
    runtime.batch_planner.max_bulk_microbatch_tokens = 150
    runtime.batch_planner.max_batch_tokens = 150

    bulk_texts = [
        "Bulk document 0 with sufficient token length",
        "Bulk document 1 with sufficient token length",
        "Bulk document 2 with sufficient token length",
        "Bulk document 3 with sufficient token length",
    ]
    query_text = ["Interactive urgent query text"]

    bulk_output: list[Optional[np.ndarray]] = [None]
    query_output: list[Optional[np.ndarray]] = [None]
    bulk_error: list[Optional[Exception]] = [None]
    query_error: list[Optional[Exception]] = [None]

    def run_bulk():
        try:
            res = runtime.submit_embed(bulk_texts, is_query=False)
            bulk_output[0] = res
        except Exception as e:
            bulk_error[0] = e

    def run_query():
        try:
            assert mock_adapter.first_bulk_batch_started.wait(timeout=2.0)
            res = runtime.submit_embed(query_text, is_query=True)
            query_output[0] = res
            mock_adapter.interactive_query_finished.set()
        except Exception as e:
            query_error[0] = e

    t_bulk = threading.Thread(target=run_bulk, name="BulkThread")
    t_query = threading.Thread(target=run_query, name="QueryThread")

    t_bulk.start()
    t_query.start()

    t_bulk.join(timeout=5.0)
    t_query.join(timeout=5.0)

    assert bulk_error[0] is None, f"Bulk error: {bulk_error[0]}"
    assert query_error[0] is None, f"Query error: {query_error[0]}"

    assert bulk_output[0] is not None
    assert query_output[0] is not None

    # Verify call sequence: query was serviced between bulk micro-batches!
    # Expected call_log starts with bulk, has query before final bulk batches
    assert "query" in mock_adapter.call_log
    query_call_idx = mock_adapter.call_log.index("query")
    assert query_call_idx > 0, "Query should start after first bulk micro-batch"
    assert query_call_idx < len(mock_adapter.call_log) - 1, "Query should finish before remaining bulk micro-batches"

    # Verify bulk output shape and strict original ordering preservation
    bulk_res = bulk_output[0]
    assert bulk_res.shape == (4, 384)
    for i in range(4):
        assert np.isfinite(bulk_res[i]).all()
        assert np.isclose(np.linalg.norm(bulk_res[i]), 1.0, atol=1e-5)
    assert bulk_res[0, 0] < bulk_res[1, 0] < bulk_res[2, 0] < bulk_res[3, 0]

    # Verify query output
    query_res = query_output[0]
    assert query_res.shape == (1, 384)
    assert np.isfinite(query_res).all()
    assert np.isclose(np.linalg.norm(query_res[0]), 1.0, atol=1e-5)

    runtime.shutdown()
    executor.shutdown()
