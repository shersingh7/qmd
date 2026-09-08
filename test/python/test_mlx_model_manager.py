"""
test_mlx_model_manager.py — Deterministic tests for shared residency budgets (8/16/24/32/64GB),
single-flight admission, evict-before-load LRU policy, failure recovery, and memory reporting.
"""

import threading
import time
import pytest
from scripts.qmd_mlx.executor import GPUExecutor
from scripts.qmd_mlx.model_manager import (
    ModelResidencyManager,
    ModelState,
    calculate_residency_budget_mb,
)
from scripts.qmd_mlx.protocol import ModelUnavailableError, OutOfMemoryError


class MockAdapter:
    def __init__(self, name: str, memory_mb: float = 1000.0, fail_load: bool = False):
        self.model_name = name
        self.model_memory_mb = memory_mb
        self.fail_load = fail_load
        self._loaded = False
        self.load_count = 0
        self.unload_count = 0
        self.lock = threading.Lock()

    def is_loaded(self) -> bool:
        with self.lock:
            return self._loaded

    def load(self):
        time.sleep(0.05)  # Simulate non-trivial load duration
        with self.lock:
            if self.fail_load:
                raise RuntimeError(f"Simulated load failure for {self.model_name}")
            self._loaded = True
            self.load_count += 1

    def unload(self):
        with self.lock:
            self._loaded = False
            self.unload_count += 1


def test_calculate_residency_budget_deterministic_profiles():
    """Verify exact conservative resident weight budgets across RAM tiers."""
    assert calculate_residency_budget_mb(8.0) == 2048.0   # 8 GB Mac
    assert calculate_residency_budget_mb(16.0) == 6144.0  # 16 GB Mac
    assert calculate_residency_budget_mb(24.0) == 10240.0 # 24 GB Mac
    assert calculate_residency_budget_mb(32.0) == 16384.0 # 32 GB Mac
    assert calculate_residency_budget_mb(64.0) == 36864.0 # 64 GB Mac
    assert calculate_residency_budget_mb(128.0) == 128.0 * 0.6 * 1024.0


def test_model_manager_single_flight_concurrent_loads():
    """Verify that multiple concurrent cold calls serialize and perform load only ONCE."""
    executor = GPUExecutor(max_queue_size=20)
    manager = ModelResidencyManager(executor, residency_budget_mb=8000.0)

    adapter = MockAdapter("test-embed", memory_mb=1500.0)
    manager.register_adapter("embed", adapter)

    assert manager.get_state("embed") == ModelState.UNLOADED

    # Spawn 5 threads concurrently calling ensure_loaded("embed")
    errors = []
    threads = []

    def caller():
        try:
            manager.ensure_loaded("embed", timeout_s=5.0)
        except Exception as e:
            errors.append(e)

    for _ in range(5):
        t = threading.Thread(target=caller)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert not errors
    assert adapter.load_count == 1  # Exactly one load performed!
    assert manager.get_state("embed") == ModelState.READY
    assert adapter.is_loaded()

    executor.shutdown()


def test_model_manager_evict_before_load_lru():
    """Verify that loading models exceeding residency budget evicts least-recently-used models."""
    executor = GPUExecutor(max_queue_size=20)
    # Budget: 3000 MB
    manager = ModelResidencyManager(executor, residency_budget_mb=3000.0)

    ad_embed = MockAdapter("embed-model", memory_mb=1800.0)
    ad_rerank = MockAdapter("rerank-model", memory_mb=1800.0)

    manager.register_adapter("embed", ad_embed)
    manager.register_adapter("rerank", ad_rerank)

    # 1. Load embed model (takes 1800MB <= 3000MB)
    manager.ensure_loaded("embed")
    assert ad_embed.is_loaded()
    assert manager.get_state("embed") == ModelState.READY

    # Touch embed to simulate active usage
    manager.touch("embed")
    time.sleep(0.01)

    # 2. Load rerank model (needs 1800MB, 1800 + 1800 = 3600MB > 3000MB)
    # Evict-before-load must unload "embed" before loading "rerank"
    manager.ensure_loaded("rerank")

    assert ad_rerank.is_loaded()
    assert manager.get_state("rerank") == ModelState.READY

    # Embed model must have been evicted
    assert not ad_embed.is_loaded()
    assert manager.get_state("embed") == ModelState.UNLOADED
    assert ad_embed.unload_count == 1

    # Total resident model weight must be within budget
    mem_info = manager.get_memory_info()
    assert mem_info["model_mb"] <= 3000.0

    executor.shutdown()


def test_model_manager_failure_recovery():
    """Verify that if a model load fails, subsequent requests can retry and recover."""
    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor, residency_budget_mb=4000.0)

    ad_faulty = MockAdapter("faulty-model", memory_mb=1000.0, fail_load=True)
    manager.register_adapter("embed", ad_faulty)

    with pytest.raises(ModelUnavailableError, match="Simulated load failure"):
        manager.ensure_loaded("embed")

    assert manager.get_state("embed") == ModelState.FAILED
    assert "Simulated load failure" in (manager.get_error("embed") or "")

    # Fix the adapter and retry
    ad_faulty.fail_load = False
    manager.ensure_loaded("embed")

    assert manager.get_state("embed") == ModelState.READY
    assert manager.get_error("embed") is None
    assert ad_faulty.is_loaded()

    executor.shutdown()


def test_model_manager_idle_unload_and_reload():
    """Verify that idle expiry unloads model and subsequent ensure_loaded reloads it."""
    executor = GPUExecutor(max_queue_size=10)
    # Set idle unload to 0.1 seconds for fast test
    manager = ModelResidencyManager(executor, idle_unload_s=0.1, residency_budget_mb=4000.0)

    ad = MockAdapter("test-model", memory_mb=1000.0)
    manager.register_adapter("embed", ad)

    manager.ensure_loaded("embed")
    assert ad.is_loaded()

    # Wait for idle unload to trigger
    time.sleep(0.25)
    manager._check_idle_unloads()

    assert not ad.is_loaded()
    assert manager.get_state("embed") == ModelState.UNLOADED

    # Subsequent request reloads cleanly
    manager.ensure_loaded("embed")
    assert ad.is_loaded()
    assert manager.get_state("embed") == ModelState.READY

    executor.shutdown()


def test_model_manager_memory_reporting_honest():
    """Verify get_memory_info includes residency budget, model weight, and activation disclaimer."""
    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor, residency_budget_mb=6144.0)

    ad = MockAdapter("test-embed", memory_mb=1234.5)
    manager.register_adapter("embed", ad)
    manager.ensure_loaded("embed")

    info = manager.get_memory_info()
    assert info["model_mb"] == 1234.5
    assert info["residency_budget_mb"] == 6144.0
    assert "not total activation memory" in info["notes"].lower()

    executor.shutdown()


def test_model_manager_insufficient_headroom_before_load_rejection():
    """Blocker 1: Verify before-load rejection when model exceeds residency budget."""
    executor = GPUExecutor(max_queue_size=10)
    # Budget: 100 MB; Adapter: 200 MB
    manager = ModelResidencyManager(executor, residency_budget_mb=100.0)

    ad = MockAdapter("too-large-model", memory_mb=200.0)
    manager.register_adapter("embed", ad)

    with pytest.raises(OutOfMemoryError, match="exceeds total residency budget"):
        manager.ensure_loaded("embed")

    assert not ad.is_loaded()
    assert manager.get_state("embed") == ModelState.FAILED
    info = manager.get_memory_info()
    assert info["model_mb"] == 0.0

    executor.shutdown()


def test_model_manager_insufficient_headroom_after_eviction_rejection():
    """Verify that if all evictions still leave insufficient headroom, OutOfMemoryError is raised."""
    executor = GPUExecutor(max_queue_size=10)
    # Budget: 300 MB
    manager = ModelResidencyManager(executor, residency_budget_mb=300.0)

    ad_embed = MockAdapter("embed-model", memory_mb=150.0)
    ad_rerank = MockAdapter("rerank-model", memory_mb=200.0)
    manager.register_adapter("embed", ad_embed)
    manager.register_adapter("rerank", ad_rerank)

    # 1. Load embed model (150MB <= 300MB)
    manager.ensure_loaded("embed")
    assert ad_embed.is_loaded()

    # 2. Try loading a 350MB model when budget is 300MB
    ad_large = MockAdapter("large-model", memory_mb=350.0)
    manager.register_adapter("large", ad_large)

    with pytest.raises(OutOfMemoryError):
        manager.ensure_loaded("large")

    assert not ad_large.is_loaded()
    assert manager.get_state("large") == ModelState.FAILED

    executor.shutdown()


def test_model_manager_post_load_actual_size_reconciliation_and_cleanup():
    """Blocker 1: Verify post-load actual-size reconciliation, cleanup, and typed OutOfMemoryError."""
    executor = GPUExecutor(max_queue_size=10)
    # Budget: 150 MB
    manager = ModelResidencyManager(executor, residency_budget_mb=150.0)

    class ExpandingAdapter(MockAdapter):
        def __init__(self):
            super().__init__("expanding-model", memory_mb=50.0)

        def load(self):
            super().load()
            # Under load, actual memory required is 250MB (exceeds budget 150MB)
            self.model_memory_mb = 250.0

    ad = ExpandingAdapter()
    manager.register_adapter("embed", ad)

    with pytest.raises(OutOfMemoryError, match="exceeds.*budget"):
        manager.ensure_loaded("embed")

    # Adapter must have been unloaded and cleaned up
    assert not ad.is_loaded()
    assert manager.get_state("embed") == ModelState.FAILED
    assert manager.get_memory_info()["model_mb"] == 0.0

    executor.shutdown()


def test_model_manager_owner_thread_ensure_loaded_no_deadlock():
    """Blocker 2: Verify executor owner thread calling ensure_loaded runs inline without self-wait."""
    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor, residency_budget_mb=2000.0)

    ad = MockAdapter("embed-model", memory_mb=500.0)
    manager.register_adapter("embed", ad)

    # Directly submit a lambda to executor that calls ensure_loaded on the owner thread
    t0 = time.monotonic()
    executor.submit(lambda: manager.ensure_loaded("embed", deadline=time.monotonic() + 0.5), timeout_s=1.0)
    elapsed = time.monotonic() - t0

    # Must complete fast without deadlock or timeout
    assert elapsed < 0.3
    assert ad.is_loaded()
    assert manager.get_state("embed") == ModelState.READY

    executor.shutdown()


def test_model_manager_owner_thread_encounters_pending_external_load():
    """Blocker 2: Verify owner thread encountering pending external load single-flight state resolves cleanly."""
    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor, residency_budget_mb=2000.0)

    ad = MockAdapter("embed-model", memory_mb=500.0)
    manager.register_adapter("embed", ad)

    # External thread sets loading state
    with manager._lock:
        manager._states["embed"] = ModelState.LOADING
        ext_event = threading.Event()
        manager._loading_events["embed"] = ext_event

    # Owner thread calls ensure_loaded — must take over and complete load inline
    executor.submit(lambda: manager.ensure_loaded("embed", deadline=time.monotonic() + 0.5), timeout_s=1.0)

    assert ad.is_loaded()
    assert manager.get_state("embed") == ModelState.READY
    assert ext_event.is_set()  # External waiter was notified

    executor.shutdown()

