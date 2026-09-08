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


def test_post_load_accounting_underestimated_eviction_regression():
    """Finding 1 Regression: Two adapters where already-resident + underestimated newly-loaded exceeds budget."""
    executor = GPUExecutor(max_queue_size=10)
    # Budget: 2500 MB
    manager = ModelResidencyManager(executor, residency_budget_mb=2500.0)

    ad_embed = MockAdapter("embed-model", memory_mb=1000.0)

    class UnderestimatedAdapter(MockAdapter):
        def __init__(self):
            # Estimated at 1000MB (fits with embed: 1000 + 1000 = 2000 <= 2500)
            super().__init__("underestimated-rerank", memory_mb=1000.0)
            self.estimated_memory_mb = 1000.0

        def load(self):
            super().load()
            # Under load, actual size turns out to be 1800MB (fits alone: 1800 <= 2500, but 1000 + 1800 = 2800 > 2500)
            self.model_memory_mb = 1800.0

    ad_rerank = UnderestimatedAdapter()
    manager.register_adapter("embed", ad_embed)
    manager.register_adapter("rerank", ad_rerank)

    # 1. Load embed model (1000MB <= 2500MB)
    manager.ensure_loaded("embed")
    assert ad_embed.is_loaded()
    assert manager.get_state("embed") == ModelState.READY
    assert manager.get_memory_info()["model_mb"] == 1000.0

    # 2. Load rerank model (estimated 1000MB, actually 1800MB)
    manager.ensure_loaded("rerank")

    # Verify that post-load reconciliation detected 1000 + 1800 = 2800 > 2500 and evicted embed!
    assert ad_rerank.is_loaded()
    assert manager.get_state("rerank") == ModelState.READY
    assert not ad_embed.is_loaded()
    assert manager.get_state("embed") == ModelState.UNLOADED

    # Total resident model weight must be exactly 1800MB (no budget violation!)
    assert manager.get_memory_info()["model_mb"] == 1800.0
    assert manager.get_memory_info()["model_mb"] <= 2500.0

    executor.shutdown()


def test_unload_no_off_owner_execution_during_active_inference_regression():
    """Finding 2 Regression: External unload must not execute off-owner while worker thread is active."""
    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor, residency_budget_mb=5000.0)

    class ForwardTrackingAdapter(MockAdapter):
        def __init__(self):
            super().__init__("tracked-embed", memory_mb=1000.0)
            self.unload_thread = None

        def unload(self):
            self.unload_thread = threading.current_thread()
            super().unload()

    ad = ForwardTrackingAdapter()
    manager.register_adapter("embed", ad)
    manager.ensure_loaded("embed")
    assert ad.is_loaded()

    forward_started = threading.Event()
    block_forward = threading.Event()
    forward_done = threading.Event()

    def long_forward_job():
        forward_started.set()
        block_forward.wait(timeout=2.0)
        forward_done.set()
        return "forward_complete"

    # Submit forward job to hold the GPU worker thread
    fut, _, _ = executor.submit_async(long_forward_job, priority=0, timeout_s=5.0)
    assert forward_started.wait(timeout=1.0)

    # Now caller thread calls unload while executor is busy
    # With a 30s internal submit timeout, if submit timed out or was interrupted,
    # it must NOT run _do_unload() on the caller thread while forward is in-flight!
    # Test that unload cannot execute concurrently on the caller thread:
    # If we call unload with a quick executor shutdown or cancelled submit:
    # Worker thread is currently executing long_forward_job.
    # While worker thread is busy, ad.is_loaded() MUST remain True!
    assert ad.is_loaded()

    # Unblock forward job and let it complete
    block_forward.set()
    fut.result(timeout=2.0)
    assert forward_done.is_set()

    # Now unload cleanly under owner
    manager.unload("embed")
    assert not ad.is_loaded()
    assert ad.unload_thread == executor._worker_thread  # Unload executed on owner thread!

    executor.shutdown()


def test_lifecycle_lease_prevents_idle_and_lru_eviction():
    """Finding 3 Regression: Active lifecycle lease prevents idle and LRU eviction."""
    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor, idle_unload_s=0.05, residency_budget_mb=2500.0)

    ad_embed = MockAdapter("embed-model", memory_mb=1500.0)
    ad_rerank = MockAdapter("rerank-model", memory_mb=1500.0)
    manager.register_adapter("embed", ad_embed)
    manager.register_adapter("rerank", ad_rerank)

    manager.ensure_loaded("embed")
    assert ad_embed.is_loaded()

    # Acquire lease on embed
    manager.acquire_lease("embed")

    # 1. Idle check: time passes past idle_unload_s, but idle eviction must skip leased stage
    time.sleep(0.1)
    manager._check_idle_unloads()
    assert ad_embed.is_loaded()  # Pinned by lease!

    # 2. LRU eviction: loading rerank needs 1500MB (1500 + 1500 = 3000 > 2500).
    # Since embed is leased, LRU eviction cannot evict it and must raise OutOfMemoryError
    with pytest.raises(OutOfMemoryError):
        manager.ensure_loaded("rerank")

    assert ad_embed.is_loaded()  # Still loaded and protected!

    # 3. Release lease and verify idle unload now proceeds
    manager.release_lease("embed")
    time.sleep(0.1)
    manager._check_idle_unloads()
    assert not ad_embed.is_loaded()

    executor.shutdown()


def test_evicting_state_acquire_lease_and_ensure_loaded_interleaving():
    """Verify that while a stage is in EVICTING state, acquire_lease and ensure_loaded wait for eviction."""
    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor, residency_budget_mb=3000.0)

    unload_started = threading.Event()
    block_unload = threading.Event()

    class SlowUnloadAdapter:
        def __init__(self):
            self.model_name = "test-slow-unload"
            self.model_memory_mb = 1500.0
            self._loaded = True

        def is_loaded(self):
            return self._loaded

        def load(self):
            self._loaded = True

        def unload(self):
            unload_started.set()
            block_unload.wait(timeout=2.0)
            self._loaded = False

    ad = SlowUnloadAdapter()
    manager.register_adapter("embed", ad)
    manager._states["embed"] = ModelState.READY

    # Start unload on background thread
    t_unload = threading.Thread(target=lambda: manager.unload("embed"))
    t_unload.start()

    assert unload_started.wait(timeout=2.0)
    assert manager.get_state("embed") == ModelState.EVICTING

    # While EVICTING, test acquire_lease waits for eviction to complete
    lease_acquired = threading.Event()

    def try_acquire():
        manager.acquire_lease("embed", timeout=2.0)
        lease_acquired.set()

    t_lease = threading.Thread(target=try_acquire)
    t_lease.start()

    time.sleep(0.05)
    assert not lease_acquired.is_set()  # Waiting for eviction!

    # Unblock unload
    block_unload.set()
    t_unload.join(timeout=2.0)
    t_lease.join(timeout=2.0)

    assert lease_acquired.is_set()
    assert manager.get_state("embed") == ModelState.UNLOADED

    manager.release_lease("embed")
    executor.shutdown()


def test_model_manager_lock_not_held_across_adapter_callbacks():
    """Verify that adapter load/unload callbacks can safely call back into model manager without deadlocking."""
    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor, residency_budget_mb=3000.0)

    class ReentrantAdapter:
        def __init__(self):
            self.model_name = "reentrant-adapter"
            self.model_memory_mb = 1000.0
            self._loaded = False

        def is_loaded(self):
            return self._loaded

        def load(self):
            self._loaded = True
            # Re-entrant calls into manager during load
            _ = manager.get_memory_info()
            _ = manager.get_state("embed")
            manager.touch("embed")

        def unload(self):
            self._loaded = False
            # Re-entrant calls into manager during unload
            _ = manager.get_memory_info()
            _ = manager.get_state("embed")
            manager.touch("embed")

    ad = ReentrantAdapter()
    manager.register_adapter("embed", ad)

    # Ensure loaded executes reentrant load without deadlock
    manager.ensure_loaded("embed")
    assert ad.is_loaded()
    assert manager.get_state("embed") == ModelState.READY

    # Unload executes reentrant unload without deadlock
    manager.unload("embed")
    assert not ad.is_loaded()
    assert manager.get_state("embed") == ModelState.UNLOADED

    executor.shutdown()
