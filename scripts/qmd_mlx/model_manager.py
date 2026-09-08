"""
model_manager.py — Model Residency Manager for MLX Stages (Embed, Rerank, Generate)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

from .executor import GPUExecutor
from .protocol import (
    DeadlineExceededError,
    MLXServerError,
    ModelUnavailableError,
    OutOfMemoryError,
    RequestCancelledError,
    UnsupportedModelError,
)

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


class ModelState:
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


def calculate_residency_budget_mb(ram_gb: float) -> float:
    """
    Calculates conservative shared resident weight budget (in MB) based on physical RAM.
    Resident weight budgets constrain resident model parameters in unified memory,
    leaving headroom for activation buffers and OS allocations.
    """
    if ram_gb <= 8.5:
        return 2048.0   # 2 GB for 8GB Mac
    elif ram_gb <= 16.5:
        return 6144.0   # 6 GB for 16GB Mac
    elif ram_gb <= 24.5:
        return 10240.0  # 10 GB for 24GB Mac
    elif ram_gb <= 36.5:
        return 16384.0  # 16 GB for 32GB/36GB Mac
    elif ram_gb <= 64.5:
        return 36864.0  # 36 GB for 64GB Mac
    else:
        return max(36864.0, ram_gb * 0.6 * 1024.0)


def _detect_system_ram_gb() -> float:
    try:
        if sys.platform == "darwin":
            import subprocess
            res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                return int(res.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / (1024 ** 3)
    except Exception:
        return 16.0


class ModelResidencyManager:
    """
    Manages loading, evaluation, single-flight admission, and LRU/idle eviction
    for all MLX model adapters behind the single GPUExecutor execution owner.
    """

    def __init__(
        self,
        executor: GPUExecutor,
        idle_unload_s: Optional[float] = None,
        residency_budget_mb: Optional[float] = None,
        system_ram_gb_fn: Optional[Callable[[], float]] = None,
    ):
        self.executor = executor
        if idle_unload_s is not None:
            self.idle_unload_s = float(idle_unload_s)
        else:
            self.idle_unload_s = float(os.getenv("MLX_IDLE_UNLOAD_S", "300.0"))

        self._ram_fn = system_ram_gb_fn or _detect_system_ram_gb
        if residency_budget_mb is not None:
            self.residency_budget_mb = float(residency_budget_mb)
        elif "MLX_RESIDENCY_BUDGET_MB" in os.environ:
            self.residency_budget_mb = float(os.environ["MLX_RESIDENCY_BUDGET_MB"])
        else:
            self.residency_budget_mb = calculate_residency_budget_mb(self._ram_fn())

        # Registered adapters: "embed", "rerank", "generate"
        self._adapters: Dict[str, Any] = {}
        self._states: Dict[str, str] = {}
        self._last_active: Dict[str, float] = {}
        self._errors: Dict[str, Optional[str]] = {}

        # Single-flight synchronization lock
        self._lock = threading.Lock()
        self._loading_events: Dict[str, threading.Event] = {}

        self.peak_memory_mb: float = 0.0

        # Register idle check with executor
        self.executor.register_idle_callback(self._check_idle_unloads)

    def register_adapter(self, stage: str, adapter: Any):
        """Registers an adapter instance (embed, rerank, generate)."""
        with self._lock:
            self._adapters[stage] = adapter
            self._states[stage] = ModelState.UNLOADED
            self._last_active[stage] = time.monotonic()
            self._errors[stage] = None

    def get_adapter(self, stage: str) -> Optional[Any]:
        with self._lock:
            return self._adapters.get(stage)

    def get_state(self, stage: str) -> str:
        with self._lock:
            return self._states.get(stage, ModelState.UNLOADED)

    def get_error(self, stage: str) -> Optional[str]:
        with self._lock:
            return self._errors.get(stage)

    def touch(self, stage: str):
        with self._lock:
            self._last_active[stage] = time.monotonic()

    def _get_active_memory_mb(self) -> float:
        try:
            if _MLX_AVAILABLE:
                if hasattr(mx, "get_active_memory"):
                    return mx.get_active_memory() / (1024 * 1024)
                if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                    return mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def _get_peak_memory_mb(self) -> float:
        try:
            if _MLX_AVAILABLE:
                if hasattr(mx, "get_peak_memory"):
                    return mx.get_peak_memory() / (1024 * 1024)
                if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
                    return mx.metal.get_peak_memory() / (1024 * 1024)
        except Exception:
            pass
        return self.peak_memory_mb

    def update_peak_memory(self):
        cur = self._get_peak_memory_mb()
        if cur > self.peak_memory_mb:
            self.peak_memory_mb = cur

    def _get_resident_model_mb(self) -> float:
        total = 0.0
        for st, ad in self._adapters.items():
            if self._states.get(st) == ModelState.READY and getattr(ad, "is_loaded", lambda: False)():
                total += getattr(ad, "model_memory_mb", 0.0) or getattr(ad, "estimated_memory_mb", 1000.0)
        return total

    def _estimate_adapter_mb(self, adapter: Any) -> float:
        if hasattr(adapter, "model_memory_mb") and adapter.model_memory_mb > 0:
            return float(adapter.model_memory_mb)
        if hasattr(adapter, "estimated_memory_mb") and adapter.estimated_memory_mb > 0:
            return float(adapter.estimated_memory_mb)
        # Default estimates based on model attributes if available
        name = getattr(adapter, "model_name", "").lower()
        if "4b" in name or "rerank" in name:
            return 4500.0
        if "1.7b" in name or "generate" in name:
            return 2500.0
        if "nomic" in name or "bert" in name:
            return 1200.0
        return 800.0

    def _is_owner_thread(self) -> bool:
        return hasattr(self.executor, "is_owner_thread") and self.executor.is_owner_thread()

    def _evict_for_budget_under_owner(self, stage: str, needed_mb: float):
        """
        Runs on GPU executor thread: evicts least-recently-used ready models
        (other than `stage`) until `needed_mb` fits within `residency_budget_mb`.
        Raises OutOfMemoryError if insufficient headroom can be achieved.
        """
        # If needed_mb alone exceeds residency budget, reject before attempting evictions
        if needed_mb > self.residency_budget_mb:
            raise OutOfMemoryError(
                f"Model '{stage}' memory requirement ({needed_mb:.1f}MB) exceeds total residency budget ({self.residency_budget_mb:.1f}MB)"
            )

        current = self._get_resident_model_mb()
        if current + needed_mb <= self.residency_budget_mb:
            return

        # Find candidates for eviction (must be READY and not currently requested stage)
        candidates = []
        for st, ad in self._adapters.items():
            if st != stage and self._states.get(st) == ModelState.READY and getattr(ad, "is_loaded", lambda: False)():
                candidates.append((self._last_active.get(st, 0.0), st, ad))

        # Sort by oldest last_active first (LRU)
        candidates.sort(key=lambda c: c[0])

        for _, cand_stage, cand_adapter in candidates:
            if current + needed_mb <= self.residency_budget_mb:
                break
            try:
                ad_mb = getattr(cand_adapter, "model_memory_mb", 0.0) or self._estimate_adapter_mb(cand_adapter)
                cand_adapter.unload()
                with self._lock:
                    self._states[cand_stage] = ModelState.UNLOADED
                if _MLX_AVAILABLE and hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                current = max(0.0, current - ad_mb)
                print(f"[mlx-manager] Evicted LRU stage '{cand_stage}' to free memory for '{stage}'")
            except Exception as e:
                print(f"[mlx-manager] Failed to evict '{cand_stage}': {e}", file=sys.stderr)

        # Re-verify resident headroom after eviction
        current = self._get_resident_model_mb()
        if current + needed_mb > self.residency_budget_mb:
            raise OutOfMemoryError(
                f"Insufficient residency headroom to load '{stage}' (needed: {needed_mb:.1f}MB, resident: {current:.1f}MB, budget: {self.residency_budget_mb:.1f}MB)"
            )

    def _do_load_internal(
        self,
        stage: str,
        adapter: Any,
        dl: float,
        cancel_event: Optional[threading.Event],
        event: threading.Event,
    ):
        """
        Internal load execution running strictly on the GPU execution owner thread.
        Performs before-load eviction/headroom check, loading, and post-load actual size reconciliation.
        """
        try:
            # 1. Check if already loaded
            if adapter.is_loaded() and self._states.get(stage) == ModelState.READY:
                return

            # 2. Check cancellation / deadline
            if cancel_event and cancel_event.is_set():
                raise RequestCancelledError(f"Load of '{stage}' cancelled")
            if time.monotonic() > dl:
                raise DeadlineExceededError(f"Load of '{stage}' exceeded deadline")

            # 3. Before-load headroom check & evict-before-load
            needed_mb = self._estimate_adapter_mb(adapter)
            self._evict_for_budget_under_owner(stage, needed_mb)

            # 4. Perform actual model loading
            adapter.load()

            # 5. Post-load actual size reconciliation & cleanup
            actual_mb = getattr(adapter, "model_memory_mb", 0.0) or self._estimate_adapter_mb(adapter)
            if actual_mb > self.residency_budget_mb:
                try:
                    adapter.unload()
                    if _MLX_AVAILABLE and hasattr(mx, "clear_cache"):
                        mx.clear_cache()
                except Exception:
                    pass
                raise OutOfMemoryError(
                    f"Loaded model '{stage}' actual size ({actual_mb:.1f}MB) exceeds total residency budget ({self.residency_budget_mb:.1f}MB)"
                )

            current_resident = self._get_resident_model_mb()
            if current_resident > self.residency_budget_mb:
                # Evict other models if actual size turned out larger than estimate
                self._evict_for_budget_under_owner(stage, 0.0)
                current_resident = self._get_resident_model_mb()
                if current_resident > self.residency_budget_mb:
                    try:
                        adapter.unload()
                        if _MLX_AVAILABLE and hasattr(mx, "clear_cache"):
                            mx.clear_cache()
                    except Exception:
                        pass
                    raise OutOfMemoryError(
                        f"Resident memory ({current_resident:.1f}MB) exceeds residency budget ({self.residency_budget_mb:.1f}MB) after loading '{stage}'"
                    )

            with self._lock:
                self._states[stage] = ModelState.READY
                self._errors[stage] = None
                self._last_active[stage] = time.monotonic()
            self.update_peak_memory()
        except Exception as exc:
            with self._lock:
                self._states[stage] = ModelState.FAILED
                self._errors[stage] = str(exc)
            if isinstance(exc, MLXServerError):
                raise
            raise ModelUnavailableError(f"Failed to load '{stage}' model: {exc}")
        finally:
            with self._lock:
                self._loading_events.pop(stage, None)
                event.set()

    def ensure_loaded(
        self,
        stage: str,
        timeout_s: float = 120.0,
        deadline: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        """
        Ensures the requested stage's model is resident in memory.
        Owner-thread-aware serialized lifecycle:
        - If called from the executor owner thread, executes synchronously/inline without self-wait.
        - If called from an external thread, serializes through the executor with single-flight waiting.
        """
        now = time.monotonic()
        dl = deadline if deadline is not None else (now + timeout_s)
        is_owner = self._is_owner_thread()

        with self._lock:
            adapter = self._adapters.get(stage)
            if adapter is None:
                raise ModelUnavailableError(f"No adapter registered for stage '{stage}'")

            if self._states.get(stage) == ModelState.READY and adapter.is_loaded():
                self._last_active[stage] = time.monotonic()
                return

            if is_owner:
                # Owner thread executing: NEVER self-wait or submit back to self.
                # If an external thread marked LOADING, owner thread takes over execution inline.
                event = self._loading_events.get(stage)
                if event is None:
                    event = threading.Event()
                    self._loading_events[stage] = event
                self._states[stage] = ModelState.LOADING
                self._errors[stage] = None
                is_designated_loader = True
            else:
                if self._states.get(stage) == ModelState.LOADING and stage in self._loading_events:
                    # Another thread is already loading this model — wait on existing load
                    event = self._loading_events[stage]
                    is_designated_loader = False
                else:
                    self._states[stage] = ModelState.LOADING
                    self._errors[stage] = None
                    event = threading.Event()
                    self._loading_events[stage] = event
                    is_designated_loader = True

        if is_owner:
            # Owner thread runs inline directly
            self._do_load_internal(stage, adapter, dl, cancel_event, event)
            return

        if not is_designated_loader:
            rem = max(0.01, dl - time.monotonic())
            if not event.wait(timeout=rem):
                raise DeadlineExceededError(f"Timed out waiting for concurrent load of '{stage}'")
            with self._lock:
                if self._states.get(stage) == ModelState.READY and adapter.is_loaded():
                    self._last_active[stage] = time.monotonic()
                    return
                err = self._errors.get(stage) or "Concurrent model loading failed"
                if "insufficient" in err.lower() or "budget" in err.lower() or "exceeds" in err.lower():
                    raise OutOfMemoryError(f"Failed to load '{stage}' model: {err}")
                raise ModelUnavailableError(f"Failed to load '{stage}' model: {err}")

        # External designated loader thread: submits to executor
        def _job_fn():
            self._do_load_internal(stage, adapter, dl, cancel_event, event)

        rem_timeout = max(0.01, dl - time.monotonic())
        try:
            self.executor.submit(
                _job_fn,
                priority=0,
                timeout_s=rem_timeout,
                cancel_event=cancel_event,
                description=f"Load {stage}",
            )
        except Exception:
            with self._lock:
                self._loading_events.pop(stage, None)
                event.set()
            raise

    def unload(self, stage: str):
        """Unloads a stage model from unified memory (runs on owner thread)."""
        with self._lock:
            adapter = self._adapters.get(stage)
            if adapter is None:
                return

        def _do_unload():
            try:
                adapter.unload()
                with self._lock:
                    self._states[stage] = ModelState.UNLOADED
                    self._errors[stage] = None
                if _MLX_AVAILABLE and hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                print(f"[mlx-manager] Unloaded stage '{stage}' (active: {self._get_active_memory_mb():.1f}MB)")
            except Exception as e:
                print(f"[mlx-manager] Error unloading '{stage}': {e}", file=sys.stderr)

        if self._is_owner_thread():
            _do_unload()
        elif self.executor.is_alive():
            try:
                self.executor.submit(_do_unload, priority=0, timeout_s=30.0, description=f"Unload {stage}")
            except Exception:
                _do_unload()
        else:
            _do_unload()

    def unload_all(self):
        """Unloads all registered stages safely under owner thread."""
        with self._lock:
            stages = list(self._adapters.keys())
        for stage in stages:
            self.unload(stage)

    def _check_idle_unloads(self):
        """Called on the executor thread when idle to evict expired models."""
        if self.idle_unload_s <= 0:
            return

        now = time.monotonic()
        with self._lock:
            stages = list(self._adapters.keys())

        for stage in stages:
            with self._lock:
                adapter = self._adapters.get(stage)
                state = self._states.get(stage)
                last_act = self._last_active.get(stage, now)

            if adapter and state == ModelState.READY and getattr(adapter, "is_loaded", lambda: False)():
                idle_time = now - last_act
                if idle_time >= self.idle_unload_s:
                    try:
                        adapter.unload()
                        with self._lock:
                            self._states[stage] = ModelState.UNLOADED
                        if _MLX_AVAILABLE and hasattr(mx, "clear_cache"):
                            mx.clear_cache()
                        print(
                            f"[mlx-manager] Stage '{stage}' idle for {idle_time:.0f}s — "
                            f"unloaded ({self._get_active_memory_mb():.0f}MB active Metal)"
                        )
                    except Exception as e:
                        print(f"[mlx-manager] Idle unload for '{stage}' failed: {e}", file=sys.stderr)

    def get_memory_info(self) -> dict[str, Any]:
        with self._lock:
            total_model_mb = self._get_resident_model_mb()
        return {
            "active_mb": round(self._get_active_memory_mb(), 1),
            "peak_mb": round(self._get_peak_memory_mb(), 1),
            "model_mb": round(total_model_mb, 1),
            "residency_budget_mb": round(self.residency_budget_mb, 1),
            "notes": "Resident weight budgets are not total activation memory guarantees.",
        }
