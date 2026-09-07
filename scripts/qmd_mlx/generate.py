"""
generate.py — MLX-native Query Expansion (Generation) Adapter

Wraps mlx-lm generation for QMD query expansion. Shares the same bounded
execution-owner discipline as the embedding and rerank adapters: all forward
passes run on a single dedicated worker thread, one model resident.
"""

import threading
import time
from concurrent.futures import Future
from queue import Queue
from typing import Any, Optional

try:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


class GenerateError(ValueError):
    """Raised when generation input validation or inference fails."""
    pass


class MLXGenerateAdapter:
    """
    Query-expansion generation adapter on Apple Silicon MLX.
    Deterministic-by-default (temp=0) short completions.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-1.7B-4bit",
        revision: Optional[str] = None,
        max_context: int = 4096,
        max_new_tokens: int = 600,
        lazy_load: bool = False,
    ):
        if not _MLX_AVAILABLE:
            raise GenerateError("MLX or mlx-lm is not installed.")

        self.model_name = model_name
        self.revision = revision
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens

        self.model: Any = None
        self.tokenizer: Any = None
        self.raw_hf_tokenizer: Any = None
        self.model_memory_mb: float = 0.0
        self.total_requests: int = 0
        self.total_tokens_generated: int = 0
        self.total_latency_ms: float = 0.0

        self._work_queue: Queue = Queue()
        self._ready_event = threading.Event()
        self._init_error: Optional[Exception] = None
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._ready_event.wait()
        if self._init_error is not None:
            raise self._init_error

        if not lazy_load:
            # Model was already loaded synchronously on the worker via
            # _worker_loop's _load_on_worker; nothing further needed.
            pass

    # ------------------------------------------------------------------ #
    # Execution-owner plumbing (same discipline as embedding runtime)
    # ------------------------------------------------------------------ #

    def _get_active_memory_mb(self) -> float:
        try:
            if hasattr(mx, "get_active_memory"):
                return mx.get_active_memory() / (1024 * 1024)
            if hasattr(mx.metal, "get_active_memory"):
                return mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def _load_on_worker(self):
        t0 = time.time()
        mem_before = self._get_active_memory_mb()
        try:
            model, tokenizer_wrap = mlx_lm.load(self.model_name, revision=self.revision)
            self.model = model
            self.tokenizer = tokenizer_wrap
            self.raw_hf_tokenizer = getattr(tokenizer_wrap, "_tokenizer", tokenizer_wrap)
        except Exception as exc:
            raise GenerateError(f"Failed to load generation model '{self.model_name}': {exc}")
        self.model_memory_mb = max(0.0, self._get_active_memory_mb() - mem_before)
        print(
            f"[mlx-generate] Loaded '{self.model_name}' ✓ ({self.model_memory_mb:.1f}MB Metal) "
            f"in {time.time() - t0:.2f}s"
        )

    def _worker_loop(self):
        try:
            self._load_on_worker()
        except Exception as exc:
            self._init_error = exc
            self._ready_event.set()
            return
        self._ready_event.set()

        while True:
            job = self._work_queue.get()
            if job is None:
                break
            prompt, max_tokens, temperature, response_future = job
            t0 = time.time()
            try:
                text, tokens_out = self._generate_sync(prompt, max_tokens, temperature)
                self.total_requests += 1
                self.total_tokens_generated += tokens_out
                self.total_latency_ms += (time.time() - t0) * 1000
                response_future.set_result(text)
            except Exception as exc:
                response_future.set_exception(exc)
            finally:
                self._work_queue.task_done()

    # ------------------------------------------------------------------ #
    # Sync inference (owner thread only)
    # ------------------------------------------------------------------ #

    def _generate_sync(self, prompt: str, max_tokens: int, temperature: float) -> tuple[str, int]:
        # Qwen3 thinking models: strip any literal thinking blocks from output
        # and remove the no-op thinking preamble when present.
        formatted_prompt = prompt
        chat_formatted = False
        if hasattr(self.raw_hf_tokenizer, "apply_chat_template"):
            try:
                formatted_prompt = self.raw_hf_tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                chat_formatted = True
            except Exception:
                chat_formatted = False
        if not chat_formatted:
            formatted_prompt = prompt

        # Truncate prompt tokens to fit max_context (keep tail: instructions
        # live at the end of QMD expansion prompts).
        toks = self.raw_hf_tokenizer.encode(formatted_prompt)
        if len(toks) > self.max_context - self.max_new_tokens:
            keep = self.max_context - self.max_new_tokens
            formatted_prompt = self.raw_hf_tokenizer.decode(toks[-keep:])

        sampler = make_sampler(temp=temperature)
        text = mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        text = text if isinstance(text, str) else str(text)

        # Strip leading thinking-block artifacts the 1.7B emits when the
        # chat template leaves the thinking channel open. Markers use explicit
        # escapes to stay intact in source.
        OPEN = "<think>"          # literal open tag
        CLOSE = "</think>"         # literal close tag
        NL2 = "\n\n"
        if text.startswith(NL2):
            text = text[len(NL2):]
        if text.startswith(OPEN):
            end = text.find(CLOSE)
            if end != -1:
                text = text[end + len(CLOSE):]
            else:
                text = text[len(OPEN):]
        text = text.strip()

        return text, max(0, len(text))

    # ------------------------------------------------------------------ #
    # Public API (thread-safe, submits to owner queue)
    # ------------------------------------------------------------------ #

    def submit_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise GenerateError("Prompt must be a non-empty string.")
        mt = max_tokens if max_tokens is not None else self.max_new_tokens
        if not isinstance(mt, int) or mt <= 0 or mt > 4096:
            raise GenerateError(f"max_tokens must be an integer in (0, 4096], got {mt}")

        future: Future = Future()
        self._work_queue.put((prompt, mt, temperature, future))
        return future.result(timeout=timeout)

    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        """Synchronous convenience wrapper."""
        return self.submit_generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def warmup(self):
        self.submit_generate("/no_think warmup", max_tokens=4)
        print("[mlx-generate] Warmup complete ✓")

    def get_descriptor(self) -> dict[str, Any]:
        return {
            "version": 1,
            "backend": "mlx",
            "kind": "generate",
            "model": self.model_name,
            "revision": self.revision or "",
            "maxContext": self.max_context,
            "maxNewTokens": self.max_new_tokens,
        }

    def shutdown(self):
        self._work_queue.put(None)
        self._worker_thread.join(timeout=5.0)