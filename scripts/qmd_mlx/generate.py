"""
generate.py — MLX-native Query Expansion (Generation) Adapter
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from .protocol import (
    DeadlineExceededError,
    InvalidInputError,
    ModelUnavailableError,
    RequestCancelledError,
)

try:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


class GenerateError(InvalidInputError):
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
        executor: Optional[Any] = None,
        model_manager: Optional[Any] = None,
    ):
        if not _MLX_AVAILABLE:
            raise ModelUnavailableError("MLX or mlx-lm is not installed.")

        self.model_name = model_name
        self.revision = revision
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens
        self.executor = executor
        self.model_manager = model_manager

        self.model: Any = None
        self.tokenizer: Any = None
        self.raw_hf_tokenizer: Any = None
        self.model_memory_mb: float = 0.0
        self.total_requests: int = 0
        self.total_tokens_generated: int = 0
        self.total_latency_ms: float = 0.0

        if not lazy_load and self.model_manager is None:
            self.load()

    def load_via_manager(self, timeout_s: float = 120.0):
        if self.model_manager is not None:
            self.model_manager.ensure_loaded("generate", timeout_s=timeout_s)
        else:
            self.load()

    def is_loaded(self) -> bool:
        return self.model is not None and self.raw_hf_tokenizer is not None

    def _get_active_memory_mb(self) -> float:
        try:
            if hasattr(mx, "get_active_memory"):
                return mx.get_active_memory() / (1024 * 1024)
            if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                return mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def load(self):
        if self.model is not None:
            return

        t0 = time.time()
        mem_before = self._get_active_memory_mb()
        try:
            model, tokenizer_wrap = mlx_lm.load(self.model_name, revision=self.revision)
            self.model = model
            self.tokenizer = tokenizer_wrap
            self.raw_hf_tokenizer = getattr(tokenizer_wrap, "_tokenizer", tokenizer_wrap)
        except Exception as exc:
            raise ModelUnavailableError(f"Failed to load generation model '{self.model_name}': {exc}")

        self.model_memory_mb = max(0.0, self._get_active_memory_mb() - mem_before)
        print(
            f"[mlx-generate] Loaded '{self.model_name}' ✓ ({self.model_memory_mb:.1f}MB Metal) "
            f"in {time.time() - t0:.2f}s"
        )

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        self.model_memory_mb = 0.0

    def _generate_sync(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        deadline: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> tuple[str, int]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise GenerateError("Prompt must be a non-empty string.")

        if cancel_event and cancel_event.is_set():
            raise RequestCancelledError("Generation request cancelled")
        if deadline is not None and time.monotonic() > deadline:
            raise DeadlineExceededError("Generation request deadline exceeded before start")

        if not self.is_loaded():
            if self.model_manager is not None:
                self.model_manager.ensure_loaded("generate", deadline=deadline, cancel_event=cancel_event)
            else:
                self.load()

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

        toks = self.raw_hf_tokenizer.encode(formatted_prompt)
        if len(toks) > self.max_context - self.max_new_tokens:
            raise GenerateError(
                f"Prompt length ({len(toks)} tokens) exceeds max context budget ({self.max_context - self.max_new_tokens} tokens)"
            )

        sampler = make_sampler(temp=temperature)
        pieces = []
        if hasattr(mlx_lm, "stream_generate"):
            for response in mlx_lm.stream_generate(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                if cancel_event and cancel_event.is_set():
                    raise RequestCancelledError("Generation request cancelled")
                if deadline is not None and time.monotonic() > deadline:
                    raise DeadlineExceededError("Generation request deadline exceeded")
                pieces.append(getattr(response, "text", str(response)))
            text = "".join(pieces)
        else:
            text = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            )
        text = text if isinstance(text, str) else str(text)

        OPEN = "<think>"
        CLOSE = "</think>"
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

        if self.model_manager is not None:
            self.model_manager.touch("generate")

        return text, max(0, len(text))

    def submit_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        cancel_event: Optional[threading.Event] = None,
        deadline: Optional[float] = None,
    ) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise GenerateError("Prompt must be a non-empty string.")
        mt = max_tokens if max_tokens is not None else self.max_new_tokens
        if not isinstance(mt, int) or mt <= 0 or mt > 4096:
            raise GenerateError(f"max_tokens must be an integer in (0, 4096], got {mt}")

        now = time.monotonic()
        dl = deadline if deadline is not None else (now + timeout)
        rem = max(0.01, dl - time.monotonic())

        if self.executor:
            return self.executor.submit(
                lambda: self._generate_sync(
                    prompt, mt, temperature,
                    deadline=dl,
                    cancel_event=cancel_event,
                )[0],
                priority=0,  # interactive priority
                timeout_s=rem,
                cancel_event=cancel_event,
                description="Generate expansion",
            )

        text, _ = self._generate_sync(prompt, mt, temperature, deadline=dl, cancel_event=cancel_event)
        return text

    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        return self.submit_generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def warmup(self):
        self.submit_generate("/no_think warmup", max_tokens=4, timeout=30.0)
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
        if self.executor and hasattr(self.executor, "is_owner_thread") and self.executor.is_owner_thread():
            self.unload()
        elif self.executor and hasattr(self.executor, "is_alive") and self.executor.is_alive():
            try:
                self.executor.submit(self.unload, priority=0, timeout_s=10.0, description="Unload generate")
            except Exception:
                self.unload()
        else:
            self.unload()