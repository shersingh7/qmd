/**
 * config.ts — Validated Embedding Configuration Resolution
 *
 * Implements deterministic precedence:
 *   1. Explicit caller / CLI options
 *   2. Environment variables (QMD_EMBED_BACKEND, QMD_EMBED_MODEL, etc.)
 *   3. Safe fallback defaults
 *
 * Enforces fail-closed semantics: if MLX backend is selected but unavailable,
 * an actionable error is raised instead of silently switching to a different model.
 */

import type { EmbeddingBackendType } from "./contract.js";

export interface EmbeddingConfigOptions {
  backend?: EmbeddingBackendType;
  model?: string;
  dims?: number;
  mlxUrl?: string;
  mlxTimeoutMs?: number;
  mlxConcurrency?: number;
  mlxBinary?: boolean;
  failClosed?: boolean;
}

export interface ResolvedEmbeddingConfig {
  backend: EmbeddingBackendType;
  model: string;
  dims?: number;
  mlxUrl: string;
  mlxTimeoutMs: number;
  mlxConcurrency: number;
  mlxBinary: boolean;
  failClosed: boolean;
}

export const DEFAULT_GGUF_EMBED_MODEL = "hf:ggml-org/embeddinggemma-300M-Q8_0.gguf";
export const DEFAULT_MLX_URL = "http://127.0.0.1:8787";
export const DEFAULT_MLX_TIMEOUT_MS = 60_000;
export const DEFAULT_MLX_CONCURRENCY = 2;

/**
 * Resolves embedding configuration with validated precedence.
 */
export function resolveEmbeddingConfig(options?: EmbeddingConfigOptions): ResolvedEmbeddingConfig {
  // 1. Backend resolution: caller options > env > default ('gguf')
  let backend: EmbeddingBackendType = "gguf";
  const envBackend = process.env.QMD_EMBED_BACKEND?.toLowerCase();
  if (options?.backend) {
    backend = options.backend;
  } else if (envBackend === "mlx" || envBackend === "gguf") {
    backend = envBackend;
  }

  // 2. Model resolution: caller options > env > default for backend
  let model: string;
  if (options?.model) {
    model = options.model;
  } else if (process.env.QMD_EMBED_MODEL) {
    model = process.env.QMD_EMBED_MODEL;
  } else if (backend === "mlx") {
    model = process.env.MLX_EMBED_MODEL || "mlx-community/nomic-embed-text-v1.5";
  } else {
    model = DEFAULT_GGUF_EMBED_MODEL;
  }

  // 3. Dimensions: caller options > env > undefined (use native)
  let dims: number | undefined = options?.dims;
  if (dims === undefined && process.env.QMD_EMBED_DIMS) {
    const parsed = parseInt(process.env.QMD_EMBED_DIMS, 10);
    if (!isNaN(parsed) && parsed > 0) {
      dims = parsed;
    }
  }

  // 4. MLX URL: caller options > env > port env > default
  let mlxUrl = options?.mlxUrl || process.env.QMD_MLX_EMBED_URL;
  if (!mlxUrl) {
    const port = process.env.MLX_EMBED_PORT || "8787";
    mlxUrl = `http://127.0.0.1:${port}`;
  }
  mlxUrl = mlxUrl.replace(/\/$/, "");

  // 5. MLX Timeout
  let mlxTimeoutMs = options?.mlxTimeoutMs;
  if (mlxTimeoutMs === undefined && process.env.QMD_MLX_TIMEOUT_MS) {
    const parsed = parseInt(process.env.QMD_MLX_TIMEOUT_MS, 10);
    if (!isNaN(parsed) && parsed > 0) mlxTimeoutMs = parsed;
  }
  if (mlxTimeoutMs === undefined) mlxTimeoutMs = DEFAULT_MLX_TIMEOUT_MS;

  // 6. MLX Concurrency
  let mlxConcurrency = options?.mlxConcurrency;
  if (mlxConcurrency === undefined && process.env.QMD_MLX_CONCURRENCY) {
    const parsed = parseInt(process.env.QMD_MLX_CONCURRENCY, 10);
    if (!isNaN(parsed) && parsed > 0) mlxConcurrency = parsed;
  }
  if (mlxConcurrency === undefined) mlxConcurrency = DEFAULT_MLX_CONCURRENCY;

  // 7. MLX Binary preference
  let mlxBinary = options?.mlxBinary ?? true;
  if (process.env.QMD_MLX_BINARY !== undefined) {
    mlxBinary = process.env.QMD_MLX_BINARY !== "0" && process.env.QMD_MLX_BINARY !== "false";
  }

  // 8. Fail-closed: true by default when MLX is explicitly chosen
  const failClosed = options?.failClosed ?? (backend === "mlx");

  return {
    backend,
    model,
    dims,
    mlxUrl,
    mlxTimeoutMs,
    mlxConcurrency,
    mlxBinary,
    failClosed,
  };
}
