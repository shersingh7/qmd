/**
 * mlx.ts - MLX Embedding Client (HTTP bridge to Python MLX server)
 *
 * This module provides an HTTP client to a Python MLX embedding server
 * so QMD can use native Apple MLX models instead of GGUF via node-llama-cpp.
 *
 * The server runs separately as a Python process (scripts/mlx_embed_server.py).
 *
 * Usage:
 *   const mlx = new MlxEmbedClient({ url: 'http://127.0.0.1:8787' });
 *   const result = await mlx.embed("some text");
 *   const results = await mlx.embedBatch(["text1", "text2"]);
 *
 * Environment:
 *   QMD_MLX_EMBED_URL  - MLX server base URL (default: http://127.0.0.1:8787)
 *   QMD_MLX_EMBED_DIMS - Expected embedding dimensions (auto-detected if omitted)
 */

import type { EmbeddingResult } from "./llm.js";

export interface MlxEmbedConfig {
  /** MLX server base URL (default from QMD_MLX_EMBED_URL env or 127.0.0.1:8787) */
  url?: string;
  /** Expected embedding dimensions (optional, auto-detected on first request) */
  dims?: number;
  /** Request timeout in ms (default: 30s) */
  timeoutMs?: number;
}

export interface MlxHealth {
  status: string;
  model?: string;
  dims?: number;
  ready: boolean;
}

export interface MlxError {
  error: string;
}

const DEFAULT_URL = process.env.QMD_MLX_EMBED_URL || "http://127.0.0.1:8787";
const DEFAULT_TIMEOUT_MS = 30_000;

function getConfigUrl(config?: MlxEmbedConfig): string {
  const raw = config?.url ?? DEFAULT_URL;
  return raw.replace(/\/$/, ""); // strip trailing slash
}

function getTimeout(config?: MlxEmbedConfig): number {
  return config?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
}

/** Resolve a text input — single string to embed */
export async function embedWithMlx(text: string, config?: MlxEmbedConfig): Promise<EmbeddingResult | null> {
  try {
    const result = await embedBatchWithMlx([text], config);
    return result[0] ?? null;
  } catch {
    return null;
  }
}

/** Resolve multiple texts in one server request */
export async function embedBatchWithMlx(texts: string[], config?: MlxEmbedConfig): Promise<(EmbeddingResult | null)[]> {
  const url = `${getConfigUrl(config)}/embed`;
  const timeout = getTimeout(config);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const body = await response.text().catch(() => null);
      throw new Error(`MLX server returned ${response.status}: ${body ?? ""}`);
    }

    const data = await response.json() as {
      embeddings: number[][];
      model: string;
      dims: number;
    };

    return data.embeddings.map((vec) => ({
      embedding: vec,
      model: data.model,
    }));
  } finally {
    clearTimeout(timer);
  }
}

/** Ping the MLX server for health */
export async function mlxHealth(config?: MlxEmbedConfig): Promise<MlxHealth | null> {
  try {
    const url = `${getConfigUrl(config)}/health`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 2_000);
    const response = await fetch(url, { signal: controller.signal });
    clearTimeout(timer);
    if (!response.ok) return null;
    return (await response.json()) as MlxHealth;
  } catch {
    return null;
  }
}

/** Check if the MLX server is reachable and serving */
export async function isMlxAvailable(config?: MlxEmbedConfig): Promise<boolean> {
  const health = await mlxHealth(config);
  return health?.ready ?? false;
}

/**
 * Class wrapper that matches the embed/embedBatch surface of LlamaCpp.
 * Can be plugged into the LLM session layer as an alternative backend.
 */
export class MlxEmbedClient {
  private url: string;
  public dims: number | null = null;
  public modelName: string | null = null;

  constructor(config?: MlxEmbedConfig) {
    this.url = getConfigUrl(config);
  }

  /**
   * Detect dimensionality by probing the server. Caches result.
   */
  async probeDims(): Promise<number> {
    if (this.dims != null) return this.dims;
    const health = await mlxHealth({ url: this.url });
    if (health?.dims) {
      this.dims = health.dims;
      this.modelName = health.model ?? null;
      return health.dims;
    }
    // Fallback: embed a single token and measure
    const results = await embedBatchWithMlx(["test"], { url: this.url });
    if (results[0]) {
      this.dims = results[0].embedding.length;
      this.modelName = results[0].model;
      return this.dims;
    }
    throw new Error("Could not probe MLX embedding dimensions");
  }

  async embed(text: string): Promise<EmbeddingResult | null> {
    return embedWithMlx(text, { url: this.url });
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    return embedBatchWithMlx(texts, { url: this.url });
  }

  async health(): Promise<MlxHealth | null> {
    return mlxHealth({ url: this.url });
  }

  async isAvailable(): Promise<boolean> {
    return isMlxAvailable({ url: this.url });
  }
}
