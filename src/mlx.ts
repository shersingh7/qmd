/**
 * mlx.ts - MLX Embedding Client (HTTP bridge to Python MLX server)
 *
 * This module provides an HTTP client to the QMD Python MLX embedding server
 * so QMD can use native Apple MLX models instead of GGUF via node-llama-cpp.
 *
 * The server runs separately as a Python process (scripts/mlx_embed_server.py):
 *   python scripts/mlx_embed_server.py --model mlx-community/nomic-embed-text-v2-moe --port 8787 --preload
 *
 * Usage:
 *   import { MlxEmbedClient } from './mlx.js';
 *   const mlx = new MlxEmbedClient({ url: 'http://127.0.0.1:8787' });
 *   const result = await mlx.embed("some text", { isQuery: true, dims: 256 });
 *
 * Environment:
 *   QMD_MLX_EMBED_URL  - MLX server base URL (default: http://127.0.0.1:8787)
 *   QMD_MLX_EMBED_DIMS - Expected embedding dimensionality (auto-detected if omitted)
 */
import type { EmbeddingResult, EmbedOptions } from "./llm.js";

// ── Types ───────────────────────────────────────────────────────────────────

export interface MlxEmbedConfig {
  /** MLX server base URL (default from QMD_MLX_EMBED_URL env or 127.0.0.1:8787) */
  url?: string;
  /** Request timeout in ms (default: 30s) */
  timeoutMs?: number;
}

export interface MlxHealth {
  status: string;
  model?: string;
  dims?: number;
  ready: boolean;
}

export interface MlxMemory {
  active_memory_mb: number;
  peak_memory_mb: number;
}

// ── Constants ───────────────────────────────────────────────────────────────

const DEFAULT_URL = process.env.QMD_MLX_EMBED_URL || "http://127.0.0.1:8787";
const DEFAULT_TIMEOUT_MS = 30_000;

function getConfigUrl(config?: MlxEmbedConfig): string {
  const raw = config?.url ?? DEFAULT_URL;
  return raw.replace(/\/$/, ""); // strip trailing slash
}

function getTimeout(config?: MlxEmbedConfig): number {
  return config?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
}

// ── Shared fetch wrapper (connection reuse, error handling) ─────────────────

function fetchJson(url: string, options: { method: string; body?: unknown; timeoutMs: number }): Promise<Response> {
  const { method, body, timeoutMs } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  return fetch(url, {
    method,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Connection": "keep-alive",
    },
    body: body != null ? JSON.stringify(body) : undefined,
    signal: controller.signal,
  }).finally(() => clearTimeout(timer));
}

// ── Public API ──────────────────────────────────────────────────────────────

/**
 * Embed a single text via MLX server.
 * Returns null on any failure (network, server error, timeout).
 */
export async function embedWithMlx(
  text: string,
  mlxConfig?: MlxEmbedConfig,
  embedOpts?: EmbedOptions & { dims?: number },
): Promise<EmbeddingResult | null> {
  try {
    const result = await embedBatchWithMlx([text], mlxConfig, embedOpts);
    return result[0] ?? null;
  } catch {
    return null;
  }
}

/**
 * Embed multiple texts in one server request.
 * NEVER throws — returns null entries for failed items on error.
 */
export async function embedBatchWithMlx(
  texts: string[],
  mlxConfig?: MlxEmbedConfig,
  embedOpts?: EmbedOptions & { dims?: number },
): Promise<(EmbeddingResult | null)[]> {
  const url = `${getConfigUrl(mlxConfig)}/embed`;
  const timeout = getTimeout(mlxConfig);

  if (texts.length === 0) return [];

  try {
    const payload: Record<string, unknown> = { texts };
    if (embedOpts?.dims) payload.dims = embedOpts.dims;
    if (embedOpts?.isQuery) payload.is_query = embedOpts.isQuery;

    const response = await fetchJson(url, {
      method: "POST",
      body: payload,
      timeoutMs: timeout,
    });

    if (!response.ok) {
      const body = await response.text().catch(() => null);
      console.error(`MLX server returned ${response.status}: ${body ?? ""}`);
      return texts.map(() => null);
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
  } catch (error) {
    console.error("MLX batch embedding failed:", error);
    return texts.map(() => null);
  }
}

/**
 * Embed multiple texts using the binary wire format (no JSON overhead).
 * Fall back to JSON endpoint if binary returns non-ok.
 */
export async function embedBatchBinaryWithMlx(
  texts: string[],
  mlxConfig?: MlxEmbedConfig,
  embedOpts?: EmbedOptions & { dims?: number },
): Promise<number[][] | null> {
  const url = `${getConfigUrl(mlxConfig)}/embed-bin`;
  const timeout = getTimeout(mlxConfig);

  if (texts.length === 0) return [];

  try {
    const payload: Record<string, unknown> = { texts };
    if (embedOpts?.dims) payload.dims = embedOpts.dims;
    if (embedOpts?.isQuery) payload.is_query = embedOpts.isQuery;

    const response = await fetchJson(url, {
      method: "POST",
      body: payload,
      timeoutMs: timeout,
    });

    if (!response.ok) {
      // Fall back to JSON endpoint
      console.warn("MLX binary endpoint failed, falling back to JSON embed");
      const results = await embedBatchWithMlx(texts, mlxConfig, embedOpts);
      return results.map((r) => r?.embedding ?? null) as (number[] | null)[] | null;
    }

    const buffer = await response.arrayBuffer();
    const view = new DataView(buffer);
    const count = view.getInt32(0, true); // little-endian
    const dims = view.getInt32(4, true);  // little-endian

    const embeddings: number[][] = [];
    let offset = 8;
    for (let i = 0; i < count; i++) {
      const vec: number[] = [];
      for (let j = 0; j < dims; j++) {
        vec.push(view.getFloat32(offset, true));
        offset += 4;
      }
      embeddings.push(vec);
    }
    return embeddings;
  } catch (error) {
    console.error("MLX binary embedding failed:", error);
    // Fall back to JSON
    const results = await embedBatchWithMlx(texts, mlxConfig, embedOpts);
    if (results.every((r) => r === null)) return null;
    return results.map((r) => r?.embedding ?? null) as (number[] | null)[] | null;
  }
}

/** Ping the MLX server for health */
export async function mlxHealth(mlxConfig?: MlxEmbedConfig): Promise<MlxHealth | null> {
  try {
    const url = `${getConfigUrl(mlxConfig)}/health`;
    const response = await fetch(url, {
      signal: AbortSignal.timeout(2_000),
      headers: { "Connection": "keep-alive" },
    });
    if (!response.ok) return null;
    return (await response.json()) as MlxHealth;
  } catch {
    return null;
  }
}

/** Get MLX GPU memory stats */
export async function mlxMemory(mlxConfig?: MlxEmbedConfig): Promise<MlxMemory | null> {
  try {
    const url = `${getConfigUrl(mlxConfig)}/memory`;
    const response = await fetch(url, {
      signal: AbortSignal.timeout(2_000),
      headers: { "Connection": "keep-alive" },
    });
    if (!response.ok) return null;
    return (await response.json()) as MlxMemory;
  } catch {
    return null;
  }
}

/** Check if the MLX server is reachable and ready */
export async function isMlxAvailable(mlxConfig?: MlxEmbedConfig): Promise<boolean> {
  const health = await mlxHealth(mlxConfig);
  return health?.ready ?? false;
}

// ── Class Wrapper ───────────────────────────────────────────────────────────

/**
 * Class wrapper that matches the embed/embedBatch surface of LlamaCpp.
 * Can be plugged into the LLM session layer as an alternative backend.
 */
export class MlxEmbedClient {
  private url: string;
  private timeoutMs: number;
  public dims: number | null = null;
  public modelName: string | null = null;

  constructor(config?: MlxEmbedConfig) {
    this.url = getConfigUrl(config);
    this.timeoutMs = getTimeout(config);
  }

  /** Detect dimensionality by probing the server. Caches result. */
  async probeDims(): Promise<number> {
    if (this.dims != null) return this.dims;
    const health = await mlxHealth({ url: this.url, timeoutMs: this.timeoutMs });
    if (health?.dims) {
      this.dims = health.dims;
      this.modelName = health.model ?? null;
      return health.dims;
    }
    // Fallback: embed a single token and measure
    const results = await embedBatchWithMlx(["test"], { url: this.url, timeoutMs: this.timeoutMs });
    if (results[0]) {
      this.dims = results[0].embedding.length;
      this.modelName = results[0].model;
      return this.dims;
    }
    throw new Error("Could not probe MLX embedding dimensions");
  }

  /** Embed a single text. Returns null on failure. */
  async embed(text: string, options?: EmbedOptions & { dims?: number }): Promise<EmbeddingResult | null> {
    return embedWithMlx(text, { url: this.url, timeoutMs: this.timeoutMs }, options);
  }

  /** Embed multiple texts. Returns null entries for failed items. */
  async embedBatch(texts: string[], options?: EmbedOptions & { dims?: number }): Promise<(EmbeddingResult | null)[]> {
    return embedBatchWithMlx(texts, { url: this.url, timeoutMs: this.timeoutMs }, options);
  }

  /** Binary embed (faster for large batches). Returns null on failure. */
  async embedBatchBinary(texts: string[], options?: EmbedOptions & { dims?: number }): Promise<number[][] | null> {
    return embedBatchBinaryWithMlx(texts, { url: this.url, timeoutMs: this.timeoutMs }, options);
  }

  async health(): Promise<MlxHealth | null> {
    return mlxHealth({ url: this.url, timeoutMs: this.timeoutMs });
  }

  async memory(): Promise<MlxMemory | null> {
    return mlxMemory({ url: this.url, timeoutMs: this.timeoutMs });
  }

  async isAvailable(): Promise<boolean> {
    return isMlxAvailable({ url: this.url, timeoutMs: this.timeoutMs });
  }
}
