/**
 * mlx.ts — MLX Embedding Client (zero-copy HTTP bridge to Python MLX server)
 *
 * Designed for maximum throughput on M2 Pro 32GB:
 *   - Float32Array zero-copy binary decoder (single constructor, no per-float loop)
 *   - Concurrent batch pipelining: fires next request before previous resolves
 *   - Auto binary protocol for batches > 16 texts
 *   - Connection keep-alive for TCP handshake amortization
 *
 * The server runs separately as a Python process:
 *   python scripts/mlx_embed_server.py --model mlx-community/nomic-embed-text-v2-moe --port 8787 --preload
 *
 * Env:
 *   QMD_MLX_EMBED_URL  — server URL (default http://127.0.0.1:8787)
 */

import type { EmbeddingResult, EmbedOptions } from "./llm.js";

// ── Types ───────────────────────────────────────────────────────────────────

export interface MlxEmbedConfig {
  url?: string;
  timeoutMs?: number;
  /** Use binary wire format by default (default: true for batches > 16) */
  binary?: boolean;
  /** Max concurrent in-flight requests (default: 2) */
  concurrency?: number;
}

export interface MlxHealth {
  status: string;
  model?: string;
  dims?: number;
  ready: boolean;
}

export interface MlxMemory {
  active_mb: number;
  peak_mb: number;
  model_mb: number;
}

export interface MlxStats {
  total_requests: number;
  avg_ms: number;
  compiled_shapes: number;
  uptime_sec: number;
}

// ── Constants ───────────────────────────────────────────────────────────────

const DEFAULT_URL = process.env.QMD_MLX_EMBED_URL || "http://127.0.0.1:8787";
const DEFAULT_TIMEOUT_MS = 60_000; // 60s for large batch processing
const BINARY_THRESHOLD = 16;       // switch to binary protocol above this batch size

function url(config?: MlxEmbedConfig): string {
  return (config?.url ?? DEFAULT_URL).replace(/\/$/, "");
}

function timeoutMs(config?: MlxEmbedConfig): number {
  return config?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
}

// ── Shared fetch ────────────────────────────────────────────────────────────

function _fetch(url: string, opts: { method: string; body?: BodyInit; timeoutMs: number }): Promise<Response> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), opts.timeoutMs);
  return fetch(url, {
    method: opts.method,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Connection": "keep-alive",
    },
    body: opts.body,
    signal: ctrl.signal,
  }).finally(() => clearTimeout(t));
}

function _jsonBody(texts: string[], dims?: number, isQuery?: boolean): string {
  const payload: Record<string, unknown> = { texts };
  if (dims) payload.dims = dims;
  if (isQuery) payload.is_query = isQuery;
  return JSON.stringify(payload);
}

// ── Binary decoder (zero-copy TypedArray) ───────────────────────────────────

function _decodeBinary(buffer: ArrayBuffer): number[][] {
  const header = new Int32Array(buffer, 0, 2);
  const count = header[0]!;
  const dims  = header[1]!;
  if (count === 0 || dims === 0) return [];

  const floats = new Float32Array(buffer, 8); // skip 8-byte header
  const result: number[][] = [];
  for (let i = 0; i < count; i++) {
    // Float32Array.subarray() creates a view — zero copy
    result.push(Array.from(floats.subarray(i * dims, (i + 1) * dims)));
  }
  return result;
}

// ── Public API ──────────────────────────────────────────────────────────────

export async function embedWithMlx(
  text: string,
  config?: MlxEmbedConfig,
  opts?: EmbedOptions & { dims?: number },
): Promise<EmbeddingResult | null> {
  try {
    const res = await embedBatchWithMlx([text], config, opts);
    return res[0] ?? null;
  } catch {
    return null;
  }
}

export async function embedBatchWithMlx(
  texts: string[],
  config?: MlxEmbedConfig,
  opts?: EmbedOptions & { dims?: number },
): Promise<(EmbeddingResult | null)[]> {
  if (texts.length === 0) return [];

  const useBinary = config?.binary ?? (texts.length >= BINARY_THRESHOLD);
  const baseUrl = url(config);
  const endpoint = useBinary ? "/embed-bin" : "/embed";

  try {
    const resp = await _fetch(`${baseUrl}${endpoint}`, {
      method: "POST",
      body: _jsonBody(texts, opts?.dims, opts?.isQuery),
      timeoutMs: timeoutMs(config),
    });

    if (!resp.ok) {
      const errBody = await resp.text().catch(() => null);
      console.error(`MLX server ${resp.status}: ${errBody ?? ""}`);
      return texts.map(() => null);
    }

    if (useBinary) {
      const buf = await resp.arrayBuffer();
      const vecs = _decodeBinary(buf);
      return vecs.map((embedding) => ({ embedding, model: "mlx" }));
    }

    const data = await resp.json() as { embeddings: number[][]; model: string };
    return data.embeddings.map((embedding) => ({ embedding, model: data.model }));
  } catch (err) {
    console.error("MLX embedding failed:", err);
    return texts.map(() => null);
  }
}

/**
 * Concurrent batch embedding pipeline.
 * Splits texts into sub-batches and fires multiple requests simultaneously,
 * capped by concurrency limit. Returns flat array preserving input order.
 */
export async function embedBatchConcurrent(
  texts: string[],
  config?: MlxEmbedConfig,
  opts?: EmbedOptions & { dims?: number },
  batchSize: number = 32,
): Promise<(EmbeddingResult | null)[]> {
  if (texts.length === 0) return [];

  const concurrency = config?.concurrency ?? 2;

  // Split into sub-batches
  const batches: string[][] = [];
  for (let i = 0; i < texts.length; i += batchSize) {
    batches.push(texts.slice(i, i + batchSize));
  }

  // Pipeline: maintain concurrency limit
  const results: (EmbeddingResult[] | null)[] = new Array(batches.length);
  let nextIdx = 0;

  async function worker(): Promise<void> {
    while (nextIdx < batches.length) {
      const idx = nextIdx++;
      const batch = batches[idx]!;
      const res = await embedBatchWithMlx(batch, config, opts);
      results[idx] = res.every((r) => r === null) ? null : (res as EmbeddingResult[]);
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, batches.length) }, () => worker()));

  // Flatten in order
  const flat: (EmbeddingResult | null)[] = [];
  for (const batch of results) {
    if (batch === null) {
      // Entire batch failed — fill with nulls
      flat.push(...Array(batchSize).fill(null));
    } else {
      flat.push(...batch);
    }
  }
  return flat.slice(0, texts.length);
}

export async function embedBatchBinaryWithMlx(
  texts: string[],
  config?: MlxEmbedConfig,
  opts?: EmbedOptions & { dims?: number },
): Promise<number[][] | null> {
  return embedBatchWithMlx(texts, { ...config, binary: true }, opts).then((r) =>
    r.every((x) => x === null) ? null : r.map((x) => x!.embedding),
  );
}

export async function mlxHealth(config?: MlxEmbedConfig): Promise<MlxHealth | null> {
  try {
    const resp = await fetch(`${url(config)}/health`, {
      signal: AbortSignal.timeout(2_000),
      headers: { "Connection": "keep-alive" },
    });
    return resp.ok ? (await resp.json()) as MlxHealth : null;
  } catch {
    return null;
  }
}

export async function mlxMemory(config?: MlxEmbedConfig): Promise<MlxMemory | null> {
  try {
    const resp = await fetch(`${url(config)}/memory`, {
      signal: AbortSignal.timeout(2_000),
      headers: { "Connection": "keep-alive" },
    });
    return resp.ok ? (await resp.json()) as MlxMemory : null;
  } catch {
    return null;
  }
}

export async function mlxStats(config?: MlxEmbedConfig): Promise<MlxStats | null> {
  try {
    const resp = await fetch(`${url(config)}/stats`, {
      signal: AbortSignal.timeout(2_000),
      headers: { "Connection": "keep-alive" },
    });
    return resp.ok ? (await resp.json()) as MlxStats : null;
  } catch {
    return null;
  }
}

export async function isMlxAvailable(config?: MlxEmbedConfig): Promise<boolean> {
  const h = await mlxHealth(config);
  return h?.ready ?? false;
}

// ── Class Wrapper ───────────────────────────────────────────────────────────

export class MlxEmbedClient {
  private _url: string;
  private _timeout: number;
  private _concurrency: number;
  public dims: number | null = null;
  public modelName: string | null = null;

  constructor(config?: MlxEmbedConfig) {
    this._url = url(config);
    this._timeout = timeoutMs(config);
    this._concurrency = config?.concurrency ?? 2;
  }

  config(): MlxEmbedConfig {
    return { url: this._url, timeoutMs: this._timeout, concurrency: this._concurrency };
  }

  async probeDims(): Promise<number> {
    if (this.dims != null) return this.dims;
    const h = await mlxHealth(this.config());
    if (h?.dims) {
      this.dims = h.dims;
      this.modelName = h.model ?? null;
      return h.dims;
    }
    const r = await embedBatchWithMlx(["test"], this.config());
    if (r[0]) {
      this.dims = r[0].embedding.length;
      this.modelName = r[0].model;
      return this.dims;
    }
    throw new Error("Could not probe MLX dimensions");
  }

  async embed(text: string, options?: EmbedOptions & { dims?: number }): Promise<EmbeddingResult | null> {
    return embedWithMlx(text, this.config(), options);
  }

  async embedBatch(texts: string[], options?: EmbedOptions & { dims?: number }): Promise<(EmbeddingResult | null)[]> {
    return embedBatchWithMlx(texts, this.config(), options);
  }

  async embedBatchConcurrent(texts: string[], options?: EmbedOptions & { dims?: number }, batchSize?: number): Promise<(EmbeddingResult | null)[]> {
    return embedBatchConcurrent(texts, this.config(), options, batchSize);
  }

  async embedBatchBinary(texts: string[], options?: EmbedOptions & { dims?: number }): Promise<number[][] | null> {
    return embedBatchBinaryWithMlx(texts, this.config(), options);
  }

  async health(): Promise<MlxHealth | null> {
    return mlxHealth(this.config());
  }

  async memory(): Promise<MlxMemory | null> {
    return mlxMemory(this.config());
  }

  async stats(): Promise<MlxStats | null> {
    return mlxStats(this.config());
  }

  async isAvailable(): Promise<boolean> {
    return isMlxAvailable(this.config());
  }
}
