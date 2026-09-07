/**
 * mlx.ts — MLX Embedding Client (Bounded HTTP bridge to Python MLX server)
 *
 * Metal-accelerated embedding transport for Apple Silicon MacBooks:
 *   - Zero-copy Float32Array binary decoder with strict length and value checks
 *   - Full-deadline AbortSignal covering headers + body consumption
 *   - Bounded concurrent batch pipelining preserving order and cardinality
 *   - Auto binary protocol for bulk batches (>= 16 texts)
 *   - Connection keep-alive and descriptor negotiation
 *
 * Python server entrypoint:
 *   python scripts/mlx_embed_server.py --model mlx-community/nomic-embed-text-v1.5 --port 8787 --preload
 */

import type { EmbeddingResult, EmbedOptions } from "./llm.js";
import type { EmbeddingDescriptor } from "./embedding/contract.js";
import { decodeBinaryEmbeddings, validateJsonEmbedResponse } from "./embedding/protocol.js";

// ── Types ───────────────────────────────────────────────────────────────────

export interface MlxEmbedConfig {
  url?: string;
  timeoutMs?: number;
  /** Use binary wire format by default (default: true for batches >= 16) */
  binary?: boolean;
  /** Max concurrent in-flight requests (default: 2) */
  concurrency?: number;
}

export interface MlxHealth {
  status: string;
  state?: string;
  model?: string;
  dims?: number;
  ready: boolean;
  descriptor?: EmbeddingDescriptor & {
    rerank?: { model?: string };
    generate?: { model?: string };
  };
  rerank_model?: string | null;
  generate_model?: string | null;
  error?: string | null;
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
const BINARY_THRESHOLD = 16;       // switch to binary protocol above or equal to this batch size

function url(config?: MlxEmbedConfig): string {
  return (config?.url ?? DEFAULT_URL).replace(/\/$/, "");
}

function timeoutMs(config?: MlxEmbedConfig): number {
  return config?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
}

// ── Shared fetch with full-body deadline ───────────────────────────────────

async function _fetchJson(
  endpointUrl: string,
  payload: Record<string, unknown>,
  timeout: number,
): Promise<{ ok: boolean; status: number; data?: any; error?: string }> {
  try {
    const resp = await fetch(endpointUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Connection": "keep-alive",
        "Accept": "application/json",
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(timeout),
    });

    if (!resp.ok) {
      const errText = await resp.text().catch(() => "");
      return { ok: false, status: resp.status, error: `HTTP ${resp.status}: ${errText}` };
    }

    const data = await resp.json();
    return { ok: true, status: resp.status, data };
  } catch (err) {
    return { ok: false, status: 0, error: err instanceof Error ? err.message : String(err) };
  }
}

async function _fetchBinary(
  endpointUrl: string,
  payload: Record<string, unknown>,
  timeout: number,
): Promise<{ ok: boolean; status: number; buffer?: ArrayBuffer; error?: string }> {
  try {
    const resp = await fetch(endpointUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Connection": "keep-alive",
        "Accept": "application/octet-stream",
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(timeout),
    });

    if (!resp.ok) {
      const errText = await resp.text().catch(() => "");
      return { ok: false, status: resp.status, error: `HTTP ${resp.status}: ${errText}` };
    }

    const buffer = await resp.arrayBuffer();
    return { ok: true, status: resp.status, buffer };
  } catch (err) {
    return { ok: false, status: 0, error: err instanceof Error ? err.message : String(err) };
  }
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
  const timeout = timeoutMs(config);
  const payload: Record<string, unknown> = { texts };
  if (opts?.dims) payload.dims = opts.dims;
  if (opts?.isQuery) payload.is_query = opts.isQuery;

  try {
    if (useBinary) {
      const res = await _fetchBinary(`${baseUrl}/embed-bin`, payload, timeout);
      if (!res.ok || !res.buffer) {
        console.error(`MLX binary embed failed: ${res.error}`);
        return texts.map(() => null);
      }
      const decoded = decodeBinaryEmbeddings(res.buffer);
      if (decoded.count !== texts.length) {
        console.error(`MLX returned ${decoded.count} embeddings, expected ${texts.length}`);
        return texts.map(() => null);
      }
      return decoded.vectors.map((embedding) => ({
        embedding,
        model: opts?.model || "mlx",
      }));
    }

    const res = await _fetchJson(`${baseUrl}/embed`, payload, timeout);
    if (!res.ok || !res.data) {
      console.error(`MLX embed failed: ${res.error}`);
      return texts.map(() => null);
    }

    const validated = validateJsonEmbedResponse(res.data);
    if (validated.embeddings.length !== texts.length) {
      console.error(`MLX returned ${validated.embeddings.length} embeddings, expected ${texts.length}`);
      return texts.map(() => null);
    }

    return validated.embeddings.map((embedding) => ({
      embedding,
      model: validated.model,
    }));
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
  if (batchSize <= 0) batchSize = 32;

  const concurrency = Math.max(1, config?.concurrency ?? 2);

  // Split into sub-batches
  const batches: string[][] = [];
  for (let i = 0; i < texts.length; i += batchSize) {
    batches.push(texts.slice(i, i + batchSize));
  }

  // Pipeline: maintain concurrency limit
  const results: (EmbeddingResult | null)[][] = new Array(batches.length);
  let nextIdx = 0;

  async function worker(): Promise<void> {
    while (nextIdx < batches.length) {
      const idx = nextIdx++;
      const batch = batches[idx]!;
      const res = await embedBatchWithMlx(batch, config, opts);
      results[idx] = res;
    }
  }

  const workerCount = Math.min(concurrency, batches.length);
  await Promise.all(Array.from({ length: workerCount }, () => worker()));

  // Flatten in exact order
  const flat: (EmbeddingResult | null)[] = [];
  for (let b = 0; b < results.length; b++) {
    const batchRes = results[b];
    const expectedLength = batches[b]!.length;
    if (!batchRes || batchRes.length !== expectedLength) {
      flat.push(...Array(expectedLength).fill(null));
    } else {
      flat.push(...batchRes);
    }
  }

  return flat.slice(0, texts.length);
}

export async function embedBatchBinaryWithMlx(
  texts: string[],
  config?: MlxEmbedConfig,
  opts?: EmbedOptions & { dims?: number },
): Promise<number[][] | null> {
  const r = await embedBatchWithMlx(texts, { ...config, binary: true }, opts);
  if (r.length === 0 || r.some((x) => x === null)) return null;
  return r.map((x) => x!.embedding);
}

// ── Rerank / Generate client API ───────────────────────────────────────────

export type MlxRerankResult = {
  scores: number[];
  model: string;
};

export async function rerankWithMlx(
  query: string,
  documents: string[],
  config?: MlxEmbedConfig,
): Promise<MlxRerankResult | null> {
  if (!query.trim() || documents.length === 0) return null;
  const res = await _fetchJson(`${url(config)}/rerank`, { query, documents }, Math.max(timeoutMs(config), 120_000));
  if (!res.ok || !res.data) {
    console.error(`MLX rerank failed: ${res.error}`);
    return null;
  }
  const scores = res.data.scores;
  if (!Array.isArray(scores) || scores.length !== documents.length) {
    console.error(`MLX rerank returned ${Array.isArray(scores) ? scores.length : typeof scores} scores, expected ${documents.length}`);
    return null;
  }
  for (const s of scores) {
    if (typeof s !== "number" || !Number.isFinite(s)) {
      console.error("MLX rerank returned a non-finite score");
      return null;
    }
  }
  return { scores, model: res.data.model ?? "mlx" };
}

export async function generateWithMlx(
  prompt: string,
  config?: MlxEmbedConfig,
  opts?: { maxTokens?: number; temperature?: number },
): Promise<string | null> {
  if (!prompt.trim()) return null;
  const payload: Record<string, unknown> = { prompt, max_tokens: opts?.maxTokens ?? 600 };
  if (opts?.temperature !== undefined) payload.temperature = opts.temperature;
  const res = await _fetchJson(`${url(config)}/generate`, payload, Math.max(timeoutMs(config), 120_000));
  if (!res.ok || !res.data) {
    console.error(`MLX generate failed: ${res.error}`);
    return null;
  }
  const text = res.data.text;
  if (typeof text !== "string") {
    console.error("MLX generate returned non-string text");
    return null;
  }
  return text;
}

export async function mlxHealth(config?: MlxEmbedConfig): Promise<MlxHealth | null> {
  try {
    const resp = await fetch(`${url(config)}/health`, {
      signal: AbortSignal.timeout(3_000),
      headers: { "Connection": "keep-alive" },
    });
    return resp.ok ? ((await resp.json()) as MlxHealth) : null;
  } catch {
    return null;
  }
}

export async function mlxMemory(config?: MlxEmbedConfig): Promise<MlxMemory | null> {
  try {
    const resp = await fetch(`${url(config)}/memory`, {
      signal: AbortSignal.timeout(3_000),
      headers: { "Connection": "keep-alive" },
    });
    return resp.ok ? ((await resp.json()) as MlxMemory) : null;
  } catch {
    return null;
  }
}

export async function mlxStats(config?: MlxEmbedConfig): Promise<MlxStats | null> {
  try {
    const resp = await fetch(`${url(config)}/stats`, {
      signal: AbortSignal.timeout(3_000),
      headers: { "Connection": "keep-alive" },
    });
    return resp.ok ? ((await resp.json()) as MlxStats) : null;
  } catch {
    return null;
  }
}

export async function isMlxAvailable(config?: MlxEmbedConfig): Promise<boolean> {
  const h = await mlxHealth(config);
  return h?.ready ?? false;
}

export async function getMlxDescriptor(config?: MlxEmbedConfig): Promise<EmbeddingDescriptor | null> {
  try {
    const resp = await fetch(`${url(config)}/descriptor`, {
      signal: AbortSignal.timeout(3_000),
      headers: { "Connection": "keep-alive" },
    });
    if (resp.ok) {
      return (await resp.json()) as EmbeddingDescriptor;
    }
  } catch {
    // Fall back to health if /descriptor not supported
  }
  const h = await mlxHealth(config);
  if (h?.descriptor) return h.descriptor;
  return null;
}

// ── Class Wrapper ───────────────────────────────────────────────────────────

export class MlxEmbedClient {
  private _url: string;
  private _timeout: number;
  private _concurrency: number;
  public dims: number | null = null;
  public modelName: string | null = null;
  public descriptor: EmbeddingDescriptor | null = null;

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
    const desc = await getMlxDescriptor(this.config());
    if (desc) {
      this.descriptor = desc;
      this.dims = desc.outputDimensions;
      this.modelName = desc.model;
      return this.dims;
    }
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
    throw new Error("Could not probe MLX dimensions: server unreachable or returned empty embedding");
  }

  async getDescriptor(): Promise<EmbeddingDescriptor | null> {
    if (this.descriptor) return this.descriptor;
    const desc = await getMlxDescriptor(this.config());
    if (desc) {
      this.descriptor = desc;
      this.dims = desc.outputDimensions;
      this.modelName = desc.model;
    }
    return desc;
  }

  async embed(text: string, options?: EmbedOptions & { dims?: number }): Promise<EmbeddingResult | null> {
    return embedWithMlx(text, this.config(), options);
  }

  async embedBatch(texts: string[], options?: EmbedOptions & { dims?: number }): Promise<(EmbeddingResult | null)[]> {
    return embedBatchWithMlx(texts, this.config(), options);
  }

  async embedBatchConcurrent(
    texts: string[],
    options?: EmbedOptions & { dims?: number },
    batchSize?: number,
  ): Promise<(EmbeddingResult | null)[]> {
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
