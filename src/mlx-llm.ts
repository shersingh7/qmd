/**
 * mlx-llm.ts - Full MLX API-based LLM implementation for QMD
 *
 * Routes ALL operations (embeddings, reranking, generation) to the external
 * MLX server. No GGUF models needed — pure Apple Silicon via MLX.
 *
 * Configuration:
 *   QMD_EMBED_PROVIDER=mlx        - Enable MLX mode
 *   QMD_MLX_BASE_URL              - MLX API URL (default: http://127.0.0.1:8080)
 *   QMD_MLX_BATCH_SIZE            - Chunks per API call (default: 32)
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  Queryable,
  RerankDocument,
  RerankOptions,
  RerankResult,
  RerankDocumentResult,
} from "./llm.js";
import {
  formatDocForEmbedding,
  formatQueryForEmbedding,
  isQwen3EmbeddingModel,
} from "./llm.js";

const DEFAULT_MLX_BASE_URL = "http://127.0.0.1:8080";
const DEFAULT_MLX_EMBED_MODEL = "mlx-community/Qwen3-Embedding-8B-4bit-DWQ";
const DEFAULT_MLX_RERANK_MODEL = "mlx-community/Qwen3-Reranker-8B-mxfp8";
const DEFAULT_MLX_GENERATE_MODEL = "Qwen/Qwen3-8B-MLX-4bit";
const DEFAULT_MLX_BATCH_SIZE = 32;
const DEFAULT_RETRY_ATTEMPTS = 3;
const DEFAULT_RETRY_DELAY_MS = 250;

// ─── Response Types ──────────────────────────────────────────────────────────

interface MlxEmbeddingsResponse {
  data?: Array<{
    embedding?: number[];
    index?: number;
  }>;
  model?: string;
}

interface MlxHealthResponse {
  status?: string;
  embedding_dim?: number;
  models?: {
    embed?: boolean;
    rerank?: boolean;
    generate?: boolean;
  };
}

interface MlxRerankResponse {
  results?: Array<{
    index?: number;
    relevance_score?: number;
    text?: string;
  }>;
  model?: string;
}

interface MlxGenerateResponse {
  text?: string;
  model?: string;
  done?: boolean;
}

interface MlxChatResponse {
  choices?: Array<{
    message?: {
      content?: string;
    };
  }>;
  model?: string;
}

// ─── Utility ────────────────────────────────────────────────────────────────

class MlxHttpError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "MlxHttpError";
    this.status = status;
  }

  get retriable(): boolean {
    return this.status === 502 || this.status === 503 || this.status === 504;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "");
}

function parsePositiveInteger(value: number | string | undefined, fallback: number): number {
  if (value === undefined || value === "") return fallback;
  const parsed = typeof value === "number" ? value : Number.parseInt(value, 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isConnectionRefused(error: unknown): boolean {
  if (!(error instanceof Error)) return false;

  const cause = error as Error & { code?: string; cause?: { code?: string; message?: string } };
  const code = cause.code ?? cause.cause?.code;
  if (code === "ECONNREFUSED") return true;

  const message = `${error.message} ${cause.cause?.message ?? ""}`;
  return /ECONNREFUSED|connection refused|fetch failed/i.test(message);
}

// ─── MlxLLM ─────────────────────────────────────────────────────────────────

export class MlxLLM implements LLM {
  private readonly baseUrl: string;
  private readonly _embedModelName: string;
  private readonly _rerankModelName: string;
  private readonly _generateModelName: string;
  private readonly batchSize: number;
  private _embeddingDimensions: number | null = null;

  constructor(config: {
    embedModel?: string;
    generateModel?: string;
    rerankModel?: string;
    modelCacheDir?: string;
    mlxBaseUrl?: string;
    mlxBatchSize?: number;
  } = {}) {
    this.baseUrl = normalizeBaseUrl(
      config.mlxBaseUrl || process.env.QMD_MLX_BASE_URL || DEFAULT_MLX_BASE_URL
    );
    this._embedModelName = config.embedModel || process.env.QMD_EMBED_MODEL || DEFAULT_MLX_EMBED_MODEL;
    this._rerankModelName = config.rerankModel || process.env.QMD_RERANK_MODEL || DEFAULT_MLX_RERANK_MODEL;
    this._generateModelName = config.generateModel || process.env.QMD_GENERATE_MODEL || DEFAULT_MLX_GENERATE_MODEL;
    this.batchSize = parsePositiveInteger(
      config.mlxBatchSize ?? process.env.QMD_MLX_BATCH_SIZE,
      DEFAULT_MLX_BATCH_SIZE
    );
  }

  get embedModelName(): string {
    return this._embedModelName;
  }

  // ─── Helpers ────────────────────────────────────────────────────────────

  private resolveEmbedModel(options?: EmbedOptions): string {
    return options?.model || this._embedModelName;
  }

  private updateEmbeddingDimensions(embedding: number[] | null | undefined): void {
    if (!embedding?.length) return;

    if (this._embeddingDimensions === null) {
      this._embeddingDimensions = embedding.length;
    } else if (this._embeddingDimensions !== embedding.length) {
      console.warn(
        `MLX embedding dimension changed from ${this._embeddingDimensions} to ${embedding.length}`
      );
      this._embeddingDimensions = embedding.length;
    }
  }

  private updateEmbeddingDimensionsFromHealth(health: MlxHealthResponse): void {
    if (typeof health.embedding_dim === "number" && health.embedding_dim > 0) {
      this._embeddingDimensions = health.embedding_dim;
    }
  }

  private isFormattedQwenQuery(text: string): boolean {
    return text.startsWith("Instruct:") && text.includes("\nQuery:");
  }

  private formatText(text: string, options: EmbedOptions = {}): string {
    const model = this.resolveEmbedModel(options);
    const isQuery = options.isQuery ?? false;

    if (isQuery && this.isFormattedQwenQuery(text)) {
      return text;
    }

    if (!isQuery && options.title === undefined) {
      return text;
    }

    if (isQwen3EmbeddingModel(model)) {
      return isQuery
        ? `Instruct: Given a search query, retrieve relevant documents\nQuery: ${text}`
        : text;
    }

    return isQuery
      ? formatQueryForEmbedding(text, model)
      : formatDocForEmbedding(text, options.title, model);
  }

  private async withRetry<T>(label: string, operation: () => Promise<T>): Promise<T> {
    let lastError: unknown;

    for (let attempt = 1; attempt <= DEFAULT_RETRY_ATTEMPTS; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        const retryable = isConnectionRefused(error) || (error instanceof MlxHttpError && error.retriable);
        if (!retryable || attempt === DEFAULT_RETRY_ATTEMPTS) {
          break;
        }

        const delayMs = DEFAULT_RETRY_DELAY_MS * attempt;
        console.warn(
          `${label} failed (${formatError(error)}); retrying in ${delayMs}ms (${attempt}/${DEFAULT_RETRY_ATTEMPTS})`
        );
        await sleep(delayMs);
      }
    }

    throw lastError instanceof Error ? lastError : new Error(String(lastError));
  }

  private async fetchFromServer(path: string, body: unknown): Promise<unknown> {
    return await this.withRetry(`MLX ${path}`, async () => {
      const response = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new MlxHttpError(response.status, `MLX API error (${response.status}): ${errorText}`);
      }

      return await response.json();
    });
  }

  private async fetchHealth(): Promise<MlxHealthResponse> {
    return await this.withRetry("MLX health check", async () => {
      const response = await fetch(`${this.baseUrl}/health`);

      if (!response.ok) {
        const errorText = await response.text();
        throw new MlxHttpError(response.status, `MLX health check failed (${response.status}): ${errorText}`);
      }

      return await response.json() as MlxHealthResponse;
    });
  }

  // ─── Embeddings ──────────────────────────────────────────────────────────

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    const [result] = await this.embedBatch([text], options);
    return result ?? null;
  }

  async embedBatch(texts: string[], options: EmbedOptions = {}): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    const model = this.resolveEmbedModel(options);
    const formattedTexts = texts.map((text) => this.formatText(text, { ...options, model }));
    const results: (EmbeddingResult | null)[] = Array.from({ length: texts.length }, () => null);

    for (let start = 0; start < formattedTexts.length; start += this.batchSize) {
      const batch = formattedTexts.slice(start, start + this.batchSize);

      try {
        const response = await this.fetchFromServer("/v1/embeddings", {
          input: batch,
          model,
        }) as MlxEmbeddingsResponse;

        const data = response.data ?? [];
        for (const item of data) {
          const index = item.index;
          if (typeof index === "number" && index >= 0 && index < batch.length && item.embedding) {
            this.updateEmbeddingDimensions(item.embedding);
            results[start + index] = {
              embedding: item.embedding,
              model: response.model || model,
            };
          }
        }
      } catch (error) {
        console.error(`MLX batch embedding error (batch starting at ${start}):`, error);
      }
    }

    return results;
  }

  // ─── Reranking ──────────────────────────────────────────────────────────

  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    const texts = documents.map((doc) => doc.text);

    try {
      const response = await this.fetchFromServer("/v1/rerank", {
        query,
        documents: texts,
        model: this._rerankModelName,
      }) as MlxRerankResponse;

      const results: RerankDocumentResult[] = (response.results ?? []).map((item) => {
        const idx = item.index ?? 0;
        return {
          file: documents[idx]?.file ?? "",
          score: item.relevance_score ?? 0,
          index: idx,
        };
      });

      return {
        results,
        model: response.model || this._rerankModelName,
      };
    } catch (error) {
      console.error("MLX rerank error, returning original order:", error);
      // Fallback: return documents in original order with score 0
      return {
        results: documents.map((doc, i) => ({
          file: doc.file,
          score: 0,
          index: i,
        })),
        model: this._rerankModelName,
      };
    }
  }

  // ─── Generation / Query Expansion ────────────────────────────────────────

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    try {
      const response = await this.fetchFromServer("/v1/generate", {
        prompt,
        max_tokens: options?.maxTokens ?? 150,
        temperature: options?.temperature ?? 0.7,
      }) as MlxGenerateResponse;

      return {
        text: response.text ?? "",
        model: response.model || this._generateModelName,
        done: response.done ?? true,
      };
    } catch (error) {
      console.error("MLX generate error:", error);
      return null;
    }
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    const includeLexical = options?.includeLexical ?? true;
    const intent = (options as any)?.intent;
    const prompt = intent
      ? `/no_think Expand this search query: ${query}\nQuery intent: ${intent}`
      : `/no_think Expand this search query: ${query}`;

    try {
      const response = await this.fetchFromServer("/v1/chat/completions", {
        model: this._generateModelName,
        messages: [
          {
            role: "system",
            content: "You are a search query expansion assistant. Expand the given query into multiple search variations. Output each variation on a new line with a type prefix (lex, vec, or hyde) followed by a colon and the expanded query. Example:\nlex: exact terms\nvec: semantic description\nhyde: hypothetical answer",
          },
          {
            role: "user",
            content: prompt,
          },
        ],
        max_tokens: 600,
        temperature: 0.7,
      }) as MlxChatResponse;

      const content = response.choices?.[0]?.message?.content ?? "";

      // Parse the response into Queryable objects
      const lines = content.trim().split("\n");
      const queryLower = query.toLowerCase();
      const queryTerms = queryLower.replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);

      const hasQueryTerm = (text: string): boolean => {
        const lower = text.toLowerCase();
        if (queryTerms.length === 0) return true;
        return queryTerms.some(term => lower.includes(term));
      };

      const queryables: Queryable[] = lines.map(line => {
        const colonIdx = line.indexOf(":");
        if (colonIdx === -1) return null;
        const type = line.slice(0, colonIdx).trim();
        if (type !== 'lex' && type !== 'vec' && type !== 'hyde') return null;
        const text = line.slice(colonIdx + 1).trim();
        if (!hasQueryTerm(text)) return null;
        return { type: type as Queryable['type'], text };
      }).filter((q): q is Queryable => q !== null);

      const filtered = includeLexical ? queryables : queryables.filter(q => q.type !== 'lex');
      if (filtered.length > 0) return filtered;

      // Fallback
      const fallback: Queryable[] = [
        { type: 'hyde', text: `Information about ${query}` },
        { type: 'lex', text: query },
        { type: 'vec', text: query },
      ];
      return includeLexical ? fallback : fallback.filter(q => q.type !== 'lex');
    } catch (error) {
      console.error("MLX expandQuery error:", error);
      // Fallback to original query
      const fallback: Queryable[] = [{ type: 'vec', text: query }];
      if (includeLexical) fallback.unshift({ type: 'lex', text: query });
      return fallback;
    }
  }

  // ─── Model Info ──────────────────────────────────────────────────────────

  async modelExists(model: string): Promise<ModelInfo> {
    if (model === this._embedModelName || model === this._rerankModelName || model === this._generateModelName) {
      try {
        const health = await this.fetchHealth();
        this.updateEmbeddingDimensionsFromHealth(health);
        return {
          name: model,
          exists: health.status === "ok",
        };
      } catch {
        return {
          name: model,
          exists: false,
        };
      }
    }

    return { name: model, exists: false };
  }

  async getDeviceInfo(): Promise<{
    gpu: string | false;
    gpuOffloading: boolean;
    gpuDevices: string[];
    vram?: { total: number; used: number; free: number };
    cpuCores: number;
  }> {
    // MLX always uses Apple GPU
    return {
      gpu: "Apple Silicon (MLX)",
      gpuOffloading: true,
      gpuDevices: ["Metal"],
      cpuCores: 1,
    };
  }

  async tokenize(text: string): Promise<number[]> {
    // Approximate tokenization — Qwen3 models average ~4 chars/token
    // For precise counts, we'd need the tokenizer, but this is good enough for QMD
    return Array.from({ length: Math.ceil(text.length / 4) }, (_, i) => i);
  }

  async dispose(): Promise<void> {
    this._embeddingDimensions = null;
    // No models to dispose — the MLX server manages its own lifecycle
  }
}