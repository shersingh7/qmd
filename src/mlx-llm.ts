/**
 * mlx-llm.ts - MLX API-based embedding implementation for QMD
 *
 * Uses an external FastAPI MLX embedding server for embeddings while delegating
 * generation and reranking to node-llama-cpp.
 *
 * Configuration:
 *   QMD_EMBED_PROVIDER=mlx        - Enable MLX embeddings
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
} from "./llm.js";
import {
  LlamaCpp,
  formatDocForEmbedding,
  formatQueryForEmbedding,
  isQwen3EmbeddingModel,
} from "./llm.js";

const DEFAULT_MLX_BASE_URL = "http://127.0.0.1:8080";
const DEFAULT_MLX_EMBED_MODEL = "mlx-community/Qwen3-Embedding-8B-4bit-DWQ";
const DEFAULT_MLX_BATCH_SIZE = 32;
const DEFAULT_RETRY_ATTEMPTS = 3;
const DEFAULT_RETRY_DELAY_MS = 250;

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
}

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

export class MlxLLM implements LLM {
  private readonly baseUrl: string;
  private readonly _embedModelName: string;
  private readonly batchSize: number;
  private readonly _llamaCpp: LlamaCpp;
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
    this.batchSize = parsePositiveInteger(
      config.mlxBatchSize ?? process.env.QMD_MLX_BATCH_SIZE,
      DEFAULT_MLX_BATCH_SIZE
    );

    this._llamaCpp = new LlamaCpp({
      generateModel: config.generateModel,
      rerankModel: config.rerankModel,
      modelCacheDir: config.modelCacheDir,
    });
  }

  get embedModelName(): string {
    return this._embedModelName;
  }

  get llamaCpp(): LlamaCpp {
    return this._llamaCpp;
  }

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

  private async fetchEmbeddings(texts: string[], model: string): Promise<MlxEmbeddingsResponse> {
    return await this.withRetry("MLX embedding request", async () => {
      const response = await fetch(`${this.baseUrl}/v1/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input: texts,
          model,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new MlxHttpError(response.status, `MLX API error (${response.status}): ${errorText}`);
      }

      return await response.json() as MlxEmbeddingsResponse;
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

  private mapEmbeddingsResponse(
    response: MlxEmbeddingsResponse,
    expectedCount: number,
    model: string
  ): (EmbeddingResult | null)[] {
    const results: (EmbeddingResult | null)[] = Array.from({ length: expectedCount }, () => null);
    const data = response.data ?? [];

    for (const item of data) {
      const index = item.index;
      if (typeof index !== "number" || index < 0 || index >= expectedCount || !item.embedding) {
        continue;
      }

      this.updateEmbeddingDimensions(item.embedding);
      results[index] = {
        embedding: item.embedding,
        model: response.model || model,
      };
    }

    return results;
  }

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
        const response = await this.fetchEmbeddings(batch, model);
        const batchResults = this.mapEmbeddingsResponse(response, batch.length, model);
        for (let i = 0; i < batchResults.length; i++) {
          results[start + i] = batchResults[i] ?? null;
        }
      } catch (error) {
        console.error(`MLX batch embedding error (batch starting at ${start}):`, error);
      }
    }

    return results;
  }

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    return this._llamaCpp.generate(prompt, options);
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    return this._llamaCpp.expandQuery(query, options);
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    return this._llamaCpp.rerank(query, documents, options);
  }

  async modelExists(model: string): Promise<ModelInfo> {
    if (model === this._embedModelName) {
      try {
        const health = await this.fetchHealth();
        this.updateEmbeddingDimensionsFromHealth(health);
        return {
          name: this._embedModelName,
          exists: health.status === "ok",
        };
      } catch {
        return {
          name: this._embedModelName,
          exists: false,
        };
      }
    }

    return this._llamaCpp.modelExists(model);
  }

  async getDeviceInfo(): Promise<{
    gpu: string | false;
    gpuOffloading: boolean;
    gpuDevices: string[];
    vram?: { total: number; used: number; free: number };
    cpuCores: number;
  }> {
    return this._llamaCpp.getDeviceInfo();
  }

  async tokenize(text: string): Promise<number[]> {
    const tokens = await this._llamaCpp.tokenize(text);
    return Array.from(tokens) as number[];
  }

  async dispose(): Promise<void> {
    this._embeddingDimensions = null;
    await this._llamaCpp.dispose();
  }
}
