/**
 * openai-llm.ts - OpenAI API-based LLM implementation for QMD
 *
 * Routes embedding operations to the OpenAI embeddings API.
 * Reranking and generation fall back to the built-in LlamaCpp (GGUF) models,
 * since OpenAI doesn't offer rerank/generate endpoints we want to use here.
 *
 * Configuration:
 *   QMD_EMBED_PROVIDER=openai          - Enable OpenAI embed mode
 *   QMD_OPENAI_API_KEY                 - OpenAI API key (required)
 *   QMD_OPENAI_BASE_URL                - Custom base URL (default: https://api.openai.com/v1)
 *   QMD_EMBED_MODEL                    - Model name (default: text-embedding-3-large)
 *   QMD_OPENAI_EMBED_BATCH_SIZE        - Texts per API call (default: 2048, max OpenAI limit)
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

const DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1";
const DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-large";
const DEFAULT_OPENAI_EMBED_BATCH_SIZE = 2048; // OpenAI supports up to 2048 inputs per request
const DEFAULT_EMBED_DIMENSIONS = 3072; // text-embedding-3-large native dim

interface OpenAIEmbeddingData {
  embedding: number[];
  index: number;
}

interface OpenAIEmbeddingResponse {
  object: string;
  data: OpenAIEmbeddingData[];
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

interface OpenAIErrorResponse {
  error?: {
    message: string;
    type: string;
    code: string;
  };
}

class OpenAIError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(status: number, message: string, code?: string) {
    super(message);
    this.name = "OpenAIError";
    this.status = status;
    this.code = code;
  }

  get retriable(): boolean {
    return this.status === 429 || this.status >= 500;
  }
}

// ─── OpenAILLM Class ──────────────────────────────────────────────────────

export class OpenAILLM implements LLM {
  readonly apiKey: string;
  readonly baseUrl: string;
  readonly embedModelName: string;
  readonly generateModelName: string;
  readonly rerankModelName: string;
  readonly batchSize: number;
  readonly embedDimensions: number;

  constructor(opts?: {
    embedModel?: string;
    generateModel?: string;
    rerankModel?: string;
  }) {
    const apiKey = process.env.QMD_OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error(
        "QMD_OPENAI_API_KEY is required for OpenAI embed provider. " +
        "Set it in your environment: export QMD_OPENAI_API_KEY=sk-..."
      );
    }
    this.apiKey = apiKey;
    this.baseUrl = (process.env.QMD_OPENAI_BASE_URL || DEFAULT_OPENAI_BASE_URL).replace(/\/+$/, '');
    this.embedModelName = opts?.embedModel || process.env.QMD_EMBED_MODEL || DEFAULT_OPENAI_EMBED_MODEL;
    this.generateModelName = opts?.generateModel || process.env.QMD_GENERATE_MODEL || "qmd-generated"; // unused for OpenAI
    this.rerankModelName = opts?.rerankModel || process.env.QMD_RERANK_MODEL || "qmd-reranker"; // unused for OpenAI
    this.batchSize = parseInt(process.env.QMD_OPENAI_EMBED_BATCH_SIZE || String(DEFAULT_OPENAI_EMBED_BATCH_SIZE), 10);
    this.embedDimensions = parseInt(process.env.QMD_OPENAI_EMBED_DIMENSIONS || String(DEFAULT_EMBED_DIMENSIONS), 10);

    process.stderr.write(
      `[OpenAI] Initialized: model=${this.embedModelName}, dimensions=${this.embedDimensions}, ` +
      `batch_size=${this.batchSize}, base_url=${this.baseUrl}\n`
    );
  }

  // ─── Core: Embed Single Text ──────────────────────────────────────────

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    const results = await this.embedBatch([text], options);
    return results[0] ?? null;
  }

  // ─── Core: Embed Batch ────────────────────────────────────────────────

  async embedBatch(texts: string[], options?: EmbedOptions & { signal?: AbortSignal }): Promise<(EmbeddingResult | null)[]> {
    // Split into batches of batchSize
    const results: (EmbeddingResult | null)[] = [];
    const batchSize = this.batchSize;

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const batchResults = await this._embedBatchInternal(batch, options?.signal);
      results.push(...batchResults);
    }

    return results;
  }

  private async _embedBatchInternal(
    texts: string[],
    signal?: AbortSignal
  ): Promise<(EmbeddingResult | null)[]> {
    const url = `${this.baseUrl}/embeddings`;

    let retries = 10; // More retries to handle rate limits
    let lastError: Error | null = null;

    while (retries > 0) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120_000); // 2 min timeout per batch

        // Wire up external abort signal
        if (signal) {
          signal.addEventListener('abort', () => controller.abort(), { once: true });
        }

        const response = await fetch(url, {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${this.apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: this.embedModelName,
            input: texts,
            dimensions: this.embedDimensions,
          }),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          let errorBody = "";
          try {
            const errJson = await response.json() as OpenAIErrorResponse;
            errorBody = errJson.error?.message || JSON.stringify(errJson);
          } catch {
            errorBody = await response.text().catch(() => `HTTP ${response.status}`);
          }

          const err = new OpenAIError(response.status, errorBody);

          // For rate limits (429), parse retry-after hint and wait longer
          if (response.status === 429 && retries > 1) {
            retries--;
            lastError = err;
            // Parse "Please try again in X.XXs" from OpenAI error message
            const waitMatch = errorBody.match(/try again in ([\d.]+)s/i);
            let waitMs = waitMatch ? parseFloat(waitMatch[1]!) * 1000 : 10000;
            waitMs = Math.max(waitMs, 5000); // At least 5 seconds
            waitMs += Math.random() * 1000; // Jitter to avoid thundering herd
            process.stderr.write(`[OpenAI] Rate limited (429), retrying in ${Math.round(waitMs/1000)}s (${retries} retries left): ${errorBody.substring(0, 120)}\n`);
            await new Promise(r => setTimeout(r, waitMs));
            continue;
          }

          if (err.retriable && retries > 1) {
            retries--;
            lastError = err;
            const backoff = Math.pow(2, 10 - retries) * 500; // Exponential backoff
            process.stderr.write(`[OpenAI] Retrying (${retries} left) after ${response.status}: ${errorBody.substring(0, 120)} (waiting ${backoff}ms)\n`);
            await new Promise(r => setTimeout(r, backoff));
            continue;
          }
          throw err;
        }

        const data = await response.json() as OpenAIEmbeddingResponse;

        // Map results back to original order using index field
        const resultMap = new Map<number, EmbeddingResult>();
        for (const item of data.data) {
          resultMap.set(item.index, {
            embedding: item.embedding,
            model: data.model,
          });
        }

        process.stderr.write(
          `[OpenAI] Embedded ${data.data.length} texts, ` +
          `${data.usage?.prompt_tokens ?? '?'} prompt tokens, ` +
          `model=${data.model}\n`
        );

        return texts.map((_, idx) => resultMap.get(idx) ?? null);
      } catch (err) {
        if (err instanceof OpenAIError) {
          if (err.retriable && retries > 1) {
            retries--;
            lastError = err;
            const backoff = Math.pow(2, 10 - retries) * 500;
            await new Promise(r => setTimeout(r, backoff));
            continue;
          }
          throw err;
        }
        // AbortError from timeout or signal
        if ((err as Error).name === 'AbortError') {
          throw new Error(`OpenAI embed request aborted (timeout or signal). Last error: ${lastError?.message || 'none'}`);
        }
        throw err;
      }
    }

    throw lastError || new Error("OpenAI embed failed after all retries");
  }

  // ─── Embed Dimensions ──────────────────────────────────────────────────

  /** Returns the configured embedding dimensions for this provider. */
  get embeddingDimensions(): number {
    return this.embedDimensions;
  }

  // ─── Unsupported Operations (fall back to LlamaCpp) ────────────────────

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    throw new Error(
      "OpenAILLM does not support generation. Use LlamaCpp or MlxLLM for query expansion. " +
      "Set QMD_GENERATE_MODEL to configure the GGUF generate model."
    );
  }

  async modelExists(model: string): Promise<ModelInfo> {
    // OpenAI models are always "available" — no local download needed
    return {
      name: model,
      exists: true,
      path: `openai:${model}`,
    };
  }

  async expandQuery(query: string, options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]> {
    throw new Error(
      "OpenAILLM does not support query expansion. Use LlamaCpp or MlxLLM for this."
    );
  }

  async rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    throw new Error(
      "OpenAILLM does not support reranking. Use LlamaCpp or MlxLLM for this."
    );
  }

  async dispose(): Promise<void> {
    // No resources to dispose — OpenAI is stateless
    process.stderr.write("[OpenAI] Disposed (no-op, stateless API client)\n");
  }
}