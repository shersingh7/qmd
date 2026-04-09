/**
 * ollama-llm.ts - Ollama API-based LLM implementation for QMD
 *
 * Uses Ollama's /api/embeddings endpoint for embedding generation,
 * falling back to node-llama-cpp for generation and reranking.
 *
 * This enables using large embedding models (e.g., Qwen3-Embedding-8B)
 * that would cause OOM crashes when loaded in-process via node-llama-cpp,
 * since Ollama manages memory and GPU offloading properly.
 *
 * Configuration:
 *   QMD_EMBED_PROVIDER=ollama   - Enable Ollama embeddings
 *   QMD_OLLAMA_BASE_URL         - Ollama API URL (default: http://127.0.0.1:11434)
 *   QMD_OLLAMA_EMBED_MODEL     - Model name in Ollama (default: qwen3-embedding)
 *   QMD_OLLAMA_BATCH_SIZE      - Chunks per API call (default: 32)
 */

import type { LLM, EmbedOptions, EmbeddingResult, GenerateOptions, GenerateResult, RerankOptions, RerankResult, Queryable, RerankDocument, ModelInfo } from "./llm.js";
import { LlamaCpp, isQwen3EmbeddingModel, formatQueryForEmbedding, formatDocForEmbedding } from "./llm.js";

const DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434";
const DEFAULT_OLLAMA_EMBED_MODEL = "qwen3-embedding";
const DEFAULT_OLLAMA_BATCH_SIZE = 32;

/**
 * Ollama API embedding response
 */
interface OllamaEmbedResponse {
  model: string;
  embeddings: number[][];
  total_duration?: number;
  load_duration?: number;
}

/**
 * OllamaLLM - Uses Ollama API for embeddings, delegates everything else to LlamaCpp
 *
 * This hybrid approach gives us:
 * - Best embedding quality via Ollama (4096d vectors from large models)
 * - Local generation/reranking via node-llama-cpp (small models, in-process)
 * - No OOM crashes since Ollama manages memory properly
 */
export class OllamaLLM implements LLM {
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
    ollamaBaseUrl?: string;
    ollamaEmbedModel?: string;
    ollamaBatchSize?: number;
  } = {}) {
    this.baseUrl = config.ollamaBaseUrl || process.env.QMD_OLLAMA_BASE_URL || DEFAULT_OLLAMA_BASE_URL;
    this._embedModelName = config.ollamaEmbedModel || process.env.QMD_OLLAMA_EMBED_MODEL || DEFAULT_OLLAMA_EMBED_MODEL;
    this.batchSize = config.ollamaBatchSize || parseInt(process.env.QMD_OLLAMA_BATCH_SIZE || String(DEFAULT_OLLAMA_BATCH_SIZE), 10);

    // Delegate generation and reranking to LlamaCpp (small models, in-process)
    this._llamaCpp = new LlamaCpp({
      generateModel: config.generateModel,
      rerankModel: config.rerankModel,
      modelCacheDir: config.modelCacheDir,
      // Don't load embed model in llamaCpp — we use Ollama instead
      // Use embeddinggemma as placeholder (it won't be used for embedding)
      // This avoids loading a large model unnecessarily
    });
  }

  get embedModelName(): string {
    return this._embedModelName;
  }

  /**
   * Get the underlying LlamaCpp instance for session management
   */
  get llamaCpp(): LlamaCpp {
    return this._llamaCpp;
  }

  /**
   * Call Ollama's /api/embeddings endpoint for a batch of texts
   */
  private async ollamaEmbed(texts: string[]): Promise<OllamaEmbedResponse> {
    const response = await fetch(`${this.baseUrl}/api/embed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: this._embedModelName,
        input: texts,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Ollama API error (${response.status}): ${errorText}`);
    }

    return await response.json() as OllamaEmbedResponse;
  }

  /**
   * Detect embedding dimensions by making a test call
   */
  private async detectDimensions(): Promise<number> {
    if (this._embeddingDimensions !== null) {
      return this._embeddingDimensions;
    }

    const result = await this.ollamaEmbed(["test"]);
    if (!result.embeddings || result.embeddings.length === 0) {
      throw new Error("Failed to get embedding dimensions from Ollama");
    }

    this._embeddingDimensions = result.embeddings[0]!.length;
    return this._embeddingDimensions;
  }

  /**
   * Format text for embedding based on model type
   */
  private formatText(text: string, options: EmbedOptions = {}): string {
    const isQuery = options.isQuery ?? false;
    if (isQwen3EmbeddingModel(this._embedModelName)) {
      // Qwen3-Embedding uses instruct format
      if (isQuery) {
        return `Instruct: Given a search query, retrieve relevant documents\nQuery: ${text}`;
      }
      return text;
    }
    // Default: nomic/embeddinggemma format
    if (isQuery) {
      return formatQueryForEmbedding(text);
    }
    return formatDocForEmbedding(text, options.title);
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      // Text is already formatted by the store (formatDocForEmbedding/formatQueryForEmbedding)
      // Just send it directly to Ollama
      const result = await this.ollamaEmbed([text]);

      if (!result.embeddings || result.embeddings.length === 0) {
        console.error("Ollama returned no embeddings");
        return null;
      }

      return {
        embedding: result.embeddings[0]!,
        model: this._embedModelName,
      };
    } catch (error) {
      console.error("Ollama embedding error:", error);
      return null;
    }
  }

  async embedBatch(texts: string[], options: EmbedOptions = {}): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    // Texts are already formatted by the store
    const results: (EmbeddingResult | null)[] = [];

    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize);
      try {
        const response = await this.ollamaEmbed(batch);

        if (!response.embeddings || response.embeddings.length !== batch.length) {
          console.error(`Ollama returned ${response.embeddings?.length ?? 0} embeddings for ${batch.length} texts`);
          // Fill with nulls for this batch
          for (let j = 0; j < batch.length; j++) {
            results.push(null);
          }
          continue;
        }

        for (const embedding of response.embeddings) {
          results.push({
            embedding,
            model: this._embedModelName,
          });
        }
      } catch (error) {
        console.error(`Ollama batch embedding error (batch starting at ${i}):`, error);
        for (let j = 0; j < batch.length; j++) {
          results.push(null);
        }
      }
    }

    return results;
  }

  // Delegate generation and reranking to LlamaCpp
  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    return this._llamaCpp.generate(prompt, options);
  }

  async expandQuery(query: string, options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]> {
    return this._llamaCpp.expandQuery(query, options);
  }

  async rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    return this._llamaCpp.rerank(query, documents, options);
  }

  async modelExists(model: string): Promise<ModelInfo> {
    // Check if model exists in Ollama
    if (model === this._embedModelName) {
      try {
        const response = await fetch(`${this.baseUrl}/api/show`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: this._embedModelName }),
        });
        return { name: this._embedModelName, exists: response.ok };
      } catch {
        return { name: this._embedModelName, exists: false };
      }
    }
    // Delegate other models to LlamaCpp
    return this._llamaCpp.modelExists(model);
  }

  async dispose(): Promise<void> {
    this._embeddingDimensions = null;
    await this._llamaCpp.dispose();
  }

  /**
   * Tokenize text using the underlying LlamaCpp instance
   */
  async tokenize(text: string): Promise<number[]> {
    const tokens = await this._llamaCpp.tokenize(text);
    return Array.from(tokens) as number[];
  }
}