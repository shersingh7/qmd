/**
 * embedding-contract.test.ts — Unit tests for embedding descriptor contract and configuration
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import {
  computeEmbeddingSpaceId,
  areDescriptorsCompatible,
  assertEmbeddingCompatibility,
  formatQueryForDescriptor,
  formatDocForDescriptor,
  isQwenEmbeddingModel,
  isNomicEmbeddingModel,
  type EmbeddingDescriptor,
} from "../src/embedding/contract.js";
import {
  resolveEmbeddingConfig,
  DEFAULT_GGUF_EMBED_MODEL,
  DEFAULT_MLX_URL,
} from "../src/embedding/config.js";

describe("Embedding Contract Space ID & Fingerprinting", () => {
  const baseDescriptor: EmbeddingDescriptor = {
    version: 1,
    backend: "gguf",
    model: "hf:ggml-org/embeddinggemma-300M-Q8_0.gguf",
    pooling: "mean",
    nativeDimensions: 768,
    outputDimensions: 768,
    maxTokens: 2048,
    normalized: true,
  };

  test("computes deterministic space ID for identical descriptor", () => {
    const id1 = computeEmbeddingSpaceId(baseDescriptor);
    const id2 = computeEmbeddingSpaceId({ ...baseDescriptor });
    expect(id1).toBe(id2);
    expect(typeof id1).toBe("string");
    expect(id1.length).toBeGreaterThan(10);
  });

  test("produces different space ID for same dimensions but different model", () => {
    const nomicDescriptor: EmbeddingDescriptor = {
      ...baseDescriptor,
      backend: "mlx",
      model: "mlx-community/nomic-embed-text-v1.5",
    };
    const id1 = computeEmbeddingSpaceId(baseDescriptor);
    const id2 = computeEmbeddingSpaceId(nomicDescriptor);
    expect(id1).not.toBe(id2);
    expect(areDescriptorsCompatible(baseDescriptor, nomicDescriptor)).toBe(false);
  });

  test("produces different space ID when pooling strategy changes", () => {
    const clsDescriptor: EmbeddingDescriptor = {
      ...baseDescriptor,
      pooling: "cls",
    };
    expect(computeEmbeddingSpaceId(baseDescriptor)).not.toBe(computeEmbeddingSpaceId(clsDescriptor));
    expect(areDescriptorsCompatible(baseDescriptor, clsDescriptor)).toBe(false);
  });

  test("produces different space ID when output dimension is reduced (Matryoshka)", () => {
    const reducedDescriptor: EmbeddingDescriptor = {
      ...baseDescriptor,
      outputDimensions: 256,
    };
    expect(computeEmbeddingSpaceId(baseDescriptor)).not.toBe(computeEmbeddingSpaceId(reducedDescriptor));
    expect(areDescriptorsCompatible(baseDescriptor, reducedDescriptor)).toBe(false);
  });

  test("produces different space ID when normalization flag changes", () => {
    const unnormalized: EmbeddingDescriptor = {
      ...baseDescriptor,
      normalized: false,
    };
    expect(computeEmbeddingSpaceId(baseDescriptor)).not.toBe(computeEmbeddingSpaceId(unnormalized));
    expect(areDescriptorsCompatible(baseDescriptor, unnormalized)).toBe(false);
  });

  test("produces different space ID when tokenizer identity changes", () => {
    const withTokenizer1: EmbeddingDescriptor = {
      ...baseDescriptor,
      tokenizer: "tokenizer-v1",
    };
    const withTokenizer2: EmbeddingDescriptor = {
      ...baseDescriptor,
      tokenizer: "tokenizer-v2",
    };
    expect(computeEmbeddingSpaceId(withTokenizer1)).not.toBe(computeEmbeddingSpaceId(withTokenizer2));
    expect(computeEmbeddingSpaceId(baseDescriptor)).not.toBe(computeEmbeddingSpaceId(withTokenizer1));
    expect(areDescriptorsCompatible(withTokenizer1, withTokenizer2)).toBe(false);
  });

  test("produces different space ID for case-sensitive model paths", () => {
    const uppercaseModel: EmbeddingDescriptor = {
      ...baseDescriptor,
      model: "/Models/Qwen-Embedding-4B",
    };
    const lowercaseModel: EmbeddingDescriptor = {
      ...baseDescriptor,
      model: "/models/qwen-embedding-4b",
    };
    expect(computeEmbeddingSpaceId(uppercaseModel)).not.toBe(computeEmbeddingSpaceId(lowercaseModel));
    expect(areDescriptorsCompatible(uppercaseModel, lowercaseModel)).toBe(false);
  });

  test("assertEmbeddingCompatibility throws descriptive error on mismatch", () => {
    const other: EmbeddingDescriptor = {
      ...baseDescriptor,
      model: "other-model",
    };
    expect(() => assertEmbeddingCompatibility(baseDescriptor, other, "StoreTest")).toThrow(
      /\[StoreTest\] Embedding space mismatch/
    );
  });

  test("assertEmbeddingCompatibility passes for compatible descriptors", () => {
    expect(() => assertEmbeddingCompatibility(baseDescriptor, { ...baseDescriptor })).not.toThrow();
  });
});

describe("Embedding Formatting for Descriptors", () => {
  test("formats queries according to model family", () => {
    expect(formatQueryForDescriptor("search text", { model: "mlx-community/Qwen3-Embedding-0.6B" })).toContain(
      "Instruct: Retrieve relevant documents for the given query"
    );
    expect(formatQueryForDescriptor("search text", { model: "nomic-embed-text-v1.5" })).toBe(
      "search_query: search text"
    );
    expect(formatQueryForDescriptor("search text", { model: "embeddinggemma" })).toBe(
      "task: search result | query: search text"
    );
  });

  test("formats documents according to model family", () => {
    expect(
      formatDocForDescriptor("body content", "Title", { model: "mlx-community/Qwen3-Embedding-0.6B" })
    ).toBe("Title\nbody content");

    expect(
      formatDocForDescriptor("body content", "Title", { model: "nomic-embed-text-v1.5" })
    ).toBe("search_document: Title\nbody content");

    expect(
      formatDocForDescriptor("body content", "Title", { model: "embeddinggemma" })
    ).toBe("title: Title | text: body content");
  });

  test("model family detection helpers", () => {
    expect(isQwenEmbeddingModel("Qwen/Qwen2.5-Coder")).toBe(false);
    expect(isQwenEmbeddingModel("mlx-community/Qwen3-Embedding-0.6B")).toBe(true);
    expect(isNomicEmbeddingModel("nomic-embed-text-v1.5")).toBe(true);
  });
});

describe("Embedding Config Resolution & Precedence", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    delete process.env.QMD_EMBED_BACKEND;
    delete process.env.QMD_EMBED_MODEL;
    delete process.env.QMD_EMBED_DIMS;
    delete process.env.QMD_MLX_EMBED_URL;
    delete process.env.MLX_EMBED_PORT;
  });

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  test("resolves default GGUF config when no options or env set", () => {
    const config = resolveEmbeddingConfig();
    expect(config.backend).toBe("gguf");
    expect(config.model).toBe(DEFAULT_GGUF_EMBED_MODEL);
    expect(config.dims).toBeUndefined();
    expect(config.failClosed).toBe(false);
  });

  test("resolves from environment variables", () => {
    process.env.QMD_EMBED_BACKEND = "mlx";
    process.env.QMD_EMBED_MODEL = "mlx-community/custom-model";
    process.env.QMD_EMBED_DIMS = "512";
    process.env.QMD_MLX_EMBED_URL = "http://127.0.0.1:9999";

    const config = resolveEmbeddingConfig();
    expect(config.backend).toBe("mlx");
    expect(config.model).toBe("mlx-community/custom-model");
    expect(config.dims).toBe(512);
    expect(config.mlxUrl).toBe("http://127.0.0.1:9999");
    expect(config.failClosed).toBe(true); // MLX fails closed
  });

  test("caller options override environment variables", () => {
    process.env.QMD_EMBED_BACKEND = "gguf";
    process.env.QMD_EMBED_MODEL = "env-model";

    const config = resolveEmbeddingConfig({
      backend: "mlx",
      model: "caller-model",
      dims: 256,
      mlxUrl: "http://localhost:8000",
    });

    expect(config.backend).toBe("mlx");
    expect(config.model).toBe("caller-model");
    expect(config.dims).toBe(256);
    expect(config.mlxUrl).toBe("http://localhost:8000");
  });
});
