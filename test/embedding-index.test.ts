/**
 * embedding-index.test.ts - Unit and integration tests for embedding space identity,
 * index safety, progressive overfetch, and probe reuse in store.ts.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { openDatabase, loadSqliteVec } from "../src/db.js";
import type { Database } from "../src/db.js";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createStore,
  searchVec,
  generateEmbeddings,
  type Store,
} from "../src/store.js";
import { setDefaultLlamaCpp } from "../src/llm.js";
import {
  type EmbeddingDescriptor,
  computeEmbeddingSpaceId,
} from "../src/embedding/contract.js";

describe("Embedding Index Identity & Safety", () => {
  let tempDir: string;
  let dbPath: string;
  let store: Store;
  let db: Database;

  const nomicDescriptor: EmbeddingDescriptor = {
    version: 1,
    backend: "mlx",
    model: "mlx-community/nomic-embed-text-v1.5",
    pooling: "mean",
    nativeDimensions: 768,
    outputDimensions: 768,
    maxTokens: 2048,
    normalized: true,
  };

  const bgeDescriptor: EmbeddingDescriptor = {
    version: 1,
    backend: "mlx",
    model: "BAAI/bge-small-en-v1.5",
    pooling: "cls",
    nativeDimensions: 384,
    outputDimensions: 384,
    maxTokens: 512,
    normalized: true,
  };

  const qwenDescriptor: EmbeddingDescriptor = {
    version: 1,
    backend: "mlx",
    model: "Qwen/Qwen3-Embedding-0.6B",
    pooling: "last_token",
    nativeDimensions: 768,
    outputDimensions: 768,
    maxTokens: 2048,
    normalized: true,
  };

  beforeEach(async () => {
    tempDir = await mkdtemp(join(tmpdir(), "qmd-embed-test-"));
    dbPath = join(tempDir, "test.sqlite");
    store = createStore(dbPath);
    db = openDatabase(dbPath);
    loadSqliteVec(db);
  });

  afterEach(async () => {
    try {
      db.close();
      store.close();
    } catch {}
    try {
      await rm(tempDir, { recursive: true, force: true });
    } catch {}
  });

  test("initializes vector table and records embedding space metadata in store_config", () => {
    store.ensureVecTable(768, nomicDescriptor);

    const spaceRow = db.prepare("SELECT value FROM store_config WHERE key = 'embedding_space_id'").get() as { value: string };
    const modelRow = db.prepare("SELECT value FROM store_config WHERE key = 'embedding_model'").get() as { value: string };
    const descRow = db.prepare("SELECT value FROM store_config WHERE key = 'embedding_descriptor'").get() as { value: string };

    const expectedSpaceId = computeEmbeddingSpaceId(nomicDescriptor);
    expect(spaceRow?.value).toBe(expectedSpaceId);
    expect(modelRow?.value).toBe("mlx-community/nomic-embed-text-v1.5");
    expect(JSON.parse(descRow?.value)).toEqual(nomicDescriptor);
  });

  test("allows subsequent ensureVecTable calls with the same descriptor", () => {
    store.ensureVecTable(768, nomicDescriptor);
    expect(() => store.ensureVecTable(768, nomicDescriptor)).not.toThrow();
  });

  test("rejects ensureVecTable when embedding space mismatch occurs (different model, same dims)", () => {
    store.ensureVecTable(768, nomicDescriptor);

    // qwenDescriptor has 768 dims but different model name and pooling
    expect(() => store.ensureVecTable(768, qwenDescriptor)).toThrow(/Embedding space mismatch/);
    expect(() => store.ensureVecTable(768, qwenDescriptor)).toThrow(/qmd embed -f/);
  });

  test("rejects ensureVecTable when embedding dimension mismatch occurs", () => {
    store.ensureVecTable(768, nomicDescriptor);

    // bgeDescriptor has 384 dims
    expect(() => store.ensureVecTable(384, bgeDescriptor)).toThrow(/Embedding space mismatch/);
  });

  test("re-creates vector table when force cleared", () => {
    store.ensureVecTable(768, nomicDescriptor);

    // Simulate force clear by dropping vector table and deleting config
    db.exec("DROP TABLE IF EXISTS vectors_vec");
    db.exec("DELETE FROM store_config WHERE key LIKE 'embedding_%'");

    // Now re-initializing with Qwen descriptor succeeds
    expect(() => store.ensureVecTable(768, qwenDescriptor)).not.toThrow();

    const spaceRow = db.prepare("SELECT value FROM store_config WHERE key = 'embedding_space_id'").get() as { value: string };
    expect(spaceRow?.value).toBe(computeEmbeddingSpaceId(qwenDescriptor));
  });
});

describe("Progressive Overfetch in searchVec", () => {
  let tempDir: string;
  let dbPath: string;
  let store: Store;
  let db: Database;

  beforeEach(async () => {
    tempDir = await mkdtemp(join(tmpdir(), "qmd-overfetch-test-"));
    dbPath = join(tempDir, "test.sqlite");
    store = createStore(dbPath);
    db = openDatabase(dbPath);
    loadSqliteVec(db);
  });

  afterEach(async () => {
    try {
      db.close();
      store.close();
    } catch {}
    try {
      await rm(tempDir, { recursive: true, force: true });
    } catch {}
  });

  test("finds scoped collection documents when top global KNN results belong to other collections", async () => {
    const dims = 4;
    store.ensureVecTable(dims);

    const now = new Date().toISOString();

    // Query vector is [1.0, 0.0, 0.0, 0.0]
    const queryVec = [1.0, 0.0, 0.0, 0.0];

    // Insert 30 docs into other_col very close to queryVec: angle ~ 0.01..0.3
    const insertDoc = db.prepare("INSERT INTO documents (collection, path, title, hash, active, created_at, modified_at) VALUES (?, ?, ?, ?, 1, ?, ?)");
    const insertContent = db.prepare("INSERT INTO content (hash, doc, created_at) VALUES (?, ?, ?)");
    const insertCV = db.prepare("INSERT INTO content_vectors (hash, seq, pos, model, embedded_at) VALUES (?, 0, 0, 'test', ?)");
    const insertVec = db.prepare("INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)");

    for (let i = 0; i < 30; i++) {
      const hash = `other_hash_${i}`;
      insertContent.run(hash, `Other doc ${i}`, now);
      insertDoc.run("other_col", `doc_${i}.md`, `Other Doc ${i}`, hash, now, now);
      insertCV.run(hash, now);
      const angle = 0.01 * (i + 1);
      const vec = [Math.cos(angle), Math.sin(angle), 0.0, 0.0];
      insertVec.run(`${hash}_0`, new Float32Array(vec));
    }

    // Insert 3 docs into target_col slightly further from queryVec: angle ~ 0.5..0.52
    for (let i = 0; i < 3; i++) {
      const hash = `target_hash_${i}`;
      insertContent.run(hash, `Target doc ${i}`, now);
      insertDoc.run("target_col", `target_${i}.md`, `Target Doc ${i}`, hash, now, now);
      insertCV.run(hash, now);
      const angle = 0.5 + 0.01 * i;
      const vec = [Math.cos(angle), Math.sin(angle), 0.0, 0.0];
      insertVec.run(`${hash}_0`, new Float32Array(vec));
    }

    // Search with limit=3 scoped to target_col
    // Top 30 global KNN vectors all belong to other_col.
    // Without progressive overfetch (limit*3 = 9), searchVec would find 0 results.
    // With progressive overfetch, it expands fetchK and retrieves all 3 target docs.
    const results = await searchVec(db, "test query", "dummy-model", 3, "target_col", undefined, queryVec);

    expect(results.length).toBe(3);
    for (const r of results) {
      expect(r.collectionName).toBe("target_col");
      expect(r.filepath).toMatch(/^qmd:\/\/target_col\//);
    }
  });
});

describe("Probe and descriptor initialization in generateEmbeddings", () => {
  let tempDir: string;
  let dbPath: string;
  let store: Store;
  let db: Database;

  beforeEach(async () => {
    tempDir = await mkdtemp(join(tmpdir(), "qmd-probe-test-"));
    dbPath = join(tempDir, "test.sqlite");
    store = createStore(dbPath);
    db = openDatabase(dbPath);
    loadSqliteVec(db);
  });

  afterEach(async () => {
    setDefaultLlamaCpp(null);
    try {
      db.close();
      store.close();
    } catch {}
    try {
      await rm(tempDir, { recursive: true, force: true });
    } catch {}
  });

  test("probes dimensions on uninitialized table when descriptor is absent", async () => {
    const now = new Date().toISOString();
    // Insert 2 test docs
    db.prepare("INSERT INTO content (hash, doc, created_at) VALUES (?, ?, ?)").run("hash_1", "Doc 1 text", now);
    db.prepare("INSERT INTO documents (collection, path, title, hash, active, created_at, modified_at) VALUES (?, ?, ?, ?, 1, ?, ?)").run("docs", "doc1.md", "Doc 1", "hash_1", now, now);

    db.prepare("INSERT INTO content (hash, doc, created_at) VALUES (?, ?, ?)").run("hash_2", "Doc 2 text", now);
    db.prepare("INSERT INTO documents (collection, path, title, hash, active, created_at, modified_at) VALUES (?, ?, ?, ?, 1, ?, ?)").run("docs", "doc2.md", "Doc 2", "hash_2", now, now);

    let embedCallCount = 0;
    let embedBatchCallCount = 0;
    let totalTextsBatchEmbedded = 0;

    const fakeLlm = {
      async embed(text: string) {
        embedCallCount++;
        return { embedding: [0.1, 0.2, 0.3, 0.4], model: "test-model" };
      },
      async embedBatch(texts: string[]) {
        embedBatchCallCount++;
        totalTextsBatchEmbedded += texts.length;
        return texts.map(() => ({ embedding: [0.1, 0.2, 0.3, 0.4], model: "test-model" }));
      },
      async tokenize(text: string) {
        return new Array(Math.max(1, Math.ceil(text.length / 16))).fill(1);
      },
      async findContextSize() { return 2048; },
      async isModelAvailable() { return true; },
      async acquireContextSession() { return {} as any; },
      releaseContextSession() {},
    };

    setDefaultLlamaCpp(fakeLlm as any);
    store.llm = fakeLlm as any;

    const res = await generateEmbeddings(store);

    expect(res.chunksEmbedded).toBe(2);
    // Probe called once for chunk 0 to determine dimension
    expect(embedCallCount).toBe(1);
    // Batch embedding executes for all chunks
    expect(embedBatchCallCount).toBe(1);
    expect(totalTextsBatchEmbedded).toBe(2);

    const cvCount = db.prepare("SELECT COUNT(*) as count FROM content_vectors").get() as { count: number };
    expect(cvCount.count).toBe(2);
  });

  test("skips probe call when descriptor is available from session", async () => {
    const now = new Date().toISOString();
    db.prepare("INSERT INTO content (hash, doc, created_at) VALUES (?, ?, ?)").run("hash_1", "Doc 1 text", now);
    db.prepare("INSERT INTO documents (collection, path, title, hash, active, created_at, modified_at) VALUES (?, ?, ?, ?, 1, ?, ?)").run("docs", "doc1.md", "Doc 1", "hash_1", now, now);

    let embedCallCount = 0;
    let embedBatchCallCount = 0;

    const fakeLlmWithDescriptor = {
      async embed(text: string) {
        embedCallCount++;
        return { embedding: [0.1, 0.2, 0.3, 0.4], model: "test-model" };
      },
      async embedBatch(texts: string[]) {
        embedBatchCallCount++;
        return texts.map(() => ({ embedding: [0.1, 0.2, 0.3, 0.4], model: "test-model" }));
      },
      async tokenize(text: string) {
        return new Array(Math.max(1, Math.ceil(text.length / 16))).fill(1);
      },
      async findContextSize() { return 2048; },
      async isModelAvailable() { return true; },
      async getDescriptor(): Promise<EmbeddingDescriptor> {
        return {
          version: 1,
          backend: "mlx" as const,
          model: "test-model",
          pooling: "mean" as const,
          nativeDimensions: 4,
          outputDimensions: 4,
          maxTokens: 2048,
          normalized: true,
        };
      },
      async acquireContextSession() {
        return {} as any;
      },
      releaseContextSession() {},
    };

    setDefaultLlamaCpp(fakeLlmWithDescriptor as any);
    store.llm = fakeLlmWithDescriptor as any;

    const res = await generateEmbeddings(store);

    expect(res.chunksEmbedded).toBe(1);
    // Zero probe calls since descriptor was known upfront
    expect(embedCallCount).toBe(0);
    expect(embedBatchCallCount).toBe(1);
  });
});
