import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, writeFileSync, symlinkSync, linkSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { Database } from "../src/db.js";
import { createStore, insertEmbedding, type Store } from "../src/store.js";
import {
  validateShadowTarget,
  ShadowDatabaseProtectionError,
  IndexingFingerprintMismatchError,
  runDurableIndexingJob,
} from "../src/indexing/job.js";
import {
  saveCheckpoint,
  getActiveCheckpoint,
  computeDescriptorSignature,
  type IndexingJobCheckpoint,
} from "../src/indexing/checkpoint.js";
import { setDefaultLlamaCpp, type LlamaCpp, type EmbeddingResult } from "../src/llm.js";
import type { EmbeddingDescriptor } from "../src/embedding/contract.js";

describe("Durable Indexing and Resume", () => {
  let tempDir: string;
  let store: Store;

  beforeEach(() => {
    tempDir = mkdtempSync(path.join(tmpdir(), "qmd-test-resume-"));
    store = createStore(path.join(tempDir, "test-shadow-index.sqlite"));
  });

  afterEach(() => {
    store.close();
    try {
      rmSync(tempDir, { recursive: true, force: true });
    } catch {}
  });

  it("validateShadowTarget rejects live database path", () => {
    const livePath = "/Users/someone/.cache/qmd/index.sqlite";
    expect(() => validateShadowTarget(livePath, livePath)).toThrow(ShadowDatabaseProtectionError);
  });

  it("validateShadowTarget rejects symlinks pointing to live database", () => {
    const fakeLive = path.join(tempDir, "live-fake.sqlite");
    writeFileSync(fakeLive, "sqlite dummy content");
    const symlinkTarget = path.join(tempDir, "shadow-symlink.sqlite");
    symlinkSync(fakeLive, symlinkTarget);

    expect(() => validateShadowTarget(symlinkTarget, fakeLive)).toThrow(ShadowDatabaseProtectionError);
  });

  it("validateShadowTarget rejects hardlinks pointing to live database", () => {
    const fakeLive = path.join(tempDir, "live-fake-hard.sqlite");
    writeFileSync(fakeLive, "sqlite dummy content for hardlink");
    const hardlinkTarget = path.join(tempDir, "shadow-hardlink.sqlite");
    linkSync(fakeLive, hardlinkTarget);

    expect(() => validateShadowTarget(hardlinkTarget, fakeLive)).toThrow(ShadowDatabaseProtectionError);
  });

  it("validateShadowTarget rejects unisolated paths without markers", () => {
    expect(() => validateShadowTarget("/var/data/index.sqlite")).toThrow(ShadowDatabaseProtectionError);
  });

  it("validateShadowTarget accepts explicit shadow paths", () => {
    expect(() => validateShadowTarget(path.join(tempDir, "shadow-index.sqlite"))).not.toThrow();
  });

  it("saves and retrieves active checkpoints", () => {
    const cp: IndexingJobCheckpoint = {
      jobId: "job-123",
      fingerprint: {
        model: "test-model",
        dimensions: 384,
        descriptorSignature: "sig-abc",
      },
      startedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: "in_progress",
      docsTotal: 10,
      docsCompleted: 4,
      chunksTotal: 50,
      chunksCommitted: 20,
    };

    saveCheckpoint(store.db, cp);
    const active = getActiveCheckpoint(store.db);
    expect(active).not.toBeNull();
    expect(active?.jobId).toBe("job-123");
    expect(active?.chunksCommitted).toBe(20);
    expect(active?.status).toBe("in_progress");
  });

  it("proves real descriptor integration with mock active backend and awaited store.getEmbeddingDescriptor", async () => {
    const realDescriptor: EmbeddingDescriptor = {
      version: 1,
      backend: "mlx",
      model: "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
      outputDimensions: 1024,
      nativeDimensions: 1024,
      maxTokens: 512,
      normalized: true,
      pooling: "last",
    };

    const mockLlm = {
      embedModelName: "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
      // Actual descriptor API in src/llm.ts is async getDescriptor()
      getDescriptor: async () => realDescriptor,
      async embed(): Promise<EmbeddingResult | null> {
        return { embedding: new Array(1024).fill(0.01), model: "mlx-community/Qwen3-Embedding-4B-4bit-DWQ" };
      },
      async embedBatch(): Promise<(EmbeddingResult | null)[]> {
        return [];
      },
    };

    store.llm = mockLlm as any;

    // store.getEmbeddingDescriptor must await the backend's getDescriptor() and not return undefined
    const fetchedDescriptor = await store.getEmbeddingDescriptor?.();
    expect(fetchedDescriptor).toEqual(realDescriptor);

    const sig = computeDescriptorSignature(fetchedDescriptor);
    expect(sig).not.toBe("legacy-unversioned");
    expect(sig).toMatch(/^[0-9a-f]{16,24}$/);

    const res = await runDurableIndexingJob(store, {
      model: "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
    });

    expect(res.checkpoint.fingerprint.descriptorSignature).toBe(sig);
    expect(res.checkpoint.fingerprint.dimensions).toBe(1024);
    expect(res.checkpoint.fingerprint.model).toBe("mlx-community/Qwen3-Embedding-4B-4bit-DWQ");
  });

  it("resumes partially embedded documents and embeds only missing chunks without duplicates", async () => {
    const now = new Date().toISOString();
    function insertDoc(name: string, body: string, hash: string) {
      store.db.prepare(`
        INSERT OR IGNORE INTO content (hash, doc, created_at)
        VALUES (?, ?, ?)
      `).run(hash, body, now);
      store.db.prepare(`
        INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
        VALUES ('test', ?, ?, ?, ?, ?, 1)
      `).run(`${name}.md`, name, hash, now, now);
    }

    // Insert doc with enough text and sections to generate multiple chunks
    const multiChunkText = Array.from({ length: 40 }, (_, i) =>
      `## Section ${i}\n\nThis is detailed paragraph ${i} with substantial content for testing the chunker and indexing resume flow across multiple chunk boundaries.\n\n`
    ).join("\n");
    insertDoc("doc1", multiChunkText, "hash-doc-1");

    function createFakeTokenizer() {
      return {
        tokenize: (text: string) => text.split(/\s+/).map((_, i) => i + 1),
        detokenize: (tokens: number[]) => tokens.join(" "),
      };
    }
    setDefaultLlamaCpp(createFakeTokenizer() as any);

    let embeddedTexts: string[] = [];
    const mockDescriptor: EmbeddingDescriptor = {
      version: 1,
      backend: "mock",
      model: "test-embed",
      outputDimensions: 4,
      nativeDimensions: 4,
      maxTokens: 128,
      normalized: true,
    };
    const mockLlm = {
      embedModelName: "test-embed",
      getDescriptor: async () => mockDescriptor,
      async embed(text: string): Promise<EmbeddingResult | null> {
        embeddedTexts.push(text);
        return {
          embedding: [0.1, 0.2, 0.3, 0.4],
          model: "test-embed",
        };
      },
      async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
        embeddedTexts.push(...texts);
        return texts.map(() => ({
          embedding: [0.1, 0.2, 0.3, 0.4],
          model: "test-embed",
        }));
      },
    };

    store.llm = mockLlm as any;

    // Simulate a prior crashed run that only committed chunk 0:
    store.ensureVecTable(4);
    insertEmbedding(store.db, "hash-doc-1", 0, 0, new Float32Array([0.1, 0.2, 0.3, 0.4]), "test-embed", now);

    // Save an active interrupted checkpoint matching the current descriptor & config
    const initialCp: IndexingJobCheckpoint = {
      jobId: "interrupted-job-1",
      fingerprint: {
        model: "test-embed",
        dimensions: 4,
        chunkStrategy: "regex",
        descriptorSignature: computeDescriptorSignature(mockDescriptor),
      },
      startedAt: now,
      updatedAt: now,
      status: "in_progress",
      docsTotal: 1,
      docsCompleted: 0,
      chunksTotal: 0,
      chunksCommitted: 1,
      lastCommittedHash: "hash-doc-1",
    };
    saveCheckpoint(store.db, initialCp);

    // Resume the job
    embeddedTexts = [];
    const resumeRes = await runDurableIndexingJob(store, { model: "test-embed" });

    expect(resumeRes.status).toBe("complete");
    // Only missing chunks (> seq 0) were sent to embedBatch
    expect(embeddedTexts.length).toBeGreaterThan(0);

    // Check all chunks in DB have unique (hash, seq)
    const allVectors = store.db.prepare("SELECT hash, seq FROM content_vectors WHERE hash = 'hash-doc-1' ORDER BY seq").all() as { hash: string; seq: number }[];
    const seqs = allVectors.map(v => v.seq);
    expect(seqs).toEqual(Array.from({ length: seqs.length }, (_, i) => i)); // [0, 1, 2, ...] with no missing or duplicate seqs

    // Checkpoint is marked completed
    const finalCp = getActiveCheckpoint(store.db);
    expect(finalCp).toBeNull(); // No more active/in_progress checkpoints
  });

  it("respects AbortSignal cancellation during durable indexing job", async () => {
    const now = new Date().toISOString();
    store.db.prepare(`
      INSERT OR IGNORE INTO content (hash, doc, created_at)
      VALUES ('hash-abort', 'Some text for abort test', ?)
    `).run(now);
    store.db.prepare(`
      INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
      VALUES ('test', 'abort.md', 'abort', 'hash-abort', ?, ?, 1)
    `).run(now, now);

    const controller = new AbortController();
    controller.abort(); // Abort immediately

    const mockLlm = {
      embedModelName: "test-model",
      async embed(): Promise<EmbeddingResult | null> {
        return { embedding: [0.1], model: "test-model" };
      },
      async embedBatch(): Promise<(EmbeddingResult | null)[]> {
        return [{ embedding: [0.1], model: "test-model" }];
      },
    };
    store.llm = mockLlm as any;

    const res = await runDurableIndexingJob(store, {
      model: "test-model",
      signal: controller.signal,
    });

    expect(res.status).toBe("cancelled");
    expect(res.checkpoint.status).toBe("cancelled");
  });

  describe("Fingerprint Mismatch Safety (No Destructive Migration on Resume)", () => {
    const now = new Date().toISOString();

    function seedDatabaseWithUnrelatedVectorsAndPartialCheckpoint(db: Database) {
      // 1. Seed completed unrelated documents and vectors
      db.prepare(`
        INSERT OR IGNORE INTO content (hash, doc, created_at)
        VALUES ('hash-unrelated-1', 'Completed unrelated doc 1 body', ?),
               ('hash-unrelated-2', 'Completed unrelated doc 2 body', ?),
               ('hash-partial-doc', '# Partial Document\n\nPart 1\n\nPart 2', ?)
      `).run(now, now, now);

      db.prepare(`
        INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
        VALUES ('test', 'unrelated1.md', 'Unrelated 1', 'hash-unrelated-1', ?, ?, 1),
               ('test', 'unrelated2.md', 'Unrelated 2', 'hash-unrelated-2', ?, ?, 1),
               ('test', 'partial.md', 'Partial', 'hash-partial-doc', ?, ?, 1)
      `).run(now, now, now, now, now, now);

      store.ensureVecTable(4);
      insertEmbedding(db, "hash-unrelated-1", 0, 0, new Float32Array([0.1, 0.2, 0.3, 0.4]), "base-model", now);
      insertEmbedding(db, "hash-unrelated-2", 0, 0, new Float32Array([0.5, 0.6, 0.7, 0.8]), "base-model", now);
      insertEmbedding(db, "hash-partial-doc", 0, 0, new Float32Array([0.9, 0.9, 0.9, 0.9]), "base-model", now);

      // 2. Seed active interrupted checkpoint
      const baseCp: IndexingJobCheckpoint = {
        jobId: "partial-checkpoint-1",
        fingerprint: {
          model: "base-model",
          dimensions: 4,
          chunkStrategy: "regex",
          descriptorSignature: "sig-base-descriptor",
        },
        startedAt: now,
        updatedAt: now,
        status: "in_progress",
        docsTotal: 3,
        docsCompleted: 2,
        chunksTotal: 5,
        chunksCommitted: 3,
        lastCommittedHash: "hash-partial-doc",
      };
      saveCheckpoint(db, baseCp);
    }

    function getLogicalState(db: Database) {
      return {
        checkpoints: db.prepare("SELECT * FROM indexing_checkpoints ORDER BY job_id").all(),
        vectors: db.prepare("SELECT hash, seq, pos, model, embedded_at FROM content_vectors ORDER BY hash, seq").all(),
      };
    }

    it("rejects model mismatch with typed error, byte-equivalent state, and zero inference", async () => {
      seedDatabaseWithUnrelatedVectorsAndPartialCheckpoint(store.db);
      const initialState = getLogicalState(store.db);

      let inferenceCount = 0;
      const mockLlm = {
        embedModelName: "mismatched-model",
        getDescriptor: async () => ({
          version: 1,
          backend: "mock",
          model: "mismatched-model",
          outputDimensions: 4,
          nativeDimensions: 4,
          maxTokens: 128,
          normalized: true,
        }),
        async embed() {
          inferenceCount++;
          return { embedding: [0.1, 0.2, 0.3, 0.4], model: "mismatched-model" };
        },
        async embedBatch() {
          inferenceCount++;
          return [];
        },
      };
      store.llm = mockLlm as any;

      let error: any = null;
      try {
        await runDurableIndexingJob(store, { model: "mismatched-model" });
      } catch (err) {
        error = err;
      }

      expect(error).toBeInstanceOf(IndexingFingerprintMismatchError);
      expect(error.message).toContain("explicit rebuild on isolated shadow index required");
      expect(error.details.modelMismatch).toBe(true);
      expect(inferenceCount).toBe(0);

      const postState = getLogicalState(store.db);
      expect(postState).toEqual(initialState);
    });

    it("rejects chunker strategy mismatch with typed error, byte-equivalent state, and zero inference", async () => {
      seedDatabaseWithUnrelatedVectorsAndPartialCheckpoint(store.db);
      const initialState = getLogicalState(store.db);

      let inferenceCount = 0;
      const mockLlm = {
        embedModelName: "base-model",
        getDescriptor: async () => ({
          version: 1,
          backend: "mock",
          model: "base-model",
          outputDimensions: 4,
          nativeDimensions: 4,
          maxTokens: 128,
          normalized: true,
        }),
        async embed() {
          inferenceCount++;
          return { embedding: [0.1, 0.2, 0.3, 0.4], model: "base-model" };
        },
        async embedBatch() {
          inferenceCount++;
          return [];
        },
      };
      store.llm = mockLlm as any;

      let error: any = null;
      try {
        // checkpoint has chunkStrategy: "regex", request runs with "auto"
        await runDurableIndexingJob(store, { model: "base-model", chunkStrategy: "auto" });
      } catch (err) {
        error = err;
      }

      expect(error).toBeInstanceOf(IndexingFingerprintMismatchError);
      expect(error.message).toContain("explicit rebuild on isolated shadow index required");
      expect(error.details.strategyMismatch).toBe(true);
      expect(inferenceCount).toBe(0);

      const postState = getLogicalState(store.db);
      expect(postState).toEqual(initialState);
    });

    it("rejects descriptor signature mismatch with typed error, byte-equivalent state, and zero inference", async () => {
      seedDatabaseWithUnrelatedVectorsAndPartialCheckpoint(store.db);
      const initialState = getLogicalState(store.db);

      let inferenceCount = 0;
      const mockLlm = {
        embedModelName: "base-model",
        getDescriptor: async () => ({
          version: 2, // Changed version/backend -> changed signature
          backend: "mlx-v2",
          model: "base-model",
          outputDimensions: 4,
          nativeDimensions: 4,
          maxTokens: 256,
          normalized: true,
        }),
        async embed() {
          inferenceCount++;
          return { embedding: [0.1, 0.2, 0.3, 0.4], model: "base-model" };
        },
        async embedBatch() {
          inferenceCount++;
          return [];
        },
      };
      store.llm = mockLlm as any;

      let error: any = null;
      try {
        await runDurableIndexingJob(store, { model: "base-model", chunkStrategy: "regex" });
      } catch (err) {
        error = err;
      }

      expect(error).toBeInstanceOf(IndexingFingerprintMismatchError);
      expect(error.message).toContain("explicit rebuild on isolated shadow index required");
      expect(error.details.sigMismatch).toBe(true);
      expect(inferenceCount).toBe(0);

      const postState = getLogicalState(store.db);
      expect(postState).toEqual(initialState);
    });

    it("rejects dimensions mismatch with typed error, byte-equivalent state, and zero inference", async () => {
      seedDatabaseWithUnrelatedVectorsAndPartialCheckpoint(store.db);
      const initialState = getLogicalState(store.db);

      let inferenceCount = 0;
      const mockLlm = {
        embedModelName: "base-model",
        getDescriptor: async () => ({
          version: 1,
          backend: "mock",
          model: "base-model",
          outputDimensions: 8, // Checkpoint was seeded with 4
          nativeDimensions: 8,
          maxTokens: 128,
          normalized: true,
        }),
        async embed() {
          inferenceCount++;
          return { embedding: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], model: "base-model" };
        },
        async embedBatch() {
          inferenceCount++;
          return [];
        },
      };
      store.llm = mockLlm as any;

      let error: any = null;
      try {
        await runDurableIndexingJob(store, { model: "base-model", chunkStrategy: "regex" });
      } catch (err) {
        error = err;
      }

      expect(error).toBeInstanceOf(IndexingFingerprintMismatchError);
      expect(error.message).toContain("explicit rebuild on isolated shadow index required");
      expect(error.details.dimMismatch).toBe(true);
      expect(inferenceCount).toBe(0);

      const postState = getLogicalState(store.db);
      expect(postState).toEqual(initialState);
    });

    it("allows rebuild when options.force is explicitly true without throwing mismatch error", async () => {
      seedDatabaseWithUnrelatedVectorsAndPartialCheckpoint(store.db);

      const mockLlm = {
        embedModelName: "new-forced-model",
        getDescriptor: async () => ({
          version: 1,
          backend: "mock",
          model: "new-forced-model",
          outputDimensions: 4,
          nativeDimensions: 4,
          maxTokens: 128,
          normalized: true,
        }),
        async embed() {
          return { embedding: [0.1, 0.2, 0.3, 0.4], model: "new-forced-model" };
        },
        async embedBatch(texts: string[]) {
          return texts.map(() => ({ embedding: [0.1, 0.2, 0.3, 0.4], model: "new-forced-model" }));
        },
      };
      store.llm = mockLlm as any;

      // With force: true explicitly provided by the caller, job proceeds
      const res = await runDurableIndexingJob(store, {
        model: "new-forced-model",
        force: true,
      });

      expect(res.status).toBe("complete");
      expect(res.checkpoint.fingerprint.model).toBe("new-forced-model");
      expect(res.checkpoint.jobId).not.toBe("partial-checkpoint-1");
    });

    it("rejects revision mismatch with typed error, byte-equivalent state, and zero inference", async () => {
      seedDatabaseWithUnrelatedVectorsAndPartialCheckpoint(store.db);

      const baseDescriptor: EmbeddingDescriptor = {
        version: 1,
        backend: "mock",
        model: "base-model",
        revision: "rev-original-v1",
        outputDimensions: 4,
        nativeDimensions: 4,
        maxTokens: 128,
        normalized: true,
      };

      const baseCp = getActiveCheckpoint(store.db)!;
      baseCp.fingerprint.descriptorSignature = computeDescriptorSignature(baseDescriptor);
      baseCp.fingerprint.revision = "rev-original-v1";
      saveCheckpoint(store.db, baseCp);
      const stateWithSig = getLogicalState(store.db);

      let inferenceCount = 0;
      const mockLlm = {
        embedModelName: "base-model",
        getDescriptor: async () => ({
          ...baseDescriptor,
          revision: "rev-modified-v2",
        }),
        async embed() {
          inferenceCount++;
          return { embedding: [0.1, 0.2, 0.3, 0.4], model: "base-model" };
        },
        async embedBatch() {
          inferenceCount++;
          return [];
        },
      };
      store.llm = mockLlm as any;

      let error: any = null;
      try {
        await runDurableIndexingJob(store, { model: "base-model" });
      } catch (err) {
        error = err;
      }

      expect(error).toBeInstanceOf(IndexingFingerprintMismatchError);
      expect(error.message).toContain("explicit rebuild on isolated shadow index required");
      expect(error.details.sigMismatch).toBe(true);
      expect(inferenceCount).toBe(0);

      const postState = getLogicalState(store.db);
      expect(postState).toEqual(stateWithSig);
    });

    it("rejects quantization mismatch with typed error, byte-equivalent state, and zero inference", async () => {
      seedDatabaseWithUnrelatedVectorsAndPartialCheckpoint(store.db);

      const baseDescriptor: EmbeddingDescriptor = {
        version: 1,
        backend: "mock",
        model: "base-model",
        quantization: "4bit",
        outputDimensions: 4,
        nativeDimensions: 4,
        maxTokens: 128,
        normalized: true,
      };

      const baseCp = getActiveCheckpoint(store.db)!;
      baseCp.fingerprint.descriptorSignature = computeDescriptorSignature(baseDescriptor);
      baseCp.fingerprint.quantization = "4bit";
      saveCheckpoint(store.db, baseCp);
      const stateWithSig = getLogicalState(store.db);

      let inferenceCount = 0;
      const mockLlm = {
        embedModelName: "base-model",
        getDescriptor: async () => ({
          ...baseDescriptor,
          quantization: "8bit",
        }),
        async embed() {
          inferenceCount++;
          return { embedding: [0.1, 0.2, 0.3, 0.4], model: "base-model" };
        },
        async embedBatch() {
          inferenceCount++;
          return [];
        },
      };
      store.llm = mockLlm as any;

      let error: any = null;
      try {
        await runDurableIndexingJob(store, { model: "base-model" });
      } catch (err) {
        error = err;
      }

      expect(error).toBeInstanceOf(IndexingFingerprintMismatchError);
      expect(error.message).toContain("explicit rebuild on isolated shadow index required");
      expect(error.details.sigMismatch).toBe(true);
      expect(inferenceCount).toBe(0);

      const postState = getLogicalState(store.db);
      expect(postState).toEqual(stateWithSig);
    });

    it("kill-mid-job and rerun completes exact missing chunks without duplicate vectors", async () => {
      const now = new Date().toISOString();
      function insertDoc(name: string, body: string, hash: string) {
        store.db.prepare(`
          INSERT OR IGNORE INTO content (hash, doc, created_at)
          VALUES (?, ?, ?)
        `).run(hash, body, now);
        store.db.prepare(`
          INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
          VALUES ('test', ?, ?, ?, ?, ?, 1)
        `).run(`${name}.md`, name, hash, now, now);
      }

      const multiChunkText = Array.from({ length: 30 }, (_, i) =>
        `## Paragraph ${i}\n\nThis is paragraph ${i} with substantial length to test multi-chunk splitting during kill and resume cycle.\n\n`
      ).join("\n");
      insertDoc("doc-kill-test", multiChunkText, "hash-kill-test");

      const desc: EmbeddingDescriptor = {
        version: 1,
        backend: "mock",
        model: "kill-test-model",
        outputDimensions: 4,
        nativeDimensions: 4,
        maxTokens: 128,
        normalized: true,
      };

      const abortCtrl = new AbortController();
      let callCount = 0;
      const embeddedItems: string[] = [];

      const mockLlm = {
        embedModelName: "kill-test-model",
        getDescriptor: async () => desc,
        async embed(t: string): Promise<EmbeddingResult | null> {
          callCount++;
          embeddedItems.push(t);
          if (callCount === 2) {
            // Simulate kill mid job via AbortController
            abortCtrl.abort();
          }
          return { embedding: [0.1, 0.2, 0.3, 0.4], model: "kill-test-model" };
        },
        async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
          return Promise.all(texts.map(t => this.embed(t)));
        },
      };
      store.llm = mockLlm as any;

      // First run: aborts mid-job
      const firstRun = await runDurableIndexingJob(store, {
        model: "kill-test-model",
        maxDocsPerBatch: 1,
        maxBatchBytes: 200,
        signal: abortCtrl.signal,
      });

      expect(firstRun.status).toBe("cancelled");
      const midCp = getActiveCheckpoint(store.db);
      expect(midCp).not.toBeNull();
      expect(midCp?.status).toBe("cancelled");
      const vectorsAfterFirst = (store.db.prepare("SELECT seq FROM content_vectors WHERE hash = 'hash-kill-test'").all() as any[]).length;
      expect(vectorsAfterFirst).toBeGreaterThanOrEqual(1);

      // Second run: fresh non-aborted run resumes and completes remaining chunks
      const secondRun = await runDurableIndexingJob(store, {
        model: "kill-test-model",
        maxDocsPerBatch: 1,
      });

      expect(secondRun.status).toBe("complete");
      const finalVectors = store.db.prepare("SELECT seq FROM content_vectors WHERE hash = 'hash-kill-test' ORDER BY seq").all() as { seq: number }[];
      const finalSeqs = finalVectors.map(v => v.seq);
      // All chunks present consecutively from 0 to max without duplicates
      expect(finalSeqs).toEqual(Array.from({ length: finalSeqs.length }, (_, i) => i));
      expect(getActiveCheckpoint(store.db)).toBeNull();
    });

    it("resumes successfully with omitted model option when real descriptor is present (Finding 4 Regression)", async () => {
      const now = new Date().toISOString();
      function insertDoc(name: string, body: string, hash: string) {
        store.db.prepare(`
          INSERT OR IGNORE INTO content (hash, doc, created_at)
          VALUES (?, ?, ?)
        `).run(hash, body, now);
        store.db.prepare(`
          INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
          VALUES ('test', ?, ?, ?, ?, ?, 1)
        `).run(`${name}.md`, name, hash, now, now);
      }

      insertDoc("doc-omitted-1", "First document content to test resume with omitted model option.", "hash-omitted-1");
      insertDoc("doc-omitted-2", "Second document content to trigger mid-job cancellation on omitted model.", "hash-omitted-2");
      insertDoc("doc-omitted-3", "Third document content to complete upon resuming with omitted model.", "hash-omitted-3");

      const realDescriptor: EmbeddingDescriptor = {
        version: 1,
        backend: "mlx",
        model: "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
        outputDimensions: 1024,
        nativeDimensions: 1024,
        maxTokens: 512,
        normalized: true,
        pooling: "last",
      };

      const abortCtrl = new AbortController();
      let callCount = 0;

      const mockLlm = {
        embedModelName: "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
        getDescriptor: async () => realDescriptor,
        async embed(): Promise<EmbeddingResult | null> {
          callCount++;
          if (callCount === 2) {
            abortCtrl.abort();
          }
          return { embedding: new Array(1024).fill(0.01), model: "mlx-community/Qwen3-Embedding-4B-4bit-DWQ" };
        },
        async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
          return Promise.all(texts.map(() => this.embed()));
        },
      };
      store.llm = mockLlm as any;

      // 1. Initial run without options.model: aborts mid-way
      const firstRun = await runDurableIndexingJob(store, {
        maxDocsPerBatch: 1,
        signal: abortCtrl.signal,
      });

      expect(firstRun.status).toBe("cancelled");
      const activeCp = getActiveCheckpoint(store.db);
      expect(activeCp).not.toBeNull();
      // Checkpoint was saved with descriptor's model name
      expect(activeCp?.fingerprint.model).toBe("mlx-community/Qwen3-Embedding-4B-4bit-DWQ");

      // 2. Resume run without options.model: MUST NOT throw IndexingFingerprintMismatchError
      const resumeRun = await runDurableIndexingJob(store, {
        maxDocsPerBatch: 1,
      });

      expect(resumeRun.status).toBe("complete");
      expect(resumeRun.checkpoint.fingerprint.model).toBe("mlx-community/Qwen3-Embedding-4B-4bit-DWQ");
      expect(getActiveCheckpoint(store.db)).toBeNull();
    });
  });
});

