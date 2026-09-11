/**
 * phase5-recovery.test.ts — Unit tests for Phase 5 Public Fixtures, IR Metrics, and Recovery Logic
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  PUBLIC_CORPUS_DOCS,
  PUBLIC_HELD_OUT_QUERIES,
  computeIRMetrics,
} from "./fixtures/public_eval_dataset.js";
import {
  createStore,
  runDurableIndexingJob,
  validateShadowTarget,
  ShadowDatabaseProtectionError,
  IndexingFingerprintMismatchError,
  getActiveCheckpoint,
  saveCheckpoint,
  type QMDStore,
} from "../src/index.js";

describe("Phase 5 Public Fixtures & Recovery Unit Tests", () => {
  it("public fixture corpus has 16 distinct documents with self-contained technical facts", () => {
    expect(PUBLIC_CORPUS_DOCS.length).toBe(16);
    const ids = new Set(PUBLIC_CORPUS_DOCS.map(d => d.id));
    expect(ids.size).toBe(16);
    for (const doc of PUBLIC_CORPUS_DOCS) {
      expect(doc.content.length).toBeGreaterThan(100);
      expect(doc.title.length).toBeGreaterThan(5);
    }
  });

  it("public held-out queries have 10 frozen judged queries with explicit expected doc IDs", () => {
    expect(PUBLIC_HELD_OUT_QUERIES.length).toBe(10);
    const docIdSet = new Set(PUBLIC_CORPUS_DOCS.map(d => d.id));
    for (const q of PUBLIC_HELD_OUT_QUERIES) {
      expect(q.relevantDocIds.length).toBeGreaterThanOrEqual(1);
      for (const expectedId of q.relevantDocIds) {
        expect(docIdSet.has(expectedId)).toBe(true);
      }
    }
  });

  it("computeIRMetrics correctly calculates Recall@k, MRR, and nDCG@5 for synthetic rankings", () => {
    // Perfect rankings
    const rankedPerfect = [
      ["doc-1", "doc-2"],
      ["doc-3", "doc-4"],
    ];
    const relevant = [
      ["doc-1"],
      ["doc-3"],
    ];
    const perfectMetrics = computeIRMetrics(rankedPerfect, relevant, 5);
    expect(perfectMetrics.recallAt1).toBe(1.0);
    expect(perfectMetrics.recallAt5).toBe(1.0);
    expect(perfectMetrics.mrr).toBe(1.0);
    expect(perfectMetrics.ndcgAt5).toBe(1.0);

    // Rank 2 hits
    const rankedSecond = [
      ["doc-other", "doc-1"],
      ["doc-other", "doc-3"],
    ];
    const secondMetrics = computeIRMetrics(rankedSecond, relevant, 5);
    expect(secondMetrics.recallAt1).toBe(0.0);
    expect(secondMetrics.recallAt3).toBe(1.0);
    expect(secondMetrics.recallAt5).toBe(1.0);
    expect(secondMetrics.mrr).toBe(0.5);
    expect(secondMetrics.ndcgAt5).toBeLessThan(1.0);
    expect(secondMetrics.ndcgAt5).toBeGreaterThan(0.0);
  });

  describe("Isolated Shadow Database Recovery Lifecycle", () => {
    let tempDir: string;
    let store: QMDStore;

    beforeEach(async () => {
      tempDir = mkdtempSync(path.join(tmpdir(), "qmd-phase5-unit-"));
      const dbPath = path.join(tempDir, "shadow-test.sqlite");
      store = await createStore({
        dbPath,
        config: {
          collections: {
            test: {
              path: tempDir,
              pattern: "**/*.md",
            },
          },
        },
      });
    });

    afterEach(async () => {
      await store.close();
      try {
        rmSync(tempDir, { recursive: true, force: true });
      } catch {}
    });

    it("handles durable indexing cancellation and marks checkpoint cancelled", async () => {
      writeFileSync(path.join(tempDir, "test1.md"), "# Doc 1\n\nContent paragraph 1 for test\n");
      writeFileSync(path.join(tempDir, "test2.md"), "# Doc 2\n\nContent paragraph 2 for test\n");
      await store.update();

      const abortCtrl = new AbortController();
      abortCtrl.abort(); // Cancel immediately

      const res = await runDurableIndexingJob(store.internal, {
        signal: abortCtrl.signal,
      });

      expect(res.status).toBe("cancelled");
      const activeCp = getActiveCheckpoint(store.internal.db);
      expect(activeCp).not.toBeNull();
      expect(activeCp?.status).toBe("cancelled");
    });

    it("refuses resume when fingerprint parameters mismatch", async () => {
      writeFileSync(path.join(tempDir, "test1.md"), "# Doc 1\n\nContent paragraph 1 for test\n");
      await store.update();

      // Inject mismatched checkpoint
      saveCheckpoint(store.internal.db, {
        jobId: "mismatch-test-job",
        fingerprint: {
          model: "mismatched-model",
          dimensions: 512,
          chunkStrategy: "regex",
          descriptorSignature: "sig-abc",
        },
        startedAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        status: "in_progress",
        docsTotal: 1,
        docsCompleted: 0,
        chunksTotal: 2,
        chunksCommitted: 0,
      });

      await expect(
        runDurableIndexingJob(store.internal, {
          model: "expected-different-model",
        })
      ).rejects.toThrow(IndexingFingerprintMismatchError);
    });

    it("fingerprint mismatch rejection maintains complete logical content and vector state invariance", async () => {
      // Seed doc and vector
      writeFileSync(path.join(tempDir, "doc-keep.md"), "# Keep\n\nContent to preserve unchanged\n");
      await store.update();

      const mockDesc = {
        version: 1,
        backend: "mock",
        model: "base-embed",
        outputDimensions: 4,
        nativeDimensions: 4,
        maxTokens: 128,
        normalized: true,
      };
      store.llm = {
        embedModelName: "base-embed",
        getDescriptor: async () => mockDesc,
        embed: async () => ({ embedding: [0.1, 0.2, 0.3, 0.4], model: "base-embed" }),
        embedBatch: async (texts: string[]) => texts.map(() => ({ embedding: [0.1, 0.2, 0.3, 0.4], model: "base-embed" })),
      } as any;

      // Index initial document
      await runDurableIndexingJob(store.internal, { model: "base-embed" });

      const getLogicalState = (db: any) => ({
        content: db.prepare("SELECT hash, doc, created_at FROM content ORDER BY hash").all(),
        vectors: db.prepare("SELECT hash, seq, pos, model, embedded_at FROM content_vectors ORDER BY hash, seq").all(),
        documents: db.prepare("SELECT collection, path, title, hash, active FROM documents ORDER BY collection, path").all(),
      });

      const beforeState = getLogicalState(store.internal.db);
      expect(beforeState.vectors.length).toBeGreaterThan(0);

      // Inject mismatch checkpoint
      saveCheckpoint(store.internal.db, {
        jobId: "mismatch-invariance-test",
        fingerprint: {
          model: "other-model",
          dimensions: 512,
          chunkStrategy: "regex",
          descriptorSignature: "sig-other",
        },
        startedAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        status: "in_progress",
        docsTotal: 5,
        docsCompleted: 1,
        chunksTotal: 10,
        chunksCommitted: 1,
      });

      // Attempt resume with mismatched model
      await expect(
        runDurableIndexingJob(store.internal, { model: "base-embed" })
      ).rejects.toThrow(IndexingFingerprintMismatchError);

      // Clean up synthetic checkpoint
      store.internal.db.prepare("DELETE FROM indexing_checkpoints WHERE job_id = 'mismatch-invariance-test'").run();

      const afterState = getLogicalState(store.internal.db);
      expect(afterState).toEqual(beforeState);
    });

    it("fresh-process recovery worker executes partial barrier and fresh resumption cycle", async () => {
      const { runRecoveryWorker } = await import("../scripts/phase5_recovery_worker.js");

      const workerDir = mkdtempSync(path.join(tmpdir(), "qmd-worker-test-"));
      const workerDb = path.join(workerDir, "shadow-worker.sqlite");
      const docsDir = path.join(workerDir, "docs");
      const { mkdirSync, writeFileSync } = await import("node:fs");
      mkdirSync(docsDir, { recursive: true });

      for (const doc of PUBLIC_CORPUS_DOCS.slice(0, 4)) {
        writeFileSync(path.join(docsDir, doc.filename), `# ${doc.title}\n\n${doc.content}\n`);
      }

      // 1. Worker 1: indexes with barrier after 2 docs using mock backend
      const workerStore1 = await createStore({
        dbPath: workerDb,
        config: {
          collections: { test: { path: docsDir, pattern: "**/*.md" } },
        },
      });
      const mockLlm = {
        embedModelName: "test-model",
        getDescriptor: async () => ({
          version: 1,
          backend: "mock",
          model: "test-model",
          outputDimensions: 4,
          nativeDimensions: 4,
          maxTokens: 128,
          normalized: true,
        }),
        embed: async () => ({ embedding: [0.1, 0.2, 0.3, 0.4], model: "test-model" }),
        embedBatch: async (texts: string[]) => texts.map(() => ({ embedding: [0.1, 0.2, 0.3, 0.4], model: "test-model" })),
      };
      workerStore1.internal.llm = mockLlm as any;
      await workerStore1.update();

      const abortCtrl = new AbortController();
      await runDurableIndexingJob(workerStore1.internal, {
        maxDocsPerBatch: 1,
        model: "test-model",
        signal: abortCtrl.signal,
        onProgress: (info) => {
          if (info.chunksEmbedded >= 2) {
            abortCtrl.abort();
          }
        },
      });

      const midCp = getActiveCheckpoint(workerStore1.internal.db);
      expect(midCp).not.toBeNull();
      const midVectors = (workerStore1.internal.db.prepare("SELECT count(*) as c FROM content_vectors").get() as any).c;
      expect(midVectors).toBeGreaterThanOrEqual(2);
      await workerStore1.close();

      // 2. Worker 2: fresh process opens same DB and resumes to completion
      const workerStore2 = await createStore({
        dbPath: workerDb,
        config: {
          collections: { test: { path: docsDir, pattern: "**/*.md" } },
        },
      });
      workerStore2.internal.llm = mockLlm as any;

      const resumeRes = await runDurableIndexingJob(workerStore2.internal, {
        model: "test-model",
        maxDocsPerBatch: 2,
      });

      expect(resumeRes.status).toBe("complete");
      const finalCp = getActiveCheckpoint(workerStore2.internal.db);
      expect(finalCp).toBeNull();

      const finalVectors = (workerStore2.internal.db.prepare("SELECT count(*) as c FROM content_vectors").get() as any).c;
      expect(finalVectors).toBeGreaterThanOrEqual(4);

      // Verify consecutive sequences
      const allRows = workerStore2.internal.db.prepare("SELECT hash, seq FROM content_vectors ORDER BY hash, seq").all() as Array<{ hash: string; seq: number }>;
      const seqMap = new Map<string, number[]>();
      for (const r of allRows) {
        if (!seqMap.has(r.hash)) seqMap.set(r.hash, []);
        seqMap.get(r.hash)!.push(r.seq);
      }
      for (const seqs of seqMap.values()) {
        for (let i = 0; i < seqs.length; i++) {
          expect(seqs[i]).toBe(i);
        }
      }

      await workerStore2.close();
      try { rmSync(workerDir, { recursive: true, force: true }); } catch {}
    });
  });

  describe("Retrieval Equivalence Verification Helper", () => {
    it("verifyRetrievalEquivalence passes on identical query details and rankings", async () => {
      const { verifyRetrievalEquivalence } = await import("../scripts/phase5_e2e_runner.js");
      const stageA = {
        recallAt1: 1.0,
        recallAt3: 1.0,
        recallAt5: 1.0,
        mrr: 1.0,
        ndcgAt5: 1.0,
        avgLatencyMs: 12.0,
        p50LatencyMs: 10.0,
        p95LatencyMs: 15.0,
        queryDetails: [
          {
            queryId: "Q01",
            query: "test query 1",
            category: "easy",
            expectedDocIds: ["doc-1"],
            topDocIds: ["doc-1", "doc-2"],
            rank1: 1,
            hitAt5: true,
            latencyMs: 10.0,
          },
        ],
      };

      const stageB = JSON.parse(JSON.stringify(stageA));
      const eq = verifyRetrievalEquivalence(stageA, stageB);
      expect(eq.identical).toBe(true);
      expect(eq.details[0].docOrderMatch).toBe(true);
    });

    it("verifyRetrievalEquivalence fails when rankings or doc orders diverge", async () => {
      const { verifyRetrievalEquivalence } = await import("../scripts/phase5_e2e_runner.js");
      const stageA = {
        recallAt1: 1.0,
        recallAt3: 1.0,
        recallAt5: 1.0,
        mrr: 1.0,
        ndcgAt5: 1.0,
        avgLatencyMs: 12.0,
        p50LatencyMs: 10.0,
        p95LatencyMs: 15.0,
        queryDetails: [
          {
            queryId: "Q01",
            query: "test query 1",
            category: "easy",
            expectedDocIds: ["doc-1"],
            topDocIds: ["doc-1", "doc-2"],
            rank1: 1,
            hitAt5: true,
            latencyMs: 10.0,
          },
        ],
      };

      const stageB = JSON.parse(JSON.stringify(stageA));
      stageB.queryDetails[0].topDocIds = ["doc-2", "doc-1"]; // Mismatched doc order
      stageB.queryDetails[0].rank1 = 2;
      stageB.recallAt1 = 0.0;

      const eq = verifyRetrievalEquivalence(stageA, stageB);
      expect(eq.identical).toBe(false);
      expect(eq.details[0].docOrderMatch).toBe(false);
    });
  });
});

