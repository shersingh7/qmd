/**
 * phase5_e2e_runner.ts — End-to-End Public Retrieval & Recovery Qualification Runner
 *
 * Runs isolated production QMD SDK indexing, recovery, retrieval, and MCP transport probing
 * against curated public fixtures in disposable shadow databases.
 */

import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync, symlinkSync, linkSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import {
  createStore,
  runDurableIndexingJob,
  validateShadowTarget,
  ShadowDatabaseProtectionError,
  IndexingFingerprintMismatchError,
  getActiveCheckpoint,
  saveCheckpoint,
  type QMDStore,
  type IndexingJobCheckpoint,
} from "../src/index.js";
import {
  startMcpHttpServer,
  type HttpServerHandle,
} from "../src/mcp/server.js";
import {
  PUBLIC_CORPUS_DOCS,
  PUBLIC_HELD_OUT_QUERIES,
  computeIRMetrics,
} from "../test/fixtures/public_eval_dataset.js";
import { getDefaultLlamaCpp, disposeDefaultLlamaCpp } from "../src/llm.js";

export interface StageMetrics {
  recallAt1: number;
  recallAt3: number;
  recallAt5: number;
  mrr: number;
  ndcgAt5: number;
  avgLatencyMs: number;
  p50LatencyMs: number;
  p95LatencyMs: number;
  queryDetails: Array<{
    queryId: string;
    query: string;
    category: string;
    expectedDocIds: string[];
    topDocIds: string[];
    scores: number[];
    rank1: number | null;
    hitAt5: boolean;
    latencyMs: number;
    error?: string;
  }>;
}

export function verifyRetrievalEquivalence(
  pre: StageMetrics,
  post: StageMetrics,
  tolerance: number = 1e-4
): { identical: boolean; details: Array<{ queryId: string; docOrderMatch: boolean; scoreToleranceMatch: boolean; diff?: string }> } {
  const details: Array<{ queryId: string; docOrderMatch: boolean; scoreToleranceMatch: boolean; diff?: string }> = [];
  let allMatch = true;

  if (!pre.queryDetails.length || pre.queryDetails.length !== post.queryDetails.length) {
    return {
      identical: false,
      details: [{ queryId: "all", docOrderMatch: false, scoreToleranceMatch: false, diff: `Query counts differ: ${pre.queryDetails.length} vs ${post.queryDetails.length}` }],
    };
  }

  for (let i = 0; i < pre.queryDetails.length; i++) {
    const qPre = pre.queryDetails[i];
    const qPost = post.queryDetails[i];
    const docOrderMatch = JSON.stringify(qPre.topDocIds) === JSON.stringify(qPost.topDocIds);
    const hitMatch = qPre.hitAt5 === qPost.hitAt5 && qPre.rank1 === qPost.rank1;

    const scoreToleranceMatch = Array.isArray(qPre.scores) && Array.isArray(qPost.scores)
      && qPre.scores.length > 0 && qPre.scores.length === qPost.scores.length
      && qPre.scores.length === qPre.topDocIds.length
      && qPost.scores.length === qPost.topDocIds.length
      && qPre.scores.every((score, j) => Number.isFinite(score) && Number.isFinite(qPost.scores[j]) && Math.abs(score - qPost.scores[j]) <= tolerance);
    if (!docOrderMatch || !hitMatch || !scoreToleranceMatch || qPre.queryId !== qPost.queryId) {
      allMatch = false;
      details.push({
        queryId: qPre.queryId,
        docOrderMatch,
        scoreToleranceMatch,
        diff: `Doc order or rank mismatch: pre=[${qPre.topDocIds.join(",")}] post=[${qPost.topDocIds.join(",")}] rank1=${qPre.rank1}/${qPost.rank1}`,
      });
    } else {
      details.push({
        queryId: qPre.queryId,
        docOrderMatch: true,
        scoreToleranceMatch,
      });
    }
  }

  if (
    pre.recallAt1 !== post.recallAt1 ||
    pre.recallAt3 !== post.recallAt3 ||
    pre.recallAt5 !== post.recallAt5 ||
    pre.mrr !== post.mrr ||
    pre.ndcgAt5 !== post.ndcgAt5
  ) {
    allMatch = false;
  }

  return { identical: allMatch, details };
}

export interface Phase5ExecutionReport {
  timestamp: string;
  durationMs: number;
  status: "passed" | "failed";
  checks: {
    shadowTargetSafety: boolean;
    durableIndexingCancellation: boolean;
    durableIndexingResumption: boolean;
    vectorDeduplicationReconciled: boolean;
    fingerprintMismatchRefused: boolean;
    sqliteOnlineBackupRestored: boolean;
    restoredRetrievalIdentical: boolean;
    gguf06bRetrievalComplete: boolean;
    mlx4bRetrievalComplete: boolean;
    mcpTransportProbePassed: boolean;
  };
  corpus: {
    totalDocs: number;
    totalQueries: number;
    docIds: string[];
  };
  recovery: {
    mode: string;
    freshProcessRecovery: string;
    initialCancelledChunks: number;
    resumedTotalChunks: number;
    consecutiveSequenceVerified: boolean;
    mismatchErrorCaught: boolean;
    logicalStateInvariantOnMismatch?: boolean;
    backupVectorCount: number;
    backupDocCount: number;
  };
  retrieval: {
    gguf06b?: {
      status?: string;
      reason?: string;
      bm25?: StageMetrics;
      vector?: StageMetrics;
      hybrid?: StageMetrics;
      reranked?: StageMetrics;
    };
    mlx4b?: {
      status?: string;
      reason?: string;
      bm25?: StageMetrics;
      vector?: StageMetrics;
      hybrid?: StageMetrics;
      reranked?: StageMetrics;
    };
  };
  mcpProbe: {
    port: number;
    toolsCount: number;
    searchResultCount: number;
    firstResultDocId: string;
    getContentLength: number;
  };
  errors: string[];
}

function computeLatencies(latencies: number[]): { avg: number; p50: number; p95: number } {
  if (latencies.length === 0) return { avg: 0, p50: 0, p95: 0 };
  const sorted = [...latencies].sort((a, b) => a - b);
  const avg = sorted.reduce((sum, v) => sum + v, 0) / sorted.length;
  const p50 = sorted[Math.floor(sorted.length * 0.5)];
  const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];
  return {
    avg: Math.round(avg * 100) / 100,
    p50: Math.round(p50 * 100) / 100,
    p95: Math.round(p95 * 100) / 100,
  };
}

async function evaluateRetrievalStage(
  store: QMDStore,
  stageName: "bm25" | "vector" | "hybrid" | "reranked"
): Promise<StageMetrics> {
  const rankedDocIdsList: string[][] = [];
  const relevantDocIdsList: string[][] = [];
  const latencies: number[] = [];
  const queryDetails: StageMetrics["queryDetails"] = [];

  for (const q of PUBLIC_HELD_OUT_QUERIES) {
    const t0 = performance.now();
    let results: Array<{ file?: string; displayPath?: string; filepath?: string; score: number }> = [];
    let queryError: string | undefined = undefined;

    try {
      if (stageName === "bm25") {
        results = await store.searchLex(q.query, { limit: 5 });
      } else if (stageName === "vector") {
        results = await store.searchVector(q.query, { limit: 5 });
      } else if (stageName === "hybrid") {
        // Use typed queries API to combine lex + vec without loading local expansion model
        results = await store.search({
          queries: [
            { type: "lex", query: q.query },
            { type: "vec", query: q.query },
          ],
          limit: 5,
          rerank: false,
        });
      } else if (stageName === "reranked") {
        results = await store.search({
          queries: [
            { type: "lex", query: q.query },
            { type: "vec", query: q.query },
          ],
          limit: 5,
          rerank: true,
        });
      }
    } catch (err: any) {
      queryError = err?.message || String(err);
      console.error(`Error querying [${stageName}] for "${q.query}":`, queryError);
      throw new Error(`Query execution failure in [${stageName}] for "${q.query}": ${queryError}`);
    }

    const elapsed = performance.now() - t0;
    latencies.push(elapsed);

    // Extract doc IDs from file paths (e.g. "public-corpus/doc-api-versioning.md" -> "doc-api-versioning")
    const topIds: string[] = results.map(r => {
      const p = r.displayPath || r.filepath || r.file || "";
      const base = path.basename(p, ".md");
      return base;
    });

    rankedDocIdsList.push(topIds);
    relevantDocIdsList.push(q.relevantDocIds);

    const relSet = new Set(q.relevantDocIds);
    let rank1: number | null = null;
    for (let i = 0; i < topIds.length; i++) {
      if (relSet.has(topIds[i])) {
        rank1 = i + 1;
        break;
      }
    }

    queryDetails.push({
      queryId: q.id,
      query: q.query,
      category: q.category,
      expectedDocIds: q.relevantDocIds,
      topDocIds: topIds,
      scores: results.map(r => r.score),
      rank1,
      hitAt5: rank1 !== null && rank1 <= 5,
      latencyMs: Math.round(elapsed * 100) / 100,
      ...(queryError ? { error: queryError } : {}),
    });
  }

  const ir = computeIRMetrics(rankedDocIdsList, relevantDocIdsList, 5);
  const lStats = computeLatencies(latencies);

  return {
    ...ir,
    avgLatencyMs: lStats.avg,
    p50LatencyMs: lStats.p50,
    p95LatencyMs: lStats.p95,
    queryDetails,
  };
}

export async function runPhase5Qualification(options?: {
  skipGguf?: boolean;
  skipMlx?: boolean;
  mlxPort?: number;
  ggufModelPath?: string;
  mlxModelPath?: string;
}): Promise<Phase5ExecutionReport> {
  const startTime = Date.now();
  const errors: string[] = [];
  const checks = {
    shadowTargetSafety: false,
    durableIndexingCancellation: false,
    durableIndexingResumption: false,
    vectorDeduplicationReconciled: false,
    fingerprintMismatchRefused: false,
    sqliteOnlineBackupRestored: false,
    restoredRetrievalIdentical: false,
    gguf06bRetrievalComplete: false,
    mlx4bRetrievalComplete: false,
    mcpTransportProbePassed: false,
  };

  const recoveryReport = {
    mode: "fresh_process_recovery",
    freshProcessRecovery: "pending",
    initialCancelledChunks: 0,
    resumedTotalChunks: 0,
    consecutiveSequenceVerified: false,
    mismatchErrorCaught: false,
    logicalStateInvariantOnMismatch: false,
    backupVectorCount: 0,
    backupDocCount: 0,
  };

  const retrievalReport: Phase5ExecutionReport["retrieval"] = {};
  let mcpReport: Phase5ExecutionReport["mcpProbe"] = {
    port: 0,
    toolsCount: 0,
    searchResultCount: 0,
    firstResultDocId: "",
    getContentLength: 0,
  };

  const tempRootDir = mkdtempSync(path.join(tmpdir(), "qmd-phase5-shadow-root-"));

  try {
    // -------------------------------------------------------------------------
    // 1. Shadow Target Safety Checks
    // -------------------------------------------------------------------------
    console.log("[Phase 5] 1. Validating shadow target isolation safeguards...");
    const fakeLive = path.join(tempRootDir, "live-fake.sqlite");
    writeFileSync(fakeLive, "sqlite content");
    const fakeSymlink = path.join(tempRootDir, "shadow-symlink.sqlite");
    symlinkSync(fakeLive, fakeSymlink);
    const fakeHardlink = path.join(tempRootDir, "shadow-hardlink.sqlite");
    linkSync(fakeLive, fakeHardlink);

    let safetyPass = true;
    try {
      validateShadowTarget(fakeLive, fakeLive);
      safetyPass = false;
    } catch (e) {
      if (!(e instanceof ShadowDatabaseProtectionError)) safetyPass = false;
    }

    try {
      validateShadowTarget(fakeSymlink, fakeLive);
      safetyPass = false;
    } catch (e) {
      if (!(e instanceof ShadowDatabaseProtectionError)) safetyPass = false;
    }

    try {
      validateShadowTarget(fakeHardlink, fakeLive);
      safetyPass = false;
    } catch (e) {
      if (!(e instanceof ShadowDatabaseProtectionError)) safetyPass = false;
    }

    const validShadowPath = path.join(tempRootDir, "shadow-test-index.sqlite");
    try {
      validateShadowTarget(validShadowPath, fakeLive);
    } catch {
      safetyPass = false;
    }

    checks.shadowTargetSafety = safetyPass;
    if (!safetyPass) errors.push("Shadow target validation safeguards failed");

    // -------------------------------------------------------------------------
    // 2. Production Indexing, Fresh-Process Interruption & Resumption Qualification
    // -------------------------------------------------------------------------
    console.log("[Phase 5] 2. Testing production fresh-process recovery & resumption on shadow database...");
    const docsDir = path.join(tempRootDir, "public-docs");
    mkdirSync(docsDir, { recursive: true });
    for (const doc of PUBLIC_CORPUS_DOCS) {
      writeFileSync(path.join(docsDir, doc.filename), `# ${doc.title}\n\n${doc.content}\n`);
    }

    const isolatedModels = {
      embed: options?.mlxModelPath || path.join(process.env.HOME!, ".cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine"),
      embedBackend: "mlx" as const,
      mlxUrl: `http://127.0.0.1:${options?.mlxPort || 8797}`,
      mlxFallback: false,
    };
    const recoveryDbPath = path.join(tempRootDir, "shadow-recovery.sqlite");
    const workerScript = path.join(import.meta.dirname || path.dirname(process.argv[1]), "phase5_recovery_worker.ts");
    const launchToken = process.env.QMD_PHASE5_LAUNCH_TOKEN || "phase5-worker-token";

    // Step 2a & 2b: Worker 1 (Fresh Subprocess) - partial indexing stopping at deterministic barrier
    console.log("[Phase 5] 2b. Launching Worker 1 fresh process with barrier after 2 documents...");
    const worker1ReportPath = path.join(tempRootDir, "worker1-report.json");
    const worker1Proc = spawnSync(
      "bun",
      [
        workerScript,
        "--db", recoveryDbPath,
        "--docs-dir", docsDir,
        "--mlx-port", String(options?.mlxPort || 8797),
        "--barrier-after-docs", "2",
        "--launch-token", launchToken,
        "--output-json", worker1ReportPath,
      ],
      {
        env: { ...process.env, QMD_PHASE5_LAUNCH_TOKEN: launchToken },
        encoding: "utf-8",
      }
    );

    if (worker1Proc.error) {
      errors.push(`Worker 1 spawn error: ${worker1Proc.error.message}`);
    }

    // Inspect intermediate DB state via lightweight store inspect
    const inspectStore1 = await createStore({
      dbPath: recoveryDbPath,
      config: {
        models: isolatedModels,
        collections: {
          public: {
            path: docsDir,
            pattern: "**/*.md",
          },
        },
      },
    });

    const activeCp1 = getActiveCheckpoint(inspectStore1.internal.db);
    const vectorsMid = (inspectStore1.internal.db.prepare("SELECT count(*) as c FROM content_vectors").get() as any).c;
    recoveryReport.initialCancelledChunks = vectorsMid;
    checks.durableIndexingCancellation = activeCp1 !== null && vectorsMid > 0;
    await inspectStore1.close();

    // Step 2c: Worker 2 (Fresh Subprocess) - resumes from active checkpoint to completion
    console.log("[Phase 5] 2c. Launching Worker 2 fresh process to resume indexing...");
    const worker2ReportPath = path.join(tempRootDir, "worker2-report.json");
    const worker2Proc = spawnSync(
      "bun",
      [
        workerScript,
        "--db", recoveryDbPath,
        "--docs-dir", docsDir,
        "--mlx-port", String(options?.mlxPort || 8797),
        "--action", "resume",
        "--launch-token", launchToken,
        "--output-json", worker2ReportPath,
      ],
      {
        env: { ...process.env, QMD_PHASE5_LAUNCH_TOKEN: launchToken },
        encoding: "utf-8",
      }
    );

    if (worker2Proc.error) {
      errors.push(`Worker 2 spawn error: ${worker2Proc.error.message}`);
    }

    // Open store on completed recovery DB
    const recoveryStore = await createStore({
      dbPath: recoveryDbPath,
      config: {
        models: isolatedModels,
        collections: {
          public: {
            path: docsDir,
            pattern: "**/*.md",
          },
        },
      },
    });

    const activeCp2 = getActiveCheckpoint(recoveryStore.internal.db);
    const vectorsFinal = (recoveryStore.internal.db.prepare("SELECT count(*) as c FROM content_vectors").get() as any).c;
    recoveryReport.resumedTotalChunks = vectorsFinal;
    checks.durableIndexingResumption = activeCp2 === null && vectorsFinal >= PUBLIC_CORPUS_DOCS.length;

    // Step 2d: Reconcile sequence numbers for all documents to guarantee no duplicate chunks
    const allDocVectors = recoveryStore.internal.db.prepare(
      "SELECT hash, seq FROM content_vectors ORDER BY hash, seq"
    ).all() as Array<{ hash: string; seq: number }>;

    const seqByHash = new Map<string, number[]>();
    for (const row of allDocVectors) {
      if (!seqByHash.has(row.hash)) seqByHash.set(row.hash, []);
      seqByHash.get(row.hash)!.push(row.seq);
    }

    let allSeqsConsecutive = true;
    for (const seqs of seqByHash.values()) {
      for (let i = 0; i < seqs.length; i++) {
        if (seqs[i] !== i) {
          allSeqsConsecutive = false;
          break;
        }
      }
    }
    recoveryReport.consecutiveSequenceVerified = allSeqsConsecutive;
    checks.vectorDeduplicationReconciled = allSeqsConsecutive && vectorsFinal >= PUBLIC_CORPUS_DOCS.length;
    recoveryReport.mode = "fresh_process_recovery";
    recoveryReport.freshProcessRecovery = (checks.durableIndexingCancellation && checks.durableIndexingResumption && checks.vectorDeduplicationReconciled) ? "passed" : "failed";

    // Step 2e: Fingerprint Mismatch Protection & Logical Database State Invariance Check
    let mismatchCaught = false;
    const getLogicalDbState = (db: any) => ({
      content: db.prepare("SELECT hash, doc, created_at FROM content ORDER BY hash").all(),
      vectors: db.prepare("SELECT hash, seq, pos, model, embedded_at FROM content_vectors ORDER BY hash, seq").all(),
      documents: db.prepare("SELECT collection, path, title, hash, active FROM documents ORDER BY collection, path").all(),
    });

    const stateBeforeMismatch = getLogicalDbState(recoveryStore.internal.db);

    // Inject a synthetic in-progress checkpoint with mismatched model fingerprint
    saveCheckpoint(recoveryStore.internal.db, {
      jobId: "mismatch-test-job",
      fingerprint: {
        model: "mismatched-nonexistent-model",
        dimensions: 512,
        chunkStrategy: "regex",
        descriptorSignature: "fake-sig-12345",
      },
      startedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: "in_progress",
      docsTotal: 16,
      docsCompleted: 5,
      chunksTotal: 20,
      chunksCommitted: 5,
    });

    try {
      await runDurableIndexingJob(recoveryStore.internal, {
        model: "different-model-attempt",
      });
    } catch (err) {
      if (err instanceof IndexingFingerprintMismatchError) {
        mismatchCaught = true;
      }
    }

    // Clean up synthetic checkpoint so DB returns to prior clean state
    recoveryStore.internal.db.prepare("DELETE FROM indexing_checkpoints WHERE job_id = 'mismatch-test-job'").run();

    const stateAfterMismatch = getLogicalDbState(recoveryStore.internal.db);
    const logicalStateInvariant = JSON.stringify(stateBeforeMismatch) === JSON.stringify(stateAfterMismatch);

    recoveryReport.mismatchErrorCaught = mismatchCaught;
    checks.fingerprintMismatchRefused = mismatchCaught && logicalStateInvariant;

    // Step 2f: SQLite Online-Backup & Pre/Post Full Retrieval Equivalence Verification
    console.log("[Phase 5] 2f. Verifying pre/post SQLite VACUUM INTO retrieval equivalence across 10 queries...");
    const preVectorMetrics = await evaluateRetrievalStage(recoveryStore, "vector");
    const preHybridMetrics = await evaluateRetrievalStage(recoveryStore, "hybrid");

    const backupDbPath = path.join(tempRootDir, "shadow-backup-restored.sqlite");
    // SQLite VACUUM INTO creates an atomic, consistent online backup
    recoveryStore.internal.db.exec(`VACUUM INTO '${backupDbPath}'`);
    await recoveryStore.close();

    const restoredStore = await createStore({
      dbPath: backupDbPath,
      config: {
        models: isolatedModels,
        collections: {
          public: {
            path: docsDir,
            pattern: "**/*.md",
          },
        },
      },
    });

    const restoredStatus = await restoredStore.getStatus();
    const restoredVectors = (restoredStore.internal.db.prepare("SELECT count(*) as c FROM content_vectors").get() as any).c;
    recoveryReport.backupDocCount = restoredStatus.totalDocuments;
    recoveryReport.backupVectorCount = restoredVectors;
    checks.sqliteOnlineBackupRestored = restoredStatus.totalDocuments === PUBLIC_CORPUS_DOCS.length && restoredVectors === vectorsFinal;

    // Execute exact same 10 held-out queries on restored store
    const postVectorMetrics = await evaluateRetrievalStage(restoredStore, "vector");
    const postHybridMetrics = await evaluateRetrievalStage(restoredStore, "hybrid");

    const vectorEquivalence = verifyRetrievalEquivalence(preVectorMetrics, postVectorMetrics);
    const hybridEquivalence = verifyRetrievalEquivalence(preHybridMetrics, postHybridMetrics);

    checks.restoredRetrievalIdentical = vectorEquivalence.identical && hybridEquivalence.identical;
    await restoredStore.close();

    // -------------------------------------------------------------------------
    // 3. Baseline GGUF 0.6B Indexing & Retrieval Qualification (BLOCKED)
    // -------------------------------------------------------------------------
    if (options?.skipGguf) {
      (retrievalReport as any).gguf06b = {
        status: "blocked",
        reason: "GGUF stage explicitly BLOCKED pending supervised integration instead of unmanaged subprocess.run",
      };
    } else {
      console.log("[Phase 5] 3. Qualifying Baseline GGUF 0.6B in dedicated shadow database...");
      const ggufDbPath = path.join(tempRootDir, "shadow-gguf-06b.sqlite");
      const explicitGgufPath = options?.ggufModelPath || path.join(process.env.HOME || "~", ".cache/qmd/models/hf_Qwen_Qwen3-Embedding-0.6B-Q8_0.gguf");

      if (!existsSync(explicitGgufPath)) {
        throw new Error(`Explicit GGUF model path does not exist: ${explicitGgufPath}. Network autodownloads are strictly prohibited.`);
      }

      const ggufStore = await createStore({
        dbPath: ggufDbPath,
        config: {
          collections: {
            public: {
              path: docsDir,
              pattern: "**/*.md",
            },
          },
          models: {
            embed: explicitGgufPath,
            embedBackend: "gguf",
          },
        },
      });

      await ggufStore.update();
      await ggufStore.embed({ model: explicitGgufPath, force: true });

      const bm25Metrics = await evaluateRetrievalStage(ggufStore, "bm25");
      const vecMetrics = await evaluateRetrievalStage(ggufStore, "vector");
      const hybridMetrics = await evaluateRetrievalStage(ggufStore, "hybrid");

      retrievalReport.gguf06b = {
        bm25: bm25Metrics,
        vector: vecMetrics,
        hybrid: hybridMetrics,
      };

      checks.gguf06bRetrievalComplete = vecMetrics.recallAt5 >= 0.8 && hybridMetrics.recallAt5 >= 0.9;
      await ggufStore.close();
    }

    // -------------------------------------------------------------------------
    // 4. Candidate MLX 4B Indexing & Retrieval Qualification
    // -------------------------------------------------------------------------
    if (options?.skipMlx) {
      (retrievalReport as any).mlx4b = {
        status: "not_run",
        reason: "MLX stage skipped by options",
      };
    } else {
      console.log("[Phase 5] 4. Qualifying Candidate MLX 4B in dedicated shadow database...");
      const mlxPort = options?.mlxPort || 8797;
      const mlxUrl = `http://127.0.0.1:${mlxPort}`;
      const mlxDbPath = path.join(tempRootDir, "shadow-mlx-4b.sqlite");
      const defaultMlxModel = path.join(process.env.HOME || "~", ".cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine");
      const explicitMlxModel = options?.mlxModelPath || defaultMlxModel;

      if (!existsSync(explicitMlxModel)) {
        throw new Error(`Explicit MLX model path does not exist: ${explicitMlxModel}. Network autodownloads are strictly prohibited.`);
      }

      // Test health of MLX server
      let mlxAvailable = false;
      try {
        const hRes = await fetch(`${mlxUrl}/health`);
        if (hRes.ok) mlxAvailable = true;
      } catch {}

      if (mlxAvailable) {
        const mlxStore = await createStore({
          dbPath: mlxDbPath,
          config: {
            collections: {
              public: {
                path: docsDir,
                pattern: "**/*.md",
              },
            },
            models: {
              embed: explicitMlxModel,
              embedBackend: "mlx",
              mlxUrl: mlxUrl,
            },
          },
        });

        await mlxStore.update();
        await mlxStore.embed({ model: explicitMlxModel, force: true });

        const bm25Metrics = await evaluateRetrievalStage(mlxStore, "bm25");
        const vecMetrics = await evaluateRetrievalStage(mlxStore, "vector");
        const hybridMetrics = await evaluateRetrievalStage(mlxStore, "hybrid");

        retrievalReport.mlx4b = {
          bm25: bm25Metrics,
          vector: vecMetrics,
          hybrid: hybridMetrics,
        };

        checks.mlx4bRetrievalComplete = vecMetrics.recallAt5 >= 0.8 && hybridMetrics.recallAt5 >= 0.9;
        await mlxStore.close();
      } else {
        console.log(`[Phase 5] MLX server not reachable on ${mlxUrl}; skipping MLX live retrieval stage`);
      }
    }

    // -------------------------------------------------------------------------
    // 5. Integrated MCP Transport Retrieval Probe
    // -------------------------------------------------------------------------
    console.log("[Phase 5] 5. Running integrated MCP Streamable HTTP transport retrieval probe...");
    const mcpDbPath = path.join(tempRootDir, "shadow-mcp.sqlite");
    const mcpStore = await createStore({
      dbPath: mcpDbPath,
      config: {
        models: isolatedModels,
        collections: {
          public: {
            path: docsDir,
            pattern: "**/*.md",
          },
        },
      },
    });
    await mcpStore.update();

    // Start MCP server on ephemeral port (port 0) with injected store
    const mcpHandle: HttpServerHandle = await startMcpHttpServer(0, { store: mcpStore, quiet: true });
    const baseUrl = `http://localhost:${mcpHandle.port}`;

    let mcpSessionId: string | null = null;
    async function postMcp(body: object): Promise<any> {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
      };
      if (mcpSessionId) headers["mcp-session-id"] = mcpSessionId;

      const res = await fetch(`${baseUrl}/mcp`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      });

      const sid = res.headers.get("mcp-session-id");
      if (sid) mcpSessionId = sid;
      return await res.json();
    }

    // Initialize MCP Session
    const initRes = await postMcp({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-03-26",
        capabilities: {},
        clientInfo: { name: "phase5-probe", version: "1.0.0" },
      },
    });

    // List Tools
    const toolsRes = await postMcp({
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
      params: {},
    });

    const tools = toolsRes.result?.tools || [];
    mcpReport.toolsCount = tools.length;

    // Call Search Tool
    const searchCallRes = await postMcp({
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: {
        name: "query",
        arguments: {
          searches: [{ type: "lex", query: "Raft" }],
          rerank: false,
        },
      },
    });

    const searchStructured = searchCallRes.result?.structuredContent?.results || [];
    mcpReport.searchResultCount = searchStructured.length;
    const firstResult = searchStructured[0];
    mcpReport.firstResultDocId = firstResult?.docid || "";

    // Call Get Tool
    let docContentLen = 0;
    if (firstResult) {
      const getCallRes = await postMcp({
        jsonrpc: "2.0",
        id: 4,
        method: "tools/call",
        params: {
          name: "get",
          arguments: {
            file: firstResult.file || firstResult.docid,
          },
        },
      });
      const resText = getCallRes.result?.content?.[0]?.resource?.text || "";
      docContentLen = resText.length;
      mcpReport.getContentLength = docContentLen;
    }

    mcpReport.port = mcpHandle.port;
    checks.mcpTransportProbePassed = (
      initRes?.result?.serverInfo?.name === "qmd" &&
      tools.length >= 4 &&
      searchStructured.length > 0 &&
      docContentLen > 50
    );

    await mcpHandle.stop();
    await mcpStore.close();

  } catch (err: any) {
    console.error("[Phase 5] Unhandled execution error:", err);
    errors.push(err.message || String(err));
  } finally {
    // Release native llama resources
    await disposeDefaultLlamaCpp();
    try {
      rmSync(tempRootDir, { recursive: true, force: true });
    } catch {}
  }

  const durationMs = Date.now() - startTime;
  const passed = (
    checks.shadowTargetSafety &&
    checks.durableIndexingCancellation &&
    checks.durableIndexingResumption &&
    checks.vectorDeduplicationReconciled &&
    checks.fingerprintMismatchRefused &&
    checks.sqliteOnlineBackupRestored &&
    checks.restoredRetrievalIdentical &&
    checks.mcpTransportProbePassed &&
    (options?.skipGguf ? true : checks.gguf06bRetrievalComplete) &&
    (options?.skipMlx ? true : checks.mlx4bRetrievalComplete) &&
    errors.length === 0
  );

  return {
    timestamp: new Date().toISOString(),
    durationMs,
    status: passed ? "passed" : "failed",
    checks,
    corpus: {
      totalDocs: PUBLIC_CORPUS_DOCS.length,
      totalQueries: PUBLIC_HELD_OUT_QUERIES.length,
      docIds: PUBLIC_CORPUS_DOCS.map(d => d.id),
    },
    recovery: recoveryReport,
    retrieval: retrievalReport,
    mcpProbe: mcpReport,
    errors,
  };
}

// CLI Execution Entrypoint
if (import.meta.url === `file://${process.argv[1]}`) {
  const args = process.argv.slice(2);
  const launchTokenIdx = args.indexOf("--launch-token");
  const launchToken = (launchTokenIdx !== -1 && args[launchTokenIdx + 1])
    ? args[launchTokenIdx + 1]
    : process.env.QMD_PHASE5_LAUNCH_TOKEN;

  if (!launchToken || !launchToken.trim()) {
    console.error("Error: Direct TypeScript execution rejected. Phase 5 runner requires Python supervisor approval via --launch-token or QMD_PHASE5_LAUNCH_TOKEN.");
    process.exit(1);
  }

  const skipGguf = args.includes("--skip-gguf");
  const skipMlx = args.includes("--skip-mlx");
  const mlxPortIdx = args.indexOf("--mlx-port");
  const mlxPort = mlxPortIdx !== -1 && args[mlxPortIdx + 1] ? parseInt(args[mlxPortIdx + 1], 10) : 8797;
  const outIdx = args.indexOf("--output-json");
  const outPath = outIdx !== -1 && args[outIdx + 1] ? args[outIdx + 1] : undefined;

  runPhase5Qualification({
    skipGguf,
    skipMlx,
    mlxPort,
  }).then(report => {
    console.log("\n=== Phase 5 Execution Report ===");
    console.log(`Status: ${report.status.toUpperCase()} (${report.durationMs}ms)`);
    console.log(`Checks: ${Object.entries(report.checks).map(([k, v]) => `${k}=${v}`).join(", ")}`);
    if (outPath) {
      writeFileSync(outPath, JSON.stringify(report, null, 2));
      console.log(`Written report to: ${outPath}`);
    }
    process.exit(report.status === "passed" ? 0 : 1);
  }).catch(err => {
    console.error("Phase 5 runner error:", err);
    process.exit(1);
  });
}
