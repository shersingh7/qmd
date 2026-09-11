/**
 * phase5_recovery_worker.ts — Isolated TS Subprocess Worker for Durable Indexing & Recovery
 *
 * Runs production `runDurableIndexingJob` in a separate fresh OS process against an
 * isolated shadow database, pointing exclusively to the supervised MLX embedding port (8797).
 * Strictly forbids local GGUF/model loading and remote network autodownloads.
 */

import { existsSync, writeFileSync } from "node:fs";
import path from "node:path";
import {
  createStore,
  runDurableIndexingJob,
  validateShadowTarget,
  getActiveCheckpoint,
  type QMDStore,
} from "../src/index.js";
import { PUBLIC_CORPUS_DOCS } from "../test/fixtures/public_eval_dataset.js";

export interface RecoveryWorkerOptions {
  dbPath: string;
  docsDir?: string;
  mlxPort?: number;
  mlxModelPath?: string;
  barrierAfterDocs?: number;
  barrierAfterChunks?: number;
  stopMode?: "exit" | "pause";
  action?: "index" | "resume";
  outputJson?: string;
  launchToken?: string;
  quiet?: boolean;
}

export interface RecoveryWorkerResult {
  status: "barrier_reached" | "completed" | "cancelled" | "failed";
  dbPath: string;
  action: string;
  docsCompleted: number;
  chunksCommitted: number;
  totalVectors: number;
  checkpointStatus: string | null;
  consecutiveSequenceVerified: boolean;
  hashes: Array<{ hash: string; seqs: number[] }>;
  error?: string;
}

export async function runRecoveryWorker(options: RecoveryWorkerOptions): Promise<RecoveryWorkerResult> {
  const dbPath = path.resolve(options.dbPath);
  const mlxPort = options.mlxPort || 8797;
  const mlxUrl = `http://127.0.0.1:${mlxPort}`;
  const action = options.action || "index";
  const stopMode = options.stopMode || "exit";
  const docsDir = options.docsDir ? path.resolve(options.docsDir) : path.join(path.dirname(dbPath), "public-docs");

  // 1. Safety validation: verify shadow target isolation
  validateShadowTarget(dbPath);

  const isolatedModels = {
    embed: options.mlxModelPath || path.join(process.env.HOME || "~", ".cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine"),
    embedBackend: "mlx" as const,
    mlxUrl,
    mlxFallback: false,
  };

  let store: QMDStore | null = null;
  try {
    store = await createStore({
      dbPath,
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

    if (action === "index") {
      await store.update();

      const barrierLimit = options.barrierAfterChunks ?? options.barrierAfterDocs ?? null;

      const jobResult = await runDurableIndexingJob(store.internal, {
        maxDocsPerBatch: 1,
        signal: abortCtrl.signal,
        onProgress: (info, _cp) => {
          if (barrierLimit !== null && info.chunksEmbedded >= barrierLimit) {
            abortCtrl.abort();
          }
        },
      });

      const vectorsMid = (store.internal.db.prepare("SELECT count(*) as c FROM content_vectors").get() as any).c;
      const activeCp = getActiveCheckpoint(store.internal.db);
      const cpStatus = activeCp ? activeCp.status : null;

      const hashesRows = store.internal.db.prepare(
        "SELECT hash, seq FROM content_vectors ORDER BY hash, seq"
      ).all() as Array<{ hash: string; seq: number }>;

      const seqByHash = new Map<string, number[]>();
      for (const row of hashesRows) {
        if (!seqByHash.has(row.hash)) seqByHash.set(row.hash, []);
        seqByHash.get(row.hash)!.push(row.seq);
      }

      let allConsecutive = true;
      for (const seqs of seqByHash.values()) {
        for (let i = 0; i < seqs.length; i++) {
          if (seqs[i] !== i) {
            allConsecutive = false;
            break;
          }
        }
      }

      const result: RecoveryWorkerResult = {
        status: barrierDocs !== null ? "barrier_reached" : (jobResult.status === "complete" ? "completed" : "cancelled"),
        dbPath,
        action,
        docsCompleted: docCount,
        chunksCommitted: vectorsMid,
        totalVectors: vectorsMid,
        checkpointStatus: cpStatus,
        consecutiveSequenceVerified: allConsecutive,
        hashes: Array.from(seqByHash.entries()).map(([hash, seqs]) => ({ hash, seqs })),
      };

      if (barrierDocs !== null) {
        console.log(`[BARRIER_REACHED] Committed ${docCount} docs, ${vectorsMid} vectors; active checkpoint ${cpStatus}`);
        if (stopMode === "pause") {
          // Pause and wait for parent process termination
          while (true) {
            await new Promise(r => setTimeout(r, 200));
          }
        }
      }

      await store.close();
      store = null;
      return result;

    } else if (action === "resume") {
      const activeCpBefore = getActiveCheckpoint(store.internal.db);
      if (!activeCpBefore) {
        console.warn("[Recovery Worker] Warning: Resuming without active checkpoint on DB");
      }

      const jobResult = await runDurableIndexingJob(store.internal, {
        maxDocsPerBatch: 2,
      });

      const vectorsFinal = (store.internal.db.prepare("SELECT count(*) as c FROM content_vectors").get() as any).c;
      const activeCpAfter = getActiveCheckpoint(store.internal.db);

      const hashesRows = store.internal.db.prepare(
        "SELECT hash, seq FROM content_vectors ORDER BY hash, seq"
      ).all() as Array<{ hash: string; seq: number }>;

      const seqByHash = new Map<string, number[]>();
      for (const row of hashesRows) {
        if (!seqByHash.has(row.hash)) seqByHash.set(row.hash, []);
        seqByHash.get(row.hash)!.push(row.seq);
      }

      let allConsecutive = true;
      for (const seqs of seqByHash.values()) {
        for (let i = 0; i < seqs.length; i++) {
          if (seqs[i] !== i) {
            allConsecutive = false;
            break;
          }
        }
      }

      const result: RecoveryWorkerResult = {
        status: jobResult.status === "complete" ? "completed" : "failed",
        dbPath,
        action,
        docsCompleted: jobResult.docsProcessed,
        chunksCommitted: jobResult.chunksCommitted,
        totalVectors: vectorsFinal,
        checkpointStatus: activeCpAfter ? activeCpAfter.status : null,
        consecutiveSequenceVerified: allConsecutive,
        hashes: Array.from(seqByHash.entries()).map(([hash, seqs]) => ({ hash, seqs })),
      };

      console.log(`[RESUME_COMPLETE] Resumed to ${vectorsFinal} vectors; checkpoint cleared: ${activeCpAfter === null}`);
      await store.close();
      store = null;
      return result;
    } else {
      throw new Error(`Unknown action: ${action}`);
    }

  } catch (err: any) {
    if (store) {
      try { await store.close(); } catch {}
    }
    const errMsg = err?.message || String(err);
    console.error(`[Recovery Worker Error] ${errMsg}`);
    return {
      status: "failed",
      dbPath,
      action,
      docsCompleted: 0,
      chunksCommitted: 0,
      totalVectors: 0,
      checkpointStatus: null,
      consecutiveSequenceVerified: false,
      hashes: [],
      error: errMsg,
    };
  }
}

// CLI Entrypoint
if (import.meta.url === `file://${process.argv[1]}`) {
  const args = process.argv.slice(2);
  const getArg = (flag: string): string | undefined => {
    const idx = args.indexOf(flag);
    return idx !== -1 && args[idx + 1] ? args[idx + 1] : undefined;
  };

  const launchToken = getArg("--launch-token") || process.env.QMD_PHASE5_LAUNCH_TOKEN;
  if (!launchToken || !launchToken.trim()) {
    console.error("Error: Direct TypeScript recovery worker execution rejected. Requires supervisor approval via --launch-token or QMD_PHASE5_LAUNCH_TOKEN.");
    process.exit(1);
  }

  const dbPath = getArg("--db");
  if (!dbPath) {
    console.error("Error: --db <path> is required.");
    process.exit(1);
  }

  const docsDir = getArg("--docs-dir");
  const mlxPort = getArg("--mlx-port") ? parseInt(getArg("--mlx-port")!, 10) : 8797;
  const mlxModelPath = getArg("--mlx-model-path");
  const barrierDocs = getArg("--barrier-after-docs") ? parseInt(getArg("--barrier-after-docs")!, 10) : undefined;
  const stopMode = (getArg("--stop-mode") as "exit" | "pause") || "exit";
  const action = (getArg("--action") as "index" | "resume") || "index";
  const outputJson = getArg("--output-json");

  runRecoveryWorker({
    dbPath,
    docsDir,
    mlxPort,
    mlxModelPath,
    barrierAfterDocs: barrierDocs,
    stopMode,
    action,
    outputJson,
    launchToken,
  }).then(res => {
    if (outputJson) {
      writeFileSync(outputJson, JSON.stringify(res, null, 2));
    }
    process.exit(res.status === "failed" ? 1 : 0);
  }).catch(err => {
    console.error("Recovery worker top-level error:", err);
    process.exit(1);
  });
}
