/**
 * job.ts — Durable indexing job orchestration and shadow database protections
 */

import { randomUUID } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { Database } from "../db.js";
import type { Store, EmbedResult, ChunkStrategy, EmbedProgress } from "../store.js";
import { generateEmbeddings } from "../store.js";
import {
  type IndexingJobCheckpoint,
  type IndexingFingerprint,
  computeDescriptorSignature,
  getActiveCheckpoint,
  saveCheckpoint,
} from "./checkpoint.js";
import type { EmbeddingDescriptor } from "../embedding/contract.js";

export interface IndexingJobOptions {
  jobId?: string;
  force?: boolean;
  model?: string;
  chunkStrategy?: ChunkStrategy;
  maxDocsPerBatch?: number;
  maxBatchBytes?: number;
  maxDuration?: number;
  onProgress?: (info: EmbedProgress, checkpoint?: IndexingJobCheckpoint) => void;
  signal?: AbortSignal;
}

export class ShadowDatabaseProtectionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ShadowDatabaseProtectionError";
  }
}

export class IndexingFingerprintMismatchError extends Error {
  readonly existingFingerprint: IndexingFingerprint;
  readonly expectedFingerprint: Partial<IndexingFingerprint>;
  readonly details: {
    sigMismatch: boolean;
    modelMismatch: boolean;
    strategyMismatch: boolean;
    dimMismatch: boolean;
  };

  constructor(
    message: string,
    existingFingerprint: IndexingFingerprint,
    expectedFingerprint: Partial<IndexingFingerprint>,
    details: {
      sigMismatch: boolean;
      modelMismatch: boolean;
      strategyMismatch: boolean;
      dimMismatch: boolean;
    }
  ) {
    super(message);
    this.name = "IndexingFingerprintMismatchError";
    this.existingFingerprint = existingFingerprint;
    this.expectedFingerprint = expectedFingerprint;
    this.details = details;
  }
}

export { IndexingFingerprintMismatchError as FingerprintMismatchError };

/**
 * Validates that a target database path is an explicit, safe shadow path
 * and NOT the active/live database file, symlink, or hardlink alias.
 */
export function validateShadowTarget(targetDbPath: string, liveDbPath?: string): void {
  if (!targetDbPath || typeof targetDbPath !== "string") {
    throw new ShadowDatabaseProtectionError("Shadow target database path must be a non-empty string.");
  }

  const defaultLivePath = path.resolve(process.env.HOME || "", ".cache/qmd/index.sqlite");
  const livePath = path.resolve(liveDbPath || defaultLivePath);

  // 1. Resolve canonical realpath for target and live
  let resolvedTarget = path.resolve(targetDbPath);
  try {
    if (fs.existsSync(resolvedTarget)) {
      resolvedTarget = fs.realpathSync(resolvedTarget);
    } else {
      const parentDir = path.dirname(resolvedTarget);
      if (fs.existsSync(parentDir)) {
        resolvedTarget = path.join(fs.realpathSync(parentDir), path.basename(resolvedTarget));
      }
    }
  } catch {}

  let resolvedLive = livePath;
  try {
    if (fs.existsSync(livePath)) {
      resolvedLive = fs.realpathSync(livePath);
    }
  } catch {}

  if (resolvedTarget === resolvedLive) {
    throw new ShadowDatabaseProtectionError(
      `Refusing to execute shadow workflow on live database: ${resolvedTarget}`
    );
  }

  // 2. Check stat / inode match (hardlinks) if both files exist
  try {
    if (fs.existsSync(resolvedTarget) && fs.existsSync(resolvedLive)) {
      const targetStat = fs.statSync(resolvedTarget);
      const liveStat = fs.statSync(resolvedLive);
      if (targetStat.dev === liveStat.dev && targetStat.ino === liveStat.ino) {
        throw new ShadowDatabaseProtectionError(
          `Refusing to execute shadow workflow on hardlink/alias of live database: ${resolvedTarget}`
        );
      }
    }
  } catch (err) {
    if (err instanceof ShadowDatabaseProtectionError) throw err;
  }

  // 3. Ensure shadow target has explicit isolation marker
  if (
    resolvedTarget.endsWith("index.sqlite") &&
    !resolvedTarget.includes("shadow") &&
    !resolvedTarget.includes("test") &&
    !resolvedTarget.includes("tmp") &&
    !resolvedTarget.includes("scratch")
  ) {
    throw new ShadowDatabaseProtectionError(
      `Shadow path '${resolvedTarget}' does not contain explicit isolation marker ('shadow', 'test', 'tmp', or 'scratch')`
    );
  }
}

/**
 * Runs a resumable, durable indexing job with checkpoint tracking.
 */
export async function runDurableIndexingJob(
  store: Store,
  options?: IndexingJobOptions,
): Promise<EmbedResult & { checkpoint: IndexingJobCheckpoint }> {
  const db = store.db;
  const jobId = options?.jobId || randomUUID();
  const startedAt = new Date().toISOString();

  let descriptor: EmbeddingDescriptor | undefined = undefined;
  if (typeof store.getEmbeddingDescriptor === "function") {
    const d = await store.getEmbeddingDescriptor();
    descriptor = d ?? undefined;
  } else if (typeof (store as any).getDescriptor === "function") {
    const d = await (store as any).getDescriptor();
    descriptor = d ?? undefined;
  }

  const currentSig = computeDescriptorSignature(descriptor);
  const effectiveModel = options?.model || descriptor?.model || "default";
  const targetStrategy: ChunkStrategy = options?.chunkStrategy ?? "regex";

  let existing = getActiveCheckpoint(db);
  if (existing && !options?.force) {
    const existingStrategy = existing.fingerprint.chunkStrategy ?? "regex";
    const existingModel = existing.fingerprint.model || "default";
    const sigMismatch = existing.fingerprint.descriptorSignature !== currentSig;
    const modelMismatch = existingModel !== effectiveModel;
    const strategyMismatch = existingStrategy !== targetStrategy;
    const dimMismatch =
      descriptor?.outputDimensions !== undefined &&
      existing.fingerprint.dimensions > 0 &&
      existing.fingerprint.dimensions !== descriptor.outputDimensions;

    if (sigMismatch || modelMismatch || strategyMismatch || dimMismatch) {
      throw new IndexingFingerprintMismatchError(
        `Checkpoint fingerprint mismatch detected (descriptor: ${sigMismatch}, model: ${modelMismatch}, chunkStrategy: ${strategyMismatch}, dims: ${dimMismatch}). Automatic destructive migration is prohibited to prevent vector loss. Actionable message: explicit rebuild on isolated shadow index required (or specify options.force: true to explicitly overwrite).`,
        existing.fingerprint,
        {
          model: effectiveModel,
          dimensions: descriptor?.outputDimensions || 0,
          chunkStrategy: targetStrategy,
          descriptorSignature: currentSig,
        },
        { sigMismatch, modelMismatch, strategyMismatch, dimMismatch }
      );
    }
  }

  const cp: IndexingJobCheckpoint = (existing && !options?.force) ? existing : {
    jobId,
    fingerprint: {
      model: effectiveModel,
      dimensions: descriptor?.outputDimensions || 0,
      chunkStrategy: targetStrategy,
      descriptorSignature: currentSig,
      revision: descriptor?.revision || undefined,
      quantization: descriptor?.quantization || undefined,
    },
    startedAt,
    updatedAt: startedAt,
    status: "in_progress",
    docsTotal: 0,
    docsCompleted: 0,
    chunksTotal: 0,
    chunksCommitted: 0,
  };

  saveCheckpoint(db, cp);

  const result = await generateEmbeddings(store, {
    force: Boolean(options?.force),
    model: options?.model,
    chunkStrategy: options?.chunkStrategy ?? "regex",
    maxDocsPerBatch: options?.maxDocsPerBatch,
    maxBatchBytes: options?.maxBatchBytes,
    maxDuration: options?.maxDuration,
    signal: options?.signal,
    onProgress: (info) => {
      cp.chunksTotal = info.totalChunks;
      cp.chunksCommitted = info.chunksEmbedded;
      cp.updatedAt = new Date().toISOString();
      saveCheckpoint(db, cp);
      options?.onProgress?.(info, cp);
    },
  });

  cp.status = result.status === "complete" ? "completed" : result.status === "cancelled" ? "cancelled" : "failed";
  cp.docsCompleted = result.docsProcessed;
  cp.chunksCommitted = result.chunksCommitted;
  cp.updatedAt = new Date().toISOString();
  saveCheckpoint(db, cp);

  return {
    ...result,
    checkpoint: cp,
  };
}
