/**
 * checkpoint.ts — Durable indexing checkpoint and fingerprint validation
 *
 * Tracks indexing job state, content/chunker/descriptor fingerprints,
 * and chunk commit boundaries to enable safe resume after interruption.
 */

import { createHash } from "node:crypto";
import type { Database } from "../db.js";
import { type EmbeddingDescriptor, computeEmbeddingSpaceId } from "../embedding/contract.js";
import type { ChunkStrategy } from "../store.js";

export interface IndexingFingerprint {
  model: string;
  dimensions: number;
  chunkStrategy?: ChunkStrategy;
  descriptorSignature: string;
  revision?: string;
  quantization?: string;
}

export interface IndexingJobCheckpoint {
  jobId: string;
  fingerprint: IndexingFingerprint;
  startedAt: string;
  updatedAt: string;
  status: "in_progress" | "paused" | "completed" | "failed" | "cancelled";
  docsTotal: number;
  docsCompleted: number;
  chunksTotal: number;
  chunksCommitted: number;
  lastCommittedHash?: string;
}

export function computeDescriptorSignature(descriptor?: EmbeddingDescriptor): string {
  if (!descriptor) return "legacy-unversioned";
  return computeEmbeddingSpaceId(descriptor);
}

export function initCheckpointTable(db: Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS indexing_checkpoints (
      job_id TEXT PRIMARY KEY,
      fingerprint TEXT NOT NULL,
      status TEXT NOT NULL,
      docs_total INTEGER NOT NULL,
      docs_completed INTEGER NOT NULL,
      chunks_total INTEGER NOT NULL,
      chunks_committed INTEGER NOT NULL,
      last_hash TEXT,
      started_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
  `);
}

export function saveCheckpoint(db: Database, cp: IndexingJobCheckpoint): void {
  initCheckpointTable(db);
  const stmt = db.prepare(`
    INSERT OR REPLACE INTO indexing_checkpoints (
      job_id, fingerprint, status, docs_total, docs_completed,
      chunks_total, chunks_committed, last_hash, started_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  stmt.run(
    cp.jobId,
    JSON.stringify(cp.fingerprint),
    cp.status,
    cp.docsTotal,
    cp.docsCompleted,
    cp.chunksTotal,
    cp.chunksCommitted,
    cp.lastCommittedHash || null,
    cp.startedAt,
    cp.updatedAt
  );
}

export function getActiveCheckpoint(db: Database): IndexingJobCheckpoint | null {
  initCheckpointTable(db);
  const row = db.prepare(`
    SELECT * FROM indexing_checkpoints
    WHERE status IN ('in_progress', 'paused', 'cancelled', 'failed')
    ORDER BY updated_at DESC LIMIT 1
  `).get() as any;

  if (!row) return null;
  return {
    jobId: row.job_id,
    fingerprint: JSON.parse(row.fingerprint),
    status: row.status,
    docsTotal: row.docs_total,
    docsCompleted: row.docs_completed,
    chunksTotal: row.chunks_total,
    chunksCommitted: row.chunks_committed,
    lastCommittedHash: row.last_hash || undefined,
    startedAt: row.started_at,
    updatedAt: row.updated_at,
  };
}
