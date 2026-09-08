/**
 * embed-outcome.test.ts — Regression tests for truthful bulk outcomes and atomic counters
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, unlink, writeFile, readdir, rmdir, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import http from "node:http";
import YAML from "yaml";
import {
  createStore,
  generateEmbeddings,
  type Store,
} from "../src/store.js";
import { setDefaultLlamaCpp } from "../src/llm.js";
import type { CollectionConfig } from "../src/collections.js";

const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const qmdScript = join(projectRoot, "src", "cli", "qmd.ts");
const tsxBin = join(projectRoot, "node_modules", ".bin", "tsx");

let testDir: string;
let testConfigDir: string;
let currentStore: Store | null = null;

async function createTestStore(): Promise<Store> {
  const dbPath = join(testDir, `test-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
  const configPrefix = join(testDir, `config-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  testConfigDir = await mkdtemp(configPrefix);
  process.env.QMD_CONFIG_DIR = testConfigDir;

  const emptyConfig: CollectionConfig = { collections: {} };
  await writeFile(join(testConfigDir, "index.yml"), YAML.stringify(emptyConfig));

  const store = createStore(dbPath);
  currentStore = store;
  return store;
}

async function cleanupStore(store: Store): Promise<void> {
  currentStore = null;
  store.close();
  try {
    await unlink(store.dbPath);
  } catch {}

  try {
    const files = await readdir(testConfigDir);
    for (const file of files) {
      await unlink(join(testConfigDir, file));
    }
    await rmdir(testConfigDir);
  } catch {}

  delete process.env.QMD_CONFIG_DIR;
}

function createFakeTokenizer() {
  return {
    tokenize: (text: string) => text.split(/\s+/).map((_, i) => i + 1),
    detokenize: (tokens: number[]) => tokens.join(" "),
  };
}

async function insertDoc(db: any, name: string, body: string, hash: string) {
  const now = new Date().toISOString();
  db.prepare(`
    INSERT OR IGNORE INTO content (hash, doc, created_at)
    VALUES (?, ?, ?)
  `).run(hash, body, now);

  db.prepare(`
    INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
    VALUES ('docs', ?, ?, ?, ?, ?, 1)
  `).run(`${name}.md`, name, hash, now, now);
}

describe("Task 2: Truthful Bulk Outcomes and Atomic Counters", () => {
  beforeEach(async () => {
    testDir = await mkdtemp(join(tmpdir(), "qmd-outcome-test-"));
  });

  afterEach(async () => {
    if (currentStore) {
      await cleanupStore(currentStore);
    }
    setDefaultLlamaCpp(null);
  });

  test("returns complete status with 0 counts on empty database", async () => {
    const store = await createTestStore();
    const res = await generateEmbeddings(store);
    expect(res.status).toBe("complete");
    expect(res.docsProcessed).toBe(0);
    expect(res.docsSelected).toBe(0);
    expect(res.chunksCommitted).toBe(0);
    expect(res.chunksFailed).toBe(0);
    expect(res.chunksSkipped).toBe(0);
    expect(res.errors).toBe(0);
  });

  test("partial failure reports status 'partial', exact committed/failed counts, and does not report unattempted work as failed", async () => {
    const store = await createTestStore();
    const db = store.db;

    for (let i = 1; i <= 3; i++) {
      await insertDoc(db, `Doc ${i}`, `# Doc ${i}\n\nContent ${i}`, `hash${i}`);
    }

    setDefaultLlamaCpp(createFakeTokenizer() as any);

    let callCount = 0;
    const fakeLlm = {
      embedModelName: "mock-model",
      embedBatch: async (texts: string[]) => {
        callCount++;
        return texts.map((t) => {
          if (t.includes("Doc 2")) {
            return null; // simulated inference failure for doc 2
          }
          return { embedding: [0.1, 0.2, 0.3], model: "mock-model" };
        });
      },
      embed: async (text: string) => {
        if (text.includes("Doc 2")) return null;
        return { embedding: [0.1, 0.2, 0.3], model: "mock-model" };
      },
    };
    store.llm = fakeLlm as any;

    const res = await generateEmbeddings(store);
    expect(res.status).toBe("partial");
    expect(res.docsSelected).toBe(3);
    expect(res.docsProcessed).toBe(2); // Only doc 1 and doc 3 fully succeeded
    expect(res.chunksCommitted).toBe(2);
    expect(res.chunksFailed).toBe(1);
    expect(res.chunksSkipped).toBe(0);
    expect(res.firstFailureReason).toBeDefined();

    const count = db.prepare("SELECT COUNT(*) as count FROM content_vectors").get() as { count: number };
    expect(count.count).toBe(2);
  });

  test("database write error triggers rollback and does not advance committed counters or retry GPU calls", async () => {
    const store = await createTestStore();
    const db = store.db;

    await insertDoc(db, "Doc 1", "# Doc 1\n\nContent 1", "hash1");

    setDefaultLlamaCpp(createFakeTokenizer() as any);

    let gpuInferenceCalls = 0;
    const fakeLlm = {
      embedModelName: "mock-model",
      getDescriptor: async () => ({
        version: 1,
        backend: "mlx",
        model: "mock-model",
        pooling: "mean",
        nativeDimensions: 3,
        outputDimensions: 3,
        maxTokens: 512,
        normalized: true,
      }),
      embedBatch: async (texts: string[]) => {
        gpuInferenceCalls++;
        return texts.map(() => ({ embedding: [0.1, 0.2, 0.3], model: "mock-model" }));
      },
      embed: async () => {
        gpuInferenceCalls++;
        return { embedding: [0.1, 0.2, 0.3], model: "mock-model" };
      },
    };
    store.llm = fakeLlm as any;

    store.ensureVecTable(3);
    db.exec(`CREATE TRIGGER block_insert BEFORE INSERT ON content_vectors BEGIN SELECT RAISE(FAIL, 'simulated disk write error'); END;`);

    const res = await generateEmbeddings(store);
    expect(res.status).toBe("failed");
    expect(res.chunksCommitted).toBe(0);
    expect(res.chunksFailed).toBe(1);
    expect(res.firstFailureReason).toContain("simulated disk write error");
    expect(gpuInferenceCalls).toBe(1);
  });

  test("high error rate abort marks unattempted items as skipped, not failed", async () => {
    const store = await createTestStore();
    const db = store.db;

    for (let i = 1; i <= 40; i++) {
      await insertDoc(db, `Doc ${i}`, `# Doc ${i}\n\nContent ${i}`, `hash${i}`);
    }

    setDefaultLlamaCpp(createFakeTokenizer() as any);

    const fakeLlm = {
      embedModelName: "mock-model",
      getDescriptor: async () => ({
        version: 1,
        backend: "mlx",
        model: "mock-model",
        pooling: "mean",
        nativeDimensions: 3,
        outputDimensions: 3,
        maxTokens: 512,
        normalized: true,
      }),
      embedBatch: async (texts: string[]) => {
        return texts.map(() => null);
      },
      embed: async () => null,
    };
    store.llm = fakeLlm as any;

    const res = await generateEmbeddings(store);
    expect(res.status).toBe("failed");
    expect(res.chunksAttempted).toBe(32);
    expect(res.chunksFailed).toBe(32);
    expect(res.chunksSkipped).toBe(8);
    expect(res.chunksCommitted).toBe(0);
    expect(res.firstFailureReason).toBeDefined();
  });

  test("CLI subprocess exits nonzero and prints truthful failure summary on partial/failed runs", async () => {
    // Start a mock MLX server on ephemeral port that responds with error
    let mockServer: http.Server;
    let mockPort = 0;

    await new Promise<void>((resolve) => {
      mockServer = http.createServer((req, res) => {
        const url = new URL(req.url ?? "/", `http://${req.headers.host || "127.0.0.1"}`);
        if (url.pathname === "/health" || url.pathname === "/ready") {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            status: "ok",
            ready: true,
            model: "mock-model",
            dims: 4,
            descriptor: {
              version: 1,
              backend: "mlx",
              model: "mock-model",
              pooling: "mean",
              nativeDimensions: 4,
              outputDimensions: 4,
              maxTokens: 512,
              normalized: true,
            },
          }));
          return;
        }
        if (url.pathname === "/descriptor") {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            version: 1,
            backend: "mlx",
            model: "mock-model",
            pooling: "mean",
            nativeDimensions: 4,
            outputDimensions: 4,
            maxTokens: 512,
            normalized: true,
          }));
          return;
        }
        if (url.pathname === "/embed" || url.pathname === "/embed-bin") {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Simulated GPU inference crash" }));
          return;
        }
        res.writeHead(404);
        res.end();
      });

      mockServer.listen(0, "127.0.0.1", () => {
        const addr = mockServer.address() as any;
        mockPort = addr.port;
        resolve();
      });
    });

    try {
      const cliDbPath = join(testDir, `cli-test-${Date.now()}.sqlite`);
      const cliConfigDir = join(testDir, `cli-config-${Date.now()}`);
      const notesDir = join(testDir, "notes");
      await mkdir(cliConfigDir, { recursive: true });
      await mkdir(notesDir, { recursive: true });

      await writeFile(join(notesDir, "test.md"), "# Test Note\n\nSome test content to embed.");

      const cliConfig: CollectionConfig = {
        collections: {
          notes: {
            path: notesDir,
            pattern: "**/*.md",
          },
        },
      };
      await writeFile(join(cliConfigDir, "index.yml"), YAML.stringify(cliConfig));

      // Run qmd update to index document
      await new Promise<void>((resolve, reject) => {
        const proc = spawn(tsxBin, [qmdScript, "update"], {
          env: {
            ...process.env,
            INDEX_PATH: cliDbPath,
            QMD_CONFIG_DIR: cliConfigDir,
            PWD: testDir,
          },
          stdio: ["ignore", "pipe", "pipe"],
        });
        proc.on("close", (code) => {
          if (code === 0) resolve();
          else reject(new Error(`qmd update failed with code ${code}`));
        });
      });

      // Now run qmd embed with MLX backend pointing to mock server that fails
      const embedRun = await new Promise<{ stdout: string; stderr: string; code: number }>((resolve) => {
        const proc = spawn(tsxBin, [qmdScript, "embed"], {
          env: {
            ...process.env,
            INDEX_PATH: cliDbPath,
            QMD_CONFIG_DIR: cliConfigDir,
            PWD: testDir,
            QMD_EMBED_BACKEND: "mlx",
            QMD_MLX_EMBED_URL: `http://127.0.0.1:${mockPort}`,
            QMD_MLX_FALLBACK: "0",
          },
          stdio: ["ignore", "pipe", "pipe"],
        });
        let stdout = "";
        let stderr = "";
        proc.stdout?.on("data", (d) => { stdout += d.toString(); });
        proc.stderr?.on("data", (d) => { stderr += d.toString(); });
        proc.on("close", (code) => {
          resolve({ stdout, stderr, code: code ?? 0 });
        });
      });

      expect(embedRun.code).not.toBe(0);
      expect(embedRun.stdout).not.toContain("✓ Done!");
      expect(embedRun.stdout).toContain("failed");
    } finally {
      mockServer!.close();
    }
  });
});
