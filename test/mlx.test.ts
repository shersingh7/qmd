/**
 * mlx.test.ts — Unit tests for MLX client and wire protocol
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import {
  MlxEmbedClient,
  embedWithMlx,
  embedBatchWithMlx,
  embedBatchConcurrent,
  embedBatchBinaryWithMlx,
  mlxHealth,
  mlxMemory,
  mlxStats,
  isMlxAvailable,
  getMlxDescriptor,
} from "../src/mlx.js";
import {
  decodeBinaryEmbeddings,
  encodeBinaryEmbeddings,
  validateJsonEmbedResponse,
  EmbeddingProtocolError,
} from "../src/embedding/protocol.js";
import type { EmbeddingDescriptor } from "../src/embedding/contract.js";

import http from "node:http";

// ── Mock Server ─────────────────────────────────────────────────────────────

let server: http.Server;
let serverPort: number;
let serverUrl: string;

const mockDescriptor: EmbeddingDescriptor = {
  version: 1,
  backend: "mlx",
  model: "mlx-community/nomic-embed-text-v1.5",
  pooling: "mean",
  nativeDimensions: 768,
  outputDimensions: 768,
  maxTokens: 2048,
  normalized: true,
};

beforeAll(async () => {
  await new Promise<void>((resolve) => {
    server = http.createServer(async (req, res) => {
      const url = new URL(req.url ?? "/", `http://${req.headers.host || "127.0.0.1"}`);

      if (url.pathname === "/health") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            status: "ok",
            model: mockDescriptor.model,
            dims: mockDescriptor.outputDimensions,
            ready: true,
            descriptor: mockDescriptor,
          })
        );
        return;
      }

      if (url.pathname === "/ready") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ready: true }));
        return;
      }

      if (url.pathname === "/descriptor") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(mockDescriptor));
        return;
      }

      if (url.pathname === "/memory") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ active_mb: 250.5, peak_mb: 512.0, model_mb: 400.0 }));
        return;
      }

      if (url.pathname === "/stats") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ total_requests: 42, avg_ms: 3.2, compiled_shapes: 3, uptime_sec: 120 }));
        return;
      }

      if (url.pathname === "/embed") {
        const chunks: Buffer[] = [];
        for await (const chunk of req) chunks.push(chunk);
        const body = JSON.parse(Buffer.concat(chunks).toString("utf-8"));
        const texts = body.texts || [];
        const dims = body.dims || mockDescriptor.outputDimensions;
        const embeddings = texts.map((_: string, idx: number) =>
          Array.from({ length: dims }, (__, d) => (idx + 1) * 0.01 + d * 0.001)
        );
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            embeddings,
            model: mockDescriptor.model,
            dims,
          })
        );
        return;
      }

      if (url.pathname === "/embed-bin") {
        const chunks: Buffer[] = [];
        for await (const chunk of req) chunks.push(chunk);
        const body = JSON.parse(Buffer.concat(chunks).toString("utf-8"));
        const texts = body.texts || [];
        const dims = body.dims || mockDescriptor.outputDimensions;
        const embeddings = texts.map((_: string, idx: number) =>
          Array.from({ length: dims }, (__, d) => (idx + 1) * 0.01 + d * 0.001)
        );
        const bin = encodeBinaryEmbeddings(embeddings, dims);
        res.writeHead(200, { "Content-Type": "application/octet-stream" });
        res.end(bin);
        return;
      }

      if (url.pathname === "/slow") {
        setTimeout(() => {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true }));
        }, 5000);
        return;
      }

      res.writeHead(404);
      res.end("Not Found");
    });

    server.listen(0, "127.0.0.1", () => {
      const addr = server.address() as { port: number };
      serverPort = addr.port;
      serverUrl = `http://127.0.0.1:${serverPort}`;
      resolve();
    });
  });
});

afterAll(async () => {
  if (server) {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

// ── Tests ───────────────────────────────────────────────────────────────────

describe("Binary Protocol Framing & Validation", () => {
  test("encodes and decodes binary embeddings correctly", () => {
    const vectors = [
      [0.1, 0.2, 0.3, 0.4],
      [0.5, 0.6, 0.7, 0.8],
    ];
    const dims = 4;
    const bin = encodeBinaryEmbeddings(vectors, dims);
    expect(bin.byteLength).toBe(8 + 2 * 4 * 4); // 8-byte header + 32 bytes data

    const decoded = decodeBinaryEmbeddings(bin.buffer);
    expect(decoded.count).toBe(2);
    expect(decoded.dims).toBe(4);
    expect(decoded.vectors.length).toBe(2);
    expect(decoded.vectors[0]![0]).toBeCloseTo(0.1);
    expect(decoded.vectors[1]![3]).toBeCloseTo(0.8);
  });

  test("rejects truncated buffer (< 8 bytes)", () => {
    const truncated = new ArrayBuffer(6);
    expect(() => decodeBinaryEmbeddings(truncated)).toThrow(EmbeddingProtocolError);
  });

  test("rejects payload with mismatched length", () => {
    // Header claims 2 vectors of 4 dims (requires 40 bytes), but buffer is only 20 bytes
    const buffer = new ArrayBuffer(20);
    const view = new DataView(buffer);
    view.setInt32(0, 2, true);
    view.setInt32(4, 4, true);

    expect(() => decodeBinaryEmbeddings(buffer)).toThrow(EmbeddingProtocolError);
  });

  test("rejects NaN or Infinity in binary payload", () => {
    const badVectors = [
      [0.1, NaN, 0.3, 0.4],
    ];
    expect(() => encodeBinaryEmbeddings(badVectors, 4)).toThrow(EmbeddingProtocolError);
  });

  test("decodes empty payload cleanly", () => {
    const buffer = new ArrayBuffer(8);
    const view = new DataView(buffer);
    view.setInt32(0, 0, true);
    view.setInt32(4, 0, true);

    const decoded = decodeBinaryEmbeddings(buffer);
    expect(decoded.count).toBe(0);
    expect(decoded.dims).toBe(0);
    expect(decoded.vectors).toEqual([]);
  });
});

describe("JSON Protocol Validation", () => {
  test("validates valid JSON response", () => {
    const payload = {
      embeddings: [[0.1, 0.2], [0.3, 0.4]],
      model: "test-model",
      dims: 2,
    };
    const validated = validateJsonEmbedResponse(payload);
    expect(validated.embeddings.length).toBe(2);
    expect(validated.model).toBe("test-model");
    expect(validated.dims).toBe(2);
  });

  test("rejects non-object or missing embeddings", () => {
    expect(() => validateJsonEmbedResponse(null)).toThrow(EmbeddingProtocolError);
    expect(() => validateJsonEmbedResponse({ model: "x" })).toThrow(EmbeddingProtocolError);
  });

  test("rejects non-finite values in JSON embeddings", () => {
    expect(() =>
      validateJsonEmbedResponse({
        embeddings: [[0.1, "bad" as any]],
        dims: 2,
      })
    ).toThrow(EmbeddingProtocolError);
  });
});

describe("MLX Client Functions", () => {
  test("mlxHealth returns healthy status and descriptor", async () => {
    const h = await mlxHealth({ url: serverUrl });
    expect(h).not.toBeNull();
    expect(h?.ready).toBe(true);
    expect(h?.model).toBe(mockDescriptor.model);
    expect(h?.dims).toBe(mockDescriptor.outputDimensions);
    expect(h?.descriptor?.backend).toBe("mlx");
  });

  test("isMlxAvailable returns true for running server", async () => {
    const avail = await isMlxAvailable({ url: serverUrl });
    expect(avail).toBe(true);
  });

  test("isMlxAvailable returns false for unreachable port", async () => {
    const avail = await isMlxAvailable({ url: "http://127.0.0.1:59999" });
    expect(avail).toBe(false);
  });

  test("getMlxDescriptor fetches descriptor", async () => {
    const desc = await getMlxDescriptor({ url: serverUrl });
    expect(desc).not.toBeNull();
    expect(desc?.model).toBe(mockDescriptor.model);
    expect(desc?.outputDimensions).toBe(768);
  });

  test("mlxMemory and mlxStats return metadata", async () => {
    const mem = await mlxMemory({ url: serverUrl });
    expect(mem?.active_mb).toBe(250.5);

    const stats = await mlxStats({ url: serverUrl });
    expect(stats?.total_requests).toBe(42);
  });

  test("embedWithMlx embeds single text via JSON", async () => {
    const res = await embedWithMlx("hello world", { url: serverUrl, binary: false });
    expect(res).not.toBeNull();
    expect(res?.embedding.length).toBe(768);
    expect(res?.model).toBe(mockDescriptor.model);
  });

  test("embedBatchWithMlx embeds batch via binary protocol", async () => {
    const texts = Array.from({ length: 20 }, (_, i) => `document chunk ${i}`);
    const res = await embedBatchWithMlx(texts, { url: serverUrl, binary: true });
    expect(res.length).toBe(20);
    expect(res[0]?.embedding.length).toBe(768);
    expect(res[19]?.embedding.length).toBe(768);
  });

  test("embedBatchConcurrent maintains order and concurrency", async () => {
    const texts = Array.from({ length: 50 }, (_, i) => `text ${i}`);
    const res = await embedBatchConcurrent(texts, { url: serverUrl, concurrency: 3 }, undefined, 16);
    expect(res.length).toBe(50);
    for (let i = 0; i < 50; i++) {
      expect(res[i]).not.toBeNull();
      expect(res[i]?.embedding.length).toBe(768);
    }
  });

  test("embedBatchBinaryWithMlx returns raw float vectors", async () => {
    const texts = ["alpha", "beta", "gamma"];
    const res = await embedBatchBinaryWithMlx(texts, { url: serverUrl });
    expect(res).not.toBeNull();
    expect(res?.length).toBe(3);
    expect(res?.[0]?.length).toBe(768);
  });

  test("deadline timeout aborts slow requests cleanly", async () => {
    const start = Date.now();
    const res = await embedWithMlx("slow", { url: `${serverUrl}/slow`, timeoutMs: 150 });
    const duration = Date.now() - start;
    expect(res).toBeNull();
    expect(duration).toBeLessThan(1000); // timed out rapidly, did not wait 5s
  });
});

describe("MlxEmbedClient Class", () => {
  test("probeDims discovers dimensions and caches descriptor", async () => {
    const client = new MlxEmbedClient({ url: serverUrl });
    const dims = await client.probeDims();
    expect(dims).toBe(768);
    expect(client.modelName).toBe(mockDescriptor.model);
    expect(client.descriptor?.backend).toBe("mlx");
  });

  test("embed and embedBatch on client work seamlessly", async () => {
    const client = new MlxEmbedClient({ url: serverUrl });
    const single = await client.embed("test text");
    expect(single?.embedding.length).toBe(768);

    const batch = await client.embedBatch(["test 1", "test 2"]);
    expect(batch.length).toBe(2);
    expect(batch[0]?.embedding.length).toBe(768);
  });
});
