#!/usr/bin/env node
/**
 * gguf_embed_server.mjs — Isolated HTTP Server for GGUF Embedding Qualification
 *
 * Implements the standard QMD embedding HTTP protocol over node-llama-cpp with Metal acceleration.
 * Exposes:
 *   - Inference port (/embed, /tokenize)
 *   - Dedicated control port (/health, /ready, /descriptor, /memory)
 *   - L2 normalization, prompt formatting for Qwen3, boundary token verification (2048 limit),
 *     context pooling for concurrent requests, typed watchdog health fields, and clean lifecycle management.
 */

import http from "node:http";
import process from "node:process";
import { getLlama, LlamaLogLevel } from "node-llama-cpp";

function parseArgs() {
  const args = process.argv.slice(2);
  const options = {
    model: null,
    port: 8795,
    controlPort: 8796,
    host: "127.0.0.1",
    instanceToken: process.env.GGUF_INSTANCE_TOKEN || "default_token",
    contextSize: 2048,
    contexts: 2,
    synthetic: false,
    syntheticDims: 2560,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--model" && i + 1 < args.length) {
      options.model = args[++i];
    } else if (arg === "--port" && i + 1 < args.length) {
      options.port = parseInt(args[++i], 10);
    } else if (arg === "--control-port" && i + 1 < args.length) {
      options.controlPort = parseInt(args[++i], 10);
    } else if (arg === "--host" && i + 1 < args.length) {
      options.host = args[++i];
    } else if (arg === "--instance-token" && i + 1 < args.length) {
      options.instanceToken = args[++i];
    } else if (arg === "--context-size" && i + 1 < args.length) {
      options.contextSize = parseInt(args[++i], 10);
    } else if (arg === "--contexts" && i + 1 < args.length) {
      options.contexts = parseInt(args[++i], 10);
    } else if (arg === "--synthetic") {
      options.synthetic = true;
    } else if (arg === "--synthetic-dims" && i + 1 < args.length) {
      options.syntheticDims = parseInt(args[++i], 10);
    }
  }

  if (!options.synthetic && !options.model) {
    console.error("Error: --model <path> is required unless --synthetic is passed");
    process.exit(1);
  }

  return options;
}

function l2Normalize(vec) {
  let sumSq = 0;
  for (let i = 0; i < vec.length; i++) {
    sumSq += vec[i] * vec[i];
  }
  const norm = Math.sqrt(sumSq);
  if (norm === 0) return vec;
  const out = new Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    out[i] = vec[i] / norm;
  }
  return out;
}

function formatPrompt(text, isQuery) {
  if (isQuery) {
    return `Instruct: Retrieve relevant documents for the given query\nQuery: ${text}`;
  }
  return text;
}

async function main() {
  const options = parseArgs();

  let llama = null;
  let model = null;
  let contextPool = [];
  let nativeDims = options.synthetic ? options.syntheticDims : 2560;
  let isReady = false;

  let completedSequence = 0;
  let inFlightRequests = 0;
  let currentJobStartTime = null;
  let currentJobDesc = null;

  console.log(`[gguf-server] Starting GGUF embedding server on ${options.host}:${options.port} (control: ${options.controlPort})`);
  console.log(`[gguf-server] Synthetic mode: ${options.synthetic}, Model: ${options.model}`);

  if (!options.synthetic) {
    try {
      const t0 = performance.now();
      llama = await getLlama({
        build: "autoAttempt",
        logLevel: LlamaLogLevel.error,
        gpu: "auto",
      });
      console.log(`[gguf-server] llama initialized (GPU: ${llama.gpu})`);

      model = await llama.loadModel({ modelPath: options.model });
      console.log(`[gguf-server] Model loaded from ${options.model} in ${(performance.now() - t0).toFixed(2)}ms`);

      const numContexts = Math.max(1, options.contexts);
      for (let i = 0; i < numContexts; i++) {
        const ctx = await model.createEmbeddingContext({ contextSize: options.contextSize });
        contextPool.push(ctx);
      }
      console.log(`[gguf-server] Created ${contextPool.length} embedding context(s) with contextSize ${options.contextSize}`);

      // Probe native dimensions
      const probeRes = await contextPool[0].getEmbeddingFor("probe");
      nativeDims = probeRes.vector.length;
      console.log(`[gguf-server] Probed embedding dimensions: ${nativeDims}`);
      isReady = true;
    } catch (err) {
      console.error(`[gguf-server] Failed to initialize model:`, err);
      process.exit(1);
    }
  } else {
    isReady = true;
  }

  // Helper to count tokens
  function countTokens(text) {
    if (options.synthetic) {
      // In synthetic mode, approximate tokens by splitting on whitespace
      const words = text.trim().split(/\s+/).filter(Boolean);
      return words.length || 1;
    }
    const tokens = model.tokenize(text);
    return tokens.length;
  }

  // Acquire a context from the pool with round-robin / queue
  let contextIndex = 0;
  async function withContext(fn) {
    if (options.synthetic) {
      return await fn(null);
    }
    const ctx = contextPool[contextIndex % contextPool.length];
    contextIndex++;
    return await fn(ctx);
  }

  // Request body reader helper
  function readJsonBody(req, maxBytes = 10 * 1024 * 1024) {
    return new Promise((resolve, reject) => {
      let body = "";
      let bytes = 0;
      req.on("data", (chunk) => {
        bytes += chunk.length;
        if (bytes > maxBytes) {
          reject(new Error("Request body too large"));
          req.destroy();
          return;
        }
        body += chunk;
      });
      req.on("end", () => {
        try {
          if (!body) return resolve({});
          resolve(JSON.parse(body));
        } catch (err) {
          reject(err);
        }
      });
      req.on("error", reject);
    });
  }

  function sendJson(res, statusCode, data) {
    const jsonStr = JSON.stringify(data);
    res.writeHead(statusCode, {
      "Content-Type": "application/json; charset=utf-8",
      "Content-Length": Buffer.byteLength(jsonStr),
      "Connection": "close",
    });
    res.end(jsonStr);
  }

  function getHealthObject() {
    return {
      status: isReady ? "ok" : "starting",
      ready: isReady,
      instance_token: options.instanceToken,
      pid: process.pid,
      model: options.model || "synthetic",
      backend: "gguf",
      worker_idle: inFlightRequests === 0,
      worker_alive: true,
      completed_sequence: completedSequence,
      active_job_age_s: currentJobStartTime ? Math.round(((performance.now() - currentJobStartTime) / 1000) * 100) / 100 : null,
      active_job_description: currentJobDesc,
      queue_depth: 0,
      is_overloaded: false,
    };
  }

  // --- Control HTTP Server ---
  const controlServer = http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${options.host}:${options.controlPort}`);
    const pathname = url.pathname.replace(/\/$/, "");

    if (req.method !== "GET") {
      return sendJson(res, 405, { error: "Method not allowed on control port", type: "method_not_allowed" });
    }

    if (pathname === "/health" || pathname === "") {
      return sendJson(res, 200, getHealthObject());
    }

    if (pathname === "/ready") {
      if (isReady) {
        return sendJson(res, 200, { ready: true });
      } else {
        return sendJson(res, 503, { ready: false, state: "starting" });
      }
    }

    if (pathname === "/descriptor") {
      let quant = "Q4_K_M";
      if (options.model && options.model.includes("Q8_0")) {
        quant = "Q8_0";
      } else if (options.model && options.model.includes("Q4_K_M")) {
        quant = "Q4_K_M";
      }
      return sendJson(res, 200, {
        version: 1,
        backend: "gguf",
        model: options.model || "synthetic-qwen3-4b",
        revision: "unknown",
        pooling: "last_token",
        nativeDimensions: nativeDims,
        outputDimensions: nativeDims,
        maxTokens: options.contextSize,
        normalized: true,
        dtype: "float32",
        quantization: quant,
      });
    }

    if (pathname === "/memory") {
      let vramInfo = { total: 0, free: 0, used: 0 };
      if (llama && typeof llama.getVramState === "function") {
        try {
          vramInfo = await llama.getVramState();
        } catch {}
      }
      const mem = process.memoryUsage();
      return sendJson(res, 200, {
        active_mb: Math.round((mem.rss / (1024 * 1024)) * 10) / 10,
        peak_mb: Math.round((mem.rss / (1024 * 1024)) * 10) / 10,
        model_mb: Math.round((vramInfo.used / (1024 * 1024)) * 10) / 10,
        vram_free_mb: Math.round((vramInfo.free / (1024 * 1024)) * 10) / 10,
        vram_total_mb: Math.round((vramInfo.total / (1024 * 1024)) * 10) / 10,
      });
    }

    return sendJson(res, 404, { error: `Not found: ${pathname}` });
  });

  // --- Inference HTTP Server ---
  const inferenceServer = http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${options.host}:${options.port}`);
    const pathname = url.pathname.replace(/\/$/, "");

    if (req.method === "GET") {
      if (pathname === "/health") {
        return sendJson(res, 200, getHealthObject());
      }
      return sendJson(res, 404, { error: `Not found: ${pathname}` });
    }

    if (req.method !== "POST") {
      return sendJson(res, 405, { error: "Method not allowed", type: "method_not_allowed" });
    }

    let payload;
    try {
      payload = await readJsonBody(req);
    } catch (err) {
      return sendJson(res, 400, { error: `Invalid JSON body: ${err.message}`, type: "invalid_input" });
    }

    inFlightRequests++;
    currentJobStartTime = performance.now();
    currentJobDesc = pathname;

    try {
      if (pathname === "/tokenize") {
        const texts = payload.texts;
        if (!Array.isArray(texts) || texts.length === 0) {
          return sendJson(res, 400, { error: "'texts' must be a non-empty array of strings", type: "invalid_input" });
        }

        const counts = [];
        for (let i = 0; i < texts.length; i++) {
          const text = texts[i];
          if (typeof text !== "string") {
            return sendJson(res, 400, { error: `Text at index ${i} is not a string`, type: "invalid_input" });
          }
          const cnt = countTokens(text);
          if (cnt > options.contextSize) {
            return sendJson(res, 400, {
              error: `Text at index ${i} token length (${cnt}) exceeds max_length (${options.contextSize})`,
              type: "invalid_input",
            });
          }
          counts.push(cnt);
        }
        return sendJson(res, 200, { counts });
      }

      if (pathname === "/embed") {
        const texts = payload.texts;
        const isQuery = !!payload.is_query;

        if (!Array.isArray(texts) || texts.length === 0) {
          return sendJson(res, 400, { error: "'texts' must be a non-empty array of strings", type: "invalid_input" });
        }

        // Check max length across all texts
        for (let i = 0; i < texts.length; i++) {
          const text = texts[i];
          if (typeof text !== "string") {
            return sendJson(res, 400, { error: `Text at index ${i} is not a string`, type: "invalid_input" });
          }
          const cnt = countTokens(text);
          if (cnt > options.contextSize) {
            return sendJson(res, 400, {
              error: `Text at index ${i} token length (${cnt}) exceeds max_length (${options.contextSize})`,
              type: "invalid_input",
            });
          }
        }

        try {
          const embeddings = [];
          if (options.synthetic) {
            // Add a small synthetic workload delay so concurrent test clients can overlap
            if (texts.length > 1) {
              await new Promise((resolve) => setTimeout(resolve, 30 * texts.length));
            }
            for (let i = 0; i < texts.length; i++) {
              // Generate deterministic synthetic unit vector
              const vec = new Array(nativeDims);
              for (let d = 0; d < nativeDims; d++) {
                vec[d] = Math.sin(i + 1 + d * 0.01);
              }
              embeddings.push(l2Normalize(vec));
            }
          } else {
            // Parallel embedding using available context pool
            const results = await Promise.all(
              texts.map((text) =>
                withContext(async (ctx) => {
                  const prompt = formatPrompt(text, isQuery);
                  const res = await ctx.getEmbeddingFor(prompt);
                  return l2Normalize(Array.from(res.vector));
                })
              )
            );
            embeddings.push(...results);
          }

          return sendJson(res, 200, { embeddings });
        } catch (err) {
          console.error("[gguf-server] Embedding error:", err);
          return sendJson(res, 500, { error: `Embedding execution failed: ${err.message}`, type: "internal_error" });
        }
      }

      return sendJson(res, 404, { error: `Not found: ${pathname}` });
    } finally {
      inFlightRequests--;
      completedSequence++;
      if (inFlightRequests === 0) {
        currentJobStartTime = null;
        currentJobDesc = null;
      }
    }
  });

  // Start listeners
  controlServer.listen(options.controlPort, options.host, () => {
    console.log(`[gguf-server] Control listener ready on http://${options.host}:${options.controlPort}`);
  });

  inferenceServer.listen(options.port, options.host, () => {
    console.log(`[gguf-server] Inference listener ready on http://${options.host}:${options.port}`);
  });

  // Graceful shutdown handling
  let isShuttingDown = false;
  async function cleanup() {
    if (isShuttingDown) return;
    isShuttingDown = true;
    console.log("[gguf-server] Shutting down...");

    controlServer.close();
    inferenceServer.close();

    for (const ctx of contextPool) {
      try {
        await ctx.dispose();
      } catch {}
    }
    contextPool = [];

    if (model) {
      try {
        await model.dispose();
      } catch {}
    }
    console.log("[gguf-server] Cleanup complete.");
    process.exit(0);
  }

  process.on("SIGTERM", cleanup);
  process.on("SIGINT", cleanup);
}

main().catch((err) => {
  console.error("[gguf-server] Fatal error:", err);
  process.exit(1);
});
