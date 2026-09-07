// Same-model head-to-head: GGUF Qwen3-Embedding-4B Q4_K_M (llama.cpp) vs
// MLX Qwen3-Embedding-4B affine-4bit, identical texts, warm process.
// Run: bun scripts/timing-embed-4b-gguf.ts
import { LlamaCpp } from "../dist/llm.js";

const llm = new LlamaCpp({
  embedBackend: "gguf",
  mlxFallback: false,
  embedModel: "/Users/shersingh/.cache/qmd/models/hf_Qwen_Qwen3-Embedding-4B-Q4_K_M.gguf",
});

const texts = Array.from({ length: 32 }, (_, i) =>
  `Benchmark document ${i}: distributed systems trade consistency for availability under partition, indexing strategies, and CAP theorem discussions.`
);

// Warmup (model load + first inference)
let t0 = Date.now();
await llm.embedBatch(texts.slice(0, 4));
console.log(JSON.stringify({ warmup_ms: Date.now() - t0 }));

for (let rep = 0; rep < 2; rep++) {
  t0 = Date.now();
  const embs = await llm.embedBatch(texts);
  const ms = Date.now() - t0;
  console.log(JSON.stringify({
    run: rep, embed32_gguf_4b: {
      ms,
      textsPerSec: Math.round(32000 / ms * 10) / 10,
      nulls: embs.filter(e => !e).length,
    }
  }));
}
process.exit(0);