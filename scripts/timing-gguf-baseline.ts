// GGUF head-to-head timing: embed 32 texts + rerank 15/40 docs.
// Run: bun scripts/timing-gguf-baseline.ts  (repo root, uses dist/)
// Prints JSON timings. No index mutation. Loads GGUF embed + rerank models.
import { LlamaCpp } from "../dist/llm.js";
import { readFileSync, readdirSync } from "fs";

const llm = new LlamaCpp({ embedBackend: "gguf", mlxFallback: false });

const texts = Array.from({ length: 32 }, (_, i) =>
  `Benchmark document ${i}: distributed systems trade consistency for availability under partition.`
);

let t0 = Date.now();
const embs = await llm.embedBatch(texts);
const embedMs = Date.now() - t0;
console.log(JSON.stringify({
  embed32: { ms: embedMs, textsPerSec: Math.round(32000 / embedMs * 10) / 10, nulls: embs.filter(e => !e).length }
}));

const files = readdirSync("test/eval-docs").filter(f => f.endsWith(".md"));
const bodies = files.map(f => readFileSync(`test/eval-docs/${f}`, "utf-8"));
const mkDocs = (n: number) => Array.from({ length: n }, (_, i) => ({
  file: `doc${i}.md`,
  text: bodies[i % bodies.length]!,
}));

for (const n of [15, 40]) {
  t0 = Date.now();
  const rr = await llm.rerank("tradeoff between data consistency and availability", mkDocs(n));
  const ms = Date.now() - t0;
  console.log(JSON.stringify({
    [`rerank${n}`]: { ms, pairsPerSec: Math.round(n / ms * 1000 * 100) / 100, top: (rr.results[0] as any)?.file }
  }));
}
process.exit(0);
