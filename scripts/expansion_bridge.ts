#!/usr/bin/env bun
/**
 * expansion_bridge.ts — Shared Contract CLI Bridge for TS Expansion Protocol
 *
 * Provides a lossless IPC bridge between Python test harnesses/smokes and the
 * pure TypeScript expansion protocol module (buildMlxExpansionPrompt & parseExpansionOutput).
 *
 * Usage:
 *   bun run scripts/expansion_bridge.ts --json < input.json
 *   bun run scripts/expansion_bridge.ts prompt "search query" "optional intent"
 *   bun run scripts/expansion_bridge.ts parse "search query" "raw completion text" [true/false]
 */

import { buildMlxExpansionPrompt, parseExpansionOutput } from "../src/expansion/protocol.js";

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString("utf-8");
}

async function main() {
  const args = process.argv.slice(2);

  if (args.includes("--json")) {
    const rawInput = await readStdin();
    if (!rawInput.trim()) {
      console.error(JSON.stringify({ error: "Empty JSON input received on stdin" }));
      process.exit(1);
    }
    const input = JSON.parse(rawInput);
    const action = input.action;

    if (action === "prompt") {
      const prompt = buildMlxExpansionPrompt(input.query || "", { intent: input.intent });
      console.log(JSON.stringify({ prompt }));
      return;
    }

    if (action === "parse") {
      const parsed = parseExpansionOutput(input.text || "", {
        query: input.query,
        includeLexical: input.includeLexical ?? true,
      });
      console.log(JSON.stringify(parsed));
      return;
    }

    console.error(JSON.stringify({ error: `Unknown action: ${action}` }));
    process.exit(1);
  }

  const cmd = args[0];
  if (cmd === "prompt") {
    const query = args[1] || "";
    const intent = args[2];
    const prompt = buildMlxExpansionPrompt(query, { intent });
    console.log(prompt);
    return;
  }

  if (cmd === "parse") {
    const query = args[1] || "";
    const text = args[2] || "";
    const includeLexical = args[3] !== "false";
    const parsed = parseExpansionOutput(text, { query, includeLexical });
    console.log(JSON.stringify(parsed, null, 2));
    return;
  }

  console.error("Usage: expansion_bridge.ts [--json | prompt <query> [intent] | parse <query> <text> [includeLexical]]");
  process.exit(1);
}

main().catch((err) => {
  console.error(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
  process.exit(1);
});
