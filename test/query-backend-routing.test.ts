import { afterEach, describe, expect, test } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { createStore } from "../src/store.js";
import { createStore as createSdkStore } from "../src/index.js";
import { setDefaultLlamaCpp } from "../src/llm.js";
import { OpenAILLM } from "../src/openai-llm.js";
import { MlxLLM } from "../src/mlx-llm.js";

async function freshDbPath(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "qmd-routing-test-"));
  return join(dir, "test.sqlite");
}

afterEach(async () => {
  delete process.env.QMD_EMBED_PROVIDER;
  delete process.env.QMD_QUERY_PROVIDER;
  delete process.env.QMD_OPENAI_API_KEY;
  delete process.env.QMD_EMBED_MODEL;
  delete process.env.QMD_MLX_BASE_URL;
  setDefaultLlamaCpp(null);
});

describe("query backend routing", () => {
  test("store.expandQuery prefers store.queryLlm over OpenAI embed backend", async () => {
    const dbPath = await freshDbPath();
    const store = createStore(dbPath);

    const openaiLlm = {
      embedModelName: "text-embedding-3-large",
      embed: async () => null,
      embedBatch: async () => [],
    } as unknown as OpenAILLM;

    const queryLlm = {
      expandQuery: async (query: string) => [{ type: "vec", text: `${query} expanded` }],
      rerank: async () => ({ results: [], model: "fake-reranker" }),
      embedModelName: "fake-query-llm",
    } as unknown as MlxLLM;

    store.llm = openaiLlm;
    store.queryLlm = queryLlm;

    const expanded = await store.expandQuery("auth");
    expect(expanded).toEqual([{ type: "vec", query: "auth expanded" }]);

    store.close();
    await rm(dirname(dbPath), { recursive: true, force: true });
  });

  test("store.rerank prefers store.queryLlm over OpenAI embed backend", async () => {
    const dbPath = await freshDbPath();
    const store = createStore(dbPath);

    const openaiLlm = {
      embedModelName: "text-embedding-3-large",
      embed: async () => null,
      embedBatch: async () => [],
    } as unknown as OpenAILLM;

    const queryLlm = {
      expandQuery: async () => [],
      rerank: async (_query: string, documents: { file: string; text: string }[]) => ({
        results: documents.map((doc, index) => ({ file: doc.file, score: 100 - index, index })),
        model: "fake-reranker",
      }),
      embedModelName: "fake-query-llm",
    } as unknown as MlxLLM;

    store.llm = openaiLlm;
    store.queryLlm = queryLlm;

    const reranked = await store.rerank("auth", [
      { file: "a.md", text: "first" },
      { file: "b.md", text: "second" },
    ]);

    expect(reranked.map(r => r.file)).toEqual(["a.md", "b.md"]);

    store.close();
    await rm(dirname(dbPath), { recursive: true, force: true });
  });

  test("SDK createStore wires OpenAI embeddings to MLX query backend when requested", async () => {
    process.env.QMD_EMBED_PROVIDER = "openai";
    process.env.QMD_QUERY_PROVIDER = "mlx";
    process.env.QMD_OPENAI_API_KEY = "test-key";
    process.env.QMD_EMBED_MODEL = "text-embedding-3-large";
    process.env.QMD_MLX_BASE_URL = "http://127.0.0.1:8080";

    const dbPath = await freshDbPath();
    const store = await createSdkStore({ dbPath, config: { collections: {} } });

    expect(store.internal.llm).toBeInstanceOf(OpenAILLM);
    expect(store.internal.queryLlm).toBeInstanceOf(MlxLLM);

    await store.close();
    await rm(dirname(dbPath), { recursive: true, force: true });
  });
});
