#!/usr/bin/env python3
"""Apply production dist patches to the INSTALLED qmd package.

Run after every fork dist swap:
    bun run build && cp -a dist/. ~/.hermes/node/lib/node_modules/@tobilu/qmd/dist/ \\
        && python3 scripts/apply-fork-dist-patches.py

Covers the three local tunings from apply-qmd-local-patches.py that the
stock script can no longer apply itself (it is version-locked to upstream
2.5.3 anchors, while production runs fork dist):
  1. generateEmbeddings session timeout 30m -> 120m (large re-embeds)
  2. RERANK_CANDIDATE_LIMIT 40 -> 15 (David's tuned default)
  3. node-llama-cpp session timeout 10m -> 120m
  4. MCP origin-guard backport adapted to the fork's 2.1.0-based
     dist/mcp/server.js handler shape + candidateLimit describe string.
Idempotent. Verifies every replacement landed.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PKG = Path.home() / ".hermes/node/lib/node_modules/@tobilu/qmd"
PATCH_DIR = Path.home() / ".hermes/scripts/qmd-patches"


def replace_once(text: str, old: str, new: str, where: str) -> str:
    if new in text and old not in text:
        return text  # already applied
    if old not in text:
        raise SystemExit(f"PATCH ANCHOR MISSING in {where}: {old[:70]!r}")
    return text.replace(old, new, 1)


def main() -> int:
    store = PKG / "dist/store.js"
    llm = PKG / "dist/llm.js"
    mcp = PKG / "dist/mcp/server.js"
    for f in (store, llm, mcp):
        if not f.exists():
            print(f"ERROR: missing {f}", file=sys.stderr)
            return 1

    t = store.read_text()
    t = replace_once(t,
        "maxDuration: 30 * 60 * 1000, name: 'generateEmbeddings'",
        "maxDuration: 120 * 60 * 1000, name: 'generateEmbeddings'", "store.js")
    t = replace_once(t,
        "export const RERANK_CANDIDATE_LIMIT = 40;",
        "export const RERANK_CANDIDATE_LIMIT = 15;", "store.js")
    store.write_text(t)

    t = llm.read_text()
    t = replace_once(t, "?? 10 * 60 * 1000", "?? 120 * 60 * 1000", "llm.js")
    llm.write_text(t)

    shutil.copy2(PATCH_DIR / "origin-guard.js", PKG / "dist/mcp/origin-guard.js")
    t = mcp.read_text()
    imp = 'import { checkRequestOrigin, resolveOriginGuard } from "./origin-guard.js";\n'
    anchor = 'import { getConfigPath } from "../collections.js";\n'
    if './origin-guard.js' not in t:
        if anchor not in t:
            raise SystemExit("PATCH ANCHOR MISSING: mcp import anchor")
        t = t.replace(anchor, anchor + imp, 1)
    init_old = """    const store = await createStore({
        dbPath: getDefaultDbPath(),
        ...(existsSync(configPath) ? { configPath } : {}),
    });
    // Pre-fetch default collection names for REST endpoint"""
    init_new = """    const store = await createStore({
        dbPath: getDefaultDbPath(),
        ...(existsSync(configPath) ? { configPath } : {}),
    });
    const originGuard = resolveOriginGuard({ host: "localhost" });
    // Pre-fetch default collection names for REST endpoint"""
    if 'const originGuard = resolveOriginGuard' not in t:
        if init_old not in t:
            raise SystemExit("PATCH ANCHOR MISSING: mcp init anchor")
        t = t.replace(init_old, init_new, 1)
    call_old = ('        const pathname = nodeReq.url || "/";\n        try {\n'
                '            if (pathname === "/health" && nodeReq.method === "GET") {')
    call_new = '''        const pathname = nodeReq.url || "/";
        try {
            const originVerdict = checkRequestOrigin({
                origin: typeof nodeReq.headers.origin === "string" ? nodeReq.headers.origin : undefined,
                host: typeof nodeReq.headers.host === "string" ? nodeReq.headers.host : undefined,
            }, originGuard);
            if (!originVerdict.ok) {
                nodeRes.writeHead(403, { "Content-Type": "application/json" });
                nodeRes.end(JSON.stringify({ error: originVerdict.reason }));
                log(`${ts()} 403 ${pathname} ${originVerdict.reason}`);
                return;
            }
            if (pathname === "/health" && nodeReq.method === "GET") {'''
    if 'originVerdict' not in t:
        if call_old not in t:
            raise SystemExit("PATCH ANCHOR MISSING: mcp call anchor")
        t = t.replace(call_old, call_new, 1)
    t = t.replace(
        "Maximum candidates to rerank (default: 40, lower = faster but may miss results)",
        "Maximum candidates to rerank (default: 15, lower = faster but may miss results)")
    mcp.write_text(t)
    print(f"OK: production dist patches applied to {PKG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
