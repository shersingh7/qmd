import { describe, expect, it } from "vitest";
import { validateSemanticQuery } from "../src/store.js";

describe("semantic query hyphens", () => {
  it.each(["non-blocking database writes", "end-to-end retrieval", "B-tree indexes"])("accepts natural hyphenation: %s", query => {
    expect(validateSemanticQuery(query)).toBeNull();
  });
  it.each(["-term", "search -excluded", 'search -"excluded phrase"'])("rejects lexical exclusions: %s", query => {
    expect(validateSemanticQuery(query)).not.toBeNull();
  });
});
