/**
 * protocol.ts — MLX Query Expansion Protocol (Prompt Builder & Response Validation)
 *
 * Implements strict prompt construction and response parsing for fast-path MLX
 * query expansion on Apple Silicon.
 *
 * Requirements:
 * 1. Clear prompt asking ONLY for concise typed lines (lex:, vec:, hyde:)
 * 2. Delimits input query as raw data (prevents prompt injection / instruction hallucination)
 * 3. Topic & query-term relevance verification
 * 4. Lexical filtering (includeLexical)
 * 5. Rejection of prose, markdown bullet points, preamble, placeholder echo, and empty completions
 */

export type QueryType = 'lex' | 'vec' | 'hyde';

export interface Queryable {
  type: QueryType;
  text: string;
}

export interface ParsedExpansion {
  queryables: Queryable[];
  lex: string[];
  vec: string[];
  hyde: string[];
  invalidLines: string[];
  usable: boolean;
  qualityMessage: string;
  rawText: string;
}

export interface ExpansionPromptOptions {
  intent?: string;
}

export interface ExpansionParseOptions {
  query?: string;
  includeLexical?: boolean;
}

/**
 * Builds a strict, structured query expansion prompt for generic / chat instruction models.
 * Delimits the search query within <query>...</query> tags to treat user input purely as data.
 */
export function buildMlxExpansionPrompt(query: string, options?: ExpansionPromptOptions): string {
  const cleanQuery = query.replace(/<\/?query>/gi, "").trim();
  const cleanIntent = options?.intent ? options.intent.replace(/<\/?intent>/gi, "").trim() : undefined;

  let prompt = `/no_think
Generate search query expansions for the query delimited below.
Output each expansion on a new line using one of these three prefixes:
lex: add specific related technical terms or synonyms, not a copy of the query
vec: a natural search reformulation using concrete related concepts
hyde: a short factual-looking hypothetical answer passage, never a question

Example for the query "SQL joins":
lex: SQL joins inner outer left foreign key
vec: combining rows from relational tables using matching columns
hyde: SQL joins combine table rows on matching keys. Inner joins keep matches, while outer joins retain unmatched rows.

Write expansions for the actual query below, NOT the example. Do not write "semantic reformulation" or other task labels in the content. Output ONLY three prefixed lines. No preamble, no explanations, no code fences.

<query>
${cleanQuery}
</query>`;

  if (cleanIntent) {
    prompt += `\n<intent>\n${cleanIntent}\n</intent>`;
  }

  return prompt;
}

/**
 * Parses and validates raw LLM completion text into typed Queryable items (lex/vec/hyde).
 * Rejects unstructured prose, conversational fluff, unstripped thinking tags, and irrelevant lines.
 */
export function parseExpansionOutput(
  text: string,
  options?: ExpansionParseOptions,
): ParsedExpansion {
  const includeLexical = options?.includeLexical ?? true;
  const query = options?.query ?? "";
  const rawText = text ?? "";

  const result: ParsedExpansion = {
    queryables: [],
    lex: [],
    vec: [],
    hyde: [],
    invalidLines: [],
    usable: false,
    qualityMessage: "",
    rawText,
  };

  if (!rawText.trim()) {
    result.qualityMessage = "Completion text is empty";
    return result;
  }

  if (rawText.includes("<think>") || rawText.includes("</think>")) {
    result.qualityMessage = "Completion contains unstripped thinking tags";
    return result;
  }

  const queryLower = query.toLowerCase().trim();
  const queryTerms = queryLower
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length >= 2);

  const hasQueryRelevance = (content: string): boolean => {
    if (queryTerms.length === 0) return true;
    const lower = content.toLowerCase();
    return queryTerms.some((term) => lower.includes(term)) || lower.includes(queryLower);
  };

  const seen = new Set<string>();

  const lines = rawText.split("\n");
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;

    // Reject markdown code block fence markers
    if (line.startsWith("```")) {
      result.invalidLines.push(line);
      continue;
    }

    // Match typed line: optional leading bullet/number, then lex/vec/hyde: <content>
    const match = line.match(/^[-*•\d.\s]*(lex|vec|hyde)\s*:\s*(.+)$/i);
    if (!match) {
      result.invalidLines.push(line);
      continue;
    }

    const typeStr = match[1]!.toLowerCase() as QueryType;
    let content = match[2]!.trim();

    // Strip wrapping quotes or markdown bolding around content
    content = content.replace(/^["'`*]+|["'`*]+$/g, "").trim();

    if (!content) {
      result.invalidLines.push(line);
      continue;
    }

    const contentLower = content.toLowerCase();

    // Reject echo of prompt delimiter tags or instruction placeholders
    if (
      contentLower === "<query>" ||
      contentLower === "</query>" ||
      contentLower === "<intent>" ||
      contentLower === "</intent>" ||
      contentLower.startsWith("/no_think") ||
      contentLower === "(keyword or synonym phrase)" ||
      contentLower === "(conceptual or semantic reformulation)" ||
      contentLower === "(hypothetical sentence answering the query)"
    ) {
      result.invalidLines.push(line);
      continue;
    }

    // Topic / query-term overlap check
    if (query && !hasQueryRelevance(content)) {
      result.invalidLines.push(line);
      continue;
    }

    // Deduplicate
    const key = `${typeStr}:${content}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);

    // Record accepted typed items
    if (typeStr === "lex") {
      result.lex.push(content);
    } else if (typeStr === "vec") {
      result.vec.push(content);
    } else if (typeStr === "hyde") {
      result.hyde.push(content);
    }

    if (typeStr !== "lex" || includeLexical) {
      result.queryables.push({ type: typeStr, text: content });
    }
  }

  if (result.queryables.length === 0) {
    result.usable = false;
    result.qualityMessage =
      result.invalidLines.length > 0
        ? `No valid typed lines parsed; ${result.invalidLines.length} invalid lines rejected`
        : "No expansion lines generated";
    return result;
  }

  result.usable = true;
  result.qualityMessage = "Valid usable query expansion";
  return result;
}
