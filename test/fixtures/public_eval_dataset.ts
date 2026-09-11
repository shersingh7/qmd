/**
 * public_eval_dataset.ts — Curated Public Technical Corpus & Held-Out Judged Queries
 *
 * Pre-defined, frozen evaluation fixtures for Phase 5 qualification.
 * Judgments are defined before execution with zero outcome tuning.
 */

export interface PublicDocFixture {
  id: string;
  filename: string;
  title: string;
  content: string;
}

export interface PublicJudgedQuery {
  id: string;
  query: string;
  relevantDocIds: string[];
  category: "easy" | "medium" | "hard";
  intent: string;
}

export const PUBLIC_CORPUS_DOCS: PublicDocFixture[] = [
  {
    id: "doc-api-versioning",
    filename: "doc-api-versioning.md",
    title: "REST API Versioning Strategies",
    content: `URI path versioning (/v1/users) is transparent for caching proxies and documentation. Custom request headers (X-API-Version: 2) keep URLs clean but complicate intermediate caching. Accept header content negotiation (application/vnd.company.v1+json) conforms strictly to REST principles and media type versioning. When breaking changes occur, semantic versioning guidelines mandate incrementing the major version. Deprecation schedules should provide client consumers at least a six-month sunset window with Sunset HTTP headers.`
  },
  {
    id: "doc-raft-consensus",
    filename: "doc-raft-consensus.md",
    title: "Raft Consensus Algorithm",
    content: `Raft is a distributed consensus algorithm designed to be understandable and equivalent to Multi-Paxos in fault tolerance. Raft decomposes consensus into three independent subproblems: leader election, log replication, and safety invariants. Nodes operate in one of three states: Follower, Candidate, or Leader. If followers receive no heartbeats within an election timeout (typically 150ms to 300ms), they increment term numbers and transition to candidate state to request votes. A candidate wins when it receives majority votes. Log entries only commit once replicated to a quorum of nodes.`
  },
  {
    id: "doc-sqlite-wal",
    filename: "doc-sqlite-wal.md",
    title: "SQLite Write-Ahead Logging",
    content: `Write-Ahead Logging (WAL) is a high-performance journal mode in SQLite that enables concurrent readers alongside an active writer. Instead of modifying the database pages directly or rolling back via a rollback journal file, committed transactions append frame changes to a separate .wal file. Readers traverse both the main database and WAL index (.shm shared memory) without acquiring exclusive database table locks. Periodic checkpointing transfers accumulated WAL frames back into the main database file and resets log offset.`
  },
  {
    id: "doc-apple-metal-mem",
    filename: "doc-apple-metal-mem.md",
    title: "Apple Silicon Unified Memory Architecture",
    content: `Apple Silicon integrates the CPU, GPU, and Neural Engine onto a single unified memory bus. This unified memory architecture eliminates PCIe host-to-device memory copies between CPU RAM and GPU VRAM. Metal compute shaders and MLX tensor operations map unified memory allocations directly via zero-copy MTLBuffer instances. Quantized weight representations, such as 4-bit affine and 8-bit integer formats, maximize effective memory bandwidth for on-device Large Language Model inference.`
  },
  {
    id: "doc-ann-vector-index",
    filename: "doc-ann-vector-index.md",
    title: "Approximate Nearest Neighbor Vector Search",
    content: `Approximate Nearest Neighbor (ANN) search algorithms enable sub-linear vector retrieval across high-dimensional semantic spaces. Hierarchical Navigable Small World (HNSW) graphs organize vectors into multi-layer proximity graphs, allowing greedy logarithmic search routing. Inverted File with Product Quantization (IVFPQ) clusters embedding vectors into Voronoi cells and compresses residuals with sub-vector codebooks. Metric choices like cosine similarity, inner product, and L2 Euclidean distance determine vector normalization requirements.`
  },
  {
    id: "doc-cache-invalidation",
    filename: "doc-cache-invalidation.md",
    title: "Cache Invalidation Patterns and Consistency",
    content: `Cache invalidation is a classic distributed systems challenge. Common caching topologies include cache-aside (lazy loading), write-through (synchronous cache and database store update), and write-back (asynchronous write behind). Cache stampede (thundering herd) occurs when a high-traffic key expires simultaneously across concurrent workers; mitigation techniques include mutex locking (single-flight fetching), probabilistic early expiration (XFetch algorithm), and active cache warming.`
  },
  {
    id: "doc-b-tree-indexing",
    filename: "doc-b-tree-indexing.md",
    title: "B-Trees and Log-Structured Merge-Trees",
    content: `Storage engines choose between B-Trees and Log-Structured Merge-Trees (LSM-Trees) based on read vs write workload trade-offs. B-Trees offer predictable O(log N) point reads and range scans by maintaining sorted balanced tree nodes on disk, but suffer from write amplification due to random in-place page overwrites. LSM-Trees append writes sequentially into an in-memory MemTable and flush immutable SSTables to disk, minimizing write amplification at the expense of read amplification during multi-table compactions.`
  },
  {
    id: "doc-jwt-auth",
    filename: "doc-jwt-auth.md",
    title: "JSON Web Token Authentication and Security",
    content: `JSON Web Tokens (JWT) encode user identity and claims inside base64url-encoded header, payload, and signature components. Asymmetric signing algorithms like RS256 and ES256 allow authentication servers to sign tokens using private keys while API gateways verify validity using public JWKS endpoints. To mitigate token leakage, short-lived access tokens (5 to 15 minutes) are paired with long-lived refresh tokens stored in secure HTTP-only cookies, combined with refresh token rotation and token family revocation.`
  },
  {
    id: "doc-compiler-jit",
    filename: "doc-compiler-jit.md",
    title: "Just-In-Time Compilation and Dynamic Optimization",
    content: `Just-In-Time (JIT) compilers improve interpreter execution speed by translating hot bytecode regions into native machine code at runtime. Multi-tier compilation architectures use a baseline fast compiler to generate unoptimized machine code quickly, followed by an optimizing compiler that performs loop unrolling, escape analysis, inline caching, and dead code elimination based on runtime type feedback. When dynamic assumptions fail, execution safely deoptimizes back to the interpreter.`
  },
  {
    id: "doc-linux-epoll",
    filename: "doc-linux-epoll.md",
    title: "Linux Epoll Scalable I/O Multiplexing",
    content: `The Linux epoll subsystem provides O(1) event notification for scalable asynchronous network servers. Unlike select and poll which scan all file descriptors on each invocation (O(N)), epoll registers descriptors into a kernel event table backed by a red-black tree. Readiness events are delivered via an epoll ready list. Edge-triggered (EPOLLET) mode signals state changes only once, requiring non-blocking sockets with drained read loops, whereas level-triggered mode notifies continuously while buffers contain data.`
  },
  {
    id: "doc-tls-handshake",
    filename: "doc-tls-handshake.md",
    title: "TLS 1.3 Protocol Handshake",
    content: `Transport Layer Security (TLS) version 1.3 optimizes network security and connection latency by reducing the standard cryptographic handshake from 2-RTT to a single round-trip (1-RTT). Ephemeral Elliptic Curve Diffie-Hellman (ECDHE) key exchange ensures perfect forward secrecy for all session traffic. Static RSA key exchange is deprecated. TLS 1.3 also supports 0-RTT early data resumption for returning clients using pre-shared keys (PSK), with anti-replay mechanisms to mitigate replay attacks.`
  },
  {
    id: "doc-garbage-collection",
    filename: "doc-garbage-collection.md",
    title: "Generational Garbage Collection Algorithms",
    content: `Generational garbage collection is founded on the weak generational hypothesis: most allocated objects die shortly after creation. Heap memory is partitioned into a young generation (Eden and Survivor spaces) and an old (tenured) generation. Minor garbage collections sweep short-lived young objects rapidly using copying collectors. Card tables track inter-generational references from old-to-young objects to avoid full heap scanning during minor collections. Long-lived surviving objects are tenured to the old generation.`
  },
  {
    id: "doc-cookie-recipe",
    filename: "doc-cookie-recipe.md",
    title: "Classic Chocolate Chip Cookies Recipe",
    content: `To bake homemade chocolate chip cookies, combine 2 cups all-purpose flour, 1 teaspoon baking soda, and 1/2 teaspoon salt. Cream 1 cup softened unsalted butter with 3/4 cup brown sugar and 3/4 cup granulated white sugar until smooth. Beat in two large eggs and two teaspoons vanilla extract. Stir in 2 cups semi-sweet chocolate chips and chopped walnuts. Drop rounded tablespoons onto baking sheets and bake at 375 degrees Fahrenheit (190 degrees Celsius) for 9 to 11 minutes until golden brown.`
  },
  {
    id: "doc-vacation-policy",
    filename: "doc-vacation-policy.md",
    title: "Company Vacation and Paid Time Off Guidelines",
    content: `Full-time employees accrue 15 days of paid time off (PTO) annually, credited at the rate of 1.25 days per completed calendar month. Vacation requests exceeding three consecutive work days must be submitted to human resources through the employee portal at least fourteen days in advance. Core collaboration hours for all remote and hybrid employees are 10:00 AM to 3:00 PM local time. Unused PTO up to a maximum of five days may roll over into the following calendar year.`
  },
  {
    id: "doc-startup-fundraising",
    filename: "doc-startup-fundraising.md",
    title: "Startup Seed and Series A Fundraising Memo",
    content: `When raising institutional venture capital for a technology startup, founders track key operational metrics including Annual Recurring Revenue (ARR), Net Revenue Retention (NRR), and Customer Acquisition Cost (CAC) payback period. Seed stage funding typically utilizes Simple Agreements for Future Equity (SAFE) with valuation caps and discount rates to defer equity price setting. For Series A rounds, investors evaluate product-market fit, unit economics, and cash runway of at least 18 to 24 months.`
  },
  {
    id: "doc-phoenix-launch",
    filename: "doc-phoenix-launch.md",
    title: "Project Phoenix Launch Post-Mortem and Retrospective",
    content: `The Project Phoenix database migration deployment encountered a 22-minute partial service interruption due to connection pool starvation during automated schema migration. Root cause analysis identified that unindexed foreign key lookups generated table-level locking cascades on the primary relational cluster. Corrective actions include enforcing strict read-only shadow migrations in pre-production, implementing circuit breakers on connection pools, and automating immediate rollback scripts.`
  }
];

export const PUBLIC_HELD_OUT_QUERIES: PublicJudgedQuery[] = [
  {
    id: "Q01",
    query: "HTTP header versus URI path versioning for web APIs",
    relevantDocIds: ["doc-api-versioning"],
    category: "easy",
    intent: "REST API versioning best practices"
  },
  {
    id: "Q02",
    query: "how do distributed nodes elect a leader after heartbeat timeout",
    relevantDocIds: ["doc-raft-consensus"],
    category: "medium",
    intent: "distributed consensus and leader election"
  },
  {
    id: "Q03",
    query: "concurrent readers non-blocking during database write commits",
    relevantDocIds: ["doc-sqlite-wal"],
    category: "medium",
    intent: "sqlite write ahead log concurrency"
  },
  {
    id: "Q04",
    query: "eliminating CPU to GPU buffer copy overhead on Apple chips",
    relevantDocIds: ["doc-apple-metal-mem"],
    category: "medium",
    intent: "apple silicon unified memory zero-copy"
  },
  {
    id: "Q05",
    query: "hierarchical graph index for fast cosine similarity search",
    relevantDocIds: ["doc-ann-vector-index"],
    category: "medium",
    intent: "approximate nearest neighbor vector search algorithms"
  },
  {
    id: "Q06",
    query: "preventing thundering herd when cached entries expire",
    relevantDocIds: ["doc-cache-invalidation"],
    category: "hard",
    intent: "cache stampede invalidation mitigation"
  },
  {
    id: "Q07",
    query: "tradeoffs between write amplification in LSM trees vs B-trees",
    relevantDocIds: ["doc-b-tree-indexing"],
    category: "hard",
    intent: "database storage engine read write amplification"
  },
  {
    id: "Q08",
    query: "cryptographic key exchange with forward secrecy and 1-RTT latency",
    relevantDocIds: ["doc-tls-handshake"],
    category: "hard",
    intent: "tls 1.3 handshake security"
  },
  {
    id: "Q09",
    query: "ingredients and oven temperature for homemade chocolate cookies",
    relevantDocIds: ["doc-cookie-recipe"],
    category: "easy",
    intent: "baking chocolate chip cookies"
  },
  {
    id: "Q10",
    query: "employee paid time off accrual rules and request notice window",
    relevantDocIds: ["doc-vacation-policy"],
    category: "easy",
    intent: "workplace vacation and pto policies"
  }
];

export function computeIRMetrics(
  rankedDocIdsList: string[][],
  relevantDocIdsList: string[][],
  k: number = 5
): {
  recallAt1: number;
  recallAt3: number;
  recallAt5: number;
  mrr: number;
  ndcgAt5: number;
} {
  const total = rankedDocIdsList.length;
  if (total === 0) {
    return { recallAt1: 0, recallAt3: 0, recallAt5: 0, mrr: 0, ndcgAt5: 0 };
  }

  let hits1 = 0;
  let hits3 = 0;
  let hits5 = 0;
  let sumRR = 0;
  let sumNDCG = 0;

  for (let q = 0; q < total; q++) {
    const ranked = rankedDocIdsList[q] || [];
    const relSet = new Set(relevantDocIdsList[q] || []);

    if (ranked.length > 0 && relSet.has(ranked[0])) hits1++;
    if (ranked.slice(0, 3).some(id => relSet.has(id))) hits3++;
    if (ranked.slice(0, 5).some(id => relSet.has(id))) hits5++;

    // Reciprocal Rank (MRR)
    let rr = 0;
    for (let r = 0; r < ranked.length; r++) {
      if (relSet.has(ranked[r])) {
        rr = 1 / (r + 1);
        break;
      }
    }
    sumRR += rr;

    // nDCG@k
    let dcg = 0;
    for (let i = 0; i < Math.min(ranked.length, k); i++) {
      if (relSet.has(ranked[i])) {
        dcg += 1.0 / Math.log2(i + 2);
      }
    }

    let idcg = 0;
    for (let i = 0; i < Math.min(relSet.size, k); i++) {
      idcg += 1.0 / Math.log2(i + 2);
    }
    const ndcg = idcg > 0 ? dcg / idcg : 1.0;
    sumNDCG += ndcg;
  }

  return {
    recallAt1: Math.round((hits1 / total) * 10000) / 10000,
    recallAt3: Math.round((hits3 / total) * 10000) / 10000,
    recallAt5: Math.round((hits5 / total) * 10000) / 10000,
    mrr: Math.round((sumRR / total) * 10000) / 10000,
    ndcgAt5: Math.round((sumNDCG / total) * 10000) / 10000,
  };
}
