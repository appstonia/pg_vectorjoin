# pg_vectorjoin

> **Warning:** pg_vectorjoin has not reached a stable release yet. Do not use it in production environments.

Vectorized join execution for PostgreSQL. This extension replaces the
standard Volcano-style tuple-at-a-time join operators with batch-oriented
alternatives that process data in columnar blocks, enabling SIMD-accelerated
hashing and comparison on modern CPUs.

pg_vectorjoin hooks into the PostgreSQL planner via `set_join_pathlist_hook`
and injects custom scan paths (VectorHashJoin, VectorNestedLoop,
VectorMergeJoin) alongside native join strategies. The optimizer's
cost-based selection decides which path wins -- vectorized or native -- on
a per-join basis.

## How It Works

PostgreSQL's Volcano execution model processes one tuple at a time through
each join operator. This prevents the CPU from using SIMD instructions for
key comparison or hashing because there is only a single value to work with
at any given moment.

pg_vectorjoin changes this by buffering tuples into fixed-size blocks
(default 2000) and operating on them in bulk:

**VectorHashJoin** -- Deforms all inner tuples into flat columnar arrays at
build time. During the probe phase, outer tuples are batch-hashed and
compared against the hash table using SIMD equality checks on INT4/INT8/FLOAT8
keys. The hash table uses open-addressing with linear probing and power-of-2
capacity. Parallel execution is supported: workers can build concurrently
using CAS-based lock-free inserts, or attach to a leader-built shared hash
table via DSA.

**VectorNestedLoop** -- Loads a block of outer tuples, then scans the inner
relation once per block instead of once per tuple. For a batch size of 2000,
this reduces inner rescans by a factor of 2000x. A single-key SIMD fast path
compares the entire outer block against each inner value in one pass. The
inner relation is materialized into a tuplestore on first scan and replayed
from memory for subsequent blocks.

**VectorMergeJoin** -- Merges two sorted streams with batch processing of
match groups. When all join keys are pass-by-value types, a block merge mode
pre-deforms tuples into columnar arrays for cache-friendly access. Parallel
merge materializes the sorted inner into DSA-shared arrays; workers merge
their partial outer streams against the shared inner independently.

## SIMD Support

Runtime detection at extension load time. No compile-time flags needed.

- x86-64: SSE2 (baseline) and AVX2 (when available)
- ARM: NEON (baseline on ARMv8)

SIMD is used for block equality/inequality comparison and batch hashing on
supported key types. Non-SIMD types fall back to PostgreSQL's generic
function manager (FmgrInfo) calls.

## Supported Key Types

Only fixed-size, pass-by-value types qualify for the SIMD fast path.
Other types (text, varchar, uuid, bytea, etc.) are routed to native
PostgreSQL join strategies.

| SIMD Path | Types |
|-----------|-------|
| 32-bit integer | int4, int2, date, oid, regclass, regtype, regproc |
| 64-bit integer | int8, timestamp, timestamptz, time |
| 64-bit float | float8 |

float4 is explicitly excluded because its Datum representation does not
fill all 8 bytes, producing incorrect hash values when read as float8.

## Join Type Support

- **INNER JOIN** -- Full support across all three strategies (hash, nested
  loop, merge).
- **LEFT JOIN** -- Hash join only. Restricted to a safe subset: anti-join
  patterns (LEFT JOIN ... WHERE inner.col IS NULL) are rejected, and joins
  that project nullable inner-side variables are excluded.
- **Other join types** (SEMI, ANTI, RIGHT, FULL) -- Not supported. The
  extension does not create paths for these; native PostgreSQL handles them.

## Building from Source

Requirements: PostgreSQL server development headers (PG 14 through PG 19),
a C compiler with C99 support, and pg_config in PATH.

```
git clone https://github.com/appstonia/pg_vectorjoin.git
cd pg_vectorjoin
make
make install
```

On macOS with Xcode, the Makefile auto-detects the current SDK path.

## Configuration

Load the extension in postgresql.conf:

```
shared_preload_libraries = 'pg_vectorjoin'
```

Or load per-session (requires superuser or is granted via
`GRANT EXECUTE ON FUNCTION pg_load_shared_library() TO ...`):

```sql
LOAD 'pg_vectorjoin';
```

All parameters are settable per-session without server restart:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| pg_vectorjoin.enable | true | -- | Master switch for all vectorized joins |
| pg_vectorjoin.enable_hashjoin | true | -- | Enable VectorHashJoin paths |
| pg_vectorjoin.enable_nestloop | true | -- | Enable VectorNestedLoop paths |
| pg_vectorjoin.enable_mergejoin | true | -- | Enable VectorMergeJoin paths |
| pg_vectorjoin.batch_size | 2000 | 100 -- 10000 | Block size for batch processing |
| pg_vectorjoin.cost_factor | 0.5 | 0.01 -- 10.0 | Join overhead cost multiplier (lower = prefer vectorized) |

## Usage

Once loaded, the extension is transparent. The planner automatically
considers vectorized paths and selects them when their estimated cost is
lower than native alternatives:

```sql
LOAD 'pg_vectorjoin';

-- Tune parameters per-session
SET pg_vectorjoin.enable = on;
SET pg_vectorjoin.batch_size = 2000;
SET pg_vectorjoin.cost_factor = 0.3;

SET pg_vectorjoin.enable_hashjoin = on;
SET pg_vectorjoin.enable_mergejoin = on;
SET pg_vectorjoin.enable_nestloop = on;

EXPLAIN (ANALYZE)
SELECT count(*)
FROM orders o
JOIN customers c ON c.id = o.customer_id;
```

```
 Aggregate
   ->  Custom Scan (VectorHashJoin)
         Join Type: Inner
         Hash Table Size: 50000
         Batch Size: 2000
         SIMD: true
         ->  Seq Scan on orders o
         ->  Seq Scan on customers c
```

Disable individual strategies to compare performance:

```sql
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_nestloop = off;
-- Only VectorMergeJoin paths will be considered
```

## Limitations

- **Numeric key types only.** Text, varchar, uuid, bytea, jsonb, and other
  variable-length or pass-by-reference types are not eligible for vectorized
  join. Joins on these types use native PostgreSQL strategies.

- **No disk spill.** VectorHashJoin keeps the entire hash table in memory.
  If the estimated table exceeds work_mem * hash_mem_multiplier, the
  vectorized hash path is not created. Native hash join (which can batch to
  disk) is used instead.

- **No parameterized inner paths.** VectorNestedLoop cannot push outer
  values into inner index conditions. For index nested loop patterns
  (SELECT ... FROM a JOIN b ON a.id = b.id where b has an index on id),
  native nested loop with parameterized index scan is typically faster.

- **Limited LEFT JOIN.** Only hash join, only equi-join, no anti-join
  patterns, no projection of nullable inner columns. Complex outer join
  queries fall back to native execution.

- **No SEMI/ANTI/RIGHT/FULL joins.** These join types are handled entirely
  by native PostgreSQL.

- **Maximum 8 join keys** per join clause (VJOIN_MAX_KEYS = 8).

- **Single-key SIMD only.** The SIMD comparison fast path activates for
  single-key equi-joins on numeric types. Multi-key joins use a scalar
  comparison loop (still batched, but without SIMD acceleration).

## Compatibility

Tested with PostgreSQL 14 through 19. Version-specific differences
(API changes in PG 15, 17, 18, 19) are handled via compile-time
compatibility shims.

## License

GPL License. See [LICENSE](LICENSE) for details.
