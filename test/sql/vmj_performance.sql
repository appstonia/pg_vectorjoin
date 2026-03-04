-- =============================================================
-- VectorMergeJoin (VMJ) — Performance Benchmark
-- =============================================================
SELECT vjoin_loaded();
SET max_parallel_workers_per_gather = 0;

-- -------------------------------------------------------
-- Setup large tables
-- -------------------------------------------------------
DROP TABLE IF EXISTS vmj_perf_l, vmj_perf_r;

CREATE TABLE vmj_perf_l (id int, val int);
CREATE TABLE vmj_perf_r (id int, val int);

-- 200K rows each, full overlap
INSERT INTO vmj_perf_l SELECT g, g % 1000 FROM generate_series(1, 200000) g;
INSERT INTO vmj_perf_r SELECT g, g % 500  FROM generate_series(1, 200000) g;
ANALYZE vmj_perf_l;
ANALYZE vmj_perf_r;

-- -------------------------------------------------------
-- 1. VectorMergeJoin
-- -------------------------------------------------------
SET enable_hashjoin = off;
SET enable_mergejoin = off;
SET enable_nestloop = off;
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_nestloop = off;
SET pg_vectorjoin.enable_mergejoin = on;

EXPLAIN (ANALYZE, COSTS OFF, TIMING ON, SUMMARY ON)
SELECT COUNT(*) FROM vmj_perf_l l JOIN vmj_perf_r r ON l.id = r.id;

-- -------------------------------------------------------
-- 2. PG Native Merge Join
-- -------------------------------------------------------
SET pg_vectorjoin.enable_mergejoin = off;
SET enable_mergejoin = on;

EXPLAIN (ANALYZE, COSTS OFF, TIMING ON, SUMMARY ON)
SELECT COUNT(*) FROM vmj_perf_l l JOIN vmj_perf_r r ON l.id = r.id;

-- -------------------------------------------------------
-- 3. VectorHashJoin (for reference)
-- -------------------------------------------------------
SET enable_mergejoin = off;
SET pg_vectorjoin.enable_hashjoin = on;

EXPLAIN (ANALYZE, COSTS OFF, TIMING ON, SUMMARY ON)
SELECT COUNT(*) FROM vmj_perf_l l JOIN vmj_perf_r r ON l.id = r.id;

-- -------------------------------------------------------
-- 4. PG Native Hash Join (for reference)
-- -------------------------------------------------------
SET pg_vectorjoin.enable_hashjoin = off;
SET enable_hashjoin = on;

EXPLAIN (ANALYZE, COSTS OFF, TIMING ON, SUMMARY ON)
SELECT COUNT(*) FROM vmj_perf_l l JOIN vmj_perf_r r ON l.id = r.id;

-- -------------------------------------------------------
-- 5. VMJ with pre-sorted (indexed) input
-- -------------------------------------------------------
CREATE INDEX ON vmj_perf_l (id);
CREATE INDEX ON vmj_perf_r (id);
ANALYZE vmj_perf_l;
ANALYZE vmj_perf_r;

SET enable_hashjoin = off;
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_mergejoin = on;

EXPLAIN (ANALYZE, COSTS OFF, TIMING ON, SUMMARY ON)
SELECT COUNT(*) FROM vmj_perf_l l JOIN vmj_perf_r r ON l.id = r.id;

-- -------------------------------------------------------
-- 6. Parallel VectorMergeJoin
-- -------------------------------------------------------
SET pg_vectorjoin.enable_mergejoin = off;
SET max_parallel_workers_per_gather = 2;
SET min_parallel_table_scan_size = 0;
SET parallel_tuple_cost = 0;
SET parallel_setup_cost = 0;

EXPLAIN (ANALYZE, COSTS OFF, TIMING ON, SUMMARY ON)
SELECT COUNT(*) FROM vmj_perf_l l JOIN vmj_perf_r r ON l.id = r.id;

-- -------------------------------------------------------
-- 7. PG Native Parallel Merge Join (for reference)
-- -------------------------------------------------------
SET enable_mergejoin = on;

EXPLAIN (ANALYZE, COSTS OFF, TIMING ON, SUMMARY ON)
SELECT COUNT(*) FROM vmj_perf_l l JOIN vmj_perf_r r ON l.id = r.id;

-- Cleanup
DROP TABLE vmj_perf_l, vmj_perf_r;
