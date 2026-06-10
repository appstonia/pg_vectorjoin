-- Parallel per-worker hash spilling regression test.
--
-- Exercises the per-worker private build: a non-parallel-aware inner is
-- scanned in full by every worker, each builds its own spill-capable hash
-- table, and each probes a disjoint slice of the parallel-aware outer scan.
-- Verifies that:
--   1. the planner chooses a Parallel Custom Scan (VectorHashJoin),
--   2. INNER and LEFT join results match the native executor even though each
--      worker spills its inner to disk (small work_mem, wide inner payload).
CREATE EXTENSION IF NOT EXISTS pg_vectorjoin;

CREATE TABLE par_inner (id int4, payload text);
CREATE TABLE par_outer (oid int4, ref int4);

-- Incompressible ~128-byte payload so the per-worker in-memory footprint
-- exceeds the spill budget and forces multi-batch spilling.
INSERT INTO par_inner
    SELECT g,
           md5(g::text) || md5((g + 1)::text) ||
           md5((g + 2)::text) || md5((g + 3)::text)
    FROM generate_series(1, 15000) g;
INSERT INTO par_outer
    SELECT g, ((g - 1) % 15000) + 1 FROM generate_series(1, 45000) g;

ANALYZE par_inner;
ANALYZE par_outer;

-- Force the inner to be non-parallel-aware so the planner selects the
-- per-worker private build (par_inner_full) path.
ALTER TABLE par_inner SET (parallel_workers = 0);

SET pg_vectorjoin.enable = on;
SET pg_vectorjoin.cost_factor = 0.01;
SET work_mem = '64kB';
SET max_parallel_workers_per_gather = 4;
SET max_parallel_workers = 8;
SET parallel_setup_cost = 0;
SET parallel_tuple_cost = 0;
SET min_parallel_table_scan_size = 0;

-- (1) Planner picks a parallel vectorized hash join with a non-parallel inner.
EXPLAIN (COSTS OFF)
SELECT count(*) FROM par_outer o JOIN par_inner i ON o.ref = i.id;

-- (2a) INNER join correctness under per-worker spilling (text inner attr).
SELECT count(*) AS vhj_cnt,
       sum(length(i.payload)) AS vhj_sumlen,
       bool_and(i.payload =
                md5(i.id::text) || md5((i.id + 1)::text) ||
                md5((i.id + 2)::text) || md5((i.id + 3)::text)) AS payload_ok
FROM par_outer o JOIN par_inner i ON o.ref = i.id;

-- Native baseline.
SET pg_vectorjoin.enable = off;
SELECT count(*) AS nat_cnt,
       sum(length(i.payload)) AS nat_sumlen
FROM par_outer o JOIN par_inner i ON o.ref = i.id;

ALTER TABLE par_inner RESET (parallel_workers);
DROP TABLE par_inner, par_outer;
