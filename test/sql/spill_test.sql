-- Multi-batch hash spilling regression test.
--
-- Forces the vectorized hash join to spill the inner relation to disk by
-- using a small work_mem together with a wide, incompressible inner payload,
-- then checks that:
--   1. the planner still chooses VectorHashJoin for an inner that does not
--      fit in work_mem (only possible because spilling is enabled),
--   2. spilling actually engages at run time (Batches > 1),
--   3. results (including a pass-by-reference text inner attribute) match the
--      native executor for both INNER and LEFT joins.
CREATE EXTENSION IF NOT EXISTS pg_vectorjoin;

CREATE TABLE spill_inner (id int4, payload text);
CREATE TABLE spill_outer (oid int4, ref int4);

-- Incompressible ~128-byte payload so the in-memory footprint (which counts
-- the real datum size) reliably exceeds the spill budget.
INSERT INTO spill_inner
    SELECT g,
           md5(g::text) || md5((g + 1)::text) ||
           md5((g + 2)::text) || md5((g + 3)::text)
    FROM generate_series(1, 15000) g;
INSERT INTO spill_outer
    SELECT g, ((g - 1) % 15000) + 1 FROM generate_series(1, 45000) g;

ANALYZE spill_inner;
ANALYZE spill_outer;

SET max_parallel_workers_per_gather = 0;
SET pg_vectorjoin.enable = on;
SET pg_vectorjoin.cost_factor = 0.01;
SET work_mem = '64kB';

-- (1) Planner picks the vectorized hash join even though the inner relation
-- does not fit in work_mem.  Runtime-only fields (Hash Table Size, Batches)
-- are shown pre-execution and are therefore deterministic.
EXPLAIN (COSTS OFF)
SELECT count(*) FROM spill_outer o JOIN spill_inner i ON o.ref = i.id;

-- (2) Confirm multi-batch spilling actually engaged at run time.  The exact
-- batch count is non-deterministic, so only assert it exceeded one.
DO $$
DECLARE
    j json;
    b int;
BEGIN
    EXECUTE 'EXPLAIN (ANALYZE, FORMAT JSON, TIMING off, BUFFERS off, SUMMARY off) '
            'SELECT count(*) FROM spill_outer o JOIN spill_inner i ON o.ref = i.id'
        INTO j;
    b := (j->0->'Plan'->'Plans'->0->>'Batches')::int;
    IF b > 1 THEN
        RAISE NOTICE 'spilling engaged (multi-batch)';
    ELSE
        RAISE NOTICE 'NOT spilled (batches=%)', b;
    END IF;
END $$;

-- (3a) INNER join correctness with a pass-by-reference (text) inner attribute.
SELECT count(*) AS vhj_cnt,
       sum(length(i.payload)) AS vhj_sumlen,
       bool_and(i.payload =
                md5(i.id::text) || md5((i.id + 1)::text) ||
                md5((i.id + 2)::text) || md5((i.id + 3)::text)) AS payload_ok
FROM spill_outer o JOIN spill_inner i ON o.ref = i.id;

-- Add some unmatched outer rows for the LEFT join test.
INSERT INTO spill_outer VALUES (999001, -1), (999002, -2);

-- (3b) LEFT join correctness under spilling (unmatched outer rows preserved).
SELECT count(*) AS vhj_lj_cnt,
       count(i.id) AS vhj_lj_matched
FROM spill_outer o LEFT JOIN spill_inner i ON o.ref = i.id;

-- Native baselines for comparison.
SET pg_vectorjoin.enable = off;

SELECT count(*) AS nat_cnt,
       sum(length(i.payload)) AS nat_sumlen
FROM spill_outer o JOIN spill_inner i ON o.ref = i.id;

SELECT count(*) AS nat_lj_cnt,
       count(i.id) AS nat_lj_matched
FROM spill_outer o LEFT JOIN spill_inner i ON o.ref = i.id;

DROP TABLE spill_inner, spill_outer;
