-- =============================================================
-- VectorMergeJoin (VMJ) — Basic Correctness Tests
-- =============================================================
-- Prerequisite: extension installed, SELECT vjoin_loaded() called

SELECT vjoin_loaded();

-- Force only VectorMergeJoin
SET enable_hashjoin = off;
SET enable_mergejoin = off;
SET enable_nestloop = off;
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_nestloop = off;
SET pg_vectorjoin.enable_mergejoin = on;
SET max_parallel_workers_per_gather = 0;

-- -------------------------------------------------------
-- 1. Setup tables
-- -------------------------------------------------------
DROP TABLE IF EXISTS vmj_left, vmj_right;

CREATE TABLE vmj_left  (id int PRIMARY KEY, val int);
CREATE TABLE vmj_right (id int PRIMARY KEY, ref_id int, data text);

INSERT INTO vmj_left  SELECT g, g % 100 FROM generate_series(1, 1000) g;
INSERT INTO vmj_right SELECT g, (g % 200) + 1, 'row_' || g FROM generate_series(1, 2000) g;
ANALYZE vmj_left;
ANALYZE vmj_right;

-- -------------------------------------------------------
-- 2. Plan check - should show VectorMergeJoin with Sort
-- -------------------------------------------------------
EXPLAIN (COSTS OFF)
SELECT l.id, l.val, r.data
FROM vmj_left l JOIN vmj_right r ON l.id = r.ref_id;

-- -------------------------------------------------------
-- 3. Basic correctness — count must match PG native
-- -------------------------------------------------------
SELECT 'VMJ' AS method, COUNT(*) AS cnt
FROM vmj_left l JOIN vmj_right r ON l.id = r.ref_id;

-- Compare with PG native merge join
SET pg_vectorjoin.enable_mergejoin = off;
SET enable_mergejoin = on;
SELECT 'PG_MJ' AS method, COUNT(*) AS cnt
FROM vmj_left l JOIN vmj_right r ON l.id = r.ref_id;

-- Re-enable VMJ
SET enable_mergejoin = off;
SET pg_vectorjoin.enable_mergejoin = on;

-- -------------------------------------------------------
-- 4. SUM check — values must match exactly
-- -------------------------------------------------------
SELECT 'VMJ' AS method, SUM(l.val) AS lsum, SUM(r.ref_id) AS rsum
FROM vmj_left l JOIN vmj_right r ON l.id = r.ref_id;

SET pg_vectorjoin.enable_mergejoin = off;
SET enable_mergejoin = on;
SELECT 'PG_MJ' AS method, SUM(l.val) AS lsum, SUM(r.ref_id) AS rsum
FROM vmj_left l JOIN vmj_right r ON l.id = r.ref_id;

SET enable_mergejoin = off;
SET pg_vectorjoin.enable_mergejoin = on;

-- -------------------------------------------------------
-- 5. Select actual rows — spot check
-- -------------------------------------------------------
SELECT l.id, l.val, r.id AS rid, r.ref_id, r.data
FROM vmj_left l JOIN vmj_right r ON l.id = r.ref_id
WHERE l.id <= 5
ORDER BY l.id, r.id;

-- -------------------------------------------------------
-- 6. EXPLAIN ANALYZE — must run without error
-- -------------------------------------------------------
EXPLAIN (ANALYZE, COSTS OFF, TIMING ON, SUMMARY ON)
SELECT COUNT(*) FROM vmj_left l JOIN vmj_right r ON l.id = r.ref_id;

-- Cleanup
DROP TABLE vmj_left, vmj_right;
