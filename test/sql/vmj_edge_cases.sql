-- =============================================================
-- VectorMergeJoin (VMJ) — Edge Cases
-- =============================================================
SELECT vjoin_loaded();

SET enable_hashjoin = off;
SET enable_mergejoin = off;
SET enable_nestloop = off;
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_nestloop = off;
SET pg_vectorjoin.enable_mergejoin = on;
SET max_parallel_workers_per_gather = 0;

-- -------------------------------------------------------
-- 1. Empty tables
-- -------------------------------------------------------
DROP TABLE IF EXISTS vmj_edge_l, vmj_edge_r;
CREATE TABLE vmj_edge_l (id int, val int);
CREATE TABLE vmj_edge_r (id int, val int);
ANALYZE vmj_edge_l;
ANALYZE vmj_edge_r;

-- Both empty
SELECT 'both_empty' AS test, COUNT(*) AS cnt
FROM vmj_edge_l l JOIN vmj_edge_r r ON l.id = r.id;
-- Expected: 0

-- Left empty, right has data
INSERT INTO vmj_edge_r SELECT g, g FROM generate_series(1, 100) g;
ANALYZE vmj_edge_r;

SELECT 'left_empty' AS test, COUNT(*) AS cnt
FROM vmj_edge_l l JOIN vmj_edge_r r ON l.id = r.id;
-- Expected: 0

-- Right empty, left has data
TRUNCATE vmj_edge_r;
INSERT INTO vmj_edge_l SELECT g, g FROM generate_series(1, 100) g;
ANALYZE vmj_edge_l;
ANALYZE vmj_edge_r;

SELECT 'right_empty' AS test, COUNT(*) AS cnt
FROM vmj_edge_l l JOIN vmj_edge_r r ON l.id = r.id;
-- Expected: 0

-- -------------------------------------------------------
-- 2. Single row tables
-- -------------------------------------------------------
TRUNCATE vmj_edge_l, vmj_edge_r;
INSERT INTO vmj_edge_l VALUES (42, 1);
INSERT INTO vmj_edge_r VALUES (42, 2);
ANALYZE vmj_edge_l;
ANALYZE vmj_edge_r;

SELECT 'single_match' AS test, l.id, l.val AS lval, r.val AS rval
FROM vmj_edge_l l JOIN vmj_edge_r r ON l.id = r.id;
-- Expected: 1 row, id=42

TRUNCATE vmj_edge_r;
INSERT INTO vmj_edge_r VALUES (99, 2);
ANALYZE vmj_edge_r;

SELECT 'single_nomatch' AS test, COUNT(*) AS cnt
FROM vmj_edge_l l JOIN vmj_edge_r r ON l.id = r.id;
-- Expected: 0

-- -------------------------------------------------------
-- 3. No overlap at all (disjoint key ranges)
-- -------------------------------------------------------
TRUNCATE vmj_edge_l, vmj_edge_r;
INSERT INTO vmj_edge_l SELECT g, g FROM generate_series(1, 500) g;
INSERT INTO vmj_edge_r SELECT g, g FROM generate_series(501, 1000) g;
ANALYZE vmj_edge_l;
ANALYZE vmj_edge_r;

SELECT 'disjoint' AS test, COUNT(*) AS cnt
FROM vmj_edge_l l JOIN vmj_edge_r r ON l.id = r.id;
-- Expected: 0

-- -------------------------------------------------------
-- 4. Complete overlap (all keys match)
-- -------------------------------------------------------
TRUNCATE vmj_edge_l, vmj_edge_r;
INSERT INTO vmj_edge_l SELECT g, g * 10 FROM generate_series(1, 500) g;
INSERT INTO vmj_edge_r SELECT g, g * 20 FROM generate_series(1, 500) g;
ANALYZE vmj_edge_l;
ANALYZE vmj_edge_r;

SELECT 'full_overlap' AS test, COUNT(*) AS cnt
FROM vmj_edge_l l JOIN vmj_edge_r r ON l.id = r.id;
-- Expected: 500

SELECT 'full_overlap_sum' AS test, SUM(l.val) AS lsum, SUM(r.val) AS rsum
FROM vmj_edge_l l JOIN vmj_edge_r r ON l.id = r.id;
-- Expected: lsum=1252500, rsum=2505000

-- -------------------------------------------------------
-- 5. Negative and zero keys
-- -------------------------------------------------------
TRUNCATE vmj_edge_l, vmj_edge_r;
INSERT INTO vmj_edge_l VALUES (-10, 1), (-5, 2), (0, 3), (5, 4), (10, 5);
INSERT INTO vmj_edge_r VALUES (-10, 10), (0, 30), (10, 50), (20, 60);
ANALYZE vmj_edge_l;
ANALYZE vmj_edge_r;

SELECT l.id, l.val AS lval, r.val AS rval
FROM vmj_edge_l l JOIN vmj_edge_r r ON l.id = r.id
ORDER BY l.id;
-- Expected: 3 rows: -10, 0, 10

-- -------------------------------------------------------
-- 6. Very large values
-- -------------------------------------------------------
TRUNCATE vmj_edge_l, vmj_edge_r;
INSERT INTO vmj_edge_l VALUES (2147483647, 1), (-2147483648, 2), (0, 3);
INSERT INTO vmj_edge_r VALUES (2147483647, 10), (-2147483648, 20), (999, 30);
ANALYZE vmj_edge_l;
ANALYZE vmj_edge_r;

SELECT 'extreme_vals' AS test, l.id, l.val AS lval, r.val AS rval
FROM vmj_edge_l l JOIN vmj_edge_r r ON l.id = r.id
ORDER BY l.id;
-- Expected: 2 rows: -2147483648 and 2147483647

-- Cleanup
DROP TABLE vmj_edge_l, vmj_edge_r;
