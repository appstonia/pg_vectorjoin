-- =============================================================
-- VectorMergeJoin (VMJ) — NULL Handling Tests
-- =============================================================
SELECT vjoin_loaded();

SET enable_hashjoin = off;
SET enable_mergejoin = off;
SET enable_nestloop = off;
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_nestloop = off;
SET pg_vectorjoin.enable_mergejoin = on;
SET max_parallel_workers_per_gather = 0;

DROP TABLE IF EXISTS vmj_null_l, vmj_null_r;

CREATE TABLE vmj_null_l (id int, val text);
CREATE TABLE vmj_null_r (id int, data text);

-- -------------------------------------------------------
-- 1. NULLs on both sides — must never match
-- -------------------------------------------------------
INSERT INTO vmj_null_l VALUES (1, 'a'), (NULL, 'b'), (3, 'c'), (NULL, 'd');
INSERT INTO vmj_null_r VALUES (1, 'x'), (NULL, 'y'), (3, 'z'), (NULL, 'w');
ANALYZE vmj_null_l;
ANALYZE vmj_null_r;

-- Should return 2 rows: id=1 and id=3
SELECT l.id, l.val, r.data
FROM vmj_null_l l JOIN vmj_null_r r ON l.id = r.id
ORDER BY l.id;

SELECT 'NULL_both' AS test, COUNT(*) AS cnt
FROM vmj_null_l l JOIN vmj_null_r r ON l.id = r.id;
-- Expected: 2

-- -------------------------------------------------------
-- 2. NULLs only on left
-- -------------------------------------------------------
TRUNCATE vmj_null_l, vmj_null_r;
INSERT INTO vmj_null_l VALUES (NULL, 'a'), (NULL, 'b'), (NULL, 'c');
INSERT INTO vmj_null_r VALUES (1, 'x'), (2, 'y');
ANALYZE vmj_null_l;
ANALYZE vmj_null_r;

SELECT 'NULL_left_only' AS test, COUNT(*) AS cnt
FROM vmj_null_l l JOIN vmj_null_r r ON l.id = r.id;
-- Expected: 0

-- -------------------------------------------------------
-- 3. NULLs only on right
-- -------------------------------------------------------
TRUNCATE vmj_null_l, vmj_null_r;
INSERT INTO vmj_null_l VALUES (1, 'a'), (2, 'b');
INSERT INTO vmj_null_r VALUES (NULL, 'x'), (NULL, 'y'), (NULL, 'z');
ANALYZE vmj_null_l;
ANALYZE vmj_null_r;

SELECT 'NULL_right_only' AS test, COUNT(*) AS cnt
FROM vmj_null_l l JOIN vmj_null_r r ON l.id = r.id;
-- Expected: 0

-- -------------------------------------------------------
-- 4. All NULLs
-- -------------------------------------------------------
TRUNCATE vmj_null_l, vmj_null_r;
INSERT INTO vmj_null_l VALUES (NULL, 'a'), (NULL, 'b');
INSERT INTO vmj_null_r VALUES (NULL, 'x'), (NULL, 'y');
ANALYZE vmj_null_l;
ANALYZE vmj_null_r;

SELECT 'ALL_null' AS test, COUNT(*) AS cnt
FROM vmj_null_l l JOIN vmj_null_r r ON l.id = r.id;
-- Expected: 0

-- -------------------------------------------------------
-- 5. NULL interspersed with valid keys
-- -------------------------------------------------------
TRUNCATE vmj_null_l, vmj_null_r;
INSERT INTO vmj_null_l
  SELECT CASE WHEN g % 3 = 0 THEN NULL ELSE g END, 'l_' || g
  FROM generate_series(1, 30) g;
INSERT INTO vmj_null_r
  SELECT CASE WHEN g % 5 = 0 THEN NULL ELSE g END, 'r_' || g
  FROM generate_series(1, 30) g;
ANALYZE vmj_null_l;
ANALYZE vmj_null_r;

-- VMJ count
SELECT 'VMJ_mixed_null' AS test, COUNT(*) AS cnt
FROM vmj_null_l l JOIN vmj_null_r r ON l.id = r.id;

-- PG native for comparison
SET pg_vectorjoin.enable_mergejoin = off;
SET enable_mergejoin = on;
SELECT 'PG_mixed_null' AS test, COUNT(*) AS cnt
FROM vmj_null_l l JOIN vmj_null_r r ON l.id = r.id;

-- Cleanup
DROP TABLE vmj_null_l, vmj_null_r;
