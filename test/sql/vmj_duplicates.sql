-- =============================================================
-- VectorMergeJoin (VMJ) — Duplicates & Group Tests
-- =============================================================
-- Merge join collects groups of equal keys and emits their
-- cross-product, so duplicates are a critical edge case.

SELECT vjoin_loaded();

SET enable_hashjoin = off;
SET enable_mergejoin = off;
SET enable_nestloop = off;
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_nestloop = off;
SET pg_vectorjoin.enable_mergejoin = on;
SET max_parallel_workers_per_gather = 0;

DROP TABLE IF EXISTS vmj_dup_l, vmj_dup_r;

CREATE TABLE vmj_dup_l (id int, val text);
CREATE TABLE vmj_dup_r (id int, data text);

-- -------------------------------------------------------
-- 1. Many-to-many: 5 left × 3 right with same key
-- -------------------------------------------------------
INSERT INTO vmj_dup_l VALUES
  (1, 'L1'), (1, 'L2'), (1, 'L3'), (1, 'L4'), (1, 'L5');
INSERT INTO vmj_dup_r VALUES
  (1, 'R1'), (1, 'R2'), (1, 'R3');
ANALYZE vmj_dup_l;
ANALYZE vmj_dup_r;

-- Should return 5 × 3 = 15 rows
SELECT 'many_to_many' AS test, COUNT(*) AS cnt
FROM vmj_dup_l l JOIN vmj_dup_r r ON l.id = r.id;

-- Verify all combinations exist
SELECT l.val, r.data
FROM vmj_dup_l l JOIN vmj_dup_r r ON l.id = r.id
ORDER BY l.val, r.data;

-- -------------------------------------------------------
-- 2. Multiple distinct groups with varying sizes
-- -------------------------------------------------------
TRUNCATE vmj_dup_l, vmj_dup_r;

-- Left: key=10 (3 rows), key=20 (1 row), key=30 (4 rows)
INSERT INTO vmj_dup_l VALUES
  (10, 'A1'), (10, 'A2'), (10, 'A3'),
  (20, 'B1'),
  (30, 'C1'), (30, 'C2'), (30, 'C3'), (30, 'C4');

-- Right: key=10 (2 rows), key=20 (3 rows), key=30 (1 row), key=40 (2 rows)
INSERT INTO vmj_dup_r VALUES
  (10, 'X1'), (10, 'X2'),
  (20, 'Y1'), (20, 'Y2'), (20, 'Y3'),
  (30, 'Z1'),
  (40, 'W1'), (40, 'W2');

ANALYZE vmj_dup_l;
ANALYZE vmj_dup_r;

-- Expected: (3×2) + (1×3) + (4×1) + 0 = 6 + 3 + 4 = 13
SELECT 'multi_group' AS test, COUNT(*) AS cnt
FROM vmj_dup_l l JOIN vmj_dup_r r ON l.id = r.id;

-- Per-group counts
SELECT l.id AS key, COUNT(*) AS combos
FROM vmj_dup_l l JOIN vmj_dup_r r ON l.id = r.id
GROUP BY l.id
ORDER BY l.id;

-- -------------------------------------------------------
-- 3. Large duplicate group (stress group collection)
-- -------------------------------------------------------
TRUNCATE vmj_dup_l, vmj_dup_r;

-- 200 left rows × 150 right rows, all with key=42
INSERT INTO vmj_dup_l
  SELECT 42, 'L' || g FROM generate_series(1, 200) g;
INSERT INTO vmj_dup_r
  SELECT 42, 'R' || g FROM generate_series(1, 150) g;
ANALYZE vmj_dup_l;
ANALYZE vmj_dup_r;

-- Expected: 200 × 150 = 30000
SELECT 'large_group' AS test, COUNT(*) AS cnt
FROM vmj_dup_l l JOIN vmj_dup_r r ON l.id = r.id;

-- -------------------------------------------------------
-- 4. One-to-one (no duplicates at all)
-- -------------------------------------------------------
TRUNCATE vmj_dup_l, vmj_dup_r;

INSERT INTO vmj_dup_l SELECT g, 'L' || g FROM generate_series(1, 500) g;
INSERT INTO vmj_dup_r SELECT g, 'R' || g FROM generate_series(1, 500) g;
ANALYZE vmj_dup_l;
ANALYZE vmj_dup_r;

SELECT 'one_to_one' AS test, COUNT(*) AS cnt
FROM vmj_dup_l l JOIN vmj_dup_r r ON l.id = r.id;
-- Expected: 500

-- -------------------------------------------------------
-- 5. Partial overlap with duplicates
-- -------------------------------------------------------
TRUNCATE vmj_dup_l, vmj_dup_r;

-- Left: 1..100, each repeated 3 times
INSERT INTO vmj_dup_l
  SELECT (g - 1) / 3 + 1, 'L' || g FROM generate_series(1, 300) g;
-- Right: 50..150, each repeated 2 times
INSERT INTO vmj_dup_r
  SELECT (g - 1) / 2 + 50, 'R' || g FROM generate_series(1, 202) g;
ANALYZE vmj_dup_l;
ANALYZE vmj_dup_r;

-- Overlap on keys 50..100: 51 keys × 3 left × 2 right = 306
SELECT 'partial_overlap' AS test, COUNT(*) AS cnt
FROM vmj_dup_l l JOIN vmj_dup_r r ON l.id = r.id;

-- Compare with PG native
SET pg_vectorjoin.enable_mergejoin = off;
SET enable_mergejoin = on;
SELECT 'PG_partial_overlap' AS test, COUNT(*) AS cnt
FROM vmj_dup_l l JOIN vmj_dup_r r ON l.id = r.id;

-- Cleanup
DROP TABLE vmj_dup_l, vmj_dup_r;
