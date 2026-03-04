-- =============================================================
-- NestLoop (NL) — Full Correctness Test Suite
-- =============================================================
SELECT vjoin_loaded();

-- Force only NL
SET enable_hashjoin = off;
SET enable_mergejoin = off;
SET enable_nestloop = off;
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_nestloop = on;
SET pg_vectorjoin.enable_mergejoin = off;
SET max_parallel_workers_per_gather = 0;

-- =============================================================
-- 1. BASIC CORRECTNESS (INT4)
-- =============================================================
DROP TABLE IF EXISTS nl_left, nl_right;
CREATE TABLE nl_left  (id int PRIMARY KEY, val int);
CREATE TABLE nl_right (id int PRIMARY KEY, ref_id int, data text);

INSERT INTO nl_left  SELECT g, g % 100 FROM generate_series(1, 1000) g;
INSERT INTO nl_right SELECT g, (g % 200) + 1, 'row_' || g FROM generate_series(1, 2000) g;
ANALYZE nl_left;
ANALYZE nl_right;

-- Plan check
EXPLAIN (COSTS OFF)
SELECT l.id, l.val, r.data
FROM nl_left l JOIN nl_right r ON l.id = r.ref_id;

-- Count
SELECT 'NL_basic_count' AS test, COUNT(*) AS cnt
FROM nl_left l JOIN nl_right r ON l.id = r.ref_id;

-- Compare with PG native
SET pg_vectorjoin.enable_nestloop = off;
SET enable_nestloop = on;
SELECT 'PG_NL_basic_count' AS test, COUNT(*) AS cnt
FROM nl_left l JOIN nl_right r ON l.id = r.ref_id;
SET enable_nestloop = off;
SET pg_vectorjoin.enable_nestloop = on;

-- SUM check
SELECT 'NL_basic_sum' AS test, SUM(l.val) AS lsum, SUM(r.ref_id) AS rsum
FROM nl_left l JOIN nl_right r ON l.id = r.ref_id;

SET pg_vectorjoin.enable_nestloop = off;
SET enable_nestloop = on;
SELECT 'PG_NL_basic_sum' AS test, SUM(l.val) AS lsum, SUM(r.ref_id) AS rsum
FROM nl_left l JOIN nl_right r ON l.id = r.ref_id;
SET enable_nestloop = off;
SET pg_vectorjoin.enable_nestloop = on;

-- Spot check rows
SELECT l.id, l.val, r.id AS rid, r.ref_id, r.data
FROM nl_left l JOIN nl_right r ON l.id = r.ref_id
WHERE l.id <= 5
ORDER BY l.id, r.id;

-- EXPLAIN ANALYZE
EXPLAIN (ANALYZE, COSTS OFF, TIMING ON, SUMMARY ON)
SELECT COUNT(*) FROM nl_left l JOIN nl_right r ON l.id = r.ref_id;

DROP TABLE nl_left, nl_right;

-- =============================================================
-- 2. NULL HANDLING
-- =============================================================
DROP TABLE IF EXISTS bnl_null_l, bnl_null_r;
CREATE TABLE bnl_null_l (id int, val text);
CREATE TABLE bnl_null_r (id int, data text);

-- 2a. NULLs on both sides
INSERT INTO bnl_null_l VALUES (1, 'a'), (NULL, 'b'), (3, 'c'), (NULL, 'd');
INSERT INTO bnl_null_r VALUES (1, 'x'), (NULL, 'y'), (3, 'z'), (NULL, 'w');
ANALYZE bnl_null_l;
ANALYZE bnl_null_r;

SELECT 'NULL_both' AS test, COUNT(*) AS cnt
FROM bnl_null_l l JOIN bnl_null_r r ON l.id = r.id;
-- Expected: 2

SELECT l.id, l.val, r.data
FROM bnl_null_l l JOIN bnl_null_r r ON l.id = r.id
ORDER BY l.id;

-- 2b. NULLs only on left
TRUNCATE bnl_null_l, bnl_null_r;
INSERT INTO bnl_null_l VALUES (NULL, 'a'), (NULL, 'b'), (NULL, 'c');
INSERT INTO bnl_null_r VALUES (1, 'x'), (2, 'y');
ANALYZE bnl_null_l;
ANALYZE bnl_null_r;

SELECT 'NULL_left_only' AS test, COUNT(*) AS cnt
FROM bnl_null_l l JOIN bnl_null_r r ON l.id = r.id;
-- Expected: 0

-- 2c. NULLs only on right
TRUNCATE bnl_null_l, bnl_null_r;
INSERT INTO bnl_null_l VALUES (1, 'a'), (2, 'b');
INSERT INTO bnl_null_r VALUES (NULL, 'x'), (NULL, 'y'), (NULL, 'z');
ANALYZE bnl_null_l;
ANALYZE bnl_null_r;

SELECT 'NULL_right_only' AS test, COUNT(*) AS cnt
FROM bnl_null_l l JOIN bnl_null_r r ON l.id = r.id;
-- Expected: 0

-- 2d. All NULLs
TRUNCATE bnl_null_l, bnl_null_r;
INSERT INTO bnl_null_l VALUES (NULL, 'a'), (NULL, 'b');
INSERT INTO bnl_null_r VALUES (NULL, 'x'), (NULL, 'y');
ANALYZE bnl_null_l;
ANALYZE bnl_null_r;

SELECT 'ALL_null' AS test, COUNT(*) AS cnt
FROM bnl_null_l l JOIN bnl_null_r r ON l.id = r.id;
-- Expected: 0

-- 2e. NULLs interspersed
TRUNCATE bnl_null_l, bnl_null_r;
INSERT INTO bnl_null_l
  SELECT CASE WHEN g % 3 = 0 THEN NULL ELSE g END, 'l_' || g
  FROM generate_series(1, 30) g;
INSERT INTO bnl_null_r
  SELECT CASE WHEN g % 5 = 0 THEN NULL ELSE g END, 'r_' || g
  FROM generate_series(1, 30) g;
ANALYZE bnl_null_l;
ANALYZE bnl_null_r;

SELECT 'NL_mixed_null' AS test, COUNT(*) AS cnt
FROM bnl_null_l l JOIN bnl_null_r r ON l.id = r.id;

SET pg_vectorjoin.enable_nestloop = off;
SET enable_nestloop = on;
SELECT 'PG_mixed_null' AS test, COUNT(*) AS cnt
FROM bnl_null_l l JOIN bnl_null_r r ON l.id = r.id;
SET enable_nestloop = off;
SET pg_vectorjoin.enable_nestloop = on;

DROP TABLE bnl_null_l, bnl_null_r;

-- =============================================================
-- 3. DATA TYPES
-- =============================================================

-- 3a. INT8 (bigint)
DROP TABLE IF EXISTS bnl_i8_l, bnl_i8_r;
CREATE TABLE bnl_i8_l (id bigint, val int);
CREATE TABLE bnl_i8_r (id bigint, val int);

INSERT INTO bnl_i8_l SELECT g::bigint * 1000000000, g FROM generate_series(1, 500) g;
INSERT INTO bnl_i8_r SELECT g::bigint * 1000000000, g * 10 FROM generate_series(1, 500) g;
ANALYZE bnl_i8_l;
ANALYZE bnl_i8_r;

SELECT 'INT8' AS test, COUNT(*) AS cnt, SUM(l.val) AS lsum, SUM(r.val) AS rsum
FROM bnl_i8_l l JOIN bnl_i8_r r ON l.id = r.id;
-- Expected: 500, lsum=125250, rsum=1252500

EXPLAIN (COSTS OFF)
SELECT * FROM bnl_i8_l l JOIN bnl_i8_r r ON l.id = r.id;

DROP TABLE bnl_i8_l, bnl_i8_r;

-- 3b. FLOAT8
DROP TABLE IF EXISTS bnl_f8_l, bnl_f8_r;
CREATE TABLE bnl_f8_l (id float8, val int);
CREATE TABLE bnl_f8_r (id float8, val int);

INSERT INTO bnl_f8_l SELECT g * 1.5, g FROM generate_series(1, 200) g;
INSERT INTO bnl_f8_r SELECT g * 1.5, g * 100 FROM generate_series(50, 250) g;
ANALYZE bnl_f8_l;
ANALYZE bnl_f8_r;

SELECT 'NL_FLOAT8' AS test, COUNT(*) AS cnt
FROM bnl_f8_l l JOIN bnl_f8_r r ON l.id = r.id;

SET pg_vectorjoin.enable_nestloop = off;
SET enable_nestloop = on;
SELECT 'PG_FLOAT8' AS test, COUNT(*) AS cnt
FROM bnl_f8_l l JOIN bnl_f8_r r ON l.id = r.id;
SET enable_nestloop = off;
SET pg_vectorjoin.enable_nestloop = on;

DROP TABLE bnl_f8_l, bnl_f8_r;

-- 3c. Mixed column types (join on int, carry text/numeric)
DROP TABLE IF EXISTS bnl_mix_l, bnl_mix_r;
CREATE TABLE bnl_mix_l (id int, name text, score float8);
CREATE TABLE bnl_mix_r (id int, label text, amount numeric);

INSERT INTO bnl_mix_l
  SELECT g, 'user_' || g, random() * 100.0
  FROM generate_series(1, 300) g;
INSERT INTO bnl_mix_r
  SELECT g, 'item_' || g, (random() * 1000)::numeric(10,2)
  FROM generate_series(100, 400) g;
ANALYZE bnl_mix_l;
ANALYZE bnl_mix_r;

SELECT 'mixed_types' AS test, COUNT(*) AS cnt
FROM bnl_mix_l l JOIN bnl_mix_r r ON l.id = r.id;
-- Expected: 201 (overlap 100..300)

SELECT l.id, l.name, r.label
FROM bnl_mix_l l JOIN bnl_mix_r r ON l.id = r.id
WHERE l.id IN (100, 200, 300)
ORDER BY l.id;

DROP TABLE bnl_mix_l, bnl_mix_r;

-- =============================================================
-- 4. DUPLICATES & MANY-TO-MANY
-- =============================================================
DROP TABLE IF EXISTS bnl_dup_l, bnl_dup_r;
CREATE TABLE bnl_dup_l (id int, val text);
CREATE TABLE bnl_dup_r (id int, data text);

-- 4a. 5 left × 3 right
INSERT INTO bnl_dup_l VALUES
  (1, 'L1'), (1, 'L2'), (1, 'L3'), (1, 'L4'), (1, 'L5');
INSERT INTO bnl_dup_r VALUES
  (1, 'R1'), (1, 'R2'), (1, 'R3');
ANALYZE bnl_dup_l;
ANALYZE bnl_dup_r;

SELECT 'many_to_many' AS test, COUNT(*) AS cnt
FROM bnl_dup_l l JOIN bnl_dup_r r ON l.id = r.id;
-- Expected: 15

SELECT l.val, r.data
FROM bnl_dup_l l JOIN bnl_dup_r r ON l.id = r.id
ORDER BY l.val, r.data;

-- 4b. Multiple groups
TRUNCATE bnl_dup_l, bnl_dup_r;
INSERT INTO bnl_dup_l VALUES
  (10, 'A1'), (10, 'A2'), (10, 'A3'),
  (20, 'B1'),
  (30, 'C1'), (30, 'C2'), (30, 'C3'), (30, 'C4');
INSERT INTO bnl_dup_r VALUES
  (10, 'X1'), (10, 'X2'),
  (20, 'Y1'), (20, 'Y2'), (20, 'Y3'),
  (30, 'Z1'),
  (40, 'W1'), (40, 'W2');
ANALYZE bnl_dup_l;
ANALYZE bnl_dup_r;

-- Expected: (3×2) + (1×3) + (4×1) = 13
SELECT 'multi_group' AS test, COUNT(*) AS cnt
FROM bnl_dup_l l JOIN bnl_dup_r r ON l.id = r.id;

SELECT l.id AS key, COUNT(*) AS combos
FROM bnl_dup_l l JOIN bnl_dup_r r ON l.id = r.id
GROUP BY l.id
ORDER BY l.id;

-- 4c. Large duplicate group (200 × 150 = 30000)
TRUNCATE bnl_dup_l, bnl_dup_r;
INSERT INTO bnl_dup_l SELECT 42, 'L' || g FROM generate_series(1, 200) g;
INSERT INTO bnl_dup_r SELECT 42, 'R' || g FROM generate_series(1, 150) g;
ANALYZE bnl_dup_l;
ANALYZE bnl_dup_r;

SELECT 'large_group' AS test, COUNT(*) AS cnt
FROM bnl_dup_l l JOIN bnl_dup_r r ON l.id = r.id;
-- Expected: 30000

-- 4d. One-to-one
TRUNCATE bnl_dup_l, bnl_dup_r;
INSERT INTO bnl_dup_l SELECT g, 'L' || g FROM generate_series(1, 500) g;
INSERT INTO bnl_dup_r SELECT g, 'R' || g FROM generate_series(1, 500) g;
ANALYZE bnl_dup_l;
ANALYZE bnl_dup_r;

SELECT 'one_to_one' AS test, COUNT(*) AS cnt
FROM bnl_dup_l l JOIN bnl_dup_r r ON l.id = r.id;
-- Expected: 500

DROP TABLE bnl_dup_l, bnl_dup_r;

-- =============================================================
-- 5. EDGE CASES
-- =============================================================
DROP TABLE IF EXISTS bnl_edge_l, bnl_edge_r;
CREATE TABLE bnl_edge_l (id int, val int);
CREATE TABLE bnl_edge_r (id int, val int);

-- 5a. Both empty
ANALYZE bnl_edge_l;
ANALYZE bnl_edge_r;
SELECT 'both_empty' AS test, COUNT(*) AS cnt
FROM bnl_edge_l l JOIN bnl_edge_r r ON l.id = r.id;
-- Expected: 0

-- 5b. Left empty
INSERT INTO bnl_edge_r SELECT g, g FROM generate_series(1, 100) g;
ANALYZE bnl_edge_r;
SELECT 'left_empty' AS test, COUNT(*) AS cnt
FROM bnl_edge_l l JOIN bnl_edge_r r ON l.id = r.id;
-- Expected: 0

-- 5c. Right empty
TRUNCATE bnl_edge_r;
INSERT INTO bnl_edge_l SELECT g, g FROM generate_series(1, 100) g;
ANALYZE bnl_edge_l;
ANALYZE bnl_edge_r;
SELECT 'right_empty' AS test, COUNT(*) AS cnt
FROM bnl_edge_l l JOIN bnl_edge_r r ON l.id = r.id;
-- Expected: 0

-- 5d. Single row match
TRUNCATE bnl_edge_l, bnl_edge_r;
INSERT INTO bnl_edge_l VALUES (42, 1);
INSERT INTO bnl_edge_r VALUES (42, 2);
ANALYZE bnl_edge_l;
ANALYZE bnl_edge_r;
SELECT 'single_match' AS test, l.id, l.val AS lval, r.val AS rval
FROM bnl_edge_l l JOIN bnl_edge_r r ON l.id = r.id;

-- 5e. Single row no match
TRUNCATE bnl_edge_r;
INSERT INTO bnl_edge_r VALUES (99, 2);
ANALYZE bnl_edge_r;
SELECT 'single_nomatch' AS test, COUNT(*) AS cnt
FROM bnl_edge_l l JOIN bnl_edge_r r ON l.id = r.id;
-- Expected: 0

-- 5f. Disjoint ranges
TRUNCATE bnl_edge_l, bnl_edge_r;
INSERT INTO bnl_edge_l SELECT g, g FROM generate_series(1, 500) g;
INSERT INTO bnl_edge_r SELECT g, g FROM generate_series(501, 1000) g;
ANALYZE bnl_edge_l;
ANALYZE bnl_edge_r;
SELECT 'disjoint' AS test, COUNT(*) AS cnt
FROM bnl_edge_l l JOIN bnl_edge_r r ON l.id = r.id;
-- Expected: 0

-- 5g. Full overlap
TRUNCATE bnl_edge_l, bnl_edge_r;
INSERT INTO bnl_edge_l SELECT g, g * 10 FROM generate_series(1, 500) g;
INSERT INTO bnl_edge_r SELECT g, g * 20 FROM generate_series(1, 500) g;
ANALYZE bnl_edge_l;
ANALYZE bnl_edge_r;
SELECT 'full_overlap' AS test, COUNT(*) AS cnt
FROM bnl_edge_l l JOIN bnl_edge_r r ON l.id = r.id;
-- Expected: 500

SELECT 'full_overlap_sum' AS test, SUM(l.val) AS lsum, SUM(r.val) AS rsum
FROM bnl_edge_l l JOIN bnl_edge_r r ON l.id = r.id;
-- Expected: lsum=1252500, rsum=2505000

-- 5h. Negative and zero keys
TRUNCATE bnl_edge_l, bnl_edge_r;
INSERT INTO bnl_edge_l VALUES (-100, 1), (-1, 2), (0, 3), (1, 4), (100, 5);
INSERT INTO bnl_edge_r VALUES (-100, 10), (0, 30), (100, 50), (200, 60);
ANALYZE bnl_edge_l;
ANALYZE bnl_edge_r;

SELECT 'negative_zero' AS test, l.id, l.val AS lval, r.val AS rval
FROM bnl_edge_l l JOIN bnl_edge_r r ON l.id = r.id
ORDER BY l.id;
-- Expected: 3 rows (-100, 0, 100)

DROP TABLE bnl_edge_l, bnl_edge_r;

-- =============================================================
-- 6. BLOCK BOUNDARY (> 1024 outer rows to test multi-block)
-- =============================================================
DROP TABLE IF EXISTS bnl_block_l, bnl_block_r;
CREATE TABLE bnl_block_l (id int, val int);
CREATE TABLE bnl_block_r (id int, val int);

-- 3000 outer rows = 3 full blocks at batch_size=1024
INSERT INTO bnl_block_l SELECT g, g FROM generate_series(1, 3000) g;
INSERT INTO bnl_block_r SELECT g, g * 10 FROM generate_series(1, 3000) g;
ANALYZE bnl_block_l;
ANALYZE bnl_block_r;

SELECT 'block_boundary_count' AS test, COUNT(*) AS cnt
FROM bnl_block_l l JOIN bnl_block_r r ON l.id = r.id;
-- Expected: 3000

SELECT 'block_boundary_sum' AS test, SUM(l.val) AS lsum, SUM(r.val) AS rsum
FROM bnl_block_l l JOIN bnl_block_r r ON l.id = r.id;

SET pg_vectorjoin.enable_nestloop = off;
SET enable_nestloop = on;
SELECT 'PG_block_boundary_sum' AS test, SUM(l.val) AS lsum, SUM(r.val) AS rsum
FROM bnl_block_l l JOIN bnl_block_r r ON l.id = r.id;
SET enable_nestloop = off;
SET pg_vectorjoin.enable_nestloop = on;

-- Partial overlap across blocks
TRUNCATE bnl_block_l, bnl_block_r;
INSERT INTO bnl_block_l SELECT g, g FROM generate_series(1, 2500) g;
INSERT INTO bnl_block_r SELECT g, g FROM generate_series(500, 2000) g;
ANALYZE bnl_block_l;
ANALYZE bnl_block_r;

SELECT 'partial_overlap_block' AS test, COUNT(*) AS cnt
FROM bnl_block_l l JOIN bnl_block_r r ON l.id = r.id;
-- Expected: 1501

DROP TABLE bnl_block_l, bnl_block_r;

-- =============================================================
-- 7. SMALL BATCH SIZE (test with batch_size=64)
-- =============================================================
SET pg_vectorjoin.batch_size = 64;

DROP TABLE IF EXISTS bnl_small_l, bnl_small_r;
CREATE TABLE bnl_small_l (id int, val int);
CREATE TABLE bnl_small_r (id int, val int);

INSERT INTO bnl_small_l SELECT g, g FROM generate_series(1, 50) g;
INSERT INTO bnl_small_r SELECT g, g * 10 FROM generate_series(1, 50) g;
ANALYZE bnl_small_l;
ANALYZE bnl_small_r;

SELECT 'small_batch_count' AS test, COUNT(*) AS cnt
FROM bnl_small_l l JOIN bnl_small_r r ON l.id = r.id;
-- Expected: 50

SELECT 'small_batch_sum' AS test, SUM(l.val) AS lsum, SUM(r.val) AS rsum
FROM bnl_small_l l JOIN bnl_small_r r ON l.id = r.id;

-- Verify with PG native
SET pg_vectorjoin.enable_nestloop = off;
SET enable_nestloop = on;
SELECT 'PG_small_batch_sum' AS test, SUM(l.val) AS lsum, SUM(r.val) AS rsum
FROM bnl_small_l l JOIN bnl_small_r r ON l.id = r.id;
SET enable_nestloop = off;
SET pg_vectorjoin.enable_nestloop = on;

DROP TABLE bnl_small_l, bnl_small_r;

-- Reset batch_size
RESET pg_vectorjoin.batch_size;

-- =============================================================
-- 8. MULTI-KEY JOIN (scalar fallback path)
-- =============================================================
DROP TABLE IF EXISTS bnl_mk_l, bnl_mk_r;
CREATE TABLE bnl_mk_l (k1 int, k2 int, val text);
CREATE TABLE bnl_mk_r (k1 int, k2 int, data text);

INSERT INTO bnl_mk_l SELECT g / 10, g % 10, 'L' || g FROM generate_series(1, 200) g;
INSERT INTO bnl_mk_r SELECT g / 10, g % 10, 'R' || g FROM generate_series(1, 200) g;
ANALYZE bnl_mk_l;
ANALYZE bnl_mk_r;

SELECT 'NL_multikey' AS test, COUNT(*) AS cnt
FROM bnl_mk_l l JOIN bnl_mk_r r ON l.k1 = r.k1 AND l.k2 = r.k2;

SET pg_vectorjoin.enable_nestloop = off;
SET enable_nestloop = on;
SELECT 'PG_multikey' AS test, COUNT(*) AS cnt
FROM bnl_mk_l l JOIN bnl_mk_r r ON l.k1 = r.k1 AND l.k2 = r.k2;
SET enable_nestloop = off;
SET pg_vectorjoin.enable_nestloop = on;

DROP TABLE bnl_mk_l, bnl_mk_r;

-- =============================================================
-- 9. PERFORMANCE COMPARISON
-- =============================================================
DROP TABLE IF EXISTS bnl_perf_l, bnl_perf_r;
CREATE TABLE bnl_perf_l (id int PRIMARY KEY, val int);
CREATE TABLE bnl_perf_r (id int, ref_id int, data int);

INSERT INTO bnl_perf_l SELECT g, g FROM generate_series(1, 5000) g;
INSERT INTO bnl_perf_r SELECT g, (g % 500) + 1, g FROM generate_series(1, 5000) g;
ANALYZE bnl_perf_l;
ANALYZE bnl_perf_r;

-- NL timing
EXPLAIN (ANALYZE, COSTS OFF, TIMING ON, SUMMARY ON)
SELECT COUNT(*) FROM bnl_perf_l l JOIN bnl_perf_r r ON l.id = r.ref_id;

-- PG NL timing
SET pg_vectorjoin.enable_nestloop = off;
SET enable_nestloop = on;
EXPLAIN (ANALYZE, COSTS OFF, TIMING ON, SUMMARY ON)
SELECT COUNT(*) FROM bnl_perf_l l JOIN bnl_perf_r r ON l.id = r.ref_id;
SET enable_nestloop = off;
SET pg_vectorjoin.enable_nestloop = on;

DROP TABLE bnl_perf_l, bnl_perf_r;

-- =============================================================
-- FINAL SUMMARY
-- =============================================================
\echo '=============================='
\echo 'NL Full Test Suite Complete'
\echo '=============================='
