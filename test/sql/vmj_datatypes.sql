-- =============================================================
-- VectorMergeJoin (VMJ) — Data Type Tests
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
-- 1. INT8 (bigint) keys
-- -------------------------------------------------------
DROP TABLE IF EXISTS vmj_i8_l, vmj_i8_r;
CREATE TABLE vmj_i8_l (id bigint, val int);
CREATE TABLE vmj_i8_r (id bigint, val int);

INSERT INTO vmj_i8_l SELECT g::bigint * 1000000000, g FROM generate_series(1, 500) g;
INSERT INTO vmj_i8_r SELECT g::bigint * 1000000000, g * 10 FROM generate_series(1, 500) g;
ANALYZE vmj_i8_l;
ANALYZE vmj_i8_r;

SELECT 'INT8' AS test, COUNT(*) AS cnt, SUM(l.val) AS lsum, SUM(r.val) AS rsum
FROM vmj_i8_l l JOIN vmj_i8_r r ON l.id = r.id;
-- Expected: 500, lsum=125250, rsum=1252500

-- Verify plan
EXPLAIN (COSTS OFF)
SELECT * FROM vmj_i8_l l JOIN vmj_i8_r r ON l.id = r.id;

DROP TABLE vmj_i8_l, vmj_i8_r;

-- -------------------------------------------------------
-- 2. FLOAT8 (double precision) keys
-- -------------------------------------------------------
DROP TABLE IF EXISTS vmj_f8_l, vmj_f8_r;
CREATE TABLE vmj_f8_l (id float8, val int);
CREATE TABLE vmj_f8_r (id float8, val int);

INSERT INTO vmj_f8_l SELECT g * 1.5, g FROM generate_series(1, 200) g;
INSERT INTO vmj_f8_r SELECT g * 1.5, g * 100 FROM generate_series(50, 250) g;
ANALYZE vmj_f8_l;
ANALYZE vmj_f8_r;

-- VMJ
SELECT 'FLOAT8_vmj' AS test, COUNT(*) AS cnt
FROM vmj_f8_l l JOIN vmj_f8_r r ON l.id = r.id;

-- PG native for comparison
SET pg_vectorjoin.enable_mergejoin = off;
SET enable_mergejoin = on;
SELECT 'FLOAT8_pg' AS test, COUNT(*) AS cnt
FROM vmj_f8_l l JOIN vmj_f8_r r ON l.id = r.id;

SET enable_mergejoin = off;
SET pg_vectorjoin.enable_mergejoin = on;

DROP TABLE vmj_f8_l, vmj_f8_r;

-- -------------------------------------------------------
-- 3. Mixed column types (join on int, carry text)
-- -------------------------------------------------------
DROP TABLE IF EXISTS vmj_mix_l, vmj_mix_r;
CREATE TABLE vmj_mix_l (id int, name text, score float8);
CREATE TABLE vmj_mix_r (id int, label text, amount numeric);

INSERT INTO vmj_mix_l
  SELECT g, 'user_' || g, random() * 100.0
  FROM generate_series(1, 300) g;
INSERT INTO vmj_mix_r
  SELECT g, 'item_' || g, (random() * 1000)::numeric(10,2)
  FROM generate_series(100, 400) g;
ANALYZE vmj_mix_l;
ANALYZE vmj_mix_r;

-- Should join on id, carry all columns
SELECT 'mixed_types' AS test, COUNT(*) AS cnt
FROM vmj_mix_l l JOIN vmj_mix_r r ON l.id = r.id;
-- Expected: 201 (overlap on 100..300)

-- Verify actual data comes through
SELECT l.id, l.name, l.score, r.label, r.amount
FROM vmj_mix_l l JOIN vmj_mix_r r ON l.id = r.id
WHERE l.id IN (100, 200, 300)
ORDER BY l.id;

DROP TABLE vmj_mix_l, vmj_mix_r;

-- Cleanup
