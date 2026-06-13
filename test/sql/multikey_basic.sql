-- Multi-key (composite join key) correctness tests.
--
-- Strategy: compute each join with the extension DISABLED (reference, using
-- core PostgreSQL) into a temp table, then re-run with the extension ENABLED
-- and compare both directions with EXCEPT.  An empty diff means the vectorized
-- result is identical to core PostgreSQL.

CREATE EXTENSION IF NOT EXISTS pg_vectorjoin;

CREATE TABLE mk_outer (
    k1   int4,
    k2   int8,
    k3   float8,
    val  text
);
CREATE TABLE mk_inner (
    k1   int4,
    k2   int8,
    k3   float8,
    data text
);

INSERT INTO mk_outer
SELECT (g % 50), (g % 30)::int8, (g % 20)::float8, 'o_' || g
FROM generate_series(1, 2000) g;

INSERT INTO mk_inner
SELECT (g % 50), (g % 30)::int8, (g % 20)::float8, 'i_' || g
FROM generate_series(1, 4000) g;

ANALYZE mk_outer;
ANALYZE mk_inner;

-- Helper to compare a vectorized join against the reference set.
-- Usage: build mk_ref with extension OFF, build mk_got with extension ON,
-- then SELECT the two EXCEPT counts (both must be 0).

----------------------------------------------------------------------
-- Test 1: two-key INNER join (int4 + int8)
----------------------------------------------------------------------
SET pg_vectorjoin.enable = off;
CREATE TEMP TABLE ref1 AS
SELECT o.val, i.data
FROM mk_outer o JOIN mk_inner i
  ON o.k1 = i.k1 AND o.k2 = i.k2;

-- hash join
SET pg_vectorjoin.enable = on;
SET pg_vectorjoin.enable_hashjoin = on;
SET pg_vectorjoin.enable_mergejoin = off;
SET pg_vectorjoin.enable_nestloop = off;
CREATE TEMP TABLE got1_hash AS
SELECT o.val, i.data
FROM mk_outer o JOIN mk_inner i
  ON o.k1 = i.k1 AND o.k2 = i.k2;

-- merge join
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_mergejoin = on;
CREATE TEMP TABLE got1_merge AS
SELECT o.val, i.data
FROM mk_outer o JOIN mk_inner i
  ON o.k1 = i.k1 AND o.k2 = i.k2;

-- nested loop
SET pg_vectorjoin.enable_mergejoin = off;
SET pg_vectorjoin.enable_nestloop = on;
CREATE TEMP TABLE got1_nl AS
SELECT o.val, i.data
FROM mk_outer o JOIN mk_inner i
  ON o.k1 = i.k1 AND o.k2 = i.k2;

SELECT 'hash'  AS join, count(*) FROM (SELECT * FROM ref1 EXCEPT ALL SELECT * FROM got1_hash) d
UNION ALL
SELECT 'hash_rev', count(*) FROM (SELECT * FROM got1_hash EXCEPT ALL SELECT * FROM ref1) d
UNION ALL
SELECT 'merge', count(*) FROM (SELECT * FROM ref1 EXCEPT ALL SELECT * FROM got1_merge) d
UNION ALL
SELECT 'merge_rev', count(*) FROM (SELECT * FROM got1_merge EXCEPT ALL SELECT * FROM ref1) d
UNION ALL
SELECT 'nl', count(*) FROM (SELECT * FROM ref1 EXCEPT ALL SELECT * FROM got1_nl) d
UNION ALL
SELECT 'nl_rev', count(*) FROM (SELECT * FROM got1_nl EXCEPT ALL SELECT * FROM ref1) d
ORDER BY 1;

----------------------------------------------------------------------
-- Test 2: three-key INNER join (int4 + int8 + float8)
----------------------------------------------------------------------
SET pg_vectorjoin.enable = off;
CREATE TEMP TABLE ref2 AS
SELECT o.val, i.data
FROM mk_outer o JOIN mk_inner i
  ON o.k1 = i.k1 AND o.k2 = i.k2 AND o.k3 = i.k3;

SET pg_vectorjoin.enable = on;
SET pg_vectorjoin.enable_hashjoin = on;
SET pg_vectorjoin.enable_mergejoin = off;
SET pg_vectorjoin.enable_nestloop = off;
CREATE TEMP TABLE got2_hash AS
SELECT o.val, i.data
FROM mk_outer o JOIN mk_inner i
  ON o.k1 = i.k1 AND o.k2 = i.k2 AND o.k3 = i.k3;

SELECT 'hash3', count(*) FROM (SELECT * FROM ref2 EXCEPT ALL SELECT * FROM got2_hash) d
UNION ALL
SELECT 'hash3_rev', count(*) FROM (SELECT * FROM got2_hash EXCEPT ALL SELECT * FROM ref2) d
ORDER BY 1;

----------------------------------------------------------------------
-- Test 3: multi-key LEFT JOIN (regression for unmatched-row qual bug)
-- Outer rows with no inner match must survive as NULL-extended rows.
----------------------------------------------------------------------
-- Force many unmatched outer rows by using a key range absent on the inner.
INSERT INTO mk_outer
SELECT 999, (g)::int8, 0::float8, 'lonely_' || g
FROM generate_series(1, 100) g;
ANALYZE mk_outer;

SET pg_vectorjoin.enable = off;
CREATE TEMP TABLE ref3 AS
SELECT o.val, i.data
FROM mk_outer o LEFT JOIN mk_inner i
  ON o.k1 = i.k1 AND o.k2 = i.k2;

SET pg_vectorjoin.enable = on;
SET pg_vectorjoin.enable_hashjoin = on;
SET pg_vectorjoin.enable_mergejoin = off;
SET pg_vectorjoin.enable_nestloop = off;
CREATE TEMP TABLE got3_hash AS
SELECT o.val, i.data
FROM mk_outer o LEFT JOIN mk_inner i
  ON o.k1 = i.k1 AND o.k2 = i.k2;

SELECT 'left', count(*) FROM (SELECT * FROM ref3 EXCEPT ALL SELECT * FROM got3_hash) d
UNION ALL
SELECT 'left_rev', count(*) FROM (SELECT * FROM got3_hash EXCEPT ALL SELECT * FROM ref3) d
UNION ALL
-- explicit count of NULL-extended (unmatched) rows: must be > 0 and equal
SELECT 'left_unmatched_ref', count(*) FROM ref3 WHERE data IS NULL
UNION ALL
SELECT 'left_unmatched_got', count(*) FROM got3_hash WHERE data IS NULL
ORDER BY 1;

----------------------------------------------------------------------
-- Test 4: residual (non-equi-key) condition on top of an equi-key.
-- The executor matches on the equi-key and must re-check the residual
-- "o.k3 < i.k3" as a qual; dropping that recheck would over-produce rows.
----------------------------------------------------------------------
SET pg_vectorjoin.enable = off;
CREATE TEMP TABLE ref4 AS
SELECT o.val, i.data
FROM mk_outer o JOIN mk_inner i
  ON o.k1 = i.k1 AND o.k3 < i.k3;

SET pg_vectorjoin.enable = on;
SET pg_vectorjoin.enable_hashjoin = on;
SET pg_vectorjoin.enable_mergejoin = off;
SET pg_vectorjoin.enable_nestloop = off;
CREATE TEMP TABLE got4_hash AS
SELECT o.val, i.data
FROM mk_outer o JOIN mk_inner i
  ON o.k1 = i.k1 AND o.k3 < i.k3;

SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_nestloop = on;
CREATE TEMP TABLE got4_nl AS
SELECT o.val, i.data
FROM mk_outer o JOIN mk_inner i
  ON o.k1 = i.k1 AND o.k3 < i.k3;

SELECT 'resid_hash', count(*) FROM (SELECT * FROM ref4 EXCEPT ALL SELECT * FROM got4_hash) d
UNION ALL
SELECT 'resid_hash_rev', count(*) FROM (SELECT * FROM got4_hash EXCEPT ALL SELECT * FROM ref4) d
UNION ALL
SELECT 'resid_nl', count(*) FROM (SELECT * FROM ref4 EXCEPT ALL SELECT * FROM got4_nl) d
UNION ALL
SELECT 'resid_nl_rev', count(*) FROM (SELECT * FROM got4_nl EXCEPT ALL SELECT * FROM ref4) d
UNION ALL
-- residual must actually filter rows (otherwise the test is vacuous)
SELECT 'resid_filtered_gt0', (count(*) > 0)::int
  FROM mk_outer o JOIN mk_inner i ON o.k1 = i.k1 AND o.k3 >= i.k3
ORDER BY 1;

DROP TABLE mk_outer, mk_inner;
DROP EXTENSION pg_vectorjoin;
