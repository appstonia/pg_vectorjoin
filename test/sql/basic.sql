-- Setup
CREATE EXTENSION IF NOT EXISTS pg_vectorjoin;

CREATE TABLE vjoin_outer (id int4 PRIMARY KEY, val text);
CREATE TABLE vjoin_inner (id int4 PRIMARY KEY, ref_id int4, data text);

INSERT INTO vjoin_outer SELECT g, 'outer_' || g FROM generate_series(1, 1000) g;
INSERT INTO vjoin_inner SELECT g, (g % 100) + 1, 'inner_' || g FROM generate_series(1, 5000) g;
ANALYZE vjoin_outer;
ANALYZE vjoin_inner;

-- Test 1: Basic inner join with vectorized hash join
SET pg_vectorjoin.enable = on;
SET pg_vectorjoin.enable_bnl = off;

EXPLAIN (COSTS OFF)
SELECT o.id, o.val, i.data
FROM vjoin_outer o
JOIN vjoin_inner i ON o.id = i.ref_id;

SELECT count(*) FROM vjoin_outer o JOIN vjoin_inner i ON o.id = i.ref_id;

-- Test 2: Block nested loop
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_bnl = on;

SELECT count(*) FROM vjoin_outer o JOIN vjoin_inner i ON o.id = i.ref_id;

-- Test 3: Correctness - results must match standard join
SET pg_vectorjoin.enable = off;
SELECT count(*) FROM vjoin_outer o JOIN vjoin_inner i ON o.id = i.ref_id;

-- Test 4: NULL handling
INSERT INTO vjoin_inner VALUES (5001, NULL, 'null_ref');
SET pg_vectorjoin.enable = on;
SET pg_vectorjoin.enable_hashjoin = on;

SELECT count(*) FROM vjoin_outer o JOIN vjoin_inner i ON o.id = i.ref_id;
-- Should be same as before — NULL ref_id should not match

-- Test 5: Empty relation
CREATE TABLE vjoin_empty (id int4);
SELECT count(*) FROM vjoin_empty e JOIN vjoin_inner i ON e.id = i.ref_id;

-- Test 6: int8 keys
CREATE TABLE vjoin_big_outer (id int8 PRIMARY KEY, val text);
CREATE TABLE vjoin_big_inner (id int8, ref_id int8, data text);

INSERT INTO vjoin_big_outer SELECT g, 'big_' || g FROM generate_series(1, 500) g;
INSERT INTO vjoin_big_inner SELECT g, (g % 50) + 1, 'big_inner_' || g FROM generate_series(1, 2000) g;
ANALYZE vjoin_big_outer;
ANALYZE vjoin_big_inner;

SELECT count(*) FROM vjoin_big_outer o JOIN vjoin_big_inner i ON o.id = i.ref_id;

-- GUC test
SET pg_vectorjoin.enable = off;
SHOW pg_vectorjoin.enable;
SET pg_vectorjoin.batch_size = 256;
SHOW pg_vectorjoin.batch_size;

-- Cleanup
DROP TABLE vjoin_outer, vjoin_inner, vjoin_empty, vjoin_big_outer, vjoin_big_inner;
DROP EXTENSION pg_vectorjoin;
