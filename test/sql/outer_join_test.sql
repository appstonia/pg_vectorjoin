-- Test outer joins across all strategies
-- Setup
CREATE TEMP TABLE tn1(id int, val text);
CREATE TEMP TABLE tn2(id int, val text);
INSERT INTO tn1 VALUES (1,'a'),(NULL,'b'),(3,'c');
INSERT INTO tn2 VALUES (1,'x'),(2,'y'),(NULL,'z');
ANALYZE tn1; ANALYZE tn2;

-- Expected FULL JOIN result (5 rows):
-- (1,a,1,x)       -- match
-- (NULL,b,NULL,NULL) -- left unmatched (NULL key)
-- (3,c,NULL,NULL)  -- left unmatched
-- (NULL,NULL,2,y)  -- right unmatched
-- (NULL,NULL,NULL,z) -- right unmatched (NULL key)

-- FULL JOIN via NL
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_mergejoin = off;
SET pg_vectorjoin.enable_nestloop = on;
SELECT 'NL' as s, tn1.id t1, tn1.val v1, tn2.id t2, tn2.val v2
FROM tn1 FULL JOIN tn2 ON tn1.id = tn2.id
ORDER BY coalesce(tn1.id, tn2.id) NULLS LAST, tn1.val NULLS LAST, tn2.val NULLS LAST;
RESET ALL;

-- FULL JOIN via Hash
SET pg_vectorjoin.enable_hashjoin = on;
SET pg_vectorjoin.enable_mergejoin = off;
SET pg_vectorjoin.enable_nestloop = off;
SELECT 'Hash' as s, tn1.id t1, tn1.val v1, tn2.id t2, tn2.val v2
FROM tn1 FULL JOIN tn2 ON tn1.id = tn2.id
ORDER BY coalesce(tn1.id, tn2.id) NULLS LAST, tn1.val NULLS LAST, tn2.val NULLS LAST;
RESET ALL;

-- FULL JOIN via Merge
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_mergejoin = on;
SET pg_vectorjoin.enable_nestloop = off;
SELECT 'Merge' as s, tn1.id t1, tn1.val v1, tn2.id t2, tn2.val v2
FROM tn1 FULL JOIN tn2 ON tn1.id = tn2.id
ORDER BY coalesce(tn1.id, tn2.id) NULLS LAST, tn1.val NULLS LAST, tn2.val NULLS LAST;
RESET ALL;

-- Reference: PG native
SET pg_vectorjoin.enable = off;
SELECT 'PG' as s, tn1.id t1, tn1.val v1, tn2.id t2, tn2.val v2
FROM tn1 FULL JOIN tn2 ON tn1.id = tn2.id
ORDER BY coalesce(tn1.id, tn2.id) NULLS LAST, tn1.val NULLS LAST, tn2.val NULLS LAST;
RESET ALL;

-- Test with duplicates
CREATE TEMP TABLE td1(id int, val text);
CREATE TEMP TABLE td2(id int, val text);
INSERT INTO td1 VALUES (1,'a'),(1,'b'),(2,'c');
INSERT INTO td2 VALUES (1,'x'),(1,'y'),(3,'z');
ANALYZE td1; ANALYZE td2;

-- FULL JOIN with dups via NL
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_mergejoin = off;
SELECT 'NL-dup' as s, td1.id t1, td1.val v1, td2.id t2, td2.val v2
FROM td1 FULL JOIN td2 ON td1.id = td2.id
ORDER BY coalesce(td1.id, td2.id), td1.val, td2.val;
RESET ALL;

-- Reference
SET pg_vectorjoin.enable = off;
SELECT 'PG-dup' as s, td1.id t1, td1.val v1, td2.id t2, td2.val v2
FROM td1 FULL JOIN td2 ON td1.id = td2.id
ORDER BY coalesce(td1.id, td2.id), td1.val, td2.val;
RESET ALL;
