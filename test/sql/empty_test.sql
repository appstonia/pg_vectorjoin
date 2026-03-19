CREATE TEMP TABLE e1(id int, val text);
CREATE TEMP TABLE e2(id int, val text);
INSERT INTO e1 VALUES (1,'a'),(2,'b');
ANALYZE e1; ANALYZE e2;

-- FULL JOIN with empty right (NL)
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_mergejoin = off;
SELECT 'NL-empty-r' as s, e1.id t1, e1.val v1, e2.id t2, e2.val v2
FROM e1 FULL JOIN e2 ON e1.id = e2.id ORDER BY coalesce(e1.id, e2.id);
RESET ALL;

-- FULL JOIN empty left (NL) 
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_mergejoin = off;
SELECT 'NL-empty-l' as s, e2.id t1, e2.val v1, e1.id t2, e1.val v2
FROM e2 FULL JOIN e1 ON e2.id = e1.id ORDER BY coalesce(e2.id, e1.id);
RESET ALL;

-- LEFT JOIN empty right (NL)
SET pg_vectorjoin.enable_hashjoin = off;
SET pg_vectorjoin.enable_mergejoin = off;
SELECT 'NL-left-er' as s, e1.id t1, e1.val v1, e2.id t2, e2.val v2
FROM e1 LEFT JOIN e2 ON e1.id = e2.id ORDER BY e1.id;
RESET ALL;
