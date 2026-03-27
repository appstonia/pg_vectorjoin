CREATE EXTENSION IF NOT EXISTS pg_vectorjoin;

CREATE TEMP TABLE lj_outer(id int, payload text);
CREATE TEMP TABLE lj_inner(id int, note text);

INSERT INTO lj_outer VALUES (1,'a'),(2,'b'),(3,'c'),(NULL,'n');
INSERT INTO lj_inner VALUES (1,'x'),(4,'z'),(NULL,'skip');

ANALYZE lj_outer;
ANALYZE lj_inner;

SET pg_vectorjoin.enable = on;
SET pg_vectorjoin.enable_hashjoin = on;
SET pg_vectorjoin.enable_mergejoin = off;
SET pg_vectorjoin.enable_nestloop = off;
SET pg_vectorjoin.cost_factor = 0.01;
SET enable_hashjoin = on;
SET enable_mergejoin = off;
SET enable_nestloop = off;

-- Safe subset: project only outer columns, no anti-join filter.
EXPLAIN (COSTS OFF)
SELECT o.id, o.payload
FROM lj_outer o
LEFT JOIN lj_inner i ON o.id = i.id
ORDER BY o.id NULLS LAST, o.payload;

SELECT o.id, o.payload
FROM lj_outer o
LEFT JOIN lj_inner i ON o.id = i.id
ORDER BY o.id NULLS LAST, o.payload;

-- Anti-join pattern must still fall back to core PostgreSQL.
EXPLAIN (COSTS OFF)
SELECT o.id
FROM lj_outer o
LEFT JOIN lj_inner i ON o.id = i.id
WHERE i.id IS NULL
ORDER BY o.id NULLS LAST;

SELECT o.id
FROM lj_outer o
LEFT JOIN lj_inner i ON o.id = i.id
WHERE i.id IS NULL
ORDER BY o.id NULLS LAST;

-- Projecting nullable inner Vars must still fall back.
EXPLAIN (COSTS OFF)
SELECT o.id, i.note
FROM lj_outer o
LEFT JOIN lj_inner i ON o.id = i.id
ORDER BY o.id NULLS LAST, i.note NULLS LAST;

SELECT o.id, i.note
FROM lj_outer o
LEFT JOIN lj_inner i ON o.id = i.id
ORDER BY o.id NULLS LAST, i.note NULLS LAST;

-- FULL JOIN remains a core PostgreSQL plan.
EXPLAIN (COSTS OFF)
SELECT o.id, i.note
FROM lj_outer o
FULL JOIN lj_inner i ON o.id = i.id
ORDER BY o.id NULLS LAST, i.note NULLS LAST;

RESET ALL;

DROP EXTENSION pg_vectorjoin;
