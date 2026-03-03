/* pg_vectorjoin -- vectorized join extension */

-- Dummy function to force shared library loading on CREATE EXTENSION.
-- Hooks are registered in _PG_init() which runs when the .so is loaded.
CREATE FUNCTION vjoin_loaded() RETURNS bool
AS 'MODULE_PATHNAME'
LANGUAGE C STRICT;
