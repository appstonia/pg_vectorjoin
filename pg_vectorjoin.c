#include "postgres.h"
#include "fmgr.h"
#include "optimizer/paths.h"
#include "utils/guc.h"
#include "vjoin_compat.h"
#include "pg_vectorjoin.h"

PG_MODULE_MAGIC;

void _PG_init(void);

/* GUC variables */
bool    vjoin_enable = true;
bool    vjoin_enable_hashjoin = true;
bool    vjoin_enable_bnl = true;
int     vjoin_batch_size = VJOIN_DEFAULT_BATCH;
double  vjoin_cost_factor = 0.5;

/* Saved previous hooks */
set_join_pathlist_hook_type prev_join_pathlist_hook = NULL;

void
_PG_init(void)
{
    DefineCustomBoolVariable("pg_vectorjoin.enable",
                             "Enable vectorized join optimization.",
                             NULL,
                             &vjoin_enable,
                             true,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

    DefineCustomBoolVariable("pg_vectorjoin.enable_hashjoin",
                             "Enable vectorized hash join.",
                             NULL,
                             &vjoin_enable_hashjoin,
                             true,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

    DefineCustomBoolVariable("pg_vectorjoin.enable_bnl",
                             "Enable block nested loop join.",
                             NULL,
                             &vjoin_enable_bnl,
                             true,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

    DefineCustomIntVariable("pg_vectorjoin.batch_size",
                            "Batch/block size for vectorized processing.",
                            NULL,
                            &vjoin_batch_size,
                            VJOIN_DEFAULT_BATCH,
                            VJOIN_MIN_BATCH,
                            VJOIN_MAX_BATCH,
                            PGC_USERSET,
                            0, NULL, NULL, NULL);

    DefineCustomRealVariable("pg_vectorjoin.cost_factor",
                             "Cost scaling for vectorized join (lower = more aggressive).",
                             NULL,
                             &vjoin_cost_factor,
                             0.5,
                             0.01,
                             10.0,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

    MarkGUCPrefixReserved("pg_vectorjoin");

    /* Install join pathlist hook */
    prev_join_pathlist_hook = set_join_pathlist_hook;
    set_join_pathlist_hook = vjoin_pathlist_hook;
}
