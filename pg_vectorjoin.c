#include "postgres.h"
#include "fmgr.h"
#include "optimizer/paths.h"
#include "utils/guc.h"
#include "vjoin_compat.h"
#include "pg_vectorjoin.h"
#include "vjoin_simd.h"

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
#if VJOIN_HAS_SETUP_HOOK
static join_path_setup_hook_type prev_join_setup_hook = NULL;
#endif

/* CustomScanMethods */
CustomScanMethods vjoin_hash_scan_methods = {
    .CustomName = "VectorHashJoin",
    .CreateCustomScanState = vjoin_hash_create_state,
};

CustomScanMethods vjoin_bnl_scan_methods = {
    .CustomName = "BlockNestLoop",
    .CreateCustomScanState = vjoin_bnl_create_state,
};

/* CustomPathMethods */
CustomPathMethods vjoin_hash_path_methods = {
    .CustomName = "VectorHashJoin",
    .PlanCustomPath = vjoin_hash_plan,
};

CustomPathMethods vjoin_bnl_path_methods = {
    .CustomName = "BlockNestLoop",
    .PlanCustomPath = vjoin_bnl_plan,
};

/* CustomExecMethods */
CustomExecMethods vjoin_hash_exec_methods = {
    .CustomName = "VectorHashJoin",
    .BeginCustomScan = vjoin_hash_begin,
    .ExecCustomScan = vjoin_hash_exec,
    .EndCustomScan = vjoin_hash_end,
    .ReScanCustomScan = vjoin_hash_rescan,
    .ExplainCustomScan = vjoin_hash_explain,
};

CustomExecMethods vjoin_bnl_exec_methods = {
    .CustomName = "BlockNestLoop",
    .BeginCustomScan = vjoin_bnl_begin,
    .ExecCustomScan = vjoin_bnl_exec,
    .EndCustomScan = vjoin_bnl_end,
    .ReScanCustomScan = vjoin_bnl_rescan,
    .ExplainCustomScan = vjoin_bnl_explain,
};

void
_PG_init(void)
{
    /* Define GUC parameters */
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

    /* Detect SIMD capabilities at load time */
    vjoin_detect_simd();

    /* Register CustomScanMethods */
    RegisterCustomScanMethods(&vjoin_hash_scan_methods);
    RegisterCustomScanMethods(&vjoin_bnl_scan_methods);
    
    /* Install join pathlist hook */
    prev_join_pathlist_hook = set_join_pathlist_hook;
    set_join_pathlist_hook = vjoin_pathlist_hook;

#if VJOIN_HAS_SETUP_HOOK
    prev_join_setup_hook = join_path_setup_hook;
    join_path_setup_hook = NULL;  /* future: vjoin_setup_hook */
#endif
}
