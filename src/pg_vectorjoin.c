#include "postgres.h"
#include "fmgr.h"
#include "optimizer/paths.h"
#include "utils/guc.h"
#include "vjoin_compat.h"
#include "pg_vectorjoin.h"
#include "vjoin_simd.h"

PG_MODULE_MAGIC;

void _PG_init(void);

/* Dummy function to force library loading from CREATE EXTENSION */
PG_FUNCTION_INFO_V1(vjoin_loaded);
Datum
vjoin_loaded(PG_FUNCTION_ARGS)
{
    PG_RETURN_BOOL(true);
}

/* GUC variables */
bool    vjoin_enable = true;
bool    vjoin_enable_hashjoin = true;
bool    vjoin_enable_nestloop = true;
bool    vjoin_enable_mergejoin = true;
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

CustomScanMethods vjoin_nestloop_scan_methods = {
    .CustomName = "VectorNestedLoop",
    .CreateCustomScanState = vjoin_nestloop_create_state,
};

CustomScanMethods vjoin_merge_scan_methods = {
    .CustomName = "VectorMergeJoin",
    .CreateCustomScanState = vjoin_merge_create_state,
};

/* CustomPathMethods */
CustomPathMethods vjoin_hash_path_methods = {
    .CustomName = "VectorHashJoin",
    .PlanCustomPath = vjoin_hash_plan,
};

CustomPathMethods vjoin_nestloop_path_methods = {
    .CustomName = "VectorNestedLoop",
    .PlanCustomPath = vjoin_nestloop_plan,
};

CustomPathMethods vjoin_merge_path_methods = {
    .CustomName = "VectorMergeJoin",
    .PlanCustomPath = vjoin_merge_plan,
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

CustomExecMethods vjoin_nestloop_exec_methods = {
    .CustomName = "VectorNestedLoop",
    .BeginCustomScan = vjoin_nestloop_begin,
    .ExecCustomScan = vjoin_nestloop_exec,
    .EndCustomScan = vjoin_nestloop_end,
    .ReScanCustomScan = vjoin_nestloop_rescan,
    .ExplainCustomScan = vjoin_nestloop_explain,
};

CustomExecMethods vjoin_merge_exec_methods = {
    .CustomName = "VectorMergeJoin",
    .BeginCustomScan = vjoin_merge_begin,
    .ExecCustomScan = vjoin_merge_exec,
    .EndCustomScan = vjoin_merge_end,
    .ReScanCustomScan = vjoin_merge_rescan,
    .ExplainCustomScan = vjoin_merge_explain,
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

    DefineCustomBoolVariable("pg_vectorjoin.enable_nestloop",
                             "Enable block nested loop join.",
                             NULL,
                             &vjoin_enable_nestloop,
                             true,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

    DefineCustomBoolVariable("pg_vectorjoin.enable_mergejoin",
                             "Enable vectorized merge join.",
                             NULL,
                             &vjoin_enable_mergejoin,
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
    RegisterCustomScanMethods(&vjoin_nestloop_scan_methods);
    RegisterCustomScanMethods(&vjoin_merge_scan_methods);
    
    /* Install join pathlist hook */
    prev_join_pathlist_hook = set_join_pathlist_hook;
    set_join_pathlist_hook = vjoin_pathlist_hook;

#if VJOIN_HAS_SETUP_HOOK
    prev_join_setup_hook = join_path_setup_hook;
    join_path_setup_hook = NULL;  /* future: vjoin_setup_hook */
#endif
}
