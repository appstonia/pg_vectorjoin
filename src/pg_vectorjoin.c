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
bool    vjoin_enable = VJOIN_DEFAULT_ENABLE;
bool    vjoin_enable_hashjoin = VJOIN_DEFAULT_ENABLE_HASHJOIN;
bool    vjoin_enable_nestloop = VJOIN_DEFAULT_ENABLE_NESTLOOP;
bool    vjoin_enable_mergejoin = VJOIN_DEFAULT_ENABLE_MERGEJOIN;
int     vjoin_batch_size = VJOIN_DEFAULT_BATCH;
double  vjoin_cost_factor = VJOIN_DEFAULT_COST_FACTOR;
double  vjoin_hash_cost_factor = VJOIN_DEFAULT_HASH_COST_FACTOR;
double  vjoin_merge_cost_factor = VJOIN_DEFAULT_MERGE_COST_FACTOR;
double  vjoin_nestloop_cost_factor = VJOIN_DEFAULT_NESTLOOP_COST_FACTOR;
bool    vjoin_auto_tune = VJOIN_DEFAULT_AUTO_TUNE;
double  vjoin_auto_tune_margin = VJOIN_DEFAULT_AUTO_TUNE_MARGIN;
bool    vjoin_enable_hash_spill = VJOIN_DEFAULT_ENABLE_HASH_SPILL;
int     vjoin_hash_max_batches = VJOIN_DEFAULT_HASH_MAX_BATCHES;

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
    .EstimateDSMCustomScan = vjoin_hash_estimate_dsm,
    .InitializeDSMCustomScan = vjoin_hash_initialize_dsm,
    .ReInitializeDSMCustomScan = vjoin_hash_reinitialize_dsm,
    .InitializeWorkerCustomScan = vjoin_hash_initialize_worker,
    .ShutdownCustomScan = vjoin_hash_shutdown,
    .ExplainCustomScan = vjoin_hash_explain,
};

CustomExecMethods vjoin_nestloop_exec_methods = {
    .CustomName = "VectorNestedLoop",
    .BeginCustomScan = vjoin_nestloop_begin,
    .ExecCustomScan = vjoin_nestloop_exec,
    .EndCustomScan = vjoin_nestloop_end,
    .ReScanCustomScan = vjoin_nestloop_rescan,
    .EstimateDSMCustomScan = vjoin_nestloop_estimate_dsm,
    .InitializeDSMCustomScan = vjoin_nestloop_initialize_dsm,
    .ReInitializeDSMCustomScan = vjoin_nestloop_reinitialize_dsm,
    .InitializeWorkerCustomScan = vjoin_nestloop_initialize_worker,
    .ShutdownCustomScan = vjoin_nestloop_shutdown,
    .ExplainCustomScan = vjoin_nestloop_explain,
};

CustomExecMethods vjoin_merge_exec_methods = {
    .CustomName = "VectorMergeJoin",
    .BeginCustomScan = vjoin_merge_begin,
    .ExecCustomScan = vjoin_merge_exec,
    .EndCustomScan = vjoin_merge_end,
    .ReScanCustomScan = vjoin_merge_rescan,
    .EstimateDSMCustomScan = vjoin_merge_estimate_dsm,
    .InitializeDSMCustomScan = vjoin_merge_initialize_dsm,
    .ReInitializeDSMCustomScan = vjoin_merge_reinitialize_dsm,
    .InitializeWorkerCustomScan = vjoin_merge_initialize_worker,
    .ShutdownCustomScan = vjoin_merge_shutdown,
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
                             VJOIN_DEFAULT_ENABLE,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

    DefineCustomBoolVariable("pg_vectorjoin.enable_hashjoin",
                             "Enable vectorized hash join.",
                             NULL,
                             &vjoin_enable_hashjoin,
                             VJOIN_DEFAULT_ENABLE_HASHJOIN,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

    DefineCustomBoolVariable("pg_vectorjoin.enable_nestloop",
                             "Enable block nested loop join.",
                             NULL,
                             &vjoin_enable_nestloop,
                             VJOIN_DEFAULT_ENABLE_NESTLOOP,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

    DefineCustomBoolVariable("pg_vectorjoin.enable_mergejoin",
                             "Enable vectorized merge join.",
                             NULL,
                             &vjoin_enable_mergejoin,
                             VJOIN_DEFAULT_ENABLE_MERGEJOIN,
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
                             "Global cost scaling for vectorized join (lower = more aggressive). Stacks multiplicatively with per-jointype factors.",
                             NULL,
                             &vjoin_cost_factor,
                             VJOIN_DEFAULT_COST_FACTOR,
                             VJOIN_MIN_COST_FACTOR,
                             VJOIN_MAX_COST_FACTOR,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

    DefineCustomRealVariable("pg_vectorjoin.hash_cost_factor",
                             "Per-jointype cost scaling for vectorized hash join. Multiplied with cost_factor.",
                             NULL,
                             &vjoin_hash_cost_factor,
                             VJOIN_DEFAULT_HASH_COST_FACTOR,
                             VJOIN_MIN_COST_FACTOR,
                             VJOIN_MAX_COST_FACTOR,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

    DefineCustomRealVariable("pg_vectorjoin.merge_cost_factor",
                             "Per-jointype cost scaling for vectorized merge join. Multiplied with cost_factor.",
                             NULL,
                             &vjoin_merge_cost_factor,
                             VJOIN_DEFAULT_MERGE_COST_FACTOR,
                             VJOIN_MIN_COST_FACTOR,
                             VJOIN_MAX_COST_FACTOR,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

    DefineCustomRealVariable("pg_vectorjoin.nestloop_cost_factor",
                             "Per-jointype cost scaling for vectorized nested loop join. Multiplied with cost_factor.",
                             NULL,
                             &vjoin_nestloop_cost_factor,
                             VJOIN_DEFAULT_NESTLOOP_COST_FACTOR,
                             VJOIN_MIN_COST_FACTOR,
                             VJOIN_MAX_COST_FACTOR,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

    DefineCustomBoolVariable("pg_vectorjoin.auto_tune",
                             "Clamp each vector join path's cost down to (cheapest comparable native path) * auto_tune_margin when the native path is cheaper. Lets vjoin be selected without manually tuning per-jointype cost factors.",
                             NULL,
                             &vjoin_auto_tune,
                             VJOIN_DEFAULT_AUTO_TUNE,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

    DefineCustomRealVariable("pg_vectorjoin.auto_tune_margin",
                             "Target ratio of vector join cost to cheapest comparable native path cost when auto_tune is on (e.g. 0.95 = always 5%% cheaper than native of same type).",
                             NULL,
                             &vjoin_auto_tune_margin,
                             VJOIN_DEFAULT_AUTO_TUNE_MARGIN,
                             VJOIN_MIN_AUTO_TUNE_MARGIN,
                             VJOIN_MAX_AUTO_TUNE_MARGIN,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

    DefineCustomBoolVariable("pg_vectorjoin.enable_hash_spill",
                             "Allow vectorized hash join to spill oversized inner relations to disk in multiple batches (PostgreSQL-style multi-batch hash join). When off, joins whose inner does not fit in work_mem fall back to native join.",
                             NULL,
                             &vjoin_enable_hash_spill,
                             VJOIN_DEFAULT_ENABLE_HASH_SPILL,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

    DefineCustomIntVariable("pg_vectorjoin.hash_max_batches",
                            "Upper bound on the number of hash-join batches (must be a power of two; values are rounded up). Guards against pathological recursive splits when row estimates are wildly off.",
                            NULL,
                            &vjoin_hash_max_batches,
                            VJOIN_DEFAULT_HASH_MAX_BATCHES,
                            VJOIN_MIN_HASH_MAX_BATCHES,
                            VJOIN_MAX_HASH_MAX_BATCHES,
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
