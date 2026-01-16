#ifndef PG_VECTORJOIN_H
#define PG_VECTORJOIN_H

#include "postgres.h"
#include "nodes/extensible.h"
#include "optimizer/paths.h"

/* Constants */
#define VJOIN_MAX_KEYS          8
#define VJOIN_DEFAULT_BATCH     1024
#define VJOIN_MIN_BATCH         64
#define VJOIN_MAX_BATCH         8192
#define VJOIN_HT_LOAD_FACTOR   2       /* capacity = inner_rows * factor */

/* GUC variables */
extern bool vjoin_enable;
extern bool vjoin_enable_hashjoin;
extern bool vjoin_enable_bnl;
extern int  vjoin_batch_size;
extern double vjoin_cost_factor;

/* Saved previous hooks (needed across translation units) */
extern set_join_pathlist_hook_type prev_join_pathlist_hook;

/* CustomScanMethods for registration */
extern CustomScanMethods vjoin_hash_scan_methods;
extern CustomScanMethods vjoin_bnl_scan_methods;

/* CustomPathMethods */
extern CustomPathMethods vjoin_hash_path_methods;
extern CustomPathMethods vjoin_bnl_path_methods;

/* CustomExecMethods */
extern CustomExecMethods vjoin_hash_exec_methods;
extern CustomExecMethods vjoin_bnl_exec_methods;

/* Path generation hook */
void vjoin_pathlist_hook(PlannerInfo *root,
                         RelOptInfo *joinrel,
                         RelOptInfo *outerrel,
                         RelOptInfo *innerrel,
                         JoinType jointype,
                         JoinPathExtraData *extra);

/* PlanCustomPath callbacks */
Plan *vjoin_hash_plan(PlannerInfo *root, RelOptInfo *rel,
                      CustomPath *best_path, List *tlist,
                      List *clauses, List *custom_plans);
Plan *vjoin_bnl_plan(PlannerInfo *root, RelOptInfo *rel,
                     CustomPath *best_path, List *tlist,
                     List *clauses, List *custom_plans);

/* CreateCustomScanState callbacks */
Node *vjoin_hash_create_state(CustomScan *cscan);
Node *vjoin_bnl_create_state(CustomScan *cscan);

/* Hash join executor callbacks */
void vjoin_hash_begin(CustomScanState *node, EState *estate, int eflags);
TupleTableSlot *vjoin_hash_exec(CustomScanState *node);
void vjoin_hash_end(CustomScanState *node);
void vjoin_hash_rescan(CustomScanState *node);
void vjoin_hash_explain(CustomScanState *node, List *ancestors, ExplainState *es);

/* BNL executor callbacks */
void vjoin_bnl_begin(CustomScanState *node, EState *estate, int eflags);
TupleTableSlot *vjoin_bnl_exec(CustomScanState *node);
void vjoin_bnl_end(CustomScanState *node);
void vjoin_bnl_rescan(CustomScanState *node);
void vjoin_bnl_explain(CustomScanState *node, List *ancestors, ExplainState *es);

#endif /* PG_VECTORJOIN_H */
