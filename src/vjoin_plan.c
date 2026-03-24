#include "postgres.h"
#include "nodes/makefuncs.h"
#include "nodes/value.h"
#include "optimizer/restrictinfo.h"
#include "pg_vectorjoin.h"
#include "vjoin_state.h"

/*
 * Deserialize key info from custom_private.
 * If hash_funcs is NULL, the hash_proc field is skipped.
 * Returns the list index after the last consumed element, so callers
 * with extra trailing fields (e.g. theta info) can continue reading.
 */
int
vjoin_deserialize_keys(List *private_data,
                       JoinType *jointype,
                       int *num_keys,
                       AttrNumber *outer_keynos,
                       AttrNumber *inner_keynos,
                       Oid *key_types,
                       Oid *hash_funcs,
                       Oid *eq_funcs,
                       Oid *key_collations)
{
    int idx = 0;
    int i;

    *jointype = (JoinType) intVal(list_nth(private_data, idx++));
    *num_keys = intVal(list_nth(private_data, idx++));
    for (i = 0; i < *num_keys; i++)
    {
        outer_keynos[i] = (AttrNumber) intVal(list_nth(private_data, idx++));
        inner_keynos[i] = (AttrNumber) intVal(list_nth(private_data, idx++));
        key_types[i] = (Oid) intVal(list_nth(private_data, idx++));
        if (hash_funcs)
            hash_funcs[i] = (Oid) intVal(list_nth(private_data, idx++));
        else
            idx++;  /* skip hash_proc */
        eq_funcs[i] = (Oid) intVal(list_nth(private_data, idx++));
        key_collations[i] = (Oid) intVal(list_nth(private_data, idx++));
    }
    return idx;
}

/*
 * Build custom_scan_tlist from outer + inner child plan targetlists.
 * This defines the "wide" tuple that our scan slot will hold:
 * [all outer columns] ++ [all inner columns].
 */
static List *
vjoin_build_scan_tlist(List *outer_plan_tlist, List *inner_plan_tlist)
{
    List       *tlist = NIL;
    ListCell   *lc;
    int         resno = 1;

    foreach(lc, outer_plan_tlist)
    {
        TargetEntry *tle = lfirst_node(TargetEntry, lc);
        TargetEntry *new_tle = makeTargetEntry(
            (Expr *) copyObject(tle->expr),
            resno++, tle->resname, false);
        tlist = lappend(tlist, new_tle);
    }

    foreach(lc, inner_plan_tlist)
    {
        TargetEntry *tle = lfirst_node(TargetEntry, lc);
        TargetEntry *new_tle = makeTargetEntry(
            (Expr *) copyObject(tle->expr),
            resno++, tle->resname, false);
        tlist = lappend(tlist, new_tle);
    }

    return tlist;
}

/* PlanCustomPath for VectorHashJoin */
Plan *
vjoin_hash_plan(PlannerInfo *root, RelOptInfo *rel,
                CustomPath *best_path, List *tlist,
                List *clauses, List *custom_plans)
{
    CustomScan *cscan = makeNode(CustomScan);
    Plan       *outer_plan = (Plan *) linitial(custom_plans);
    Plan       *inner_plan = (Plan *) lsecond(custom_plans);

    cscan->scan.plan.targetlist = tlist;
    cscan->custom_scan_tlist = vjoin_build_scan_tlist(
        outer_plan->targetlist, inner_plan->targetlist);
    cscan->scan.plan.qual = NIL;
    cscan->scan.scanrelid = 0;         /* not a real relation scan */
    cscan->flags = best_path->flags;
    cscan->custom_plans = custom_plans;
    cscan->custom_exprs = NIL;
    cscan->custom_private = best_path->custom_private;
    cscan->custom_relids = rel->relids;
    cscan->methods = &vjoin_hash_scan_methods;

    return &cscan->scan.plan;
}

/* PlanCustomPath for NestLoop */
Plan *
vjoin_nestloop_plan(PlannerInfo *root, RelOptInfo *rel,
               CustomPath *best_path, List *tlist,
               List *clauses, List *custom_plans)
{
    CustomScan *cscan = makeNode(CustomScan);
    Plan       *outer_plan = (Plan *) linitial(custom_plans);
    Plan       *inner_plan = (Plan *) lsecond(custom_plans);

    cscan->scan.plan.targetlist = tlist;
    cscan->custom_scan_tlist = vjoin_build_scan_tlist(
        outer_plan->targetlist, inner_plan->targetlist);
    cscan->scan.plan.qual = NIL;
    cscan->scan.scanrelid = 0;
    cscan->flags = best_path->flags;
    cscan->custom_plans = custom_plans;
    /* Extract join qual expressions from custom_private (last element) */
    {
        List *full = best_path->custom_private;
        int   len = list_length(full);
        cscan->custom_exprs = (List *) llast(full);
        cscan->custom_private = list_truncate(list_copy(full), len - 1);
    }
    cscan->custom_relids = rel->relids;
    cscan->methods = &vjoin_nestloop_scan_methods;

    return &cscan->scan.plan;
}

/* CreateCustomScanState for VectorHashJoin */
Node *
vjoin_hash_create_state(CustomScan *cscan)
{
    VectorHashJoinState *state = (VectorHashJoinState *)
        newNode(sizeof(VectorHashJoinState), T_CustomScanState);
    state->css.methods = &vjoin_hash_exec_methods;
    return (Node *) state;
}

/* CreateCustomScanState for NestLoop */
Node *
vjoin_nestloop_create_state(CustomScan *cscan)
{
    VJoinNestLoopState *state = (VJoinNestLoopState *)
        newNode(sizeof(VJoinNestLoopState), T_CustomScanState);
    state->css.methods = &vjoin_nestloop_exec_methods;
    return (Node *) state;
}

/* PlanCustomPath for VectorMergeJoin */
Plan *
vjoin_merge_plan(PlannerInfo *root, RelOptInfo *rel,
                 CustomPath *best_path, List *tlist,
                 List *clauses, List *custom_plans)
{
    CustomScan *cscan = makeNode(CustomScan);
    Plan       *outer_plan = (Plan *) linitial(custom_plans);
    Plan       *inner_plan = (Plan *) lsecond(custom_plans);

    cscan->scan.plan.targetlist = tlist;
    cscan->custom_scan_tlist = vjoin_build_scan_tlist(
        outer_plan->targetlist, inner_plan->targetlist);
    cscan->scan.plan.qual = NIL;
    cscan->scan.scanrelid = 0;
    cscan->flags = best_path->flags;
    cscan->custom_plans = custom_plans;
    cscan->custom_exprs = NIL;
    cscan->custom_private = best_path->custom_private;
    cscan->custom_relids = rel->relids;
    cscan->methods = &vjoin_merge_scan_methods;

    return &cscan->scan.plan;
}

/* CreateCustomScanState for VectorMergeJoin */
Node *
vjoin_merge_create_state(CustomScan *cscan)
{
    VectorMergeJoinState *state = (VectorMergeJoinState *)
        newNode(sizeof(VectorMergeJoinState), T_CustomScanState);
    state->css.methods = &vjoin_merge_exec_methods;
    return (Node *) state;
}
