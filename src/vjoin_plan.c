#include "postgres.h"
#include "nodes/makefuncs.h"
#include "nodes/value.h"
#include "optimizer/restrictinfo.h"
#include "pg_vectorjoin.h"
#include "vjoin_state.h"

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
    cscan->scan.plan.qual = NIL;       /* quals applied in ExecCustomScan */
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
    cscan->custom_exprs = NIL;
    cscan->custom_private = best_path->custom_private;
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
