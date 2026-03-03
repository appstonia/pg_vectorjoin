#include "postgres.h"
#include "access/stratnum.h"
#include "catalog/pg_type.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "nodes/value.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/restrictinfo.h"
#include "optimizer/tlist.h"
#include "utils/lsyscache.h"
#include "vjoin_compat.h"
#include "pg_vectorjoin.h"

/* Supported key types for SIMD fast path */
static bool
vjoin_is_supported_type(Oid typid)
{
    return typid == INT4OID || typid == INT8OID || typid == FLOAT8OID;
}

/*
 * Analyze join clauses: find equijoin clauses with supported types.
 * Returns the number of usable keys found.
 */
static int
vjoin_analyze_clauses(List *restrictlist,
                      RelOptInfo *outerrel,
                      RelOptInfo *innerrel,
                      AttrNumber *outer_keynos,
                      AttrNumber *inner_keynos,
                      Oid *key_types,
                      List *outer_tlist,
                      List *inner_tlist)
{
    ListCell   *lc;
    int         nkeys = 0;

    foreach(lc, restrictlist)
    {
        RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);
        OpExpr     *opexpr;
        Node       *left,
                   *right;
        Oid         keytype;
        TargetEntry *outer_tle,
                    *inner_tle;

        if (nkeys >= VJOIN_MAX_KEYS)
            break;

        /* Must be a hash-joinable equality operator */
        if (rinfo->hashjoinoperator == InvalidOid)
            continue;

        if (!IsA(rinfo->clause, OpExpr))
            continue;

        opexpr = (OpExpr *) rinfo->clause;
        if (list_length(opexpr->args) != 2)
            continue;

        left = (Node *) linitial(opexpr->args);
        right = (Node *) lsecond(opexpr->args);

        /* Determine which side is outer vs inner */
        if (!IsA(left, Var) || !IsA(right, Var))
            continue;

        keytype = exprType(left);
        if (!vjoin_is_supported_type(keytype))
            continue;

        /* Match Var to outer/inner relation */
        if (bms_is_subset(rinfo->left_relids, outerrel->relids) &&
            bms_is_subset(rinfo->right_relids, innerrel->relids))
        {
            outer_tle = tlist_member((Expr *) left, outer_tlist);
            inner_tle = tlist_member((Expr *) right, inner_tlist);
        }
        else if (bms_is_subset(rinfo->right_relids, outerrel->relids) &&
                 bms_is_subset(rinfo->left_relids, innerrel->relids))
        {
            outer_tle = tlist_member((Expr *) right, outer_tlist);
            inner_tle = tlist_member((Expr *) left, inner_tlist);
        }
        else
            continue;

        if (outer_tle == NULL || inner_tle == NULL)
            continue;

        outer_keynos[nkeys] = outer_tle->resno;
        inner_keynos[nkeys] = inner_tle->resno;
        key_types[nkeys] = keytype;
        nkeys++;
    }

    return nkeys;
}

/*
 * Build custom_private list for serialization into the plan.
 * Format: [num_keys, outer_keyno1, inner_keyno1, keytype1, ...]
 */
static List *
vjoin_build_private(int num_keys, AttrNumber *outer_keynos,
                    AttrNumber *inner_keynos, Oid *key_types)
{
    List *result = NIL;
    int   i;

    result = lappend(result, makeInteger(num_keys));
    for (i = 0; i < num_keys; i++)
    {
        result = lappend(result, makeInteger((int) outer_keynos[i]));
        result = lappend(result, makeInteger((int) inner_keynos[i]));
        result = lappend(result, makeInteger((int) key_types[i]));
    }
    return result;
}

/*
 * Try to create VectorHashJoin paths (non-parallel + parallel).
 */
static void
vjoin_try_hashjoin(PlannerInfo *root,
                   RelOptInfo *joinrel,
                   RelOptInfo *outerrel,
                   RelOptInfo *innerrel,
                   JoinPathExtraData *extra,
                   int nkeys,
                   AttrNumber *outer_keynos,
                   AttrNumber *inner_keynos,
                   Oid *key_types)
{
    Path       *outer_path = outerrel->cheapest_total_path;
    Path       *inner_path = innerrel->cheapest_total_path;
    CustomPath *cpath;
    Cost        startup_cost,
                run_cost;
    double      outer_rows,
                inner_rows;

    /* --- Non-parallel path --- */
    if (outer_path != NULL && inner_path != NULL)
    {
        outer_rows = outer_path->rows;
        inner_rows = inner_path->rows;

        startup_cost = inner_path->total_cost +
                       inner_rows * cpu_operator_cost * 2.0;
        run_cost = outer_path->total_cost +
                   outer_rows * cpu_operator_cost * 2.0 * vjoin_cost_factor;

        cpath = makeNode(CustomPath);
        cpath->path.pathtype = T_CustomScan;
        cpath->path.parent = joinrel;
        cpath->path.pathtarget = joinrel->reltarget;
        cpath->path.param_info = NULL;
        cpath->path.parallel_aware = false;
        cpath->path.parallel_safe = outer_path->parallel_safe &&
                                    inner_path->parallel_safe;
        cpath->path.parallel_workers = 0;
        cpath->path.rows = joinrel->rows;
        cpath->path.startup_cost = startup_cost;
        cpath->path.total_cost = startup_cost + run_cost;
        cpath->path.pathkeys = NIL;
        cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
        cpath->custom_paths = list_make2(outer_path, inner_path);
#if VJOIN_HAS_CUSTOM_RESTRICTINFO
        cpath->custom_restrictinfo = extra->restrictlist;
#endif
        cpath->custom_private = vjoin_build_private(nkeys, outer_keynos,
                                                    inner_keynos, key_types);
        cpath->methods = &vjoin_hash_path_methods;

        add_path(joinrel, &cpath->path);
    }

    /* --- Parallel path --- */
    if (joinrel->consider_parallel && outerrel->partial_pathlist != NIL)
    {
        Path *par_outer = (Path *) linitial(outerrel->partial_pathlist);
        Path *par_inner = NULL;
        int   parallel_workers;
        ListCell *lc2;

        /* Find cheapest non-parameterized parallel-safe inner path */
        foreach(lc2, innerrel->pathlist)
        {
            Path *p = (Path *) lfirst(lc2);
            if (p->parallel_safe && p->param_info == NULL &&
                (par_inner == NULL || p->total_cost < par_inner->total_cost))
                par_inner = p;
        }

        if (par_outer != NULL && par_inner != NULL)
        {
            parallel_workers = par_outer->parallel_workers;
            if (parallel_workers <= 0)
                parallel_workers = 1;

            outer_rows = par_outer->rows;
            inner_rows = par_inner->rows;

            startup_cost = par_inner->total_cost +
                           inner_rows * cpu_operator_cost * 2.0;
            run_cost = par_outer->total_cost +
                       outer_rows * cpu_operator_cost * 2.0 * vjoin_cost_factor;

            cpath = makeNode(CustomPath);
            cpath->path.pathtype = T_CustomScan;
            cpath->path.parent = joinrel;
            cpath->path.pathtarget = joinrel->reltarget;
            cpath->path.param_info = NULL;
            cpath->path.parallel_aware = true;
            cpath->path.parallel_safe = true;
            cpath->path.parallel_workers = parallel_workers;
            cpath->path.rows = clamp_row_est(joinrel->rows / parallel_workers);
            cpath->path.startup_cost = startup_cost;
            cpath->path.total_cost = startup_cost + run_cost;
            cpath->path.pathkeys = NIL;
            cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
            cpath->custom_paths = list_make2(par_outer, par_inner);
#if VJOIN_HAS_CUSTOM_RESTRICTINFO
            cpath->custom_restrictinfo = extra->restrictlist;
#endif
            cpath->custom_private = vjoin_build_private(nkeys, outer_keynos,
                                                        inner_keynos, key_types);
            cpath->methods = &vjoin_hash_path_methods;

            add_partial_path(joinrel, &cpath->path);
        }
    }
}

/*
 * Try to create VectorNestedLoop paths (non-parallel + parallel).
 */
static void
vjoin_try_nestloop(PlannerInfo *root,
              RelOptInfo *joinrel,
              RelOptInfo *outerrel,
              RelOptInfo *innerrel,
              JoinPathExtraData *extra,
              int nkeys,
              AttrNumber *outer_keynos,
              AttrNumber *inner_keynos,
              Oid *key_types)
{
    Path       *outer_path = outerrel->cheapest_total_path;
    Path       *inner_path = innerrel->cheapest_total_path;
    CustomPath *cpath;
    Cost        startup_cost,
                run_cost;
    double      outer_rows,
                inner_rows,
                num_blocks;
    int         simd_width = 4;

    /* --- Non-parallel path --- */
    if (outer_path != NULL && inner_path != NULL)
    {
        outer_rows = outer_path->rows;
        inner_rows = inner_path->rows;
        num_blocks = ceil(outer_rows / vjoin_batch_size);

        startup_cost = outer_path->startup_cost;
        run_cost = outer_path->total_cost - outer_path->startup_cost +
                   num_blocks * inner_path->total_cost +
                   outer_rows * inner_rows * cpu_operator_cost *
                   vjoin_cost_factor / simd_width;

        /* NL is typically only competitive for smaller inner relations */
        if (!(inner_rows > 10000 && nkeys > 0))
        {
            cpath = makeNode(CustomPath);
            cpath->path.pathtype = T_CustomScan;
            cpath->path.parent = joinrel;
            cpath->path.pathtarget = joinrel->reltarget;
            cpath->path.param_info = NULL;
            cpath->path.parallel_aware = false;
            cpath->path.parallel_safe = outer_path->parallel_safe &&
                                        inner_path->parallel_safe;
            cpath->path.parallel_workers = 0;
            cpath->path.rows = joinrel->rows;
            cpath->path.startup_cost = startup_cost;
            cpath->path.total_cost = startup_cost + run_cost;
            cpath->path.pathkeys = NIL;
            cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
            cpath->custom_paths = list_make2(outer_path, inner_path);
#if VJOIN_HAS_CUSTOM_RESTRICTINFO
            cpath->custom_restrictinfo = extra->restrictlist;
#endif
            cpath->custom_private = vjoin_build_private(nkeys, outer_keynos,
                                                        inner_keynos, key_types);
            cpath->methods = &vjoin_nestloop_path_methods;

            add_path(joinrel, &cpath->path);
        }
    }

    /* --- Parallel path --- */
    if (joinrel->consider_parallel && outerrel->partial_pathlist != NIL)
    {
        Path *par_outer = (Path *) linitial(outerrel->partial_pathlist);
        int   parallel_workers;

        if (par_outer != NULL && inner_path != NULL)
        {
            outer_rows = par_outer->rows;
            inner_rows = inner_path->rows;

            if (!(inner_rows > 10000 && nkeys > 0))
            {
                num_blocks = ceil(outer_rows / vjoin_batch_size);
                parallel_workers = par_outer->parallel_workers;
                if (parallel_workers <= 0)
                    parallel_workers = 1;

                startup_cost = par_outer->startup_cost;
                run_cost = par_outer->total_cost - par_outer->startup_cost +
                           num_blocks * inner_path->total_cost +
                           outer_rows * inner_rows * cpu_operator_cost *
                           vjoin_cost_factor / simd_width;

                cpath = makeNode(CustomPath);
                cpath->path.pathtype = T_CustomScan;
                cpath->path.parent = joinrel;
                cpath->path.pathtarget = joinrel->reltarget;
                cpath->path.param_info = NULL;
                cpath->path.parallel_aware = true;
                cpath->path.parallel_safe = true;
                cpath->path.parallel_workers = parallel_workers;
                cpath->path.rows = clamp_row_est(joinrel->rows / parallel_workers);
                cpath->path.startup_cost = startup_cost;
                cpath->path.total_cost = startup_cost + run_cost;
                cpath->path.pathkeys = NIL;
                cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
                cpath->custom_paths = list_make2(par_outer, inner_path);
#if VJOIN_HAS_CUSTOM_RESTRICTINFO
                cpath->custom_restrictinfo = extra->restrictlist;
#endif
                cpath->custom_private = vjoin_build_private(nkeys, outer_keynos,
                                                            inner_keynos, key_types);
                cpath->methods = &vjoin_nestloop_path_methods;

                add_partial_path(joinrel, &cpath->path);
            }
        }
    }
}

/*
 * Try to create a VectorMergeJoin path.
 * Only supports single-key equijoin for v1.
 * If inputs aren't already sorted, Sort nodes are added.
 */
static void
vjoin_try_mergejoin(PlannerInfo *root,
                    RelOptInfo *joinrel,
                    RelOptInfo *outerrel,
                    RelOptInfo *innerrel,
                    JoinPathExtraData *extra,
                    int nkeys,
                    AttrNumber *outer_keynos,
                    AttrNumber *inner_keynos,
                    Oid *key_types)
{
    ListCell   *lc;
    RestrictInfo *merge_rinfo = NULL;
    Oid         opfamily;
    PathKey    *outer_pathkey;
    PathKey    *inner_pathkey;
    List       *outer_pathkeys;
    List       *inner_pathkeys;
    Path       *outer_path;
    Path       *inner_path;
    CustomPath *cpath;
    Cost        startup_cost,
                run_cost;
    double      outer_rows,
                inner_rows;

    /* v1: single-key merge join only */
    if (nkeys != 1)
        return;

    /*
     * Find a merge-joinable RestrictInfo for our single key.
     * We need one with mergeopfamilies set.
     */
    foreach(lc, extra->restrictlist)
    {
        RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);

        if (rinfo->mergeopfamilies == NIL)
            continue;
        if (!IsA(rinfo->clause, OpExpr))
            continue;

        /* Make sure eclasses are initialized */
        if (rinfo->left_ec == NULL)
            initialize_mergeclause_eclasses(root, rinfo);

        if (rinfo->left_ec == NULL || rinfo->right_ec == NULL)
            continue;

        merge_rinfo = rinfo;
        break;
    }

    if (merge_rinfo == NULL)
        return;

    /* Use first opfamily from the merge clause */
    opfamily = linitial_oid(merge_rinfo->mergeopfamilies);

    /*
     * Build PathKeys for the merge key on each side.
     * We need ascending sort order (BTLessStrategyNumber).
     */
    if (bms_is_subset(merge_rinfo->left_relids, outerrel->relids))
    {
        outer_pathkey = make_canonical_pathkey(root, merge_rinfo->left_ec,
                                              opfamily, BTLessStrategyNumber,
                                              false);
        inner_pathkey = make_canonical_pathkey(root, merge_rinfo->right_ec,
                                              opfamily, BTLessStrategyNumber,
                                              false);
    }
    else
    {
        outer_pathkey = make_canonical_pathkey(root, merge_rinfo->right_ec,
                                              opfamily, BTLessStrategyNumber,
                                              false);
        inner_pathkey = make_canonical_pathkey(root, merge_rinfo->left_ec,
                                              opfamily, BTLessStrategyNumber,
                                              false);
    }

    outer_pathkeys = list_make1(outer_pathkey);
    inner_pathkeys = list_make1(inner_pathkey);

    /*
     * Try to find already-sorted paths; fall back to adding Sort nodes.
     */
    outer_path = get_cheapest_path_for_pathkeys(outerrel->pathlist,
                                                outer_pathkeys,
                                                NULL,
                                                TOTAL_COST,
                                                false);
    if (outer_path == NULL)
        outer_path = (Path *) create_sort_path(root, outerrel,
                                               outerrel->cheapest_total_path,
                                               outer_pathkeys, -1.0);

    inner_path = NULL;
    {
        ListCell *lc2;
        foreach(lc2, innerrel->pathlist)
        {
            Path *p = (Path *) lfirst(lc2);
            if (p->param_info != NULL)
                continue;
            if (!pathkeys_contained_in(inner_pathkeys, p->pathkeys))
                continue;
            if (inner_path == NULL || p->total_cost < inner_path->total_cost)
                inner_path = p;
        }
    }
    if (inner_path == NULL)
        inner_path = (Path *) create_sort_path(root, innerrel,
                                               innerrel->cheapest_total_path,
                                               inner_pathkeys, -1.0);

    outer_rows = outer_path->rows;
    inner_rows = inner_path->rows;

    /*
     * Cost model: linear merge of two sorted streams.
     * Vectorized batch processing reduces per-tuple overhead significantly.
     * Apply cost_factor to the join processing portion (not child scan costs).
     */
    startup_cost = outer_path->startup_cost + inner_path->startup_cost;
    run_cost = ((outer_path->total_cost - outer_path->startup_cost) +
                (inner_path->total_cost - inner_path->startup_cost)) *
               vjoin_cost_factor +
               (outer_rows + inner_rows) * cpu_operator_cost * vjoin_cost_factor / 4.0;

    cpath = makeNode(CustomPath);
    cpath->path.pathtype = T_CustomScan;
    cpath->path.parent = joinrel;
    cpath->path.pathtarget = joinrel->reltarget;
    cpath->path.param_info = NULL;
    cpath->path.parallel_aware = false;
    cpath->path.parallel_safe = outer_path->parallel_safe &&
                                inner_path->parallel_safe;
    cpath->path.parallel_workers = 0;
    cpath->path.rows = joinrel->rows;
    cpath->path.startup_cost = startup_cost;
    cpath->path.total_cost = startup_cost + run_cost;
    /* Merge preserves outer sort order */
    cpath->path.pathkeys = outer_pathkeys;
    cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
    cpath->custom_paths = list_make2(outer_path, inner_path);
#if VJOIN_HAS_CUSTOM_RESTRICTINFO
    cpath->custom_restrictinfo = extra->restrictlist;
#endif
    cpath->custom_private = vjoin_build_private(nkeys, outer_keynos,
                                                inner_keynos, key_types);
    cpath->methods = &vjoin_merge_path_methods;

    add_path(joinrel, &cpath->path);

    /* --- Parallel merge path --- */
    {
        Path       *par_outer = NULL;
        bool        par_is_parallel = false;
        int         par_workers;
        CustomPath *pcpath;
        Cost        par_startup, par_run;
        double      par_outer_rows;

        if (joinrel->consider_parallel && outerrel->partial_pathlist != NIL)
        {
            par_outer = get_cheapest_path_for_pathkeys(
                                outerrel->partial_pathlist,
                                outer_pathkeys, NULL, TOTAL_COST, false);
            if (par_outer == NULL)
            {
                Path *cheapest_partial = (Path *) linitial(outerrel->partial_pathlist);
                par_outer = (Path *) create_sort_path(root, outerrel,
                                                      cheapest_partial,
                                                      outer_pathkeys, -1.0);
            }
            par_is_parallel = true;
        }
        else if (outerrel->partial_pathlist == NIL)
        {
            /* Non-parallel fallback for small tables */
            ListCell *lc3;
            foreach(lc3, outerrel->pathlist)
            {
                Path *p = (Path *) lfirst(lc3);
                if (p->param_info != NULL)
                    continue;
                if (!pathkeys_contained_in(outer_pathkeys, p->pathkeys))
                    continue;
                if (par_outer == NULL || p->total_cost < par_outer->total_cost)
                    par_outer = p;
            }
            if (par_outer == NULL)
                par_outer = (Path *) create_sort_path(root, outerrel,
                                                      outerrel->cheapest_total_path,
                                                      outer_pathkeys, -1.0);
        }

        if (par_outer != NULL)
        {
            if (par_is_parallel)
            {
                par_workers = par_outer->parallel_workers;
                if (par_workers <= 0)
                    par_workers = 1;
                par_outer_rows = par_outer->rows;
            }
            else
            {
                par_workers = 0;
                par_outer_rows = par_outer->rows;
            }

            par_startup = par_outer->startup_cost + inner_path->startup_cost;
            par_run = ((par_outer->total_cost - par_outer->startup_cost) +
                       (inner_path->total_cost - inner_path->startup_cost)) *
                      vjoin_cost_factor +
                      (par_outer_rows + inner_rows) * cpu_operator_cost * vjoin_cost_factor / 4.0;

            pcpath = makeNode(CustomPath);
            pcpath->path.pathtype = T_CustomScan;
            pcpath->path.parent = joinrel;
            pcpath->path.pathtarget = joinrel->reltarget;
            pcpath->path.param_info = NULL;

            if (par_is_parallel)
            {
                pcpath->path.parallel_aware = true;
                pcpath->path.parallel_safe = true;
                pcpath->path.parallel_workers = par_workers;
                pcpath->path.rows = clamp_row_est(joinrel->rows / par_workers);
            }
            else
            {
                pcpath->path.parallel_aware = false;
                pcpath->path.parallel_safe = joinrel->consider_parallel;
                pcpath->path.parallel_workers = 0;
                pcpath->path.rows = joinrel->rows;
            }

            pcpath->path.startup_cost = par_startup;
            pcpath->path.total_cost = par_startup + par_run;
            pcpath->path.pathkeys = NIL;
            pcpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
            pcpath->custom_paths = list_make2(par_outer, inner_path);
#if VJOIN_HAS_CUSTOM_RESTRICTINFO
            pcpath->custom_restrictinfo = extra->restrictlist;
#endif
            pcpath->custom_private = vjoin_build_private(nkeys, outer_keynos,
                                                         inner_keynos, key_types);
            pcpath->methods = &vjoin_merge_path_methods;

            if (par_is_parallel)
                add_partial_path(joinrel, &pcpath->path);
            else
                add_path(joinrel, &pcpath->path);
        }
    }
}

/*
 * Main hook: called by optimizer for each join pair.
 */
void
vjoin_pathlist_hook(PlannerInfo *root,
                    RelOptInfo *joinrel,
                    RelOptInfo *outerrel,
                    RelOptInfo *innerrel,
                    JoinType jointype,
                    JoinPathExtraData *extra)
{
    AttrNumber  outer_keynos[VJOIN_MAX_KEYS];
    AttrNumber  inner_keynos[VJOIN_MAX_KEYS];
    Oid         key_types[VJOIN_MAX_KEYS];
    int         nkeys;

    /* Chain to any previous hook first */
    if (prev_join_pathlist_hook)
        prev_join_pathlist_hook(root, joinrel, outerrel, innerrel,
                                jointype, extra);

    /* Only inner joins for now */
    if (jointype != JOIN_INNER)
        return;
    
    /* Master kill switch */
    if (!vjoin_enable)
        return;

    /* We need cheapest paths */
    if (outerrel->cheapest_total_path == NULL ||
        innerrel->cheapest_total_path == NULL)
        return;

    /* Build temporary targetlists from PathTarget exprs for key matching */
    {
        List       *outer_tl = NIL;
        List       *inner_tl = NIL;
        ListCell   *lc;
        int         resno;

        resno = 1;
        foreach(lc, outerrel->reltarget->exprs)
        {
            outer_tl = lappend(outer_tl,
                               makeTargetEntry((Expr *) lfirst(lc),
                                               resno++, NULL, false));
        }

        resno = 1;
        foreach(lc, innerrel->reltarget->exprs)
        {
            inner_tl = lappend(inner_tl,
                               makeTargetEntry((Expr *) lfirst(lc),
                                               resno++, NULL, false));
        }

        nkeys = vjoin_analyze_clauses(extra->restrictlist,
                                      outerrel, innerrel,
                                      outer_keynos, inner_keynos,
                                      key_types,
                                      outer_tl, inner_tl);

        list_free(outer_tl);
        list_free(inner_tl);
    }

    if (nkeys == 0)
        return;

    if (vjoin_enable_hashjoin)
        vjoin_try_hashjoin(root, joinrel, outerrel, innerrel, extra,
                           nkeys, outer_keynos, inner_keynos, key_types);

    if (vjoin_enable_nestloop)
        vjoin_try_nestloop(root, joinrel, outerrel, innerrel, extra,
                      nkeys, outer_keynos, inner_keynos, key_types);

    if (vjoin_enable_mergejoin)
        vjoin_try_mergejoin(root, joinrel, outerrel, innerrel, extra,
                            nkeys, outer_keynos, inner_keynos, key_types);
}
