#include "postgres.h"
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
 * Try to create a VectorHashJoin path.
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

    if (outer_path == NULL || inner_path == NULL)
        return;

    outer_rows = outer_path->rows;
    inner_rows = inner_path->rows;

    /* Cost model: vectorized hash build + batch probe */
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

/*
 * Try to create a BlockNestLoop path.
 */
static void
vjoin_try_bnl(PlannerInfo *root,
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

    if (outer_path == NULL || inner_path == NULL)
        return;

    outer_rows = outer_path->rows;
    inner_rows = inner_path->rows;
    num_blocks = ceil(outer_rows / vjoin_batch_size);

    /* Cost: scan outer once, scan inner num_blocks times */
    startup_cost = outer_path->startup_cost;
    run_cost = outer_path->total_cost - outer_path->startup_cost +
               num_blocks * inner_path->total_cost +
               outer_rows * inner_rows * cpu_operator_cost *
               vjoin_cost_factor / simd_width;

    /* BNL is typically only competitive for smaller inner relations */
    if (inner_rows > 10000 && nkeys > 0)
        return;

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
    cpath->methods = &vjoin_bnl_path_methods;

    add_path(joinrel, &cpath->path);
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

    if (vjoin_enable_bnl)
        vjoin_try_bnl(root, joinrel, outerrel, innerrel, extra,
                      nkeys, outer_keynos, inner_keynos, key_types);
}
