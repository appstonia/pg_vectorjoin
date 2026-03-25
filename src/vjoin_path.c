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
#include "utils/typcache.h"
#include "vjoin_compat.h"
#include "pg_vectorjoin.h"

#include <math.h>

/*
 * Check if a type is usable for vectorized hash join.
 * Fast-path for INT4/INT8/FLOAT8, generic check via type cache for others.
 */
static bool
vjoin_type_has_hash_support(Oid typid, Oid *hash_proc, Oid *eq_opr)
{
    TypeCacheEntry *typentry;

    /* Fast numeric types — always supported, no cache lookup needed */
    switch (typid)
    {
        case INT4OID:
        case INT8OID:
        case FLOAT8OID:
            *hash_proc = InvalidOid;   /* will use inline fast path */
            *eq_opr = InvalidOid;
            return true;
        default:
            break;
    }

    /* Generic: look up hash function and equality operator */
    typentry = lookup_type_cache(typid, TYPECACHE_HASH_PROC | TYPECACHE_EQ_OPR);
    if (!OidIsValid(typentry->hash_proc) || !OidIsValid(typentry->eq_opr))
        return false;

    *hash_proc = typentry->hash_proc;
    *eq_opr = typentry->eq_opr;
    return true;
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
                      Oid *hash_procs,
                      Oid *eq_oprs,
                      Oid *collations,
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
        Oid         hp, eo;
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
        if (!vjoin_type_has_hash_support(keytype, &hp, &eo))
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
        hash_procs[nkeys] = hp;
        eq_oprs[nkeys] = eo;
        collations[nkeys] = opexpr->inputcollid;
        nkeys++;
    }

    return nkeys;
}

/*
 * Build custom_private list for serialization into the plan.
 * Format: [jointype, num_keys, outer_keyno1, inner_keyno1, keytype1,
 *          hash_proc1, eq_opr1, collation1, ...]
 */
static List *
vjoin_build_private(JoinType jointype,
                    int num_keys, AttrNumber *outer_keynos,
                    AttrNumber *inner_keynos, Oid *key_types,
                    Oid *hash_procs, Oid *eq_oprs, Oid *collations)
{
    List *result = NIL;
    int   i;

    result = lappend(result, makeInteger((int) jointype));
    result = lappend(result, makeInteger(num_keys));
    for (i = 0; i < num_keys; i++)
    {
        result = lappend(result, makeInteger((int) outer_keynos[i]));
        result = lappend(result, makeInteger((int) inner_keynos[i]));
        result = lappend(result, makeInteger((int) key_types[i]));
        result = lappend(result, makeInteger((int) hash_procs[i]));
        result = lappend(result, makeInteger((int) eq_oprs[i]));
        result = lappend(result, makeInteger((int) collations[i]));
    }
    return result;
}

/*
 * Analyze join restriction list for a single theta-join clause on a
 * fast type (INT4/INT8/FLOAT8) that can be handled via SIMD comparison.
 * Returns the btree strategy number (1-5) or 6 for NE, 0 if not found.
 */
static int
vjoin_analyze_theta_clause(List *restrictlist,
                           RelOptInfo *outerrel,
                           RelOptInfo *innerrel,
                           List *outer_tlist,
                           List *inner_tlist,
                           AttrNumber *theta_outer_keyno,
                           AttrNumber *theta_inner_keyno,
                           Oid *theta_keytype)
{
    ListCell      *lc;
    int            ntheta = 0;
    int            found_strategy = 0;

    foreach(lc, restrictlist)
    {
        RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);
        OpExpr       *opexpr;
        Node         *left, *right;
        Oid           keytype;
        TargetEntry  *outer_tle, *inner_tle;
        TypeCacheEntry *typentry;
        int           strategy;
        bool          swapped;

        if (!IsA(rinfo->clause, OpExpr))
            return 0;   /* complex clause, bail out */

        opexpr = (OpExpr *) rinfo->clause;
        if (list_length(opexpr->args) != 2)
            return 0;

        left = (Node *) linitial(opexpr->args);
        right = (Node *) lsecond(opexpr->args);
        if (!IsA(left, Var) || !IsA(right, Var))
            return 0;

        ntheta++;
        if (ntheta > 1)
            return 0;   /* multiple clauses — can't SIMD optimize */

        keytype = exprType(left);
        if (keytype != INT4OID && keytype != INT8OID && keytype != FLOAT8OID)
            return 0;

        /* Look up btree operator family for this type */
        typentry = lookup_type_cache(keytype, TYPECACHE_BTREE_OPFAMILY);
        if (!OidIsValid(typentry->btree_opf))
            return 0;

        strategy = get_op_opfamily_strategy(opexpr->opno, typentry->btree_opf);

        if (strategy == 0)
        {
            /* Check for NE via negator of equality */
            Oid negator = get_negator(opexpr->opno);
            if (OidIsValid(negator) &&
                get_op_opfamily_strategy(negator, typentry->btree_opf) == BTEqualStrategyNumber)
                strategy = 6;  /* NE */
            else
                return 0;
        }

        /* Only theta (non-equality) strategies */
        if (strategy == BTEqualStrategyNumber)
            return 0;   /* equi-join, handled by regular key analysis */

        /* Determine which side is outer vs inner */
        if (bms_is_subset(rinfo->left_relids, outerrel->relids) &&
            bms_is_subset(rinfo->right_relids, innerrel->relids))
        {
            outer_tle = tlist_member((Expr *) left, outer_tlist);
            inner_tle = tlist_member((Expr *) right, inner_tlist);
            swapped = false;
        }
        else if (bms_is_subset(rinfo->right_relids, outerrel->relids) &&
                 bms_is_subset(rinfo->left_relids, innerrel->relids))
        {
            outer_tle = tlist_member((Expr *) right, outer_tlist);
            inner_tle = tlist_member((Expr *) left, inner_tlist);
            swapped = true;
        }
        else
            return 0;

        if (outer_tle == NULL || inner_tle == NULL)
            return 0;

        /*
         * If sides were swapped, mirror the strategy:
         * "left OP right" with swapped sides means "outer OP_mirror inner".
         * LT(1) <-> GT(5), LE(2) <-> GE(4), EQ(3) stable, NE(6) stable.
         */
        if (swapped)
        {
            switch (strategy)
            {
                case 1: strategy = 5; break;
                case 2: strategy = 4; break;
                case 4: strategy = 2; break;
                case 5: strategy = 1; break;
                default: break;
            }
        }

        *theta_outer_keyno = outer_tle->resno;
        *theta_inner_keyno = inner_tle->resno;
        *theta_keytype = keytype;
        found_strategy = strategy;
    }

    return found_strategy;
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
                   JoinType jointype,
                   int nkeys,
                   AttrNumber *outer_keynos,
                   AttrNumber *inner_keynos,
                   Oid *key_types,
                   Oid *hash_procs,
                   Oid *eq_oprs,
                   Oid *collations)
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
        cpath->custom_private = vjoin_build_private(jointype, nkeys,
                                                    outer_keynos,
                                                    inner_keynos, key_types,
                                                    hash_procs, eq_oprs,
                                                    collations);
        cpath->methods = &vjoin_hash_path_methods;

        add_path(joinrel, &cpath->path);
    }

    /* --- Parallel path ---
     * If inner has partial paths, all participants build concurrently
     * using CAS-based lock-free insert (parallel build).
     * Otherwise fall back to leader-only build.
     * Only INNER/LEFT are safe: RIGHT/FULL need cross-worker
     * inner_matched coordination, so they stay non-parallel for now.
     *
     * Require at least 2 workers so the DSM/barrier overhead is justified. */
    if (joinrel->consider_parallel && outerrel->partial_pathlist != NIL &&
        (jointype == JOIN_INNER || jointype == JOIN_LEFT))
    {
        Path *par_outer = (Path *) linitial(outerrel->partial_pathlist);
        int   parallel_workers;

        parallel_workers = par_outer->parallel_workers;
        if (parallel_workers <= 0)
            parallel_workers = 1;

        if (parallel_workers >= 2)
        {
            Path *par_inner_partial = NULL;
            Path *par_inner_full = NULL;
            Path *par_inner;
            bool  parallel_build;
            ListCell *lc2;

            /* Try partial inner path first (enables parallel build) */
            foreach(lc2, innerrel->partial_pathlist)
            {
                Path *p = (Path *) lfirst(lc2);
                if (p->param_info == NULL &&
                    (par_inner_partial == NULL ||
                     p->total_cost < par_inner_partial->total_cost))
                    par_inner_partial = p;
            }

            /* Also find cheapest non-parameterized parallel-safe full path */
            foreach(lc2, innerrel->pathlist)
            {
                Path *p = (Path *) lfirst(lc2);
                if (p->parallel_safe && p->param_info == NULL &&
                    (par_inner_full == NULL ||
                     p->total_cost < par_inner_full->total_cost))
                    par_inner_full = p;
            }

            /* Prefer parallel build when partial inner is available */
            if (par_inner_partial != NULL)
            {
                par_inner = par_inner_partial;
                parallel_build = true;
            }
            else if (par_inner_full != NULL)
            {
                par_inner = par_inner_full;
                parallel_build = false;
            }
            else
                par_inner = NULL;

            if (par_inner != NULL)
            {
                outer_rows = par_outer->rows;

                if (parallel_build)
                {
                    double nprocs = parallel_workers + 1.0;

                    startup_cost = par_inner->total_cost +
                                   par_inner->rows * cpu_operator_cost * 2.0;

                    run_cost = par_outer->total_cost +
                               outer_rows * cpu_operator_cost * 2.0 *
                               vjoin_cost_factor;
                    run_cost /= nprocs;
                }
                else
                {
                    double nprocs = parallel_workers + 1.0;

                    inner_rows = par_inner->rows;
                    startup_cost = par_inner->total_cost +
                                   inner_rows * cpu_operator_cost * 2.0;

                    run_cost = par_outer->total_cost +
                               outer_rows * cpu_operator_cost * 2.0 *
                               vjoin_cost_factor;
                    run_cost /= nprocs;
                }

                cpath = makeNode(CustomPath);
                cpath->path.pathtype = T_CustomScan;
                cpath->path.parent = joinrel;
                cpath->path.pathtarget = joinrel->reltarget;
                cpath->path.param_info = NULL;
                cpath->path.parallel_aware = true;
                cpath->path.parallel_safe = true;
                cpath->path.parallel_workers = parallel_workers;
                cpath->path.rows = clamp_row_est(joinrel->rows /
                                                  parallel_workers);
                cpath->path.startup_cost = startup_cost;
                cpath->path.total_cost = startup_cost + run_cost;
                cpath->path.pathkeys = NIL;
                cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
                cpath->custom_paths = list_make2(par_outer, par_inner);
#if VJOIN_HAS_CUSTOM_RESTRICTINFO
                cpath->custom_restrictinfo = extra->restrictlist;
#endif
                cpath->custom_private = vjoin_build_private(jointype, nkeys,
                                                            outer_keynos,
                                                            inner_keynos,
                                                            key_types,
                                                            hash_procs, eq_oprs,
                                                            collations);
                cpath->methods = &vjoin_hash_path_methods;

                add_partial_path(joinrel, &cpath->path);
            }
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
              JoinType jointype,
              int nkeys,
              AttrNumber *outer_keynos,
              AttrNumber *inner_keynos,
              Oid *key_types,
              Oid *hash_procs,
              Oid *eq_oprs,
              Oid *collations,
              int theta_strategy,
              AttrNumber theta_outer_keyno,
              AttrNumber theta_inner_keyno,
              Oid theta_keytype)
{
    Path       *outer_path = outerrel->cheapest_total_path;
    Path       *inner_path = innerrel->cheapest_total_path;
    CustomPath *cpath;
    Cost        startup_cost,
                run_cost;
    double      outer_rows,
                inner_rows,
                num_blocks;
    int         simd_width = (nkeys > 0 || theta_strategy != 0) ? 4 : 1;

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
        cpath->custom_private = vjoin_build_private(jointype, nkeys,
                                                    outer_keynos,
                                                    inner_keynos, key_types,
                                                    hash_procs, eq_oprs,
                                                    collations);
        /* Append theta SIMD info */
        cpath->custom_private = lappend(cpath->custom_private,
                                        makeInteger(theta_strategy));
        if (theta_strategy != 0)
        {
            cpath->custom_private = lappend(cpath->custom_private,
                                            makeInteger((int) theta_outer_keyno));
            cpath->custom_private = lappend(cpath->custom_private,
                                            makeInteger((int) theta_inner_keyno));
            cpath->custom_private = lappend(cpath->custom_private,
                                            makeInteger((int) theta_keytype));
        }
        /* Append join qual expressions for executor evaluation */
        {
            List *join_clauses = NIL;
            ListCell *rlc;
            foreach(rlc, extra->restrictlist)
            {
                RestrictInfo *ri = lfirst_node(RestrictInfo, rlc);
                join_clauses = lappend(join_clauses, copyObject(ri->clause));
            }
            cpath->custom_private = lappend(cpath->custom_private, join_clauses);
        }
        cpath->methods = &vjoin_nestloop_path_methods;

        add_path(joinrel, &cpath->path);
    }

    /* --- Parallel path ---
     * Only INNER/LEFT are safe for the same reason as hash join. */
    if (joinrel->consider_parallel && outerrel->partial_pathlist != NIL &&
        (jointype == JOIN_INNER || jointype == JOIN_LEFT))
    {
        Path *par_outer = (Path *) linitial(outerrel->partial_pathlist);
        Path *par_inner = NULL;
        int   parallel_workers;
        ListCell *lc2;

        /* Find cheapest non-parameterized parallel-safe inner path.
         * Must NOT be a GatherPath — each worker needs its own
         * independent full scan of the inner relation. */
        foreach(lc2, innerrel->pathlist)
        {
            Path *p = (Path *) lfirst(lc2);
            if (p->parallel_safe && p->param_info == NULL &&
                (par_inner == NULL || p->total_cost < par_inner->total_cost))
                par_inner = p;
        }

        if (par_outer != NULL && par_inner != NULL)
        {
            outer_rows = par_outer->rows;
            inner_rows = par_inner->rows;

            num_blocks = ceil(outer_rows / vjoin_batch_size);
            parallel_workers = par_outer->parallel_workers;
            if (parallel_workers <= 0)
                parallel_workers = 1;

            startup_cost = par_outer->startup_cost;
            run_cost = par_outer->total_cost - par_outer->startup_cost +
                       num_blocks * par_inner->total_cost +
                       outer_rows * inner_rows * cpu_operator_cost *
                       vjoin_cost_factor / simd_width;

            /* Gather tuple-queue overhead */
            run_cost += clamp_row_est(joinrel->rows / parallel_workers) *
                        cpu_tuple_cost;

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
            cpath->custom_private = vjoin_build_private(jointype, nkeys,
                                                        outer_keynos,
                                                        inner_keynos, key_types,
                                                        hash_procs, eq_oprs,
                                                        collations);
            /* Append theta SIMD info */
            cpath->custom_private = lappend(cpath->custom_private,
                                            makeInteger(theta_strategy));
            if (theta_strategy != 0)
            {
                cpath->custom_private = lappend(cpath->custom_private,
                                                makeInteger((int) theta_outer_keyno));
                cpath->custom_private = lappend(cpath->custom_private,
                                                makeInteger((int) theta_inner_keyno));
                cpath->custom_private = lappend(cpath->custom_private,
                                                makeInteger((int) theta_keytype));
            }
            /* Append join qual expressions for executor evaluation */
            {
                List *join_clauses = NIL;
                ListCell *rlc;
                foreach(rlc, extra->restrictlist)
                {
                    RestrictInfo *ri = lfirst_node(RestrictInfo, rlc);
                    join_clauses = lappend(join_clauses, copyObject(ri->clause));
                }
                cpath->custom_private = lappend(cpath->custom_private, join_clauses);
            }
            cpath->methods = &vjoin_nestloop_path_methods;

            add_partial_path(joinrel, &cpath->path);
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
                    JoinType jointype,
                    int nkeys,
                    AttrNumber *outer_keynos,
                    AttrNumber *inner_keynos,
                    Oid *key_types,
                    Oid *hash_procs,
                    Oid *eq_oprs,
                    Oid *collations)
{
    ListCell   *lc;
    List       *outer_pathkeys = NIL;
    List       *inner_pathkeys = NIL;
    Path       *outer_path;
    Path       *inner_path;
    CustomPath *cpath;
    Cost        startup_cost,
                run_cost;
    double      outer_rows,
                inner_rows;
    int         merge_keys_found = 0;

    /*
     * Build PathKeys for each merge-joinable key.
     * We iterate over the restrict list and match each clause to the
     * key arrays produced by vjoin_analyze_clauses (same iteration order).
     */
    foreach(lc, extra->restrictlist)
    {
        RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);
        Oid         opfamily;
        PathKey    *outer_pathkey;
        PathKey    *inner_pathkey;

        if (merge_keys_found >= nkeys)
            break;

        if (rinfo->mergeopfamilies == NIL)
            continue;
        if (!IsA(rinfo->clause, OpExpr))
            continue;

        /* Make sure eclasses are initialized */
        if (rinfo->left_ec == NULL)
            initialize_mergeclause_eclasses(root, rinfo);

        if (rinfo->left_ec == NULL || rinfo->right_ec == NULL)
            continue;

        opfamily = linitial_oid(rinfo->mergeopfamilies);

        if (bms_is_subset(rinfo->left_relids, outerrel->relids))
        {
            outer_pathkey = make_canonical_pathkey(root, rinfo->left_ec,
                                                  opfamily, BTLessStrategyNumber,
                                                  false);
            inner_pathkey = make_canonical_pathkey(root, rinfo->right_ec,
                                                  opfamily, BTLessStrategyNumber,
                                                  false);
        }
        else
        {
            outer_pathkey = make_canonical_pathkey(root, rinfo->right_ec,
                                                  opfamily, BTLessStrategyNumber,
                                                  false);
            inner_pathkey = make_canonical_pathkey(root, rinfo->left_ec,
                                                  opfamily, BTLessStrategyNumber,
                                                  false);
        }

        outer_pathkeys = lappend(outer_pathkeys, outer_pathkey);
        inner_pathkeys = lappend(inner_pathkeys, inner_pathkey);
        merge_keys_found++;
    }

    if (merge_keys_found == 0)
        return;

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
    run_cost = (outer_path->total_cost - outer_path->startup_cost) +
               (inner_path->total_cost - inner_path->startup_cost) +
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
    cpath->custom_private = vjoin_build_private(jointype, nkeys,
                                                outer_keynos,
                                                inner_keynos, key_types,
                                                hash_procs, eq_oprs,
                                                collations);
    cpath->methods = &vjoin_merge_path_methods;

    add_path(joinrel, &cpath->path);

    /* --- Parallel merge path ---
     * Only INNER/LEFT: RIGHT/FULL need cross-worker inner tracking. */
    {
        Path       *par_outer = NULL;
        Path       *par_inner = NULL;
        bool        par_is_parallel = false;
        int         par_workers;
        CustomPath *pcpath;
        Cost        par_startup, par_run;
        double      par_outer_rows;

        if (joinrel->consider_parallel && outerrel->partial_pathlist != NIL &&
            (jointype == JOIN_INNER || jointype == JOIN_LEFT))
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

            /* Find cheapest parallel-safe sorted inner path.
             * Must not be a GatherPath — each worker needs its own
             * independent full scan of the inner relation. */
            {
                ListCell *lc3;
                foreach(lc3, innerrel->pathlist)
                {
                    Path *p = (Path *) lfirst(lc3);
                    if (!p->parallel_safe || p->param_info != NULL)
                        continue;
                    if (!pathkeys_contained_in(inner_pathkeys, p->pathkeys))
                        continue;
                    if (par_inner == NULL || p->total_cost < par_inner->total_cost)
                        par_inner = p;
                }
            }
            if (par_inner == NULL)
            {
                /* Find cheapest parallel-safe base path, then sort it */
                Path *base_inner = NULL;
                ListCell *lc3;
                foreach(lc3, innerrel->pathlist)
                {
                    Path *p = (Path *) lfirst(lc3);
                    if (p->parallel_safe && p->param_info == NULL &&
                        (base_inner == NULL || p->total_cost < base_inner->total_cost))
                        base_inner = p;
                }
                if (base_inner != NULL)
                    par_inner = (Path *) create_sort_path(root, innerrel,
                                                          base_inner,
                                                          inner_pathkeys, -1.0);
            }
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
            par_inner = inner_path;  /* reuse the non-parallel inner */
        }

        if (par_outer != NULL && par_inner != NULL)
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

            par_startup = par_outer->startup_cost + par_inner->startup_cost;

            if (par_is_parallel)
            {
                /*
                 * Shared inner materialization: leader scans inner once,
                 * all workers read from DSA-shared pre-deformed arrays.
                 * Inner cost is paid once (startup). Merge processing
                 * is distributed across workers.
                 * Small materialization overhead: memcpy cost per inner tuple.
                 */
                par_run = (par_outer->total_cost - par_outer->startup_cost) +
                          (par_inner->total_cost - par_inner->startup_cost) +
                          par_inner->rows * cpu_tuple_cost * 0.5 +
                          (par_outer_rows + par_inner->rows) *
                          cpu_operator_cost * vjoin_cost_factor / 4.0;
            }
            else
            {
                par_run = (par_outer->total_cost - par_outer->startup_cost) +
                          (par_inner->total_cost - par_inner->startup_cost) +
                          (par_outer_rows + par_inner->rows) *
                          cpu_operator_cost * vjoin_cost_factor / 4.0;
            }

            /* Gather tuple-queue overhead (parallel paths only) */
            if (par_is_parallel)
                par_run += clamp_row_est(joinrel->rows / par_workers) *
                           cpu_tuple_cost;

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
            pcpath->custom_paths = list_make2(par_outer, par_inner);
#if VJOIN_HAS_CUSTOM_RESTRICTINFO
            pcpath->custom_restrictinfo = extra->restrictlist;
#endif
            pcpath->custom_private = vjoin_build_private(jointype, nkeys,
                                                         outer_keynos,
                                                         inner_keynos, key_types,
                                                         hash_procs, eq_oprs,
                                                         collations);
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
    Oid         hash_procs[VJOIN_MAX_KEYS];
    Oid         eq_oprs[VJOIN_MAX_KEYS];
    Oid         collations[VJOIN_MAX_KEYS];
    int         nkeys;

    /* Chain to any previous hook first */
    if (prev_join_pathlist_hook)
        prev_join_pathlist_hook(root, joinrel, outerrel, innerrel,
                                jointype, extra);

    /* Supported join types */
    if (jointype != JOIN_INNER &&
        jointype != JOIN_LEFT &&
        jointype != JOIN_FULL &&
        jointype != JOIN_RIGHT)
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
                                      key_types, hash_procs, eq_oprs,
                                      collations,
                                      outer_tl, inner_tl);

        list_free(outer_tl);
        list_free(inner_tl);
    }

    /* Hash join and merge join require at least one equality key */
    if (nkeys > 0)
    {
        if (vjoin_enable_hashjoin)
            vjoin_try_hashjoin(root, joinrel, outerrel, innerrel, extra,
                               jointype,
                               nkeys, outer_keynos, inner_keynos, key_types,
                               hash_procs, eq_oprs, collations);

        if (vjoin_enable_mergejoin)
            vjoin_try_mergejoin(root, joinrel, outerrel, innerrel, extra,
                                jointype,
                                nkeys, outer_keynos, inner_keynos, key_types,
                                hash_procs, eq_oprs, collations);
    }

    /* Nested loop works for both equi-join (nkeys>0) and theta-join (nkeys==0) */
    if (vjoin_enable_nestloop)
    {
        int        theta_strategy = 0;
        AttrNumber theta_outer_keyno = 0;
        AttrNumber theta_inner_keyno = 0;
        Oid        theta_keytype = InvalidOid;

        /* For theta joins (no equi-join keys), try to detect a single SIMD-able theta clause */
        if (nkeys == 0)
        {
            List       *outer_tl2 = NIL;
            List       *inner_tl2 = NIL;
            ListCell   *lc2;
            int         resno2;

            resno2 = 1;
            foreach(lc2, outerrel->reltarget->exprs)
                outer_tl2 = lappend(outer_tl2,
                                    makeTargetEntry((Expr *) lfirst(lc2),
                                                    resno2++, NULL, false));
            resno2 = 1;
            foreach(lc2, innerrel->reltarget->exprs)
                inner_tl2 = lappend(inner_tl2,
                                    makeTargetEntry((Expr *) lfirst(lc2),
                                                    resno2++, NULL, false));

            theta_strategy = vjoin_analyze_theta_clause(extra->restrictlist,
                                                        outerrel, innerrel,
                                                        outer_tl2, inner_tl2,
                                                        &theta_outer_keyno,
                                                        &theta_inner_keyno,
                                                        &theta_keytype);
            list_free(outer_tl2);
            list_free(inner_tl2);
        }

        vjoin_try_nestloop(root, joinrel, outerrel, innerrel, extra,
                      jointype,
                      nkeys, outer_keynos, inner_keynos, key_types,
                      hash_procs, eq_oprs, collations,
                      theta_strategy, theta_outer_keyno,
                      theta_inner_keyno, theta_keytype);
    }
}
