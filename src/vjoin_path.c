#include "postgres.h"
#include "access/stratnum.h"
#include "catalog/pg_namespace_d.h"
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
#include "parser/parsetree.h"
#include "miscadmin.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/typcache.h"
#include "vjoin_compat.h"
#include "pg_vectorjoin.h"

#include <math.h>

static int
vjoin_path_next_power_of_2(uint64 n)
{
    uint64 p = 1;

    while (p < n)
    {
        if (p > (uint64) (INT_MAX / 2))
            return 0;
        p <<= 1;
    }

    return (int) p;
}

/*
 * Estimate whether the vector hash table for a given inner relation fits
 * within PostgreSQL's hash memory budget (work_mem * hash_mem_multiplier).
 *
 * When multi-batch spilling is enabled (vjoin_enable_hash_spill) and this is
 * not a parallel build (spilling is serial-only in v1), the inner relation no
 * longer has to fit in memory: we estimate how many batches (power of two,
 * capped by vjoin_hash_max_batches) are needed so that one batch's in-memory
 * footprint fits the budget, and accept the path.  The estimated batch count
 * is returned via *nbatch_out for use by the cost model.
 *
 * Otherwise (spilling off, or parallel build) the old strict behavior applies:
 * the whole table must fit within 50% of the budget (2x safety margin for
 * planner row-count estimation errors), and *nbatch_out is set to 1.
 *
 * The vector hash table stores three flat arrays:
 *   hashvals:   capacity * sizeof(uint32)
 *   all_values: capacity * sizeof(Datum) * num_inner_attrs
 *   all_isnull: capacity * sizeof(bool)  * num_inner_attrs
 *
 * Also enforces MaxAllocSize per-array (per-batch when spilling) as a hard
 * ceiling.
 */
static bool
vjoin_hash_allocation_fits(double inner_rows, int num_inner_attrs,
                           bool parallel_build, bool is_parallel_path,
                           int *nbatch_out)
{
    uint64 target_rows;
    int    capacity;
    uint64 total_bytes;
    uint64 hash_mem_limit;
    bool   spill;

    if (nbatch_out)
        *nbatch_out = 1;

    if (num_inner_attrs <= 0)
        return false;

    if (inner_rows < 64.0)
        inner_rows = 64.0;

    target_rows = (uint64) ceil(inner_rows) * VJOIN_HT_LOAD_FACTOR;
    if (parallel_build)
        target_rows *= 2;
    if (target_rows < 128)
        target_rows = 128;

    hash_mem_limit = (uint64)((double) work_mem * hash_mem_multiplier * 1024.0);
    if (hash_mem_limit < 64 * 1024)
        hash_mem_limit = 64 * 1024;

    /*
     * Spilling is serial-only in v1: it is unsupported on every parallel
     * path (the shared DSA hash table cannot grow_batches, and the byref
     * leader build serializes only the in-memory batch to DSA).  Gate on
     * is_parallel_path, not parallel_build, so the leader-only build path
     * (parallel_build == false but still parallel) also refuses to spill.
     */
    spill = vjoin_enable_hash_spill && !is_parallel_path;

    if (spill)
    {
        uint64 row_bytes;
        uint64 est_bytes;
        int    max_nbatch;
        int    nbatch;
        uint64 per_batch_rows;
        int    per_batch_cap;

        /* Approximate per-row in-memory footprint of the flat arrays. */
        row_bytes = (uint64) sizeof(uint32) +
                    ((uint64) sizeof(Datum) + sizeof(bool)) * num_inner_attrs;
        est_bytes = target_rows * row_bytes;

        /* Choose the smallest power-of-two nbatch whose batch fits the budget. */
        max_nbatch = vjoin_path_next_power_of_2(Max(vjoin_hash_max_batches, 1));
        nbatch = 1;
        while (nbatch < max_nbatch &&
               (double) est_bytes / nbatch > (double) hash_mem_limit)
            nbatch *= 2;

        /* Per-batch capacity must still respect the palloc ceiling. */
        per_batch_rows = (target_rows + nbatch - 1) / (uint64) nbatch;
        per_batch_cap = vjoin_path_next_power_of_2(per_batch_rows);
        if (per_batch_cap <= 0)
            return false;
        if ((uint64) per_batch_cap > (uint64) MaxAllocSize / sizeof(uint32))
            return false;
        if ((uint64) per_batch_cap >
            (uint64) MaxAllocSize / ((uint64) sizeof(Datum) * num_inner_attrs))
            return false;
        if ((uint64) per_batch_cap >
            (uint64) MaxAllocSize / ((uint64) sizeof(bool) * num_inner_attrs))
            return false;

        if (nbatch_out)
            *nbatch_out = nbatch;
        return true;
    }

    capacity = vjoin_path_next_power_of_2(target_rows);
    if (capacity <= 0)
        return false;

    /* Per-array MaxAllocSize hard ceiling (palloc limit) */
    if ((uint64) capacity > (uint64) MaxAllocSize / sizeof(uint32))
        return false;
    if ((uint64) capacity >
        (uint64) MaxAllocSize / ((uint64) sizeof(Datum) * num_inner_attrs))
        return false;
    if ((uint64) capacity >
        (uint64) MaxAllocSize / ((uint64) sizeof(bool) * num_inner_attrs))
        return false;

    /* Total memory footprint vs work_mem budget.
     * Apply 2x safety margin: without spilling, vector hash join cannot
     * batch to disk, so we must leave headroom for estimation errors. */
    total_bytes = (uint64) capacity * sizeof(uint32) +
                  (uint64) capacity * sizeof(Datum) * num_inner_attrs +
                  (uint64) capacity * sizeof(bool)  * num_inner_attrs;

    if (total_bytes * 2 > hash_mem_limit)
        return false;

    return true;
}

static int
vjoin_path_num_attrs(Path *path, RelOptInfo *rel)
{
    PathTarget *target = NULL;

    if (path != NULL)
        target = path->pathtarget;
    if (target == NULL && rel != NULL)
        target = rel->reltarget;
    if (target == NULL)
        return 0;

    return list_length(target->exprs);
}

/*
 * Auto-tune helpers.
 *
 * When pg_vectorjoin.auto_tune is on we inspect the native paths the core
 * planner already produced for this joinrel and, iff the cheapest
 * comparable native path is cheaper than our path, apply a flat
 * multiplicative discount (cost *= auto_tune_margin) to our path.  Lets
 * the user enable vector joins by intent without hand-tuning per-jointype
 * cost factors, while still leaving honest model decisions alone when our
 * model already wins.
 *
 * Crucially this is a *multiplicative* discount, not an absolute clamp to
 * native*margin.  An absolute clamp maps every one of our candidate paths
 * for a joinrel onto the same value, erasing the relative cost differences
 * between join orders so the planner can no longer tell a good join order
 * from a bad one.  A flat multiplicative scale is monotone and therefore
 * preserves the raw cost model's ordering across join orders.
 *
 * We only compare against unparameterized native paths of the same
 * physical type (Hash/Merge/Nest) and the same parallel-aware flag,
 * scanned from joinrel->pathlist (for non-partial) or
 * joinrel->partial_pathlist (for partial paths).
 */
static Cost
vjoin_cheapest_native_cost(List *pathlist, NodeTag native_tag,
                           bool parallel_aware)
{
    ListCell *lc;
    Cost      best = -1.0;

    foreach(lc, pathlist)
    {
        Path *p = (Path *) lfirst(lc);

        if (nodeTag(p) != native_tag)
            continue;
        if (p->param_info != NULL)
            continue;
        if (p->parallel_aware != parallel_aware)
            continue;
        if (best < 0.0 || p->total_cost < best)
            best = p->total_cost;
    }
    return best;
}

static void
vjoin_apply_auto_tune(CustomPath *cpath, RelOptInfo *joinrel,
                      NodeTag native_tag, bool partial)
{
    List *pathlist;
    Cost  native;

    if (!vjoin_auto_tune)
        return;

    pathlist = partial ? joinrel->partial_pathlist : joinrel->pathlist;
    native = vjoin_cheapest_native_cost(pathlist, native_tag,
                                        cpath->path.parallel_aware);
    if (native < 0.0)
        return;                 /* no comparable native path to anchor on */

    /*
     * Only adjust when the native path would otherwise beat us; honest wins
     * are left untouched.
     *
     * We pull our cost down close to native*margin so the user's intent to
     * use vector joins is honored, but we must NOT clamp every candidate to
     * the same value: doing so erases the relative cost differences between
     * join orders and the planner can no longer pick a good order from a bad
     * one.  Instead we blend: scaled = native*margin + epsilon*raw_cost.
     * This is strictly monotone in raw_cost (so the raw model's join-order
     * ranking is preserved for add_path's exact-cost comparison) while the
     * tiny epsilon keeps the result ~= native*margin (so we still win vs
     * native).  The same factor is applied to startup_cost so it stays <=
     * total_cost.
     */
    if (native < cpath->path.total_cost && cpath->path.total_cost > 0.0)
    {
        Cost   target = native * vjoin_auto_tune_margin;
        double eps = 0.001;
        double factor;

        if (target < 0.0)
            target = 0.0;

        factor = (target + eps * cpath->path.total_cost) /
                 cpath->path.total_cost;

        cpath->path.total_cost *= factor;
        cpath->path.startup_cost *= factor;
    }
}

/* Forward declaration — vjoin_find_var_tle calls vjoin_strip_key_expr */
static Node *vjoin_strip_key_expr(Node *node);

/*
 * Match a Var to a target list by (varno, varattno) only.
 *
 * tlist_member() uses equal() which in PG16+ also compares varnullingrels.
 * The clause Var and the reltarget Var may have different varnullingrels
 * (PG16 outer-join analysis sets these even for inner-join Vars in some
 * contexts), causing tlist_member() to return NULL for a logically
 * identical column reference.  This helper ignores those auxiliary fields.
 */
static TargetEntry *
vjoin_find_var_tle(Var *var, List *tlist)
{
    ListCell *lc;

    foreach(lc, tlist)
    {
        TargetEntry *tle = (TargetEntry *) lfirst(lc);
        Node        *texpr;

        texpr = (Node *) tle->expr;
        texpr = vjoin_strip_key_expr(texpr);
        if (texpr == NULL || !IsA(texpr, Var))
            continue;

        {
            Var *tv = (Var *) texpr;

            if (tv->varno == var->varno &&
                tv->varattno == var->varattno &&
                tv->varlevelsup == var->varlevelsup)
                return tle;
        }
    }

    return NULL;
}

static Node *
vjoin_strip_key_expr(Node *node)
{
    node = strip_implicit_coercions(node);

    for (;;)
    {
        if (node == NULL)
            return NULL;

        if (IsA(node, RelabelType))
            node = (Node *) ((RelabelType *) node)->arg;
        else if (IsA(node, CoerceViaIO))
            node = (Node *) ((CoerceViaIO *) node)->arg;
        else if (IsA(node, CollateExpr))
            node = (Node *) ((CollateExpr *) node)->arg;
        else
            break;
    }

    return strip_implicit_coercions(node);
}

static bool
vjoin_relids_are_supported(PlannerInfo *root, Relids relids)
{
    int relid = -1;

    while ((relid = bms_next_member(relids, relid)) >= 0)
    {
        RangeTblEntry *rte = planner_rt_fetch(relid, root);

        if (rte == NULL)
            return false;

        if (rte->rtekind != RTE_RELATION)
            return false;

        if (OidIsValid(rte->relid) &&
            get_rel_namespace(rte->relid) == PG_CATALOG_NAMESPACE)
            return false;
    }

    return true;
}

/*
 * Check if a type is usable for vectorized hash join.
 * Fast-path for INT4/INT8/FLOAT8, generic check via type cache for others.
 */
static bool
vjoin_type_has_hash_support(Oid typid, Oid *hash_proc, Oid *eq_opr)
{
    TypeCacheEntry *typentry;

    /* Fast types — always supported, no cache lookup needed */
    switch (typid)
    {
        case INT4OID:
        case INT2OID:
        case DATEOID:
        case OIDOID:
        case REGCLASSOID:
        case REGTYPEOID:
        case REGPROCOID:
        case INT8OID:
        case TIMESTAMPOID:
        case TIMESTAMPTZOID:
        case TIMEOID:
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
                      List *inner_tlist,
                      List **consumed_rinfos)
{
    ListCell   *lc;
    int         nkeys = 0;

    foreach(lc, restrictlist)
    {
        RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);
        OpExpr     *opexpr;
        Node       *left,
                   *right,
                   *left_key,
                   *right_key;
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
        left_key = vjoin_strip_key_expr(left);
        right_key = vjoin_strip_key_expr(right);

        /* Determine which side is outer vs inner */
        if (!IsA(left_key, Var) || !IsA(right_key, Var))
            continue;

        keytype = exprType(left_key);
        if (!vjoin_type_has_hash_support(keytype, &hp, &eo))
            continue;

        /* Match Var to outer/inner relation */
        if (bms_is_subset(rinfo->left_relids, outerrel->relids) &&
            bms_is_subset(rinfo->right_relids, innerrel->relids))
        {
            outer_tle = vjoin_find_var_tle((Var *) left_key, outer_tlist);
            inner_tle = vjoin_find_var_tle((Var *) right_key, inner_tlist);
        }
        else if (bms_is_subset(rinfo->right_relids, outerrel->relids) &&
                 bms_is_subset(rinfo->left_relids, innerrel->relids))
        {
            outer_tle = vjoin_find_var_tle((Var *) right_key, outer_tlist);
            inner_tle = vjoin_find_var_tle((Var *) left_key, inner_tlist);
        }
        else
            continue;

        if (outer_tle == NULL || inner_tle == NULL)
            continue;

        outer_keynos[nkeys] = outer_tle->resno;
        inner_keynos[nkeys] = inner_tle->resno;
        key_types[nkeys] = vjoin_map_fast_type(keytype);
        hash_procs[nkeys] = hp;
        eq_oprs[nkeys] = eo;
        collations[nkeys] = opexpr->inputcollid;
        nkeys++;

        /* Record which clause became an executor-matched equi-key, so the
         * caller can recheck only the residual (non-key) clauses. */
        if (consumed_rinfos)
            *consumed_rinfos = lappend(*consumed_rinfos, rinfo);
    }

    return nkeys;
}

static bool
vjoin_expr_references_inner(PlannerInfo *root, Node *node, Relids inner_relids)
{
    Bitmapset  *varnos;
    bool        references_inner;

    if (node == NULL)
        return false;

    varnos = vjoin_pull_varnos(root, node);
    references_inner = bms_overlap(varnos, inner_relids);
    bms_free(varnos);

    return references_inner;
}

static bool
vjoin_is_left_join_anti_pattern(PlannerInfo *root,
                                List *restrictlist,
                                RelOptInfo *innerrel)
{
    ListCell   *lc;

    foreach(lc, restrictlist)
    {
        RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);

        if (IsA(rinfo->clause, NullTest))
        {
            NullTest *nt = (NullTest *) rinfo->clause;

            if (nt->nulltesttype == IS_NULL &&
                vjoin_expr_references_inner(root,
                                            (Node *) nt->arg,
                                            innerrel->relids))
                return true;
        }
    }

    return false;
}

static bool
vjoin_is_safe_left_join_subset(PlannerInfo *root,
                               RelOptInfo *joinrel,
                               RelOptInfo *innerrel,
                               JoinPathExtraData *extra,
                               int nkeys)
{
    ListCell   *lc;

    /* Narrow initial subset: equi-join only, no anti-join shape. */
    if (nkeys <= 0)
        return false;
    if (vjoin_is_left_join_anti_pattern(root, extra->restrictlist, innerrel))
        return false;

    /* Do not project nullable inner-side Vars yet. */
    foreach(lc, joinrel->reltarget->exprs)
    {
        if (vjoin_expr_references_inner(root,
                                        (Node *) lfirst(lc),
                                        innerrel->relids))
            return false;
    }

    /* Keep the subset to simple equi-join clauses only for now. */
    foreach(lc, extra->restrictlist)
    {
        RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);

        if (!IsA(rinfo->clause, OpExpr))
            return false;
        if (rinfo->hashjoinoperator == InvalidOid)
            return false;
    }

    return true;
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
 * Build the list of join-clause expressions the executor must re-check as a
 * qual.  Equi-key clauses that were already consumed as executor match keys
 * (recorded in consumed_rinfos) are skipped — only residual conditions (e.g.
 * a.x < b.y, NE clauses, or equi-keys beyond VJOIN_MAX_KEYS) remain.  When
 * there are no residual clauses this returns NIL and the executor avoids the
 * per-row ExecQual entirely.
 */
static List *
vjoin_build_residual_clauses(List *restrictlist, List *consumed_rinfos)
{
    List *join_clauses = NIL;
    ListCell *rlc;

    foreach(rlc, restrictlist)
    {
        RestrictInfo *ri = lfirst_node(RestrictInfo, rlc);

        if (list_member_ptr(consumed_rinfos, ri))
            continue;
        join_clauses = lappend(join_clauses, copyObject(ri->clause));
    }

    return join_clauses;
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
        Node         *left, *right, *left_key, *right_key;
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
        left_key = vjoin_strip_key_expr(left);
        right_key = vjoin_strip_key_expr(right);
        if (!IsA(left_key, Var) || !IsA(right_key, Var))
            return 0;

        ntheta++;
        if (ntheta > 1)
            return 0;   /* multiple clauses — can't SIMD optimize */

        keytype = exprType(left_key);
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
            outer_tle = vjoin_find_var_tle((Var *) left_key, outer_tlist);
            inner_tle = vjoin_find_var_tle((Var *) right_key, inner_tlist);
            swapped = false;
        }
        else if (bms_is_subset(rinfo->right_relids, outerrel->relids) &&
                 bms_is_subset(rinfo->left_relids, innerrel->relids))
        {
            outer_tle = vjoin_find_var_tle((Var *) right_key, outer_tlist);
            inner_tle = vjoin_find_var_tle((Var *) left_key, inner_tlist);
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
                   Oid *collations,
                   List *residual_clauses)
{
    Path       *outer_path = outerrel->cheapest_total_path;
    Path       *inner_path = innerrel->cheapest_total_path;
    CustomPath *cpath;
    Cost        startup_cost,
                run_cost;
    double      outer_rows,
                inner_rows;
    double      cost_factor = vjoin_cost_factor * vjoin_hash_cost_factor;
    int         i;

    /*
     * Vector hash join only benefits from SIMD on numeric types.
     * For text/generic keys the per-tuple overhead (batch deform, fmgr hash/eq)
     * is higher than native hash join with no batching advantage, and VHJ
     * cannot spill to disk.  Reject if any key is non-numeric.
     */
    for (i = 0; i < nkeys; i++)
    {
        if (!vjoin_is_fast_type(key_types[i]))
            return;
    }

    /* --- Non-parallel path --- */
    if (outer_path != NULL && inner_path != NULL)
    {
        int num_inner_attrs = vjoin_path_num_attrs(inner_path, innerrel);
        int nbatch_est = 1;

        outer_rows = outer_path->rows;
        inner_rows = inner_path->rows;

        /*
         * For base relations, cross-check with pg_class.reltuples.
         * Planner estimates can severely undercount after join selectivity;
         * reltuples is the physical reality.  Use the larger of the two.
         */
        if (innerrel->tuples > 0 && innerrel->tuples > inner_rows)
            inner_rows = innerrel->tuples;

        if (!vjoin_hash_allocation_fits(inner_rows, num_inner_attrs, false,
                                        false, &nbatch_est))
            goto try_parallel_hash;

        /*
         * Build charges every inner row (hash + insert into the table);
         * probe is lighter (hash + compare, no full tuple build) so it is
         * charged at cpu_operator_cost only.  The output-cardinality charge
         * (joinrel->rows * cpu_tuple_cost) makes the model sensitive to join
         * order: orders that materialize/forward large intermediate results
         * are penalized, steering the planner toward selective-joins-first.
         */
        startup_cost = inner_path->total_cost +
                       inner_rows * (cpu_tuple_cost + cpu_operator_cost) * cost_factor;
        run_cost = outer_path->total_cost +
                   outer_rows * cpu_operator_cost * cost_factor +
                   joinrel->rows * cpu_tuple_cost;

        /*
         * Multi-batch spill I/O cost.  Roughly (nbatch_est - 1)/nbatch_est of
         * both inputs are written to and re-read from temp files, i.e. two
         * sequential I/Os per spilled tuple.  Charge inner spill to startup
         * (happens during build) and outer spill to run.
         */
        if (nbatch_est > 1)
        {
            double spill_frac = (double) (nbatch_est - 1) / nbatch_est;
            double io_per_tuple = seq_page_cost / VJOIN_SPILL_TUPLES_PER_PAGE;

            startup_cost += inner_rows * spill_frac * 2.0 * io_per_tuple;
            run_cost += outer_rows * spill_frac * 2.0 * io_per_tuple;
        }

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
        cpath->custom_private = lappend(cpath->custom_private,
                                        (List *) copyObject(residual_clauses));
        cpath->methods = &vjoin_hash_path_methods;

        vjoin_apply_auto_tune(cpath, joinrel, T_HashPath, false);
        add_path(joinrel, &cpath->path);
    }

try_parallel_hash:

    /* --- Parallel path ---
     * Only INNER joins are supported. If inner has partial paths,
     * all participants build concurrently using CAS-based lock-free
     * insert (parallel build). Otherwise fall back to leader-only build.
     * Require at least 2 workers so the DSM/barrier overhead is justified. */
    if (joinrel->consider_parallel && outerrel->partial_pathlist != NIL &&
        jointype == JOIN_INNER)
    {
        Path *par_outer = (Path *) linitial(outerrel->partial_pathlist);
        int   parallel_workers;

        parallel_workers = par_outer->parallel_workers;
        if (parallel_workers <= 0)
            parallel_workers = 1;

        if (parallel_workers >= 2)
        {
            Path *par_inner_full = NULL;
            Path *par_inner;
            bool  parallel_build;
            ListCell *lc2;

            /*
             * Always use the per-worker private build over a full
             * (non-parallel-aware) inner scan: every worker scans the entire
             * inner, builds its own private (spill-capable) hash table, and
             * probes its disjoint slice of the parallel-aware outer scan.
             *
             * The alternative CAS shared-DSA build (all participants insert
             * into one shared hash table over a parallel-aware partial inner)
             * is intentionally disabled: its dynamic-party barrier
             * synchronization is race-prone and can deadlock.  The private
             * build is robust, supports per-worker spilling, and — since the
             * inner is typically far smaller than the outer — the redundant
             * per-worker inner builds are an acceptable cost.
             */
            foreach(lc2, innerrel->pathlist)
            {
                Path *p = (Path *) lfirst(lc2);
                if (p->parallel_safe && p->param_info == NULL &&
                    (par_inner_full == NULL ||
                     p->total_cost < par_inner_full->total_cost))
                    par_inner_full = p;
            }

            par_inner = par_inner_full;
            parallel_build = false;

            if (par_inner != NULL)
            {
                int num_inner_attrs = vjoin_path_num_attrs(par_inner, innerrel);
                double par_inner_rows = par_inner->rows;
                int    par_nbatch = 1;
                bool   par_fits;

                if (innerrel->tuples > 0 && innerrel->tuples > par_inner_rows)
                    par_inner_rows = innerrel->tuples;

                outer_rows = par_outer->rows;

                /*
                 * The CAS shared-DSA build (parallel_build) cannot spill:
                 * the inner must fit.  The leader-only (par_inner_full) path
                 * instead has every worker build its own private hash table
                 * over the full inner, so it CAN spill per worker — allow it
                 * and estimate the batch count for the I/O cost below.
                 */
                if (parallel_build)
                    par_fits = vjoin_hash_allocation_fits(par_inner_rows,
                                                          num_inner_attrs,
                                                          true, true, NULL);
                else
                    par_fits = vjoin_hash_allocation_fits(par_inner_rows,
                                                          num_inner_attrs,
                                                          false, false,
                                                          &par_nbatch);

                if (par_fits)
                {
                    if (parallel_build)
                    {
                        startup_cost = par_inner->total_cost +
                                       par_inner->rows * (cpu_tuple_cost + cpu_operator_cost) * cost_factor;

                        /* par_outer is a partial path: cost is already per-worker */
                        run_cost = par_outer->total_cost +
                                   outer_rows * (cpu_tuple_cost + cpu_operator_cost) * cost_factor;
                    }
                    else
                    {
                        inner_rows = par_inner->rows;
                        startup_cost = par_inner->total_cost +
                                       inner_rows * (cpu_tuple_cost + cpu_operator_cost) * cost_factor;

                        /*
                         * par_outer is a partial path: cost is already
                         * per-worker.  Probe is charged at cpu_operator_cost
                         * (hash + compare) and the per-worker output
                         * cardinality at cpu_tuple_cost, matching the
                         * non-parallel model so join-order signal is
                         * preserved across parallel plans too.
                         */
                        run_cost = par_outer->total_cost +
                                   outer_rows * cpu_operator_cost * cost_factor +
                                   clamp_row_est(joinrel->rows / parallel_workers) *
                                   cpu_tuple_cost;

                        /*
                         * Per-worker spill I/O when the private build spills.
                         * Each worker writes/re-reads (nbatch-1)/nbatch of the
                         * full inner during build and of its outer slice during
                         * probe (two sequential I/Os per spilled tuple).
                         */
                        if (par_nbatch > 1)
                        {
                            double spill_frac = (double) (par_nbatch - 1) / par_nbatch;
                            double io_per_tuple = seq_page_cost / VJOIN_SPILL_TUPLES_PER_PAGE;

                            startup_cost += par_inner_rows * spill_frac * 2.0 * io_per_tuple;
                            run_cost += outer_rows * spill_frac * 2.0 * io_per_tuple;
                        }
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
                    cpath->custom_private = lappend(cpath->custom_private,
                                                    (List *) copyObject(residual_clauses));
                    cpath->methods = &vjoin_hash_path_methods;

                    vjoin_apply_auto_tune(cpath, joinrel, T_HashPath, true);
                    add_partial_path(joinrel, &cpath->path);
                }
            }
        }
    }
}

/*
 * For equi-joins, VectorNestedLoop performs a full block nested loop over the
 * inner relation and -- unlike native nested loop -- cannot push outer values
 * into a parameterized index scan.  When the inner relation is a base table
 * with an index whose leading column is one of the inner join keys, native
 * PostgreSQL can run a far cheaper index nested loop.  Because join cardinality
 * is frequently underestimated, the VNL cost model can wrongly win that race
 * and cause catastrophic O(outer*inner) execution.  In that situation we
 * decline to offer a VNL path and let core use its index nested loop; the
 * vectorized hash/merge joins remain available for the equi-join.
 */
static bool
vjoin_inner_has_indexable_key(RelOptInfo *innerrel,
                              AttrNumber *inner_keynos, int nkeys)
{
    Bitmapset  *key_attnos = NULL;
    ListCell   *lc;
    int         i;
    bool        result = false;

    /* Only base relations carry an index list. */
    if (innerrel->reloptkind != RELOPT_BASEREL ||
        innerrel->indexlist == NIL)
        return false;

    /* Collect base-relation attribute numbers of the inner join keys. */
    for (i = 0; i < nkeys; i++)
    {
        Node *expr = (Node *) list_nth(innerrel->reltarget->exprs,
                                       inner_keynos[i] - 1);

        if (IsA(expr, Var))
        {
            Var *v = (Var *) expr;

            key_attnos = bms_add_member(key_attnos,
                v->varattno - FirstLowInvalidHeapAttributeNumber);
        }
    }

    if (key_attnos == NULL)
        return false;

    foreach(lc, innerrel->indexlist)
    {
        IndexOptInfo *idx = (IndexOptInfo *) lfirst(lc);

        /* Leading index column must be a join key to drive an index NL. */
        if (idx->nkeycolumns > 0 && idx->indexkeys[0] != 0 &&
            bms_is_member(idx->indexkeys[0] - FirstLowInvalidHeapAttributeNumber,
                          key_attnos))
        {
            result = true;
            break;
        }
    }

    bms_free(key_attnos);
    return result;
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
              List *residual_clauses,
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
    double      cost_factor = vjoin_cost_factor * vjoin_nestloop_cost_factor;
    int         simd_width;
    int         i;

    /*
     * For equi-joins, require all keys to be fast numeric types.
     * VNL cannot use parameterized inner paths (unlike native NL which
     * pushes outer values into index conditions).  For text/generic keys
     * VNL has no SIMD advantage and the full inner scan + fmgr comparison
     * overhead makes it strictly worse than native NL.
     * Theta joins are always numeric (enforced by vjoin_analyze_theta_clause).
     */
    for (i = 0; i < nkeys; i++)
    {
        if (!vjoin_is_fast_type(key_types[i]))
            return;
    }

    /*
     * For equi-joins, decline the VNL path when native can drive an index
     * nested loop on the inner relation.  This avoids choosing a full block
     * nested loop (which cannot use indexes) over a vastly cheaper index NL,
     * a regression that misestimated join cardinalities make dangerously easy.
     * Theta joins (nkeys == 0) have no index-NL alternative and are unaffected.
     */
    if (nkeys > 0 &&
        vjoin_inner_has_indexable_key(innerrel, inner_keynos, nkeys))
        return;

    /*
     * SIMD width for cost model: only single-key numeric equi-join or
     * theta-join (always numeric) gets the 4-wide SIMD benefit.
     * Multi-key or text keys fall back to scalar comparison.
     */
    if (theta_strategy != 0)
        simd_width = 4;    /* theta SIMD — always numeric */
    else if (nkeys == 1 && vjoin_is_fast_type(key_types[0]))
        simd_width = 4;    /* single numeric key — fully vectorized */
    else if (nkeys > 1 && vjoin_is_fast_type(key_types[0]))
        simd_width = 2;    /* composite key — leading key vectorized,
                            * residual keys verified scalar */
    else
        simd_width = 1;    /* unsupported leading key — scalar fallback */

    /* --- Non-parallel path --- */
    if (outer_path != NULL && inner_path != NULL)
    {
        outer_rows = outer_path->rows;
        inner_rows = inner_path->rows;
        num_blocks = ceil(outer_rows / vjoin_nestloop_batch_size);

        startup_cost = outer_path->startup_cost;
        run_cost = outer_path->total_cost - outer_path->startup_cost +
                   num_blocks * inner_path->total_cost +
                   outer_rows * inner_rows * cpu_operator_cost / simd_width * cost_factor;

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
        /* Append residual join qual expressions for executor evaluation.
         * For theta joins (nkeys == 0) every clause is residual; for equi
         * joins only non-key clauses remain. */
        cpath->custom_private = lappend(cpath->custom_private,
                                        (List *) copyObject(residual_clauses));
        cpath->methods = &vjoin_nestloop_path_methods;

        vjoin_apply_auto_tune(cpath, joinrel, T_NestPath, false);
        add_path(joinrel, &cpath->path);
    }

    /* --- Parallel path ---
     * Only INNER joins are supported. */
    if (joinrel->consider_parallel && outerrel->partial_pathlist != NIL &&
        jointype == JOIN_INNER)
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

            num_blocks = ceil(outer_rows / vjoin_nestloop_batch_size);
            parallel_workers = par_outer->parallel_workers;
            if (parallel_workers <= 0)
                parallel_workers = 1;

            startup_cost = par_outer->startup_cost;
            run_cost = par_outer->total_cost - par_outer->startup_cost +
                       num_blocks * par_inner->total_cost +
                       outer_rows * inner_rows * cpu_operator_cost / simd_width * cost_factor;

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

            vjoin_apply_auto_tune(cpath, joinrel, T_NestPath, true);
            add_partial_path(joinrel, &cpath->path);
        }
    }
}

/*
 * Try to create a VectorMergeJoin path.
 * Supports multi-key equijoin. If inputs aren't already sorted,
 * Sort nodes are added.
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
                    Oid *collations,
                    List *residual_clauses)
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
    double      cost_factor = vjoin_cost_factor * vjoin_merge_cost_factor;
    int         merge_keys_found = 0;
    int         i;

    /*
     * Vector merge join only benefits from SIMD on numeric types.
     * For text/generic keys the per-tuple fmgr comparison overhead exceeds
     * any batching advantage.  Reject if any key is non-numeric.
     */
    for (i = 0; i < nkeys; i++)
    {
        if (!vjoin_is_fast_type(key_types[i]))
            return;
    }

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
     * cost_factor scales the join comparison overhead only (not child scan costs).
     *
     * The output-cardinality charge (joinrel->rows * cpu_tuple_cost) makes the
     * model sensitive to join order — orders that materialize/forward large
     * intermediate results are penalized — matching the hash-join model so the
     * planner ranks join orders consistently across join methods.
     */
    startup_cost = outer_path->startup_cost + inner_path->startup_cost;
    run_cost = (outer_path->total_cost - outer_path->startup_cost) +
               (inner_path->total_cost - inner_path->startup_cost) +
               (outer_rows + inner_rows) * (cpu_tuple_cost + cpu_operator_cost) * cost_factor +
               joinrel->rows * cpu_tuple_cost;

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
    cpath->custom_private = lappend(cpath->custom_private,
                                    (List *) copyObject(residual_clauses));
    cpath->methods = &vjoin_merge_path_methods;

    vjoin_apply_auto_tune(cpath, joinrel, T_MergePath, false);
    add_path(joinrel, &cpath->path);

    /* --- Parallel merge path ---
     * Only INNER joins are supported. */
    {
        Path       *par_outer = NULL;
        Path       *par_inner = NULL;
        bool        par_is_parallel = false;
        int         par_workers;
        CustomPath *pcpath;
        Cost        par_startup, par_run;
        double      par_outer_rows;

        if (joinrel->consider_parallel && outerrel->partial_pathlist != NIL &&
            jointype == JOIN_INNER)
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
                 *
                 * Per-output-row cost is charged once below via the Gather
                 * tuple-queue overhead (per-worker rate), which doubles as the
                 * output-cardinality term for parallel paths.
                 */
                par_run = (par_outer->total_cost - par_outer->startup_cost) +
                          (par_inner->total_cost - par_inner->startup_cost) +
                          par_inner->rows * cpu_tuple_cost * 0.5 +
                          (par_outer_rows + par_inner->rows) *
                          (cpu_tuple_cost + cpu_operator_cost) * cost_factor;
            }
            else
            {
                par_run = (par_outer->total_cost - par_outer->startup_cost) +
                          (par_inner->total_cost - par_inner->startup_cost) +
                          (par_outer_rows + par_inner->rows) *
                          (cpu_tuple_cost + cpu_operator_cost) * cost_factor +
                          joinrel->rows * cpu_tuple_cost;
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

            vjoin_apply_auto_tune(pcpath, joinrel, T_MergePath, par_is_parallel);
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
    List       *consumed_rinfos = NIL;
    List       *residual_clauses = NIL;

    /* Chain to any previous hook first */
    if (prev_join_pathlist_hook)
        prev_join_pathlist_hook(root, joinrel, outerrel, innerrel,
                                jointype, extra);

    /*
     * During this cleanup branch we only execute INNER joins. LEFT join
     * analysis remains here so we can codify the future re-enable checks,
     * while still routing all non-INNER joins to core PostgreSQL.
     */
    if (jointype != JOIN_INNER && jointype != JOIN_LEFT)
        return;
    
    /* Master kill switch */
    if (!vjoin_enable)
        return;

    /* We need cheapest paths */
    if (outerrel->cheapest_total_path == NULL ||
        innerrel->cheapest_total_path == NULL)
        return;

    /*
     * Keep vector joins away from system-catalog and non-relation RTEs.
     * Metadata/introspection queries often use function RTEs (for example
     * unnest(...)) and catalog tables with plan shapes this executor does not
     * support safely yet.
     */
    
    if (!vjoin_relids_are_supported(root, outerrel->relids) ||
        !vjoin_relids_are_supported(root, innerrel->relids))
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
                                      outer_tl, inner_tl,
                                      &consumed_rinfos);

        list_free(outer_tl);
        list_free(inner_tl);
    }

    /*
     * Residual = join clauses the executor does not match as equi-keys; these
     * (and only these) must be re-checked as a qual at run time.  For pure
     * equi-key joins this is NIL, so the executor skips per-row ExecQual.
     */
    residual_clauses = vjoin_build_residual_clauses(extra->restrictlist,
                                                    consumed_rinfos);

    /* Hash join and merge join require at least one equality key */
    if (jointype == JOIN_LEFT)
    {
        /*
         * First re-enable point for LEFT joins:
         * 1. reject LEFT JOIN ... WHERE inner_col IS NULL anti-join shapes
         * 2. allow only a narrow subset that doesn't project nullable inner Vars
         */
        if (!vjoin_is_safe_left_join_subset(root, joinrel, innerrel, extra,
                                            nkeys))
            return;

        if (vjoin_enable_hashjoin)
            vjoin_try_hashjoin(root, joinrel, outerrel, innerrel, extra,
                               jointype,
                               nkeys, outer_keynos, inner_keynos, key_types,
                               hash_procs, eq_oprs, collations,
                               residual_clauses);

        return;
    }

    if (nkeys > 0)
    {
        if (vjoin_enable_hashjoin)
        {
            /* Try hash join with original inner/outer assignment */
            vjoin_try_hashjoin(root, joinrel, outerrel, innerrel, extra,
                               jointype,
                               nkeys, outer_keynos, inner_keynos, key_types,
                               hash_procs, eq_oprs, collations,
                               residual_clauses);

            /*
             * Also try with swapped sides: build hash table on the
             * original outer relation, probe with the original inner.
             * The optimizer calls set_join_pathlist_hook once per join pair
             * with a fixed outer/inner assignment.  hash_inner_and_outer()
             * tries both orderings internally for native hash joins, but
             * our hook only sees one ordering.  By trying the swap
             * explicitly we ensure the smaller relation can always be the
             * hash-build side (where memory constraints matter).
             */
            vjoin_try_hashjoin(root, joinrel, innerrel, outerrel, extra,
                               jointype,
                               nkeys, inner_keynos, outer_keynos, key_types,
                               hash_procs, eq_oprs, collations,
                               residual_clauses);
        }

        if (vjoin_enable_mergejoin)
            vjoin_try_mergejoin(root, joinrel, outerrel, innerrel, extra,
                                jointype,
                                nkeys, outer_keynos, inner_keynos, key_types,
                                hash_procs, eq_oprs, collations,
                                residual_clauses);
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
                      residual_clauses,
                      theta_strategy, theta_outer_keyno,
                      theta_inner_keyno, theta_keytype);
    }
}
