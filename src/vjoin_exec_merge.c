#include "postgres.h"
#include "access/htup_details.h"
#include "access/parallel.h"
#include "vjoin_compat.h"
#include "executor/executor.h"
#include "executor/tuptable.h"
#include "nodes/extensible.h"
#include "nodes/value.h"
#include "storage/shm_toc.h"
#include "utils/memutils.h"
#include "utils/datum.h"
#include "utils/typcache.h"
#include "utils/lsyscache.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "pg_vectorjoin.h"
#include "vjoin_state.h"
#include "vjoin_simd.h"

#define VMJ_INITIAL_GROUP_CAPACITY 64

/*
 * Compare keys: returns -1, 0, or +1.
 * NULLs are treated as greater than any non-NULL (sorted to end).
 */
static int
vmj_compare_keys(VectorMergeJoinState *state)
{
    int i;

    for (i = 0; i < state->num_keys; i++)
    {
        bool on = state->outer_null[i];
        bool in = state->inner_null[i];

        if (on && in)
            continue;  /* both NULL — equal for sort, but won't match */
        if (on)
            return 1;  /* NULL > non-NULL */
        if (in)
            return -1; /* non-NULL < NULL */

        switch (state->key_types[i])
        {
            case INT4OID:
            {
                int32 ov = DatumGetInt32(state->outer_key[i]);
                int32 iv = DatumGetInt32(state->inner_key[i]);
                if (ov < iv) return -1;
                if (ov > iv) return 1;
                break;
            }
            case INT8OID:
            {
                int64 ov = DatumGetInt64(state->outer_key[i]);
                int64 iv = DatumGetInt64(state->inner_key[i]);
                if (ov < iv) return -1;
                if (ov > iv) return 1;
                break;
            }
            case FLOAT8OID:
            {
                double ov = DatumGetFloat8(state->outer_key[i]);
                double iv = DatumGetFloat8(state->inner_key[i]);
                if (ov < iv) return -1;
                if (ov > iv) return 1;
                break;
            }
            default:
            {
                int32 r = DatumGetInt32(FunctionCall2Coll(
                              &state->cmp_finfo[i],
                              state->key_collations[i],
                              state->outer_key[i],
                              state->inner_key[i]));
                if (r != 0) return r;
                break;
            }
        }
    }
    return 0;
}

/*
 * Check if two key Datum arrays match exactly (for group collection).
 */
static bool
vmj_keys_equal(VectorMergeJoinState *state,
               Datum *key_a, bool *null_a,
               Datum *key_b, bool *null_b)
{
    int i;
    for (i = 0; i < state->num_keys; i++)
    {
        if (null_a[i] || null_b[i])
            return false;  /* NULLs never match */

        switch (state->key_types[i])
        {
            case INT4OID:
                if (DatumGetInt32(key_a[i]) != DatumGetInt32(key_b[i]))
                    return false;
                break;
            case INT8OID:
                if (DatumGetInt64(key_a[i]) != DatumGetInt64(key_b[i]))
                    return false;
                break;
            case FLOAT8OID:
                if (DatumGetFloat8(key_a[i]) != DatumGetFloat8(key_b[i]))
                    return false;
                break;
            default:
                if (!DatumGetBool(FunctionCall2Coll(
                        &state->eq_finfo[i],
                        state->key_collations[i],
                        key_a[i], key_b[i])))
                    return false;
                break;
        }
    }
    return true;
}

/*
 * Fetch the next tuple from outer, extract keys.
 * Does NOT copy MinimalTuple — just reads keys from child slot.
 * Returns false if no more tuples.
 */
static bool
vmj_fetch_outer(VectorMergeJoinState *state)
{
    TupleTableSlot *slot;
    int i;

    slot = ExecProcNode(state->outer_ps);
    if (TupIsNull(slot))
    {
        state->outer_done = true;
        state->outer_cur_slot = NULL;
        return false;
    }

    state->outer_cur_slot = slot;
    for (i = 0; i < state->num_keys; i++)
        state->outer_key[i] = slot_getattr(slot, state->outer_keynos[i],
                                            &state->outer_null[i]);
    return true;
}

/*
 * Fetch the next tuple from inner, extract keys.
 * Does NOT copy MinimalTuple — just reads keys from child slot.
 * Returns false if no more tuples.
 */
static bool
vmj_fetch_inner(VectorMergeJoinState *state)
{
    TupleTableSlot *slot;
    int i;

    slot = ExecProcNode(state->inner_ps);
    if (TupIsNull(slot))
    {
        state->inner_done = true;
        state->inner_cur_slot = NULL;
        return false;
    }

    state->inner_cur_slot = slot;
    for (i = 0; i < state->num_keys; i++)
        state->inner_key[i] = slot_getattr(slot, state->inner_keynos[i],
                                            &state->inner_null[i]);
    return true;
}

/*
 * Collect all tuples with the same key into the specified group array.
 * first_mt is the already-saved first tuple of the group.
 * If *_multi is true, the current child slot has the second tuple
 * (with the same key); we copy it and keep fetching.
 * After return, the corresponding key[] holds the NEXT key (or *_done=true).
 */
static void
vmj_collect_group(VectorMergeJoinState *state, bool is_outer,
                  MinimalTuple first_mt)
{
    MemoryContext oldctx;
    MinimalTuple **group;
    int           *count;
    int           *capacity;
    bool           multi;

    if (is_outer)
    {
        group    = &state->outer_group;
        count    = &state->outer_group_count;
        capacity = &state->outer_group_capacity;
        multi    = state->outer_multi;
    }
    else
    {
        group    = &state->inner_group;
        count    = &state->inner_group_count;
        capacity = &state->inner_group_capacity;
        multi    = state->inner_multi;
    }

    *count = 0;
    oldctx = MemoryContextSwitchTo(state->match_ctx);

    /* First tuple — already saved, copy into match_ctx */
    if (*count >= *capacity)
    {
        *capacity *= 2;
        *group = repalloc(*group, sizeof(MinimalTuple) * *capacity);
    }
    (*group)[(*count)++] = vjoin_heap_copy_minimal_tuple(first_mt);

    MemoryContextSwitchTo(oldctx);

    if (multi)
    {
        TupleTableSlot *slot = is_outer ? state->outer_cur_slot
                                        : state->inner_cur_slot;

        oldctx = MemoryContextSwitchTo(state->match_ctx);
        if (*count >= *capacity)
        {
            *capacity *= 2;
            *group = repalloc(*group, sizeof(MinimalTuple) * *capacity);
        }
        (*group)[(*count)++] = ExecCopySlotMinimalTuple(slot);
        MemoryContextSwitchTo(oldctx);

        /* Keep fetching while key matches */
        while (is_outer ? vmj_fetch_outer(state) : vmj_fetch_inner(state))
        {
            Datum *cur_key  = is_outer ? state->outer_key  : state->inner_key;
            bool  *cur_null = is_outer ? state->outer_null  : state->inner_null;

            if (!vmj_keys_equal(state, state->match_key, state->match_null,
                                cur_key, cur_null))
                break;

            slot = is_outer ? state->outer_cur_slot : state->inner_cur_slot;
            oldctx = MemoryContextSwitchTo(state->match_ctx);
            if (*count >= *capacity)
            {
                *capacity *= 2;
                *group = repalloc(*group, sizeof(MinimalTuple) * *capacity);
            }
            (*group)[(*count)++] = ExecCopySlotMinimalTuple(slot);
            MemoryContextSwitchTo(oldctx);
        }
    }
}

/*
 * Fill the scan slot from matched outer + inner tuples.
 */
static TupleTableSlot *
vmj_form_result(VectorMergeJoinState *state,
                MinimalTuple outer_mt, MinimalTuple inner_mt)
{
    TupleTableSlot *scan_slot = state->css.ss.ss_ScanTupleSlot;

    ExecStoreMinimalTuple(outer_mt, state->outer_slot, false);
    slot_getallattrs(state->outer_slot);

    ExecStoreMinimalTuple(inner_mt, state->inner_slot, false);
    slot_getallattrs(state->inner_slot);

    ExecClearTuple(scan_slot);

    memcpy(scan_slot->tts_values,
           state->outer_slot->tts_values,
           state->num_outer_attrs * sizeof(Datum));
    memcpy(scan_slot->tts_isnull,
           state->outer_slot->tts_isnull,
           state->num_outer_attrs * sizeof(bool));

    memcpy(scan_slot->tts_values + state->num_outer_attrs,
           state->inner_slot->tts_values,
           state->num_inner_attrs * sizeof(Datum));
    memcpy(scan_slot->tts_isnull + state->num_outer_attrs,
           state->inner_slot->tts_isnull,
           state->num_inner_attrs * sizeof(bool));

    ExecStoreVirtualTuple(scan_slot);
    return scan_slot;
}

/*
 * Form result using current outer child slot + NULL inner (LEFT/FULL).
 * Used in tuple-at-a-time path to avoid MinimalTuple copy.
 */
static TupleTableSlot *
vmj_form_result_left_slot(VectorMergeJoinState *state)
{
    TupleTableSlot *scan_slot = state->css.ss.ss_ScanTupleSlot;
    TupleTableSlot *os = state->outer_cur_slot;

    slot_getallattrs(os);
    ExecClearTuple(scan_slot);
    memcpy(scan_slot->tts_values, os->tts_values,
           state->num_outer_attrs * sizeof(Datum));
    memcpy(scan_slot->tts_isnull, os->tts_isnull,
           state->num_outer_attrs * sizeof(bool));
    memset(scan_slot->tts_values + state->num_outer_attrs,
           0, state->num_inner_attrs * sizeof(Datum));
    memset(scan_slot->tts_isnull + state->num_outer_attrs,
           1, state->num_inner_attrs * sizeof(bool));
    ExecStoreVirtualTuple(scan_slot);
    return scan_slot;
}

/*
 * Form result using current inner child slot + NULL outer (RIGHT/FULL).
 */
static TupleTableSlot *
vmj_form_result_right_slot(VectorMergeJoinState *state)
{
    TupleTableSlot *scan_slot = state->css.ss.ss_ScanTupleSlot;
    TupleTableSlot *is = state->inner_cur_slot;

    slot_getallattrs(is);
    ExecClearTuple(scan_slot);
    memset(scan_slot->tts_values, 0,
           state->num_outer_attrs * sizeof(Datum));
    memset(scan_slot->tts_isnull, 1,
           state->num_outer_attrs * sizeof(bool));
    memcpy(scan_slot->tts_values + state->num_outer_attrs,
           is->tts_values,
           state->num_inner_attrs * sizeof(Datum));
    memcpy(scan_slot->tts_isnull + state->num_outer_attrs,
           is->tts_isnull,
           state->num_inner_attrs * sizeof(bool));
    ExecStoreVirtualTuple(scan_slot);
    return scan_slot;
}

/* ================================================================
 *  Block merge join functions (vectorized batch mode)
 *
 *  When all output columns are pass-by-value, we pre-fetch blocks of
 *  tuples into contiguous Datum arrays and run the merge on arrays.
 *  This gives:
 *    - tight fill loop (better branch prediction & instruction cache)
 *    - binary-search advance for non-matching key ranges
 *    - zero-copy result formation from pre-deformed arrays
 * ================================================================ */

/* ----------------------------------------------------------------
 * Shared inner materialization for parallel merge join.
 *
 * Leader scans the full sorted inner, deforms all tuples into flat
 * Datum/isnull arrays + extracted keys, and stores them in DSA.
 * Workers attach and get direct pointers to the shared arrays.
 * ---------------------------------------------------------------- */
static void
vmj_materialize_shared_inner(VectorMergeJoinState *state)
{
    VJoinMergeParallelState *pstate = state->pstate;
    dsa_area   *dsa = state->dsa;

    if (state->is_leader)
    {
        /* Leader: scan full inner, store in DSA */
        int         nattrs = state->num_inner_attrs;
        int         nkeys = state->num_keys;
        int         capacity = 4096;
        int         count = 0;
        Datum      *vals;
        bool       *nulls;
        Datum      *keys;
        dsa_pointer vals_dp, nulls_dp, keys_dp;

        /* Initial allocation in DSA */
        vals_dp = dsa_allocate(dsa, sizeof(Datum) * capacity * nattrs);
        nulls_dp = dsa_allocate(dsa, sizeof(bool) * capacity * nattrs);
        keys_dp = dsa_allocate(dsa, sizeof(Datum) * capacity * nkeys);
        vals = (Datum *) dsa_get_address(dsa, vals_dp);
        nulls = (bool *) dsa_get_address(dsa, nulls_dp);
        keys = (Datum *) dsa_get_address(dsa, keys_dp);

        for (;;)
        {
            TupleTableSlot *slot = ExecProcNode(state->inner_ps);
            int k;
            bool key_null = false;

            if (TupIsNull(slot))
                break;

            /* Grow if needed */
            if (count >= capacity)
            {
                int new_cap;
                dsa_pointer nv, nn, nk;

                if (capacity > INT_MAX / 2)
                    ereport(ERROR,
                            (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                             errmsg("pg_vectorjoin: merge inner materialization capacity overflow")));
                new_cap = capacity * 2;

                nv = dsa_allocate(dsa, sizeof(Datum) * new_cap * nattrs);
                nn = dsa_allocate(dsa, sizeof(bool) * new_cap * nattrs);
                nk = dsa_allocate(dsa, sizeof(Datum) * new_cap * nkeys);

                memcpy(dsa_get_address(dsa, nv), vals,
                       sizeof(Datum) * count * nattrs);
                memcpy(dsa_get_address(dsa, nn), nulls,
                       sizeof(bool) * count * nattrs);
                memcpy(dsa_get_address(dsa, nk), keys,
                       sizeof(Datum) * count * nkeys);

                dsa_free(dsa, vals_dp);
                dsa_free(dsa, nulls_dp);
                dsa_free(dsa, keys_dp);

                vals_dp = nv;
                nulls_dp = nn;
                keys_dp = nk;
                vals = (Datum *) dsa_get_address(dsa, nv);
                nulls = (bool *) dsa_get_address(dsa, nn);
                keys = (Datum *) dsa_get_address(dsa, nk);
                capacity = new_cap;
            }

            /* Extract keys, skip NULLs */
            for (k = 0; k < nkeys; k++)
            {
                bool isnl;
                keys[count * nkeys + k] = slot_getattr(slot,
                                                       state->inner_keynos[k],
                                                       &isnl);
                if (isnl)
                {
                    key_null = true;
                    break;
                }
            }
            if (key_null)
                continue;

            /* Deform all attributes */
            slot_getallattrs(slot);
            memcpy(&vals[count * nattrs], slot->tts_values,
                   nattrs * sizeof(Datum));
            memcpy(&nulls[count * nattrs], slot->tts_isnull,
                   nattrs * sizeof(bool));
            count++;
        }

        /* Publish to shared state */
        pstate->inner_values_dp = vals_dp;
        pstate->inner_isnull_dp = nulls_dp;
        pstate->inner_keys_dp = keys_dp;
        pg_write_barrier();
        pstate->inner_count = count;
    }

    /* All participants wait here */
    BarrierArriveAndWait(&pstate->barrier, 0);

    /* Everyone attaches to the shared inner arrays */
    pg_read_barrier();
    state->shared_inner_count = pstate->inner_count;
    state->shared_inner_pos = 0;
    if (pstate->inner_count > 0)
    {
        state->shared_inner_values = (Datum *) dsa_get_address(
            dsa, pstate->inner_values_dp);
        state->shared_inner_isnull = (bool *) dsa_get_address(
            dsa, pstate->inner_isnull_dp);
        state->shared_inner_keys = (Datum *) dsa_get_address(
            dsa, pstate->inner_keys_dp);
    }
    else
    {
        state->shared_inner_values = NULL;
        state->shared_inner_isnull = NULL;
        state->shared_inner_keys = NULL;
    }
}

/*
 * Fill inner batch from shared materialized buffer (parallel merge).
 * Returns true if any tuples were loaded.
 */
static bool
vmj_batch_fill_inner_shared(VectorMergeJoinState *state)
{
    int nkeys = state->num_keys;
    int ncols = state->num_inner_attrs;
    int batch_size = state->batch_size;
    int remaining, avail, tocopy, off_src, off_dst;

    /* Shift remaining entries to front */
    remaining = state->ib_count - state->ib_pos;
    if (remaining > 0 && state->ib_pos > 0)
    {
        memmove(state->ib_keys, &state->ib_keys[state->ib_pos * nkeys],
                remaining * nkeys * sizeof(Datum));
        memmove(state->ib_values, &state->ib_values[state->ib_pos * ncols],
                remaining * ncols * sizeof(Datum));
        memmove(state->ib_isnull, &state->ib_isnull[state->ib_pos * ncols],
                remaining * ncols * sizeof(bool));
    }
    state->ib_count = remaining;
    state->ib_pos = 0;

    /* Copy from shared buffer */
    avail = state->shared_inner_count - state->shared_inner_pos;
    tocopy = batch_size - remaining;
    if (tocopy > avail)
        tocopy = avail;

    if (tocopy > 0)
    {
        off_src = state->shared_inner_pos;
        off_dst = remaining;

        /* Prefetch upcoming shared data */
        if (tocopy > 8)
        {
            __builtin_prefetch(&state->shared_inner_keys[(off_src + 8) * nkeys], 0, 1);
            __builtin_prefetch(&state->shared_inner_values[(off_src + 8) * ncols], 0, 1);
        }

        memcpy(&state->ib_keys[off_dst * nkeys],
               &state->shared_inner_keys[off_src * nkeys],
               tocopy * nkeys * sizeof(Datum));
        memcpy(&state->ib_values[off_dst * ncols],
               &state->shared_inner_values[off_src * ncols],
               tocopy * ncols * sizeof(Datum));
        memcpy(&state->ib_isnull[off_dst * ncols],
               &state->shared_inner_isnull[off_src * ncols],
               tocopy * ncols * sizeof(bool));

        state->ib_count = remaining + tocopy;
        state->shared_inner_pos += tocopy;
    }

    if (state->shared_inner_pos >= state->shared_inner_count)
        state->ib_exhausted = true;

    return state->ib_count > 0;
}

/*
 * Fill (or refill) a block from a child plan.
 * Shifts remaining entries (from *pos to *count) to the front,
 * then fills the rest up to batch_size.
 */
static void
vmj_batch_fill_side(VectorMergeJoinState *state, bool is_outer)
{
    Datum      *keys;
    Datum      *values;
    bool       *isnull;
    int        *count_p;
    int        *pos_p;
    bool       *exhausted_p;
    int         ncols;
    PlanState  *child;
    AttrNumber *keynos;
    int         nkeys = state->num_keys;
    int         batch_size = state->batch_size;
    int         remaining;

    if (is_outer)
    {
        keys        = state->ob_keys;
        values      = state->ob_values;
        isnull      = state->ob_isnull;
        count_p     = &state->ob_count;
        pos_p       = &state->ob_pos;
        exhausted_p = &state->ob_exhausted;
        ncols       = state->num_outer_attrs;
        child       = state->outer_ps;
        keynos      = state->outer_keynos;
    }
    else
    {
        keys        = state->ib_keys;
        values      = state->ib_values;
        isnull      = state->ib_isnull;
        count_p     = &state->ib_count;
        pos_p       = &state->ib_pos;
        exhausted_p = &state->ib_exhausted;
        ncols       = state->num_inner_attrs;
        child       = state->inner_ps;
        keynos      = state->inner_keynos;
    }

    /* Shift remaining entries to front */
    remaining = *count_p - *pos_p;
    if (remaining > 0 && *pos_p > 0)
    {
        memmove(keys, &keys[*pos_p * nkeys],
                remaining * nkeys * sizeof(Datum));
        memmove(values, &values[*pos_p * ncols],
                remaining * ncols * sizeof(Datum));
        memmove(isnull, &isnull[*pos_p * ncols],
                remaining * ncols * sizeof(bool));
        /* Shift ob_matched for outer side (LEFT JOIN) */
        if (is_outer && state->ob_matched)
        {
            memmove(state->ob_matched, &state->ob_matched[*pos_p],
                    remaining * sizeof(bool));
        }
    }
    *count_p = remaining;
    *pos_p = 0;

    /* Reset ob_matched for new entries + batch_left_pos */
    if (is_outer && state->ob_matched)
    {
        memset(&state->ob_matched[remaining], 0,
               (batch_size - remaining) * sizeof(bool));
        state->batch_left_pos = 0;
    }

    /* Fill the rest from child plan */
    while (*count_p < batch_size)
    {
        TupleTableSlot *slot;
        bool            key_null;
        int             idx, k;

        CHECK_FOR_INTERRUPTS();

        slot = ExecProcNode(child);
        if (TupIsNull(slot))
        {
            *exhausted_p = true;
            break;
        }

        /* Extract all keys — skip if any key is NULL */
        idx = *count_p;
        key_null = false;
        for (k = 0; k < nkeys; k++)
        {
            bool isnl;
            keys[idx * nkeys + k] = slot_getattr(slot, keynos[k], &isnl);
            if (isnl)
            {
                key_null = true;
                break;
            }
        }
        if (key_null)
            continue;

        /* Deform all attributes into columnar arrays */
        slot_getallattrs(slot);
        {
            int off = idx * ncols;
            memcpy(&values[off], slot->tts_values, ncols * sizeof(Datum));
            memcpy(&isnull[off], slot->tts_isnull, ncols * sizeof(bool));
        }

        (*count_p)++;
    }
}

/*
 * Wrapper: fill inner batch — uses shared buffer in parallel mode,
 * otherwise falls back to normal child scan.
 */
static inline void
vmj_fill_inner(VectorMergeJoinState *state)
{
    if (state->is_parallel)
        vmj_batch_fill_inner_shared(state);
    else
        vmj_batch_fill_side(state, false);
}

/*
 * Inline comparison for a single key pair.
 * Called per-key in multi-key merge loops.
 */
static inline __attribute__((always_inline)) int
vmj_batch_compare_key(Datum a, Datum b, Oid keytype,
                      FmgrInfo *cmp_finfo, Oid collation)
{
    switch (keytype)
    {
        case INT4OID:
        {
            int32 av = DatumGetInt32(a), bv = DatumGetInt32(b);
            return (av < bv) ? -1 : (av > bv) ? 1 : 0;
        }
        case INT8OID:
        {
            int64 av = DatumGetInt64(a), bv = DatumGetInt64(b);
            return (av < bv) ? -1 : (av > bv) ? 1 : 0;
        }
        case FLOAT8OID:
        {
            double av = DatumGetFloat8(a), bv = DatumGetFloat8(b);
            return (av < bv) ? -1 : (av > bv) ? 1 : 0;
        }
        default:
            return DatumGetInt32(FunctionCall2Coll(cmp_finfo, collation,
                                                   a, b));
    }
}

/*
 * INT4-specialized batch merge with binary-search advance.
 * For multi-key joins where the first key is INT4, the primary key is
 * stored at keys[row * nkeys + 0].  Secondary keys are compared via
 * the generic path when primary keys are equal.
 *
 * Pre-builds combined result rows into batch_result_values/isnull
 * for zero-copy emit in VMJ_BATCH_EMIT.
 */
static void
vmj_batch_do_merge_int4(VectorMergeJoinState *state)
{
    int     oi = state->ob_pos;
    int     ii = state->ib_pos;
    int     nr = 0;
    int     ob_count = state->ob_count;
    int     ib_count = state->ib_count;
    Datum  *ok = state->ob_keys;
    Datum  *ik = state->ib_keys;
    int     nk = state->num_keys;
    VJoinMatch *results = state->batch_results;
    int     max_results = state->batch_result_capacity;
    bool   *ob_matched = state->ob_matched;     /* NULL for INNER */
    /* Pre-built result arrays */
    Datum  *rv = state->batch_result_values;
    bool   *ri = state->batch_result_isnull;
    Datum  *ob_vals = state->ob_values;
    bool   *ob_nulls = state->ob_isnull;
    Datum  *ib_vals = state->ib_values;
    bool   *ib_nulls = state->ib_isnull;
    int     noa = state->num_outer_attrs;
    int     nia = state->num_inner_attrs;
    int     total = state->total_attrs;

    /* Resume interrupted cross-product from previous call */
    if (state->batch_cp_oi >= 0)
    {
        int o  = state->batch_cp_oi;
        int i  = state->batch_cp_ii;
        int oe = state->batch_cp_oe;
        int ie = state->batch_cp_ie;
        int ii_start = state->batch_cp_ii_start;

        while (o < oe && nr < max_results)
        {
            /* Multi-key resume: check secondary keys */
            if (nk > 1)
            {
                bool match = true;
                int k;
                for (k = 1; k < nk; k++)
                {
                    if (DatumGetInt32(ok[o * nk + k]) !=
                        DatumGetInt32(ik[i * nk + k]))
                    {
                        match = false;
                        break;
                    }
                }
                if (!match)
                {
                    if (++i >= ie) { i = ii_start; o++; }
                    continue;
                }
            }
            if (ob_matched)
                ob_matched[o] = true;
            results[nr].outer_idx = o;
            results[nr].inner_idx = i;
            {
                int roff = nr * total;
                memcpy(&rv[roff], &ob_vals[o * noa], noa * sizeof(Datum));
                memcpy(&ri[roff], &ob_nulls[o * noa], noa * sizeof(bool));
                memcpy(&rv[roff + noa], &ib_vals[i * nia], nia * sizeof(Datum));
                memcpy(&ri[roff + noa], &ib_nulls[i * nia], nia * sizeof(bool));
            }
            nr++;
            if (++i >= ie)
            {
                i = ii_start;
                o++;
            }
        }

        if (o < oe)
        {
            /* Still not finished — update saved position */
            state->batch_cp_oi = o;
            state->batch_cp_ii = i;
            state->batch_result_count = nr;
            state->batch_result_pos = 0;
            return;
        }

        /* Completed interrupted cross-product */
        state->batch_cp_oi = -1;
        oi = oe;
        ii = ie;
    }

    while (oi < ob_count && ii < ib_count && nr < max_results)
    {
        int32   outer_val = DatumGetInt32(ok[oi * nk]);
        int32   inner_val = DatumGetInt32(ik[ii * nk]);

        CHECK_FOR_INTERRUPTS();

        if (outer_val < inner_val)
        {
            /* Binary search: first oi where ok[oi*nk] >= inner_val */
            int lo = oi + 1, hi = ob_count;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                if (DatumGetInt32(ok[mid * nk]) < inner_val)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            oi = lo;
        }
        else if (outer_val > inner_val)
        {
            /* Binary search: first ii where ik[ii*nk] >= outer_val */
            int lo = ii + 1, hi = ib_count;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                if (DatumGetInt32(ik[mid * nk]) < outer_val)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            ii = lo;
        }
        else
        {
            /* Primary keys match — find group boundaries on primary key */
            int32   match_val = outer_val;
            int     oe = oi + 1;
            int     ie = ii + 1;
            int     o, i;

            while (oe < ob_count && DatumGetInt32(ok[oe * nk]) == match_val)
                oe++;
            while (ie < ib_count && DatumGetInt32(ik[ie * nk]) == match_val)
                ie++;

            /* Stop if group might span a block boundary */
            if ((oe == ob_count && !state->ob_exhausted) ||
                (ie == ib_count && !state->ib_exhausted))
                break;

            /* Emit cross-product with flat loop for clean save/resume.
             * For multi-key, only emit pairs where ALL keys match. */
            o = oi;
            i = ii;
            while (o < oe && nr < max_results)
            {
                /* Multi-key: check secondary keys */
                if (nk > 1)
                {
                    bool match = true;
                    int k;
                    for (k = 1; k < nk; k++)
                    {
                        if (DatumGetInt32(ok[o * nk + k]) !=
                            DatumGetInt32(ik[i * nk + k]))
                        {
                            match = false;
                            break;
                        }
                    }
                    if (!match)
                    {
                        if (++i >= ie)
                        {
                            i = ii;
                            o++;
                        }
                        continue;
                    }
                }
                if (ob_matched)
                    ob_matched[o] = true;
                results[nr].outer_idx = o;
                results[nr].inner_idx = i;
                {
                    int roff = nr * total;
                    memcpy(&rv[roff], &ob_vals[o * noa], noa * sizeof(Datum));
                    memcpy(&ri[roff], &ob_nulls[o * noa], noa * sizeof(bool));
                    memcpy(&rv[roff + noa], &ib_vals[i * nia], nia * sizeof(Datum));
                    memcpy(&ri[roff + noa], &ib_nulls[i * nia], nia * sizeof(bool));
                }
                nr++;
                if (++i >= ie)
                {
                    i = ii;
                    o++;
                }
            }

            if (o < oe)
            {
                /* Cross-product interrupted — save for resumption */
                state->batch_cp_oi = o;
                state->batch_cp_ii = i;
                state->batch_cp_oe = oe;
                state->batch_cp_ie = ie;
                state->batch_cp_ii_start = ii;
                state->ob_pos = oi;
                state->ib_pos = ii;
                state->batch_result_count = nr;
                state->batch_result_pos = 0;
                return;
            }

            oi = oe;
            ii = ie;
        }
    }

    state->ob_pos = oi;
    state->ib_pos = ii;
    state->batch_result_count = nr;
    state->batch_result_pos = 0;
}

/*
 * Generic batch merge for INT8, FLOAT8, and multi-key joins.
 * Uses binary-search advance on primary key for O(log N) skip.
 * Pre-builds combined result rows into batch_result_values/isnull
 * for zero-copy emit.
 */
static void
vmj_batch_do_merge_generic(VectorMergeJoinState *state)
{
    int     oi = state->ob_pos;
    int     ii = state->ib_pos;
    int     nr = 0;
    Datum  *ok = state->ob_keys;
    Datum  *ik = state->ib_keys;
    int     nk = state->num_keys;
    VJoinMatch *results = state->batch_results;
    int     max_results = state->batch_result_capacity;
    bool   *ob_matched = state->ob_matched;
    /* Primary key comparison info (for binary search) */
    Oid     pk_type = state->key_types[0];
    FmgrInfo *pk_cmp = &state->cmp_finfo[0];
    Oid     pk_coll = state->key_collations[0];
    /* Pre-built result arrays */
    Datum  *rv = state->batch_result_values;
    bool   *ri = state->batch_result_isnull;
    Datum  *ob_vals = state->ob_values;
    bool   *ob_nulls = state->ob_isnull;
    Datum  *ib_vals = state->ib_values;
    bool   *ib_nulls = state->ib_isnull;
    int     noa = state->num_outer_attrs;
    int     nia = state->num_inner_attrs;
    int     total = state->total_attrs;

    /* Resume interrupted cross-product from previous call (generic) */
    if (state->batch_cp_oi >= 0)
    {
        int o  = state->batch_cp_oi;
        int i  = state->batch_cp_ii;
        int oe = state->batch_cp_oe;
        int ie = state->batch_cp_ie;
        int ii_start = state->batch_cp_ii_start;

        while (o < oe && nr < max_results)
        {
            /* Multi-key resume: check all keys */
            if (nk > 1)
            {
                bool match = true;
                int kk;
                for (kk = 0; kk < nk; kk++)
                {
                    if (vmj_batch_compare_key(ok[o * nk + kk],
                                              ik[i * nk + kk],
                                              state->key_types[kk],
                                              &state->cmp_finfo[kk],
                                              state->key_collations[kk]) != 0)
                    {
                        match = false;
                        break;
                    }
                }
                if (!match)
                {
                    if (++i >= ie) { i = ii_start; o++; }
                    continue;
                }
            }
            if (ob_matched)
                ob_matched[o] = true;
            results[nr].outer_idx = o;
            results[nr].inner_idx = i;
            {
                int roff = nr * total;
                memcpy(&rv[roff], &ob_vals[o * noa], noa * sizeof(Datum));
                memcpy(&ri[roff], &ob_nulls[o * noa], noa * sizeof(bool));
                memcpy(&rv[roff + noa], &ib_vals[i * nia], nia * sizeof(Datum));
                memcpy(&ri[roff + noa], &ib_nulls[i * nia], nia * sizeof(bool));
            }
            nr++;
            if (++i >= ie)
            {
                i = ii_start;
                o++;
            }
        }

        if (o < oe)
        {
            state->batch_cp_oi = o;
            state->batch_cp_ii = i;
            state->batch_result_count = nr;
            state->batch_result_pos = 0;
            return;
        }

        state->batch_cp_oi = -1;
        oi = oe;
        ii = ie;
    }

    while (oi < state->ob_count && ii < state->ib_count && nr < max_results)
    {
        /* Compare primary key for advance decision */
        int cmp_pk = vmj_batch_compare_key(ok[oi * nk], ik[ii * nk],
                                           pk_type, pk_cmp, pk_coll);

        CHECK_FOR_INTERRUPTS();

        if (cmp_pk < 0)
        {
            /* Binary search: first oi where pk >= inner pk */
            Datum target = ik[ii * nk];
            int lo = oi + 1, hi = state->ob_count;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                if (vmj_batch_compare_key(ok[mid * nk], target,
                                          pk_type, pk_cmp, pk_coll) < 0)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            oi = lo;
        }
        else if (cmp_pk > 0)
        {
            /* Binary search: first ii where pk >= outer pk */
            Datum target = ok[oi * nk];
            int lo = ii + 1, hi = state->ib_count;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                if (vmj_batch_compare_key(ik[mid * nk], target,
                                          pk_type, pk_cmp, pk_coll) < 0)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            ii = lo;
        }
        else
        {
            /* Primary key match — find group boundaries on primary key */
            Datum   match_pk = ok[oi * nk];
            int     oe = oi + 1;
            int     ie = ii + 1;
            int     o, i;

            while (oe < state->ob_count &&
                   vmj_batch_compare_key(ok[oe * nk], match_pk, pk_type,
                                         pk_cmp, pk_coll) == 0)
                oe++;
            while (ie < state->ib_count &&
                   vmj_batch_compare_key(ik[ie * nk], match_pk, pk_type,
                                         pk_cmp, pk_coll) == 0)
                ie++;

            if ((oe == state->ob_count && !state->ob_exhausted) ||
                (ie == state->ib_count && !state->ib_exhausted))
                break;

            /* Flat cross-product with save/resume support.
             * For multi-key, only emit pairs where ALL keys match. */
            o = oi;
            i = ii;
            while (o < oe && nr < max_results)
            {
                /* Multi-key: check all keys */
                if (nk > 1)
                {
                    bool match = true;
                    int kk;
                    for (kk = 0; kk < nk; kk++)
                    {
                        if (vmj_batch_compare_key(ok[o * nk + kk],
                                                  ik[i * nk + kk],
                                                  state->key_types[kk],
                                                  &state->cmp_finfo[kk],
                                                  state->key_collations[kk]) != 0)
                        {
                            match = false;
                            break;
                        }
                    }
                    if (!match)
                    {
                        if (++i >= ie)
                        {
                            i = ii;
                            o++;
                        }
                        continue;
                    }
                }
                if (ob_matched)
                    ob_matched[o] = true;
                results[nr].outer_idx = o;
                results[nr].inner_idx = i;
                {
                    int roff = nr * total;
                    memcpy(&rv[roff], &ob_vals[o * noa], noa * sizeof(Datum));
                    memcpy(&ri[roff], &ob_nulls[o * noa], noa * sizeof(bool));
                    memcpy(&rv[roff + noa], &ib_vals[i * nia], nia * sizeof(Datum));
                    memcpy(&ri[roff + noa], &ib_nulls[i * nia], nia * sizeof(bool));
                }
                nr++;
                if (++i >= ie)
                {
                    i = ii;
                    o++;
                }
            }

            if (o < oe)
            {
                state->batch_cp_oi = o;
                state->batch_cp_ii = i;
                state->batch_cp_oe = oe;
                state->batch_cp_ie = ie;
                state->batch_cp_ii_start = ii;
                state->ob_pos = oi;
                state->ib_pos = ii;
                state->batch_result_count = nr;
                state->batch_result_pos = 0;
                return;
            }

            oi = oe;
            ii = ie;
        }
    }

    state->ob_pos = oi;
    state->ib_pos = ii;
    state->batch_result_count = nr;
    state->batch_result_pos = 0;
}

/*
 * Dispatch to specialized or generic batch merge.
 */
static void
vmj_batch_do_merge(VectorMergeJoinState *state)
{
    /* Use INT4 specialization when ALL keys are INT4 */
    if (state->key_types[0] == INT4OID)
    {
        int k;
        bool all_int4 = true;
        for (k = 1; k < state->num_keys; k++)
        {
            if (state->key_types[k] != INT4OID)
            {
                all_int4 = false;
                break;
            }
        }
        if (all_int4)
        {
            vmj_batch_do_merge_int4(state);
            return;
        }
    }
    vmj_batch_do_merge_generic(state);
}

/*
 * Form result with outer from batch arrays + NULL inner (LEFT JOIN).
 */
static TupleTableSlot *
vmj_batch_form_result_left(VectorMergeJoinState *state, int outer_idx)
{
    TupleTableSlot *scan_slot = state->css.ss.ss_ScanTupleSlot;
    int outer_off = outer_idx * state->num_outer_attrs;

    ExecClearTuple(scan_slot);

    memcpy(scan_slot->tts_values,
           &state->ob_values[outer_off],
           state->num_outer_attrs * sizeof(Datum));
    memcpy(scan_slot->tts_isnull,
           &state->ob_isnull[outer_off],
           state->num_outer_attrs * sizeof(bool));

    memset(scan_slot->tts_values + state->num_outer_attrs,
           0, state->num_inner_attrs * sizeof(Datum));
    memset(scan_slot->tts_isnull + state->num_outer_attrs,
           1, state->num_inner_attrs * sizeof(bool));

    ExecStoreVirtualTuple(scan_slot);
    return scan_slot;
}

/* ----------------------------------------------------------------
 * Executor callbacks
 * ---------------------------------------------------------------- */

void
vjoin_merge_begin(CustomScanState *node, EState *estate, int eflags)
{
    VectorMergeJoinState *state = (VectorMergeJoinState *) node;
    CustomScan *cscan = (CustomScan *) node->ss.ps.plan;
    PlanState  *outer_ps,
               *inner_ps;
    TupleDesc   outer_desc,
                inner_desc;

    /* Initialize child plans */
    outer_ps = ExecInitNode(linitial(cscan->custom_plans), estate, eflags);
    inner_ps = ExecInitNode(lsecond(cscan->custom_plans), estate, eflags);
    node->custom_ps = list_make2(outer_ps, inner_ps);
    state->outer_ps = outer_ps;
    state->inner_ps = inner_ps;

    outer_desc = ExecGetResultType(outer_ps);
    inner_desc = ExecGetResultType(inner_ps);
    state->num_outer_attrs = outer_desc->natts;
    state->num_inner_attrs = inner_desc->natts;
    state->outer_desc = outer_desc;
    state->inner_desc = inner_desc;

    /* Deserialize key info (jointype is first element) */
    vjoin_deserialize_keys(cscan->custom_private,
                           &state->jointype,
                           &state->num_keys,
                           state->outer_keynos,
                           state->inner_keynos,
                           state->key_types,
                           NULL,
                           state->eq_funcs,
                           state->key_collations);

    /* Set up generic comparison/equality functions and type metadata */
    {
        int i;
        for (i = 0; i < state->num_keys; i++)
        {
            get_typlenbyval(state->key_types[i],
                            &state->key_typlen[i],
                            &state->key_byval[i]);

            if (!vjoin_is_fast_type(state->key_types[i]))
            {
                TypeCacheEntry *typentry;

                /* btree comparison function for ordering */
                typentry = lookup_type_cache(state->key_types[i], TYPECACHE_CMP_PROC);
                if (OidIsValid(typentry->cmp_proc))
                    fmgr_info(typentry->cmp_proc, &state->cmp_finfo[i]);

                /* equality function */
                if (OidIsValid(state->eq_funcs[i]))
                    fmgr_info(get_opcode(state->eq_funcs[i]), &state->eq_finfo[i]);
            }
        }
    }

    /* Memory context for group buffers — reset between match groups */
    state->match_ctx = AllocSetContextCreate(CurrentMemoryContext,
                                             "VectorMergeJoin match",
                                             ALLOCSET_DEFAULT_SIZES);

    /* Allocate group arrays */
    state->outer_group_capacity = VMJ_INITIAL_GROUP_CAPACITY;
    state->inner_group_capacity = VMJ_INITIAL_GROUP_CAPACITY;
    state->outer_group = palloc(sizeof(MinimalTuple) *
                                state->outer_group_capacity);
    state->inner_group = palloc(sizeof(MinimalTuple) *
                                state->inner_group_capacity);
    state->outer_group_count = 0;
    state->inner_group_count = 0;

    /* Temp slots for result construction */
    state->outer_slot = MakeSingleTupleTableSlot(outer_desc,
                                                 &TTSOpsMinimalTuple);
    state->inner_slot = MakeSingleTupleTableSlot(inner_desc,
                                                 &TTSOpsMinimalTuple);

    state->outer_done = false;
    state->inner_done = false;
    state->outer_tuple = NULL;
    state->inner_tuple = NULL;
    state->outer_cur_slot = NULL;
    state->inner_cur_slot = NULL;
    state->saved_outer = NULL;
    state->saved_inner = NULL;
    state->outer_multi = false;
    state->inner_multi = false;

    state->use_simd = vjoin_simd_caps.has_avx2 || vjoin_simd_caps.has_sse2 ||
                      vjoin_simd_caps.has_neon;

    /* Check if all output columns are pass-by-value (enables zero-copy singleton path) */
    {
        int i;
        state->all_byval = true;
        for (i = 0; i < outer_desc->natts; i++)
        {
            if (!TupleDescAttr(outer_desc, i)->attbyval)
            {
                state->all_byval = false;
                break;
            }
        }
        if (state->all_byval)
        {
            for (i = 0; i < inner_desc->natts; i++)
            {
                if (!TupleDescAttr(inner_desc, i)->attbyval)
                {
                    state->all_byval = false;
                    break;
                }
            }
        }
    }

    /* Allocate batch buffers for block merge (all_byval, INNER/LEFT) */
    if (state->all_byval &&
        (state->jointype == JOIN_INNER || state->jointype == JOIN_LEFT))
    {
        int bs = vjoin_batch_size;

        state->batch_size = bs;

        state->ob_keys   = palloc(sizeof(Datum) * bs * state->num_keys);
        state->ob_values = palloc(sizeof(Datum) * bs * state->num_outer_attrs);
        state->ob_isnull = palloc(sizeof(bool) * bs * state->num_outer_attrs);
        state->ob_count  = 0;
        state->ob_pos    = 0;
        state->ob_exhausted = false;

        state->ib_keys   = palloc(sizeof(Datum) * bs * state->num_keys);
        state->ib_values = palloc(sizeof(Datum) * bs * state->num_inner_attrs);
        state->ib_isnull = palloc(sizeof(bool) * bs * state->num_inner_attrs);
        state->ib_count  = 0;
        state->ib_pos    = 0;
        state->ib_exhausted = false;

        state->batch_result_capacity = bs * 4;
        state->batch_results = palloc(sizeof(VJoinMatch) *
                                      state->batch_result_capacity);
        state->batch_result_count = 0;
        state->batch_result_pos   = 0;
        state->total_attrs = state->num_outer_attrs + state->num_inner_attrs;

        /* Pre-built result arrays for zero-copy emit */
        state->batch_result_values = palloc(sizeof(Datum) *
                                            state->batch_result_capacity *
                                            state->total_attrs);
        state->batch_result_isnull = palloc(sizeof(bool) *
                                            state->batch_result_capacity *
                                            state->total_attrs);
        state->saved_tts_values = NULL;
        state->saved_tts_isnull = NULL;

        state->batch_cp_oi = -1;

        /* LEFT JOIN: allocate matched-tracking array */
        if (state->jointype == JOIN_LEFT)
        {
            state->ob_matched = palloc0(sizeof(bool) * bs);
            state->batch_left_pos = 0;
        }
        else
        {
            state->ob_matched = NULL;
            state->batch_left_pos = 0;
        }
    }
    else
    {
        state->batch_size = 0;
        state->total_attrs = 0;
        state->batch_result_values = NULL;
        state->batch_result_isnull = NULL;
        state->saved_tts_values = NULL;
        state->saved_tts_isnull = NULL;
    }

    /* Initialize parallel fields (DSM callbacks set these later if parallel) */
    state->pstate = NULL;
    state->dsa = NULL;
    state->is_parallel = false;
    state->is_leader = false;
    state->shared_inner_values = NULL;
    state->shared_inner_isnull = NULL;
    state->shared_inner_keys = NULL;
    state->shared_inner_count = 0;
    state->shared_inner_pos = 0;

    state->phase = VMJ_INIT;
}

TupleTableSlot *
vjoin_merge_exec(CustomScanState *node)
{
    VectorMergeJoinState *state = (VectorMergeJoinState *) node;
    ExprContext *econtext = node->ss.ps.ps_ExprContext;
    ExprState  *qual = node->ss.ps.qual;
    ProjectionInfo *projInfo = node->ss.ps.ps_ProjInfo;

    for (;;)
    {
        switch (state->phase)
        {
            case VMJ_INIT:
                /* Parallel: materialize shared inner before any fills */
                if (state->is_parallel && state->shared_inner_values == NULL &&
                    state->shared_inner_count == 0)
                    vmj_materialize_shared_inner(state);

                if (state->all_byval && state->batch_size > 0 &&
                    (state->jointype == JOIN_INNER ||
                     state->jointype == JOIN_LEFT))
                {
                    /* Block merge mode: fill both blocks */
                    vmj_batch_fill_side(state, true);
                    vmj_fill_inner(state);
                    if (state->ob_count == 0)
                    {
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                    if (state->ib_count == 0)
                    {
                        if (state->jointype == JOIN_LEFT)
                        {
                            /* No inner at all — all outers are unmatched */
                            state->phase = VMJ_BATCH_LEFT;
                            continue;
                        }
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                    state->phase = VMJ_BATCH_MERGE;
                    continue;
                }
                /* Tuple-at-a-time path */
                if (!vmj_fetch_outer(state))
                {
                    /* No outer tuples at all */
                    if (state->jointype == JOIN_RIGHT ||
                        state->jointype == JOIN_FULL)
                    {
                        if (vmj_fetch_inner(state))
                        {
                            state->phase = VMJ_RIGHT_EMIT;
                            continue;
                        }
                    }
                    state->phase = VMJ_DONE;
                    return NULL;
                }
                if (!vmj_fetch_inner(state))
                {
                    /* No inner tuples — outer still pending */
                    if (state->jointype == JOIN_LEFT ||
                        state->jointype == JOIN_FULL)
                    {
                        state->phase = VMJ_LEFT_EMIT;
                        continue;
                    }
                    state->phase = VMJ_DONE;
                    return NULL;
                }
                state->outer_matched = false;
                state->inner_matched = false;
                state->phase = VMJ_ADVANCE;
                continue;

            case VMJ_ADVANCE:
            {
                int cmp;

                /* Skip NULLs — they can never match in equijoin.
                 * For LEFT/FULL, emit outer NULLs with NULL inner.
                 * For RIGHT/FULL, emit inner NULLs with NULL outer. */
                while (!state->outer_done && state->outer_null[0])
                {
                    if (state->jointype == JOIN_LEFT ||
                        state->jointype == JOIN_FULL)
                    {
                        TupleTableSlot *result;
                        state->outer_matched = false;
                        result = vmj_form_result_left_slot(state);
                        if (!vmj_fetch_outer(state))
                            ;
                        ResetExprContext(econtext);
                        econtext->ecxt_scantuple = result;
                        if (projInfo)
                            return ExecProject(projInfo);
                        return result;
                    }
                    if (!vmj_fetch_outer(state))
                        break;
                }
                while (!state->inner_done && state->inner_null[0])
                {
                    if (state->jointype == JOIN_RIGHT ||
                        state->jointype == JOIN_FULL)
                    {
                        TupleTableSlot *result;
                        state->inner_matched = false;
                        result = vmj_form_result_right_slot(state);
                        if (!vmj_fetch_inner(state))
                            ;
                        ResetExprContext(econtext);
                        econtext->ecxt_scantuple = result;
                        if (projInfo)
                            return ExecProject(projInfo);
                        return result;
                    }
                    if (!vmj_fetch_inner(state))
                        break;
                }

                if (state->outer_done || state->inner_done)
                {
                    /* Drain remaining side for outer joins */
                    if (!state->outer_done &&
                        (state->jointype == JOIN_LEFT ||
                         state->jointype == JOIN_FULL))
                    {
                        state->phase = VMJ_LEFT_EMIT;
                        continue;
                    }
                    if (!state->inner_done &&
                        (state->jointype == JOIN_RIGHT ||
                         state->jointype == JOIN_FULL))
                    {
                        state->phase = VMJ_RIGHT_EMIT;
                        continue;
                    }
                    state->phase = VMJ_DONE;
                    return NULL;
                }

                cmp = vmj_compare_keys(state);
                if (cmp < 0)
                {
                    /* outer < inner — advance outer */
                    if ((state->jointype == JOIN_LEFT ||
                         state->jointype == JOIN_FULL) &&
                        !state->outer_matched)
                    {
                        /* Emit unmatched outer with NULL inner */
                        TupleTableSlot *result;
                        result = vmj_form_result_left_slot(state);
                        state->outer_matched = false;
                        if (!vmj_fetch_outer(state))
                        {
                            if (!state->inner_done &&
                                (state->jointype == JOIN_RIGHT ||
                                 state->jointype == JOIN_FULL))
                            {
                                ResetExprContext(econtext);
                                econtext->ecxt_scantuple = result;
                                /* Must emit this result, then go to RIGHT_EMIT */
                                state->phase = VMJ_RIGHT_EMIT;
                                if (projInfo)
                                    return ExecProject(projInfo);
                                return result;
                            }
                            state->phase = VMJ_DONE;
                        }
                        ResetExprContext(econtext);
                        econtext->ecxt_scantuple = result;
                        if (projInfo)
                            return ExecProject(projInfo);
                        return result;
                    }
                    state->outer_matched = false;
                    if (!vmj_fetch_outer(state))
                    {
                        if (!state->inner_done &&
                            (state->jointype == JOIN_RIGHT ||
                             state->jointype == JOIN_FULL))
                        {
                            state->phase = VMJ_RIGHT_EMIT;
                            continue;
                        }
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                    continue;
                }
                else if (cmp > 0)
                {
                    /* outer > inner — advance inner */
                    if ((state->jointype == JOIN_RIGHT ||
                         state->jointype == JOIN_FULL) &&
                        !state->inner_matched)
                    {
                        /* Emit unmatched inner with NULL outer */
                        TupleTableSlot *result;
                        result = vmj_form_result_right_slot(state);
                        state->inner_matched = false;
                        if (!vmj_fetch_inner(state))
                        {
                            if (!state->outer_done &&
                                (state->jointype == JOIN_LEFT ||
                                 state->jointype == JOIN_FULL))
                            {
                                ResetExprContext(econtext);
                                econtext->ecxt_scantuple = result;
                                state->phase = VMJ_LEFT_EMIT;
                                if (projInfo)
                                    return ExecProject(projInfo);
                                return result;
                            }
                            state->phase = VMJ_DONE;
                        }
                        ResetExprContext(econtext);
                        econtext->ecxt_scantuple = result;
                        if (projInfo)
                            return ExecProject(projInfo);
                        return result;
                    }
                    state->inner_matched = false;
                    if (!vmj_fetch_inner(state))
                    {
                        if (!state->outer_done &&
                            (state->jointype == JOIN_LEFT ||
                             state->jointype == JOIN_FULL))
                        {
                            state->phase = VMJ_LEFT_EMIT;
                            continue;
                        }
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                    continue;
                }
                else
                {
                    /*
                     * Keys match. Save match key and detect singleton.
                     * For all_byval tables, use a zero-copy fast path:
                     * form result from child slots before advancing,
                     * only reconstruct MinimalTuples in the rare multi case.
                     */
                    {
                        int mk;
                        for (mk = 0; mk < state->num_keys; mk++)
                        {
                            if (!state->outer_null[mk] && !state->key_byval[mk])
                                state->match_key[mk] = datumCopy(
                                    state->outer_key[mk], false,
                                    state->key_typlen[mk]);
                            else
                                state->match_key[mk] = state->outer_key[mk];
                            state->match_null[mk] = state->outer_null[mk];
                        }
                    }

                    if (state->all_byval)
                    {
                        TupleTableSlot *scan_slot = state->css.ss.ss_ScanTupleSlot;
                        TupleTableSlot *os = state->outer_cur_slot;
                        TupleTableSlot *is = state->inner_cur_slot;

                        /* Form result directly from child slots (zero copy) */
                        slot_getallattrs(os);
                        slot_getallattrs(is);
                        ExecClearTuple(scan_slot);
                        memcpy(scan_slot->tts_values,
                               os->tts_values,
                               state->num_outer_attrs * sizeof(Datum));
                        memcpy(scan_slot->tts_isnull,
                               os->tts_isnull,
                               state->num_outer_attrs * sizeof(bool));
                        memcpy(scan_slot->tts_values + state->num_outer_attrs,
                               is->tts_values,
                               state->num_inner_attrs * sizeof(Datum));
                        memcpy(scan_slot->tts_isnull + state->num_outer_attrs,
                               is->tts_isnull,
                               state->num_inner_attrs * sizeof(bool));
                        ExecStoreVirtualTuple(scan_slot);

                        /* Advance both sides to detect singleton */
                        if (!vmj_fetch_outer(state))
                            state->outer_multi = false;
                        else
                            state->outer_multi = vmj_keys_equal(
                                state, state->match_key, state->match_null,
                                state->outer_key, state->outer_null);

                        if (!vmj_fetch_inner(state))
                            state->inner_multi = false;
                        else
                            state->inner_multi = vmj_keys_equal(
                                state, state->match_key, state->match_null,
                                state->inner_key, state->inner_null);

                        if (!state->outer_multi && !state->inner_multi)
                        {
                            /* Singleton — return pre-formed result (zero copy!) */
                            state->outer_matched = true;
                            state->inner_matched = true;
                            ResetExprContext(econtext);
                            econtext->ecxt_scantuple = scan_slot;

                            state->outer_matched = false;
                            state->inner_matched = false;

                            if (qual && !ExecQual(qual, econtext))
                                continue;

                            if (projInfo)
                                return ExecProject(projInfo);
                            return scan_slot;
                        }

                        /* Multi group — reconstruct MinimalTuples from scan_slot */
                        state->saved_outer = vjoin_heap_form_minimal_tuple(
                            state->outer_desc,
                            scan_slot->tts_values,
                            scan_slot->tts_isnull);
                        state->saved_inner = vjoin_heap_form_minimal_tuple(
                            state->inner_desc,
                            scan_slot->tts_values + state->num_outer_attrs,
                            scan_slot->tts_isnull + state->num_outer_attrs);
                        MemoryContextReset(state->match_ctx);
                        state->outer_matched = true;
                        state->inner_matched = true;
                        state->phase = VMJ_MATCH_OUTER;
                        continue;
                    }
                    else
                    {
                        /* Non-byval path: save MinimalTuples, then advance */
                        TupleTableSlot *oslot = state->outer_cur_slot;
                        TupleTableSlot *islot = state->inner_cur_slot;

                        state->saved_outer = ExecCopySlotMinimalTuple(oslot);
                        state->saved_inner = ExecCopySlotMinimalTuple(islot);

                        if (!vmj_fetch_outer(state))
                            state->outer_multi = false;
                        else
                            state->outer_multi = vmj_keys_equal(
                                state, state->match_key, state->match_null,
                                state->outer_key, state->outer_null);

                        if (!vmj_fetch_inner(state))
                            state->inner_multi = false;
                        else
                            state->inner_multi = vmj_keys_equal(
                                state, state->match_key, state->match_null,
                                state->inner_key, state->inner_null);

                        if (!state->outer_multi && !state->inner_multi)
                        {
                            TupleTableSlot *result;

                            state->outer_matched = false;
                            state->inner_matched = false;

                            result = vmj_form_result(state,
                                                     state->saved_outer,
                                                     state->saved_inner);
                            pfree(state->saved_outer);
                            pfree(state->saved_inner);
                            state->saved_outer = NULL;
                            state->saved_inner = NULL;

                            ResetExprContext(econtext);
                            econtext->ecxt_scantuple = result;

                            if (qual && !ExecQual(qual, econtext))
                                continue;

                            if (projInfo)
                                return ExecProject(projInfo);
                            return result;
                        }
                        else
                        {
                            MemoryContextReset(state->match_ctx);
                            state->outer_matched = true;
                            state->inner_matched = true;
                            state->phase = VMJ_MATCH_OUTER;
                            continue;
                        }
                    }
                }
            }

            case VMJ_MATCH_OUTER:
                vmj_collect_group(state, true, state->saved_outer);
                pfree(state->saved_outer);
                state->saved_outer = NULL;
                state->phase = VMJ_MATCH_INNER;
                continue;

            case VMJ_MATCH_INNER:
                vmj_collect_group(state, false, state->saved_inner);
                pfree(state->saved_inner);
                state->saved_inner = NULL;
                state->emit_outer_pos = 0;
                state->emit_inner_pos = 0;
                state->phase = VMJ_EMIT;
                continue;

            case VMJ_EMIT:
            {
                TupleTableSlot *result;

                if (state->emit_outer_pos >= state->outer_group_count)
                {
                    /* Exhausted cross product — both sides matched */
                    state->outer_matched = false;
                    state->inner_matched = false;
                    state->phase = VMJ_ADVANCE;
                    continue;
                }

                result = vmj_form_result(
                    state,
                    state->outer_group[state->emit_outer_pos],
                    state->inner_group[state->emit_inner_pos]);

                /* Advance emit position (cross product) */
                state->emit_inner_pos++;
                if (state->emit_inner_pos >= state->inner_group_count)
                {
                    state->emit_inner_pos = 0;
                    state->emit_outer_pos++;
                }

                ResetExprContext(econtext);
                econtext->ecxt_scantuple = result;

                if (qual && !ExecQual(qual, econtext))
                    continue;

                if (projInfo)
                    return ExecProject(projInfo);
                return result;
            }

            /* ---- Block merge phases (vectorized) ---- */

            case VMJ_BATCH_FILL:
                /* Refill exhausted block(s) */
                if (state->ob_pos >= state->ob_count && !state->ob_exhausted)
                    vmj_batch_fill_side(state, true);
                if (state->ib_pos >= state->ib_count && !state->ib_exhausted)
                    vmj_fill_inner(state);

                if (state->ob_count == 0 || state->ib_count == 0)
                {
                    state->phase = VMJ_DONE;
                    return NULL;
                }
                state->phase = VMJ_BATCH_MERGE;
                continue;

            case VMJ_BATCH_MERGE:
                vmj_batch_do_merge(state);
                if (state->batch_result_count > 0)
                {
                    state->phase = VMJ_BATCH_EMIT;
                    continue;
                }
                /*
                 * No matches: one (or both) blocks is exhausted,
                 * or we stopped at a boundary-spanning group.
                 * For LEFT JOIN, emit unmatched outers before refilling.
                 */
                if (state->ob_matched &&
                    state->batch_left_pos < state->ob_pos)
                {
                    state->phase = VMJ_BATCH_LEFT;
                    continue;
                }
                if (state->ob_pos >= state->ob_count)
                {
                    if (state->ob_exhausted)
                    {
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                    vmj_batch_fill_side(state, true);
                    if (state->ob_count == 0)
                    {
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                    continue;
                }
                if (state->ib_pos >= state->ib_count)
                {
                    if (state->ib_exhausted)
                    {
                        /* Inner exhausted — remaining outers are unmatched */
                        if (state->ob_matched)
                        {
                            state->phase = VMJ_BATCH_LEFT;
                            continue;
                        }
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                    vmj_fill_inner(state);
                    if (state->ib_count == 0)
                    {
                        if (state->ob_matched)
                        {
                            state->phase = VMJ_BATCH_LEFT;
                            continue;
                        }
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                    continue;
                }
                /*
                 * Both blocks have entries (boundary-spanning group).
                 * Refill the exhausted side to complete the group.
                 */
                if (!state->ob_exhausted)
                    vmj_batch_fill_side(state, true);
                if (!state->ib_exhausted)
                    vmj_fill_inner(state);
                if (state->ob_count == 0 || state->ib_count == 0)
                {
                    if (state->ob_matched && state->ob_count > 0)
                    {
                        state->phase = VMJ_BATCH_LEFT;
                        continue;
                    }
                    state->phase = VMJ_DONE;
                    return NULL;
                }
                continue;

            case VMJ_BATCH_EMIT:
            {
                TupleTableSlot *scan_slot = state->css.ss.ss_ScanTupleSlot;
                int total = state->total_attrs;

                if (state->batch_result_pos >= state->batch_result_count)
                {
                    /* Restore scan_slot pointers before leaving emit phase */
                    if (state->saved_tts_values)
                    {
                        scan_slot->tts_values = state->saved_tts_values;
                        scan_slot->tts_isnull = state->saved_tts_isnull;
                    }
                    state->phase = VMJ_BATCH_MERGE;
                    continue;
                }

                /* Save original pointers on first entry */
                if (state->saved_tts_values == NULL)
                {
                    state->saved_tts_values = scan_slot->tts_values;
                    state->saved_tts_isnull = scan_slot->tts_isnull;
                }

                /* Zero-copy: redirect scan_slot to pre-built result row */
                {
                    int pos = state->batch_result_pos++;
                    scan_slot->tts_values = &state->batch_result_values[pos * total];
                    scan_slot->tts_isnull = &state->batch_result_isnull[pos * total];
                    ExecStoreVirtualTuple(scan_slot);

                    ResetExprContext(econtext);
                    econtext->ecxt_scantuple = scan_slot;

                    if (qual && !ExecQual(qual, econtext))
                        continue;

                    if (projInfo)
                        return ExecProject(projInfo);
                    return scan_slot;
                }
            }

            case VMJ_BATCH_LEFT:
            {
                /*
                 * LEFT JOIN batch: emit unmatched outer tuples with NULL inner.
                 * Scans ob_matched[batch_left_pos .. limit) for false entries.
                 * limit = ob_pos normally, or ob_count when inner is exhausted
                 * (remaining outers past ob_pos can never match).
                 */
                TupleTableSlot *result;
                int limit = state->ob_pos;

                /* If inner is fully consumed, all remaining outers are unmatched */
                if (state->ib_exhausted &&
                    (state->ib_count == 0 || state->ib_pos >= state->ib_count))
                    limit = state->ob_count;

                while (state->batch_left_pos < limit)
                {
                    int idx = state->batch_left_pos++;
                    if (!state->ob_matched[idx])
                    {
                        result = vmj_batch_form_result_left(state, idx);
                        ResetExprContext(econtext);
                        econtext->ecxt_scantuple = result;

                        if (qual && !ExecQual(qual, econtext))
                            continue;

                        if (projInfo)
                            return ExecProject(projInfo);
                        return result;
                    }
                }

                /*
                 * All unmatched outers emitted for current range.
                 * Continue with refill or finish.
                 */
                if (limit == state->ob_count)
                {
                    /* Whole block scanned — mark all as consumed before refill */
                    state->ob_pos = state->ob_count;
                    if (state->ob_exhausted)
                    {
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                    vmj_batch_fill_side(state, true);
                    if (state->ob_count == 0)
                    {
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                    /* If inner is exhausted, stay in VMJ_BATCH_LEFT */
                    if (state->ib_exhausted &&
                        (state->ib_count == 0 || state->ib_pos >= state->ib_count))
                        continue;
                }
                else if (state->ob_pos >= state->ob_count)
                {
                    /* Outer block consumed — refill */
                    if (state->ob_exhausted)
                    {
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                    vmj_batch_fill_side(state, true);
                    if (state->ob_count == 0)
                    {
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                }
                state->phase = VMJ_BATCH_MERGE;
                continue;
            }

            case VMJ_LEFT_EMIT:
            {
                /* Drain remaining outer tuples with NULL inner */
                TupleTableSlot *result;

                if (state->outer_done)
                {
                    if (!state->inner_done &&
                        (state->jointype == JOIN_RIGHT ||
                         state->jointype == JOIN_FULL))
                    {
                        state->phase = VMJ_RIGHT_EMIT;
                        continue;
                    }
                    state->phase = VMJ_DONE;
                    return NULL;
                }

                result = vmj_form_result_left_slot(state);
                vmj_fetch_outer(state);

                ResetExprContext(econtext);
                econtext->ecxt_scantuple = result;
                if (projInfo)
                    return ExecProject(projInfo);
                return result;
            }

            case VMJ_RIGHT_EMIT:
            {
                /* Drain remaining inner tuples with NULL outer */
                TupleTableSlot *result;

                if (state->inner_done)
                {
                    state->phase = VMJ_DONE;
                    return NULL;
                }

                result = vmj_form_result_right_slot(state);
                vmj_fetch_inner(state);

                ResetExprContext(econtext);
                econtext->ecxt_scantuple = result;
                if (projInfo)
                    return ExecProject(projInfo);
                return result;
            }

            case VMJ_DONE:
                return NULL;
        }
    }
}

void
vjoin_merge_end(CustomScanState *node)
{
    VectorMergeJoinState *state = (VectorMergeJoinState *) node;

    /* Restore scan_slot pointers if redirected by zero-copy emit */
    if (state->saved_tts_values)
    {
        TupleTableSlot *scan_slot = node->ss.ss_ScanTupleSlot;
        scan_slot->tts_values = state->saved_tts_values;
        scan_slot->tts_isnull = state->saved_tts_isnull;
        state->saved_tts_values = NULL;
        state->saved_tts_isnull = NULL;
    }

    ExecEndNode(state->outer_ps);
    ExecEndNode(state->inner_ps);

    ExecDropSingleTupleTableSlot(state->outer_slot);
    ExecDropSingleTupleTableSlot(state->inner_slot);

    if (state->match_ctx)
        MemoryContextDelete(state->match_ctx);
}

void
vjoin_merge_rescan(CustomScanState *node)
{
    VectorMergeJoinState *state = (VectorMergeJoinState *) node;

    ExecReScan(state->outer_ps);
    ExecReScan(state->inner_ps);

    state->outer_done = false;
    state->inner_done = false;
    state->outer_tuple = NULL;
    state->inner_tuple = NULL;
    state->outer_cur_slot = NULL;
    state->inner_cur_slot = NULL;
    state->saved_outer = NULL;
    state->saved_inner = NULL;
    state->outer_multi = false;
    state->inner_multi = false;
    state->outer_group_count = 0;
    state->inner_group_count = 0;

    /* Reset batch state */
    state->ob_count = 0;
    state->ob_pos = 0;
    state->ob_exhausted = false;
    state->ib_count = 0;
    state->ib_pos = 0;
    state->ib_exhausted = false;
    state->batch_result_count = 0;
    state->batch_result_pos = 0;
    state->batch_cp_oi = -1;
    state->batch_left_pos = 0;
    if (state->ob_matched)
        memset(state->ob_matched, 0, sizeof(bool) * state->batch_size);

    /* Restore scan_slot pointers if redirected by zero-copy emit */
    if (state->saved_tts_values)
    {
        TupleTableSlot *scan_slot = node->ss.ss_ScanTupleSlot;
        scan_slot->tts_values = state->saved_tts_values;
        scan_slot->tts_isnull = state->saved_tts_isnull;
        state->saved_tts_values = NULL;
        state->saved_tts_isnull = NULL;
    }

    state->outer_matched = false;
    state->inner_matched = false;

    state->phase = VMJ_INIT;
}

void
vjoin_merge_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
    VectorMergeJoinState *state = (VectorMergeJoinState *) node;
    const char *jt;

    switch (state->jointype)
    {
        case JOIN_LEFT:  jt = "Left";  break;
        case JOIN_RIGHT: jt = "Right"; break;
        case JOIN_FULL:  jt = "Full";  break;
        default:         jt = "Inner"; break;
    }
    ExplainPropertyText("Join Type", jt, es);
    ExplainPropertyInteger("Keys", NULL, state->num_keys, es);
    ExplainPropertyBool("SIMD", state->use_simd, es);
    if (state->batch_size > 0)
        ExplainPropertyInteger("Batch Size", NULL, state->batch_size, es);
    if (state->is_parallel)
        ExplainPropertyBool("Shared Inner", true, es);
}

/* ----------------------------------------------------------------
 *      Parallel DSM callbacks
 *
 * Leader materializes the full sorted inner into DSA shared memory.
 * Workers attach to DSA and read the shared inner directly.
 * Each participant merges its partial outer against the shared inner.
 * ---------------------------------------------------------------- */

Size
vjoin_merge_estimate_dsm(CustomScanState *node, ParallelContext *pcxt)
{
    return sizeof(VJoinMergeParallelState);
}

void
vjoin_merge_initialize_dsm(CustomScanState *node, ParallelContext *pcxt,
                            void *coordinate)
{
    VectorMergeJoinState *state = (VectorMergeJoinState *) node;
    VJoinMergeParallelState *pstate = (VJoinMergeParallelState *) coordinate;

    /* Create DSA in the DSM segment */
    state->dsa = dsa_create(LWTRANCHE_PARALLEL_HASH_JOIN);
    dsa_pin_mapping(state->dsa);

    pstate->dsa_handle = dsa_get_handle(state->dsa);
    BarrierInit(&pstate->barrier, vjoin_pcxt_nworkers(pcxt) + 1);
    pstate->inner_count = 0;
    pstate->num_inner_attrs = state->num_inner_attrs;
    pstate->num_keys = state->num_keys;
    pstate->inner_values_dp = InvalidDsaPointer;
    pstate->inner_isnull_dp = InvalidDsaPointer;
    pstate->inner_keys_dp = InvalidDsaPointer;

    state->pstate = pstate;
    state->is_parallel = true;
    state->is_leader = true;
}

void
vjoin_merge_reinitialize_dsm(CustomScanState *node, ParallelContext *pcxt,
                              void *coordinate)
{
    /* Rescan not supported for parallel VMJ */
}

void
vjoin_merge_initialize_worker(CustomScanState *node, shm_toc *toc,
                               void *coordinate)
{
    VectorMergeJoinState *state = (VectorMergeJoinState *) node;
    VJoinMergeParallelState *pstate = (VJoinMergeParallelState *) coordinate;

    /* Attach to the DSA area created by the leader */
    state->dsa = dsa_attach(pstate->dsa_handle);
    dsa_pin_mapping(state->dsa);

    state->pstate = pstate;
    state->is_parallel = true;
    state->is_leader = false;
}

void
vjoin_merge_shutdown(CustomScanState *node)
{
    /* All cleanup handled by vjoin_merge_end */
}
