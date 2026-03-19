#include "postgres.h"
#include "access/htup_details.h"
#include "vjoin_compat.h"
#include "executor/executor.h"
#include "executor/tuptable.h"
#include "nodes/extensible.h"
#include "nodes/value.h"
#include "utils/memutils.h"
#include "utils/datum.h"
#include "utils/typcache.h"
#include "utils/lsyscache.h"
#include "vjoin_compat.h"
#include "fmgr.h"
#include "pg_vectorjoin.h"
#include "vjoin_state.h"
#include "vjoin_simd.h"

#define VMJ_INITIAL_GROUP_CAPACITY 64

/*
 * Deserialize key info from custom_private.
 */
static void
vmj_deserialize_keys(List *private_data,
                     int *num_keys,
                     AttrNumber *outer_keynos,
                     AttrNumber *inner_keynos,
                     Oid *key_types,
                     Oid *eq_funcs,
                     Oid *key_collations)
{
    int idx = 0;
    int i;

    *num_keys = intVal(list_nth(private_data, idx++));
    for (i = 0; i < *num_keys; i++)
    {
        outer_keynos[i] = (AttrNumber) intVal(list_nth(private_data, idx++));
        inner_keynos[i] = (AttrNumber) intVal(list_nth(private_data, idx++));
        key_types[i] = (Oid) intVal(list_nth(private_data, idx++));
        idx++;  /* skip hash_proc */
        eq_funcs[i] = (Oid) intVal(list_nth(private_data, idx++));
        key_collations[i] = (Oid) intVal(list_nth(private_data, idx++));
    }
}

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
 * Collect all outer tuples with the same key into outer_group.
 * first_mt is the already-saved first tuple of the group.
 * If outer_multi is true, the current child slot has the second tuple
 * (with the same key); we copy it and keep fetching.
 * After return, outer_key holds the NEXT different key (or outer_done=true).
 */
static void
vmj_collect_outer_group(VectorMergeJoinState *state, MinimalTuple first_mt)
{
    MemoryContext oldctx;

    state->outer_group_count = 0;
    oldctx = MemoryContextSwitchTo(state->match_ctx);

    /* First tuple — already saved, copy into match_ctx */
    if (state->outer_group_count >= state->outer_group_capacity)
    {
        state->outer_group_capacity *= 2;
        state->outer_group = repalloc(state->outer_group,
            sizeof(MinimalTuple) * state->outer_group_capacity);
    }
    state->outer_group[state->outer_group_count++] = vjoin_heap_copy_minimal_tuple(first_mt);

    MemoryContextSwitchTo(oldctx);

    /* If outer_multi, current child slot has the second tuple */
    if (state->outer_multi)
    {
        TupleTableSlot *slot = state->outer_cur_slot;

        oldctx = MemoryContextSwitchTo(state->match_ctx);
        if (state->outer_group_count >= state->outer_group_capacity)
        {
            state->outer_group_capacity *= 2;
            state->outer_group = repalloc(state->outer_group,
                sizeof(MinimalTuple) * state->outer_group_capacity);
        }
        state->outer_group[state->outer_group_count++] = ExecCopySlotMinimalTuple(slot);
        MemoryContextSwitchTo(oldctx);

        /* Keep fetching while key matches */
        while (vmj_fetch_outer(state))
        {
            if (!vmj_keys_equal(state, state->match_key, state->match_null,
                                state->outer_key, state->outer_null))
                break;

            slot = state->outer_cur_slot;
            oldctx = MemoryContextSwitchTo(state->match_ctx);
            if (state->outer_group_count >= state->outer_group_capacity)
            {
                state->outer_group_capacity *= 2;
                state->outer_group = repalloc(state->outer_group,
                    sizeof(MinimalTuple) * state->outer_group_capacity);
            }
            state->outer_group[state->outer_group_count++] = ExecCopySlotMinimalTuple(slot);
            MemoryContextSwitchTo(oldctx);
        }
    }
}

/*
 * Collect all inner tuples with the same key into inner_group.
 * first_mt is the already-saved first tuple of the group.
 * If inner_multi is true, the current child slot has the second tuple.
 */
static void
vmj_collect_inner_group(VectorMergeJoinState *state, MinimalTuple first_mt)
{
    MemoryContext oldctx;

    state->inner_group_count = 0;
    oldctx = MemoryContextSwitchTo(state->match_ctx);

    if (state->inner_group_count >= state->inner_group_capacity)
    {
        state->inner_group_capacity *= 2;
        state->inner_group = repalloc(state->inner_group,
            sizeof(MinimalTuple) * state->inner_group_capacity);
    }
    state->inner_group[state->inner_group_count++] = vjoin_heap_copy_minimal_tuple(first_mt);

    MemoryContextSwitchTo(oldctx);

    if (state->inner_multi)
    {
        TupleTableSlot *slot = state->inner_cur_slot;

        oldctx = MemoryContextSwitchTo(state->match_ctx);
        if (state->inner_group_count >= state->inner_group_capacity)
        {
            state->inner_group_capacity *= 2;
            state->inner_group = repalloc(state->inner_group,
                sizeof(MinimalTuple) * state->inner_group_capacity);
        }
        state->inner_group[state->inner_group_count++] = ExecCopySlotMinimalTuple(slot);
        MemoryContextSwitchTo(oldctx);

        while (vmj_fetch_inner(state))
        {
            if (!vmj_keys_equal(state, state->match_key, state->match_null,
                                state->inner_key, state->inner_null))
                break;

            slot = state->inner_cur_slot;
            oldctx = MemoryContextSwitchTo(state->match_ctx);
            if (state->inner_group_count >= state->inner_group_capacity)
            {
                state->inner_group_capacity *= 2;
                state->inner_group = repalloc(state->inner_group,
                    sizeof(MinimalTuple) * state->inner_group_capacity);
            }
            state->inner_group[state->inner_group_count++] = ExecCopySlotMinimalTuple(slot);
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
    AttrNumber  keyno;
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
        keyno       = state->outer_keynos[0];
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
        keyno       = state->inner_keynos[0];
    }

    /* Shift remaining entries to front */
    remaining = *count_p - *pos_p;
    if (remaining > 0 && *pos_p > 0)
    {
        memmove(keys, &keys[*pos_p],
                remaining * sizeof(Datum));
        memmove(values, &values[*pos_p * ncols],
                remaining * ncols * sizeof(Datum));
        memmove(isnull, &isnull[*pos_p * ncols],
                remaining * ncols * sizeof(bool));
    }
    *count_p = remaining;
    *pos_p = 0;

    /* Fill the rest from child plan */
    while (*count_p < batch_size)
    {
        TupleTableSlot *slot;
        bool            key_null;
        int             idx, off;

        slot = ExecProcNode(child);
        if (TupIsNull(slot))
        {
            *exhausted_p = true;
            break;
        }

        /* Extract primary key — skip NULL keys */
        idx = *count_p;
        keys[idx] = slot_getattr(slot, keyno, &key_null);
        if (key_null)
            continue;

        /* Deform all attributes into columnar arrays */
        slot_getallattrs(slot);
        off = idx * ncols;
        memcpy(&values[off], slot->tts_values, ncols * sizeof(Datum));
        memcpy(&isnull[off], slot->tts_isnull, ncols * sizeof(bool));

        (*count_p)++;
    }
}

/*
 * Inline key comparison for batch merge (single key).
 */
static inline int
vmj_batch_compare_key(Datum a, Datum b, Oid keytype)
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
            return (a < b) ? -1 : (a > b) ? 1 : 0;
    }
}

/*
 * INT4-specialized batch merge with binary-search advance.
 * Processes ob_keys[ob_pos..] vs ib_keys[ib_pos..],
 * fills batch_results with match pairs.
 * Stops at block boundary if a group might span it.
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
    VJoinMatch *results = state->batch_results;
    int     max_results = state->batch_result_capacity;

    while (oi < ob_count && ii < ib_count && nr < max_results)
    {
        int32   outer_val = DatumGetInt32(ok[oi]);
        int32   inner_val = DatumGetInt32(ik[ii]);

        if (outer_val < inner_val)
        {
            /* Binary search: first oi where ok[oi] >= inner_val */
            int lo = oi + 1, hi = ob_count;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                if (DatumGetInt32(ok[mid]) < inner_val)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            oi = lo;
        }
        else if (outer_val > inner_val)
        {
            /* Binary search: first ii where ik[ii] >= outer_val */
            int lo = ii + 1, hi = ib_count;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                if (DatumGetInt32(ik[mid]) < outer_val)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            ii = lo;
        }
        else
        {
            /* Keys match — find group boundaries */
            int32   match_val = outer_val;
            int     oe = oi + 1;
            int     ie = ii + 1;
            int     o, i;

            while (oe < ob_count && DatumGetInt32(ok[oe]) == match_val)
                oe++;
            while (ie < ib_count && DatumGetInt32(ik[ie]) == match_val)
                ie++;

            /* Stop if group might span a block boundary */
            if ((oe == ob_count && !state->ob_exhausted) ||
                (ie == ib_count && !state->ib_exhausted))
                break;

            /* Add cross product to results */
            for (o = oi; o < oe && nr < max_results; o++)
                for (i = ii; i < ie && nr < max_results; i++)
                {
                    results[nr].outer_idx = o;
                    results[nr].inner_idx = i;
                    nr++;
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
 */
static void
vmj_batch_do_merge_generic(VectorMergeJoinState *state)
{
    int     oi = state->ob_pos;
    int     ii = state->ib_pos;
    int     nr = 0;
    Datum  *ok = state->ob_keys;
    Datum  *ik = state->ib_keys;
    Oid     keytype = state->key_types[0];
    VJoinMatch *results = state->batch_results;
    int     max_results = state->batch_result_capacity;

    while (oi < state->ob_count && ii < state->ib_count && nr < max_results)
    {
        int cmp = vmj_batch_compare_key(ok[oi], ik[ii], keytype);

        if (cmp < 0)
        {
            oi++;
        }
        else if (cmp > 0)
        {
            ii++;
        }
        else
        {
            /* Match — find group boundaries */
            Datum   match_val = ok[oi];
            int     oe = oi + 1;
            int     ie = ii + 1;
            int     o, i;

            while (oe < state->ob_count &&
                   vmj_batch_compare_key(ok[oe], match_val, keytype) == 0)
                oe++;
            while (ie < state->ib_count &&
                   vmj_batch_compare_key(ik[ie], match_val, keytype) == 0)
                ie++;

            if ((oe == state->ob_count && !state->ob_exhausted) ||
                (ie == state->ib_count && !state->ib_exhausted))
                break;

            for (o = oi; o < oe && nr < max_results; o++)
                for (i = ii; i < ie && nr < max_results; i++)
                {
                    results[nr].outer_idx = o;
                    results[nr].inner_idx = i;
                    nr++;
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
    if (state->num_keys == 1 && state->key_types[0] == INT4OID)
        vmj_batch_do_merge_int4(state);
    else
        vmj_batch_do_merge_generic(state);
}

/*
 * Form result from pre-deformed batch arrays (zero-copy for byval).
 */
static TupleTableSlot *
vmj_batch_form_result(VectorMergeJoinState *state,
                      int outer_idx, int inner_idx)
{
    TupleTableSlot *scan_slot = state->css.ss.ss_ScanTupleSlot;
    int outer_off = outer_idx * state->num_outer_attrs;
    int inner_off = inner_idx * state->num_inner_attrs;

    ExecClearTuple(scan_slot);

    memcpy(scan_slot->tts_values,
           &state->ob_values[outer_off],
           state->num_outer_attrs * sizeof(Datum));
    memcpy(scan_slot->tts_isnull,
           &state->ob_isnull[outer_off],
           state->num_outer_attrs * sizeof(bool));

    memcpy(scan_slot->tts_values + state->num_outer_attrs,
           &state->ib_values[inner_off],
           state->num_inner_attrs * sizeof(Datum));
    memcpy(scan_slot->tts_isnull + state->num_outer_attrs,
           &state->ib_isnull[inner_off],
           state->num_inner_attrs * sizeof(bool));

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

    /* Deserialize key info */
    vmj_deserialize_keys(cscan->custom_private,
                         &state->num_keys,
                         state->outer_keynos,
                         state->inner_keynos,
                         state->key_types,
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

    /* Allocate batch buffers for block merge (all_byval path) */
    if (state->all_byval)
    {
        int bs = vjoin_batch_size;

        state->batch_size = bs;

        state->ob_keys   = palloc(sizeof(Datum) * bs);
        state->ob_values = palloc(sizeof(Datum) * bs * state->num_outer_attrs);
        state->ob_isnull = palloc(sizeof(bool) * bs * state->num_outer_attrs);
        state->ob_count  = 0;
        state->ob_pos    = 0;
        state->ob_exhausted = false;

        state->ib_keys   = palloc(sizeof(Datum) * bs);
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

        state->batch_cp_oi = -1;
    }
    else
    {
        state->batch_size = 0;
    }

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
                if (state->all_byval && state->batch_size > 0)
                {
                    /* Block merge mode: fill both blocks */
                    vmj_batch_fill_side(state, true);
                    vmj_batch_fill_side(state, false);
                    if (state->ob_count == 0 || state->ib_count == 0)
                    {
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                    state->phase = VMJ_BATCH_MERGE;
                    continue;
                }
                /* Legacy tuple-at-a-time path */
                if (!vmj_fetch_outer(state) || !vmj_fetch_inner(state))
                {
                    state->phase = VMJ_DONE;
                    return NULL;
                }
                state->phase = VMJ_ADVANCE;
                continue;

            case VMJ_ADVANCE:
            {
                int cmp;

                /* Skip NULLs — they can never match in equijoin */
                while (!state->outer_done && state->outer_null[0])
                {
                    if (!vmj_fetch_outer(state))
                    {
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                }
                while (!state->inner_done && state->inner_null[0])
                {
                    if (!vmj_fetch_inner(state))
                    {
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                }

                if (state->outer_done || state->inner_done)
                {
                    state->phase = VMJ_DONE;
                    return NULL;
                }

                cmp = vmj_compare_keys(state);
                if (cmp < 0)
                {
                    /* outer < inner — advance outer */
                    if (!vmj_fetch_outer(state))
                    {
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                    continue;
                }
                else if (cmp > 0)
                {
                    /* outer > inner — advance inner */
                    if (!vmj_fetch_inner(state))
                    {
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
                            ResetExprContext(econtext);
                            econtext->ecxt_scantuple = scan_slot;

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
                            state->phase = VMJ_MATCH_OUTER;
                            continue;
                        }
                    }
                }
            }

            case VMJ_MATCH_OUTER:
                vmj_collect_outer_group(state, state->saved_outer);
                pfree(state->saved_outer);
                state->saved_outer = NULL;
                state->phase = VMJ_MATCH_INNER;
                continue;

            case VMJ_MATCH_INNER:
                vmj_collect_inner_group(state, state->saved_inner);
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
                    /* Exhausted cross product — back to advance */
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
                    vmj_batch_fill_side(state, false);

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
                 * Try to refill the side that ran out.
                 */
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
                        state->phase = VMJ_DONE;
                        return NULL;
                    }
                    vmj_batch_fill_side(state, false);
                    if (state->ib_count == 0)
                    {
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
                    vmj_batch_fill_side(state, false);
                if (state->ob_count == 0 || state->ib_count == 0)
                {
                    state->phase = VMJ_DONE;
                    return NULL;
                }
                continue;

            case VMJ_BATCH_EMIT:
            {
                TupleTableSlot *result;

                if (state->batch_result_pos >= state->batch_result_count)
                {
                    /* Buffer exhausted — back to merge */
                    state->phase = VMJ_BATCH_MERGE;
                    continue;
                }

                {
                    int oi = state->batch_results[state->batch_result_pos].outer_idx;
                    int ii = state->batch_results[state->batch_result_pos].inner_idx;
                    state->batch_result_pos++;

                    result = vmj_batch_form_result(state, oi, ii);

                    ResetExprContext(econtext);
                    econtext->ecxt_scantuple = result;

                    if (qual && !ExecQual(qual, econtext))
                        continue;

                    if (projInfo)
                        return ExecProject(projInfo);
                    return result;
                }
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

    state->phase = VMJ_INIT;
}

void
vjoin_merge_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
    VectorMergeJoinState *state = (VectorMergeJoinState *) node;

    ExplainPropertyInteger("Keys", NULL, state->num_keys, es);
    ExplainPropertyBool("SIMD", state->use_simd, es);
    if (state->batch_size > 0)
        ExplainPropertyInteger("Batch Size", NULL, state->batch_size, es);
}
