#include "postgres.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "access/parallel.h"
#include "vjoin_compat.h"
#include "executor/executor.h"
#include "executor/tuptable.h"
#include "nodes/extensible.h"
#include "nodes/value.h"
#include "storage/barrier.h"
#include "storage/lwlock.h"
#include "storage/shm_toc.h"
#include "utils/datum.h"
#include "utils/dsa.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "fmgr.h"
#include "pg_vectorjoin.h"
#include "vjoin_state.h"
#include "vjoin_simd.h"

/*
 * Deserialize key info from custom_private.
 */
static void
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
        hash_funcs[i] = (Oid) intVal(list_nth(private_data, idx++));
        eq_funcs[i] = (Oid) intVal(list_nth(private_data, idx++));
        key_collations[i] = (Oid) intVal(list_nth(private_data, idx++));
    }
}

/*
 * Form result with outer values and NULL-filled inner (for LEFT/FULL).
 */
static TupleTableSlot *
vjoin_form_result_left(VectorHashJoinState *state, int outer_batch_idx)
{
    TupleTableSlot *scan_slot = state->css.ss.ss_ScanTupleSlot;
    int outer_off = outer_batch_idx * state->num_outer_attrs;

    ExecClearTuple(scan_slot);

    memcpy(scan_slot->tts_values,
           &state->batch_values[outer_off],
           state->num_outer_attrs * sizeof(Datum));
    memcpy(scan_slot->tts_isnull,
           &state->batch_isnull[outer_off],
           state->num_outer_attrs * sizeof(bool));

    memset(scan_slot->tts_values + state->num_outer_attrs,
           0, state->num_inner_attrs * sizeof(Datum));
    memset(scan_slot->tts_isnull + state->num_outer_attrs,
           1, state->num_inner_attrs * sizeof(bool));

    ExecStoreVirtualTuple(scan_slot);
    return scan_slot;
}

/*
 * Form result with NULL-filled outer and inner values (for RIGHT/FULL).
 */
static TupleTableSlot *
vjoin_form_result_right(VectorHashJoinState *state, int inner_ht_pos)
{
    TupleTableSlot *scan_slot = state->css.ss.ss_ScanTupleSlot;
    VJoinHashTable *ht = state->hashtable;
    int inner_off = inner_ht_pos * ht->num_all_attrs;

    ExecClearTuple(scan_slot);

    memset(scan_slot->tts_values, 0,
           state->num_outer_attrs * sizeof(Datum));
    memset(scan_slot->tts_isnull, 1,
           state->num_outer_attrs * sizeof(bool));

    memcpy(scan_slot->tts_values + state->num_outer_attrs,
           &ht->all_values[inner_off],
           state->num_inner_attrs * sizeof(Datum));
    memcpy(scan_slot->tts_isnull + state->num_outer_attrs,
           &ht->all_isnull[inner_off],
           state->num_inner_attrs * sizeof(bool));

    ExecStoreVirtualTuple(scan_slot);
    return scan_slot;
}

/*
 * Fill the scan slot from pre-deformed outer + inner Datum arrays.
 * No intermediate TupleTableSlot operations — just 4 memcpy.
 */
static TupleTableSlot *
vjoin_form_result(VectorHashJoinState *state,
                  int outer_batch_idx, int inner_ht_pos)
{
    TupleTableSlot *scan_slot = state->css.ss.ss_ScanTupleSlot;
    VJoinHashTable *ht = state->hashtable;
    int outer_off = outer_batch_idx * state->num_outer_attrs;
    int inner_off = inner_ht_pos * ht->num_all_attrs;

    ExecClearTuple(scan_slot);

    memcpy(scan_slot->tts_values,
           &state->batch_values[outer_off],
           state->num_outer_attrs * sizeof(Datum));
    memcpy(scan_slot->tts_isnull,
           &state->batch_isnull[outer_off],
           state->num_outer_attrs * sizeof(bool));

    memcpy(scan_slot->tts_values + state->num_outer_attrs,
           &ht->all_values[inner_off],
           state->num_inner_attrs * sizeof(Datum));
    memcpy(scan_slot->tts_isnull + state->num_outer_attrs,
           &ht->all_isnull[inner_off],
           state->num_inner_attrs * sizeof(bool));

    ExecStoreVirtualTuple(scan_slot);
    return scan_slot;
}

/*
 * Build phase: read all inner tuples into hash table.
 * Pre-deforms all inner attributes for fast result emission.
 *
 * In parallel mode only the leader builds.  After building, the leader
 * copies the flat arrays into DSA shared memory and all participants
 * (leader + workers) synchronize at the barrier.  Workers then attach
 * a read-only wrapper to the shared hash table.
 */
static void
vjoin_hash_build(VectorHashJoinState *state)
{
    TupleTableSlot *slot;
    MemoryContext oldctx;

    /*
     * Parallel worker: skip build — the leader builds and shares via DSA.
     * Wait at the barrier for the leader to finish, then attach.
     */
    if (state->is_parallel && !state->is_leader)
    {
        BarrierArriveAndWait(&state->pstate->barrier, 0);

        /* Destroy the empty local HT created during begin */
        if (state->hashtable)
        {
            vjoin_ht_destroy(state->hashtable);
            state->hashtable = NULL;
        }

        /* Attach read-only wrapper to the shared hash table */
        state->hashtable = vjoin_ht_attach_from_dsa(state->pstate,
                                                     state->dsa,
                                                     state->hash_ctx);
        state->cached_ht_entries = state->hashtable->num_entries;

        /* Allocate inner_matched after attach so we know capacity */
        if (state->jointype == JOIN_RIGHT || state->jointype == JOIN_FULL)
            state->inner_matched = palloc0(sizeof(bool) * state->hashtable->capacity);

        state->phase = VHJ_PROBE;
        return;
    }

    /* Leader (parallel) or non-parallel: build the hash table locally */
    oldctx = MemoryContextSwitchTo(state->hash_ctx);

    for (;;)
    {
        bool    has_null = false;
        uint32  hashval;
        int     i;

        /* Pull inner tuple in parent context */
        MemoryContextSwitchTo(oldctx);
        slot = ExecProcNode(state->inner_ps);
        if (TupIsNull(slot))
            break;

        /* Deform all attributes once — keys are read from here too */
        slot_getallattrs(slot);
        MemoryContextSwitchTo(state->hash_ctx);

        /* Check for NULL keys */
        for (i = 0; i < state->num_keys; i++)
        {
            if (slot->tts_isnull[state->inner_keynos[i] - 1])
            {
                has_null = true;
                break;
            }
        }

        /* Skip NULL keys — NULLs never match in equijoin.
         * For RIGHT/FULL, insert anyway so they appear as unmatched. */
        if (has_null)
        {
            if (state->jointype == JOIN_RIGHT || state->jointype == JOIN_FULL)
            {
                vjoin_ht_insert(state->hashtable, 1 /* dummy nonzero hash */,
                                slot->tts_values, slot->tts_isnull);
            }
            continue;
        }

        /* Compute hash from key values already deformed in slot */
        hashval = 0;
        for (i = 0; i < state->num_keys; i++)
        {
            Datum d = slot->tts_values[state->inner_keynos[i] - 1];
            uint32 h;
            if (vjoin_is_fast_type(state->key_types[i]))
                h = vjoin_hash_datum(d, state->key_types[i]);
            else
                h = vjoin_hash_datum_generic(d,
                                             &state->hash_finfo[i],
                                             state->key_collations[i]);
            hashval = (i == 0) ? h : vjoin_combine_hashes(hashval, h);
        }

        /* Insert into hash table with full deformed values */
        vjoin_ht_insert(state->hashtable, hashval,
                        slot->tts_values, slot->tts_isnull);
    }

    MemoryContextSwitchTo(oldctx);

    /* Cache entry count before DSA (for EXPLAIN after DSM detach) */
    state->cached_ht_entries = state->hashtable->num_entries;

    /* In parallel mode, serialize HT to DSA and signal workers */
    if (state->is_parallel)
    {
        vjoin_ht_serialize_to_dsa(state->hashtable, state->dsa, state->pstate);
        BarrierArriveAndWait(&state->pstate->barrier, 0);
    }

    /* Allocate inner_matched after build so we know capacity */
    if (state->jointype == JOIN_RIGHT || state->jointype == JOIN_FULL)
    {
        VJoinHashTable *ht = state->hashtable;
        state->inner_matched = palloc0(sizeof(bool) * ht->capacity);
    }

    state->phase = VHJ_PROBE;
}

/*
 * Copy outer tuple attributes from slot into batch Datum arrays.
 * For pass-by-value types: direct Datum copy.
 * For pass-by-ref types: datumCopy into batch_ctx.
 */
static inline void
vjoin_extract_outer_attrs(VectorHashJoinState *state,
                          TupleTableSlot *slot, int batch_idx)
{
    int off = batch_idx * state->num_outer_attrs;

    slot_getallattrs(slot);

    if (state->batch_all_byval)
    {
        /* Fast path: all pass-by-value, just memcpy */
        memcpy(&state->batch_values[off], slot->tts_values,
               state->num_outer_attrs * sizeof(Datum));
        memcpy(&state->batch_isnull[off], slot->tts_isnull,
               state->num_outer_attrs * sizeof(bool));
    }
    else
    {
        int i;
        for (i = 0; i < state->num_outer_attrs; i++)
        {
            state->batch_isnull[off + i] = slot->tts_isnull[i];
            if (slot->tts_isnull[i] || state->outer_byval[i])
                state->batch_values[off + i] = slot->tts_values[i];
            else
                state->batch_values[off + i] =
                    datumCopy(slot->tts_values[i], false,
                              state->outer_typlen[i]);
        }
    }
}

/*
 * Fetch a batch of outer tuples and probe the hash table.
 */
static void
vjoin_hash_probe_batch(VectorHashJoinState *state)
{
    MemoryContext oldctx;
    TupleTableSlot *slot;
    VJoinHashTable *ht = state->hashtable;
    int batch_idx;
    int noa = state->num_outer_attrs;

    oldctx = MemoryContextSwitchTo(state->batch_ctx);
    MemoryContextReset(state->batch_ctx);

    state->batch_count = 0;
    state->result_count = 0;
    state->result_pos = 0;

    /* Pull up to batch_size outer tuples */
    for (batch_idx = 0; batch_idx < state->batch_size; batch_idx++)
    {
        bool has_null = false;
        uint32 hashval = 0;
        int i, off;

        /* Switch to parent context for ExecProcNode */
        MemoryContextSwitchTo(oldctx);
        slot = ExecProcNode(state->outer_ps);
        MemoryContextSwitchTo(state->batch_ctx);
        if (TupIsNull(slot))
            break;

        /* Extract all outer attributes into batch Datum arrays */
        vjoin_extract_outer_attrs(state, slot, batch_idx);

        /* Compute hash from key values within batch_values */
        off = batch_idx * noa;
        for (i = 0; i < state->num_keys; i++)
        {
            int keyoff = state->outer_keynos[i] - 1;  /* 1-based → 0-based */
            if (state->batch_isnull[off + keyoff])
            {
                has_null = true;
                break;
            }
            else
            {
                Datum d = state->batch_values[off + keyoff];
                uint32 h;
                if (vjoin_is_fast_type(state->key_types[i]))
                    h = vjoin_hash_datum(d, state->key_types[i]);
                else
                    h = vjoin_hash_datum_generic(d,
                                                 &state->hash_finfo[i],
                                                 state->key_collations[i]);
                hashval = (i == 0) ? h : vjoin_combine_hashes(hashval, h);
            }
        }

        state->batch_hashes[batch_idx] = has_null ? 0 : hashval;
        state->batch_count++;
    }

    /* Reset per-batch matched tracking for LEFT/FULL */
    if (state->batch_matched)
        memset(state->batch_matched, 0, sizeof(bool) * state->batch_size);

    /* Probe hash table for each outer tuple in batch */
    for (batch_idx = 0; batch_idx < state->batch_count; batch_idx++)
    {
        uint32  hashval = state->batch_hashes[batch_idx];
        int     pos;
        int     ooff;

        /* NULL keys never match — hash was set to 0 */
        if (hashval == 0)
            continue;

        ooff = batch_idx * noa;

        /* Linear probe in hash table */
        pos = hashval & ht->mask;
        while (ht->hashvals[pos] != 0)
        {
            if (ht->hashvals[pos] == hashval)
            {
                /* Compare all keys via all_values at inner_keynos offsets */
                bool match = true;
                int ioff = pos * ht->num_all_attrs;
                int k;

                for (k = 0; k < state->num_keys; k++)
                {
                    int inner_attr = ht->inner_keynos[k] - 1;
                    Datum outer_key = state->batch_values[
                        ooff + (state->outer_keynos[k] - 1)];
                    Datum inner_key = ht->all_values[ioff + inner_attr];

                    /* NULL inner keys never match */
                    if (ht->all_isnull[ioff + inner_attr])
                    {
                        match = false;
                        break;
                    }

                    switch (state->key_types[k])
                    {
                        case INT4OID:
                            if (DatumGetInt32(outer_key) !=
                                DatumGetInt32(inner_key))
                                match = false;
                            break;
                        case INT8OID:
                            if (DatumGetInt64(outer_key) !=
                                DatumGetInt64(inner_key))
                                match = false;
                            break;
                        case FLOAT8OID:
                            if (DatumGetFloat8(outer_key) !=
                                DatumGetFloat8(inner_key))
                                match = false;
                            break;
                        default:
                            if (!DatumGetBool(
                                    FunctionCall2Coll(
                                        &state->eq_finfo[k],
                                        state->key_collations[k],
                                        outer_key, inner_key)))
                                match = false;
                            break;
                    }
                    if (!match)
                        break;
                }

                if (match)
                {
                    /* Add to result buffer, grow if needed */
                    if (state->result_count >= state->result_capacity)
                    {
                        state->result_capacity *= 2;
                        state->results = repalloc(state->results,
                            sizeof(VJoinMatch) * state->result_capacity);
                    }
                    state->results[state->result_count].outer_idx = batch_idx;
                    state->results[state->result_count].inner_idx = pos;
                    state->result_count++;

                    /* Mark matched for outer join tracking */
                    if (state->batch_matched)
                        state->batch_matched[batch_idx] = true;
                    if (state->inner_matched)
                        state->inner_matched[pos] = true;
                }
            }
            pos = (pos + 1) & ht->mask;
        }
    }

    MemoryContextSwitchTo(oldctx);

    if (state->result_count > 0)
        state->phase = VHJ_EMIT;
    else if (state->batch_count < state->batch_size)
    {
        /* Outer exhausted — check for RIGHT/FULL emit */
        if ((state->jointype == JOIN_RIGHT || state->jointype == JOIN_FULL) &&
            state->inner_matched)
            state->phase = VHJ_RIGHT_EMIT;
        else
            state->phase = VHJ_DONE;
    }
    /* else: stay in VHJ_PROBE to fetch next batch */
}

/* ---- Public callbacks ---- */

void
vjoin_hash_begin(CustomScanState *node, EState *estate, int eflags)
{
    VectorHashJoinState *state = (VectorHashJoinState *) node;
    CustomScan *cscan = (CustomScan *) node->ss.ps.plan;
    PlanState  *outer_ps,
               *inner_ps;
    TupleDesc   outer_desc,
                inner_desc;
    int         i;

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
    state->outer_desc = CreateTupleDescCopy(outer_desc);
    state->inner_desc = CreateTupleDescCopy(inner_desc);

    /* Deserialize key info (jointype is first element) */
    vjoin_deserialize_keys(cscan->custom_private,
                           &state->jointype,
                           &state->num_keys,
                           state->outer_keynos,
                           state->inner_keynos,
                           state->key_types,
                           state->hash_funcs,
                           state->eq_funcs,
                           state->key_collations);

    /* Set up FmgrInfo caches and key type metadata */
    state->keys_all_byval = true;
    for (i = 0; i < state->num_keys; i++)
    {
        get_typlenbyval(state->key_types[i],
                        &state->key_typlen[i],
                        &state->key_byval[i]);
        if (!state->key_byval[i])
            state->keys_all_byval = false;

        /* Set up generic hash/eq functions for non-numeric types */
        if (OidIsValid(state->hash_funcs[i]))
            fmgr_info(state->hash_funcs[i], &state->hash_finfo[i]);
        if (OidIsValid(state->eq_funcs[i]))
            fmgr_info(get_opcode(state->eq_funcs[i]), &state->eq_finfo[i]);
    }

    /* Detect pass-by-value for outer attrs (fast batch extraction) */
    state->outer_byval = palloc(sizeof(bool) * state->num_outer_attrs);
    state->outer_typlen = palloc(sizeof(int16) * state->num_outer_attrs);
    state->batch_all_byval = true;
    for (i = 0; i < state->num_outer_attrs; i++)
    {
        get_typlenbyval(TupleDescAttr(outer_desc, i)->atttypid,
                        &state->outer_typlen[i],
                        &state->outer_byval[i]);
        if (!state->outer_byval[i])
            state->batch_all_byval = false;
    }

    /* Memory contexts */
    state->hash_ctx = AllocSetContextCreate(CurrentMemoryContext,
                                            "VectorHashJoin hash",
                                            ALLOCSET_DEFAULT_SIZES);
    state->batch_ctx = AllocSetContextCreate(CurrentMemoryContext,
                                             "VectorHashJoin batch",
                                             ALLOCSET_DEFAULT_SIZES);

    /* Initialize parallel fields — will be set properly by DSM callbacks */
    state->is_parallel = false;
    state->is_leader = false;
    state->pstate = NULL;
    state->dsa = NULL;
    state->cached_ht_entries = 0;

    /* Create hash table (estimate based on inner plan rows).
     * In parallel-worker mode the HT is created later from DSA,
     * but we don't know yet whether we're a worker — that's set by
     * initialize_worker.  Create it unconditionally; workers will
     * destroy it and replace with the shared wrapper during build. */
    {
        double inner_rows = inner_ps->plan->plan_rows;
        bool   *inner_byval;
        int16  *inner_typlen;

        if (inner_rows < 64)
            inner_rows = 64;

        /* Gather inner attr type metadata for pass-by-ref handling */
        inner_byval = palloc(sizeof(bool) * state->num_inner_attrs);
        inner_typlen = palloc(sizeof(int16) * state->num_inner_attrs);
        for (i = 0; i < state->num_inner_attrs; i++)
        {
            get_typlenbyval(TupleDescAttr(inner_desc, i)->atttypid,
                            &inner_typlen[i],
                            &inner_byval[i]);
        }

        state->hashtable = vjoin_ht_create((int) inner_rows,
                                           state->num_keys,
                                           state->num_inner_attrs,
                                           state->hash_ctx,
                                           state->inner_keynos,
                                           inner_byval,
                                           inner_typlen);

        pfree(inner_byval);
        pfree(inner_typlen);
    }

    /* Allocate batch buffers */
    state->batch_size = vjoin_batch_size;
    state->batch_hashes = palloc(sizeof(uint32) * state->batch_size);
    state->batch_values = palloc(sizeof(Datum) * state->batch_size *
                                 state->num_outer_attrs);
    state->batch_isnull = palloc(sizeof(bool) * state->batch_size *
                                 state->num_outer_attrs);

    /* Result buffer */
    state->result_capacity = state->batch_size * 4;
    state->results = palloc(sizeof(VJoinMatch) * state->result_capacity);
    state->result_count = 0;
    state->result_pos = 0;

    /* Outer join tracking arrays */
    if (state->jointype == JOIN_LEFT || state->jointype == JOIN_FULL)
        state->batch_matched = palloc0(sizeof(bool) * state->batch_size);
    else
        state->batch_matched = NULL;

    state->inner_matched = NULL;  /* allocated after build phase */
    state->left_emit_pos = 0;
    state->right_emit_pos = 0;

    /* SIMD detection */
    state->use_simd = vjoin_simd_caps.has_avx2 || vjoin_simd_caps.has_sse2 ||
                      vjoin_simd_caps.has_neon;

    state->phase = VHJ_BUILD;
}

TupleTableSlot *
vjoin_hash_exec(CustomScanState *node)
{
    VectorHashJoinState *state = (VectorHashJoinState *) node;
    ExprContext *econtext = node->ss.ps.ps_ExprContext;
    ExprState  *qual = node->ss.ps.qual;
    ProjectionInfo *projInfo = node->ss.ps.ps_ProjInfo;

    for (;;)
    {
        TupleTableSlot *result;

        switch (state->phase)
        {
            case VHJ_BUILD:
                vjoin_hash_build(state);
                continue;

            case VHJ_PROBE:
                vjoin_hash_probe_batch(state);
                continue;

            case VHJ_EMIT:
                if (state->result_pos >= state->result_count)
                {
                    /* Buffer exhausted — check for unmatched outers */
                    if (state->batch_matched &&
                        (state->jointype == JOIN_LEFT ||
                         state->jointype == JOIN_FULL))
                    {
                        state->left_emit_pos = 0;
                        state->phase = VHJ_LEFT_EMIT;
                    }
                    else if (state->batch_count < state->batch_size)
                    {
                        if ((state->jointype == JOIN_RIGHT ||
                             state->jointype == JOIN_FULL) &&
                            state->inner_matched)
                            state->phase = VHJ_RIGHT_EMIT;
                        else
                            state->phase = VHJ_DONE;
                    }
                    else
                        state->phase = VHJ_PROBE;
                    continue;
                }

                {
                    int oi = state->results[state->result_pos].outer_idx;
                    int ii = state->results[state->result_pos].inner_idx;
                    state->result_pos++;

                    result = vjoin_form_result(state, oi, ii);

                    ResetExprContext(econtext);
                    econtext->ecxt_scantuple = result;

                    /* Apply remaining quals */
                    if (qual && !ExecQual(qual, econtext))
                        continue;

                    /* Project result */
                    if (projInfo)
                        return ExecProject(projInfo);
                    return result;
                }

            case VHJ_LEFT_EMIT:
                /* Emit unmatched outer tuples with NULL inner */
                while (state->left_emit_pos < state->batch_count)
                {
                    int idx = state->left_emit_pos++;
                    if (!state->batch_matched[idx])
                    {
                        result = vjoin_form_result_left(state, idx);
                        ResetExprContext(econtext);
                        econtext->ecxt_scantuple = result;
                        if (projInfo)
                            return ExecProject(projInfo);
                        return result;
                    }
                }
                /* All unmatched emitted — continue to next batch or done */
                if (state->batch_count < state->batch_size)
                {
                    if ((state->jointype == JOIN_RIGHT ||
                         state->jointype == JOIN_FULL) &&
                        state->inner_matched)
                        state->phase = VHJ_RIGHT_EMIT;
                    else
                        state->phase = VHJ_DONE;
                }
                else
                    state->phase = VHJ_PROBE;
                continue;

            case VHJ_RIGHT_EMIT:
                /* Emit unmatched inner tuples with NULL outer */
                {
                    VJoinHashTable *ht = state->hashtable;
                    while (state->right_emit_pos < ht->capacity)
                    {
                        int pos = state->right_emit_pos++;
                        if (ht->hashvals[pos] != 0 &&
                            !state->inner_matched[pos])
                        {
                            result = vjoin_form_result_right(state, pos);
                            ResetExprContext(econtext);
                            econtext->ecxt_scantuple = result;
                            if (projInfo)
                                return ExecProject(projInfo);
                            return result;
                        }
                    }
                }
                state->phase = VHJ_DONE;
                continue;

            case VHJ_DONE:
                return NULL;
        }
    }
}

void
vjoin_hash_end(CustomScanState *node)
{
    VectorHashJoinState *state = (VectorHashJoinState *) node;

    ExecEndNode(state->outer_ps);
    ExecEndNode(state->inner_ps);

    if (state->outer_desc)
        FreeTupleDesc(state->outer_desc);
    if (state->inner_desc)
        FreeTupleDesc(state->inner_desc);

    /*
     * For parallel workers, the HT wrapper points into DSA shared memory.
     * We must not pfree those arrays — just destroy the wrapper context.
     * For leader/non-parallel, the HT owns its own allocations.
     */
    if (state->hashtable)
        vjoin_ht_destroy(state->hashtable);
    if (state->hash_ctx)
        MemoryContextDelete(state->hash_ctx);
    if (state->batch_ctx)
        MemoryContextDelete(state->batch_ctx);

    /* Detach DSA — the shared memory persists until leader destroys it */
    if (state->dsa)
    {
        dsa_detach(state->dsa);
        state->dsa = NULL;
    }
    state->hashtable = NULL;
    state->pstate = NULL;
}

void
vjoin_hash_rescan(CustomScanState *node)
{
    VectorHashJoinState *state = (VectorHashJoinState *) node;

    ExecReScan(state->outer_ps);
    /* Inner doesn't need rescan — hash table remains valid */

    state->phase = VHJ_PROBE;
    state->batch_count = 0;
    state->result_count = 0;
    state->result_pos = 0;
    state->left_emit_pos = 0;
    state->right_emit_pos = 0;

    /* Reset inner matched tracking — hash table is reused */
    if (state->inner_matched && state->hashtable)
        memset(state->inner_matched, 0,
               sizeof(bool) * state->hashtable->capacity);
}

void
vjoin_hash_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
    VectorHashJoinState *state = (VectorHashJoinState *) node;
    const char *jt;

    switch (state->jointype)
    {
        case JOIN_LEFT:  jt = "Left";  break;
        case JOIN_RIGHT: jt = "Right"; break;
        case JOIN_FULL:  jt = "Full";  break;
        default:         jt = "Inner"; break;
    }
    ExplainPropertyText("Join Type", jt, es);

    /* Use cached count — safe even after DSM detach */
    ExplainPropertyInteger("Hash Table Size", NULL,
                           state->cached_ht_entries, es);
    ExplainPropertyInteger("Batch Size", NULL, state->batch_size, es);
    ExplainPropertyBool("SIMD", state->use_simd, es);
}

/* ----------------------------------------------------------------
 *      Parallel DSM callbacks — Leader-builds, Workers-probe
 *
 * The leader creates a DSA area and initializes a Barrier in the
 * shared VJoinParallelState.  During the build phase, only the leader
 * reads the inner child and populates the hash table.  It then copies
 * the flat arrays into DSA and arrives at the barrier.  Workers wait
 * at the barrier, then create a read-only HT wrapper pointing at the
 * shared DSA arrays.  All participants then probe with their own
 * partial outer scan.
 * ---------------------------------------------------------------- */

Size
vjoin_hash_estimate_dsm(CustomScanState *node, ParallelContext *pcxt)
{
    return sizeof(VJoinParallelState);
}

void
vjoin_hash_initialize_dsm(CustomScanState *node, ParallelContext *pcxt,
                           void *coordinate)
{
    VectorHashJoinState *state = (VectorHashJoinState *) node;
    VJoinParallelState *pstate = (VJoinParallelState *) coordinate;

    /* Create DSA in the DSM segment */
    state->dsa = dsa_create(LWTRANCHE_PARALLEL_HASH_JOIN);
    dsa_pin_mapping(state->dsa);

    /* Initialize the shared state */
    pstate->dsa_handle = dsa_get_handle(state->dsa);
    BarrierInit(&pstate->barrier, pcxt->nworkers_to_launch + 1);
    pstate->num_entries = 0;

    state->pstate = pstate;
    state->is_parallel = true;
    state->is_leader = true;
}

void
vjoin_hash_reinitialize_dsm(CustomScanState *node, ParallelContext *pcxt,
                             void *coordinate)
{
    /* Rescan is not supported for parallel VHJ — hash table stays valid */
}

void
vjoin_hash_initialize_worker(CustomScanState *node, shm_toc *toc,
                              void *coordinate)
{
    VectorHashJoinState *state = (VectorHashJoinState *) node;
    VJoinParallelState *pstate = (VJoinParallelState *) coordinate;

    /* Attach to the DSA area created by the leader */
    state->dsa = dsa_attach(pstate->dsa_handle);
    dsa_pin_mapping(state->dsa);

    state->pstate = pstate;
    state->is_parallel = true;
    state->is_leader = false;
}

void
vjoin_hash_shutdown(CustomScanState *node)
{
    /* All cleanup handled by vjoin_hash_end */
}
