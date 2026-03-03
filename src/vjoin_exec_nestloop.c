#include "postgres.h"
#include "access/htup_details.h"
#include "executor/executor.h"
#include "executor/tuptable.h"
#include "nodes/extensible.h"
#include "nodes/value.h"
#include "utils/memutils.h"
#include "utils/tuplestore.h"
#include "miscadmin.h"
#include "pg_vectorjoin.h"
#include "vjoin_state.h"
#include "vjoin_simd.h"

/*
 * Deserialize key info from custom_private.
 */
static void
nl_deserialize_keys(List *private_data,
                     int *num_keys,
                     AttrNumber *outer_keynos,
                     AttrNumber *inner_keynos,
                     Oid *key_types)
{
    int idx = 0;
    int i;

    *num_keys = intVal(list_nth(private_data, idx++));
    for (i = 0; i < *num_keys; i++)
    {
        outer_keynos[i] = (AttrNumber) intVal(list_nth(private_data, idx++));
        inner_keynos[i] = (AttrNumber) intVal(list_nth(private_data, idx++));
        key_types[i] = (Oid) intVal(list_nth(private_data, idx++));
    }
}

/*
 * Fill the scan slot from matched outer + inner tuple.
 */
static TupleTableSlot *
nl_form_result(VJoinNestLoopState *state, MinimalTuple outer_mt,
                MinimalTuple inner_mt)
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
 * Load a block of outer tuples into the block buffer.
 * Returns the number of tuples loaded.
 */
static int
nl_load_outer_block(VJoinNestLoopState *state)
{
    TupleTableSlot *slot;
    MemoryContext old;
    int loaded = 0;
    int i;

    old = MemoryContextSwitchTo(state->block_ctx);
    MemoryContextReset(state->block_ctx);

    for (i = 0; i < state->block_size; i++)
    {
        int k;

        /* Pull outer tuple in parent context */
        MemoryContextSwitchTo(old);
        slot = ExecProcNode(state->outer_ps);
        MemoryContextSwitchTo(state->block_ctx);
        if (TupIsNull(slot))
        {
            state->outer_exhausted = true;
            break;
        }

        state->block_tuples[loaded] = ExecCopySlotMinimalTuple(slot);

        /* Extract keys into columnar arrays */
        for (k = 0; k < state->num_keys; k++)
        {
            bool isnull;
            Datum d = slot_getattr(slot, state->outer_keynos[k], &isnull);
            state->block_keys[loaded * state->num_keys + k] = d;
            state->block_nulls[loaded * state->num_keys + k] = isnull;
        }

        loaded++;
    }

    MemoryContextSwitchTo(old);
    state->block_count = loaded;
    return loaded;
}

/*
 * Compare one inner tuple against the entire outer block using SIMD
 * for single int4/int8/float8 key, scalar fallback otherwise.
 */
static void
nl_compare_block(VJoinNestLoopState *state,
                  Datum *inner_keys, bool *inner_nulls,
                  MinimalTuple inner_mt)
{
    int k, i;

    /* Single int4 key with SIMD */
    if (state->num_keys == 1 && state->key_types[0] == INT4OID &&
        !inner_nulls[0] && state->use_simd)
    {
        int32  inner_val = DatumGetInt32(inner_keys[0]);
        int   *match_indices;
        int    nmatches;
        int32 *key_array;

        key_array = (int32 *) palloc(sizeof(int32) * state->block_count);
        match_indices = (int *) palloc(sizeof(int) * state->block_count);

        for (i = 0; i < state->block_count; i++)
        {
            if (state->block_nulls[i * state->num_keys])
                key_array[i] = inner_val + 1;  /* ensure no match for null */
            else
                key_array[i] = DatumGetInt32(
                    state->block_keys[i * state->num_keys]);
        }

        nmatches = vjoin_compare_int4_block(key_array, state->block_count,
                                            inner_val, match_indices);

        for (i = 0; i < nmatches; i++)
        {
            if (state->result_count >= state->result_capacity)
            {
                state->result_capacity *= 2;
                state->results = repalloc(state->results,
                    sizeof(VJoinMatch) * state->result_capacity);
                state->result_inner_tuples = repalloc(state->result_inner_tuples,
                    sizeof(MinimalTuple) * state->result_capacity);
            }
            state->results[state->result_count].outer_idx = match_indices[i];
            state->results[state->result_count].inner_idx = 0;
            state->result_inner_tuples[state->result_count] = inner_mt;
            state->result_count++;
        }

        pfree(key_array);
        pfree(match_indices);
        return;
    }

    /* Single int8 key with SIMD */
    if (state->num_keys == 1 && state->key_types[0] == INT8OID &&
        !inner_nulls[0] && state->use_simd)
    {
        int64  inner_val = DatumGetInt64(inner_keys[0]);
        int   *match_indices;
        int    nmatches;
        int64 *key_array;

        key_array = (int64 *) palloc(sizeof(int64) * state->block_count);
        match_indices = (int *) palloc(sizeof(int) * state->block_count);

        for (i = 0; i < state->block_count; i++)
        {
            if (state->block_nulls[i * state->num_keys])
                key_array[i] = inner_val + 1;
            else
                key_array[i] = DatumGetInt64(
                    state->block_keys[i * state->num_keys]);
        }

        nmatches = vjoin_compare_int8_block(key_array, state->block_count,
                                            inner_val, match_indices);

        for (i = 0; i < nmatches; i++)
        {
            if (state->result_count >= state->result_capacity)
            {
                state->result_capacity *= 2;
                state->results = repalloc(state->results,
                    sizeof(VJoinMatch) * state->result_capacity);
                state->result_inner_tuples = repalloc(state->result_inner_tuples,
                    sizeof(MinimalTuple) * state->result_capacity);
            }
            state->results[state->result_count].outer_idx = match_indices[i];
            state->results[state->result_count].inner_idx = 0;
            state->result_inner_tuples[state->result_count] = inner_mt;
            state->result_count++;
        }

        pfree(key_array);
        pfree(match_indices);
        return;
    }

    /* Single float8 key with SIMD */
    if (state->num_keys == 1 && state->key_types[0] == FLOAT8OID &&
        !inner_nulls[0] && state->use_simd)
    {
        double  inner_val = DatumGetFloat8(inner_keys[0]);
        int    *match_indices;
        int     nmatches;
        double *key_array;

        key_array = (double *) palloc(sizeof(double) * state->block_count);
        match_indices = (int *) palloc(sizeof(int) * state->block_count);

        for (i = 0; i < state->block_count; i++)
        {
            if (state->block_nulls[i * state->num_keys])
                key_array[i] = inner_val + 1.0;
            else
                key_array[i] = DatumGetFloat8(
                    state->block_keys[i * state->num_keys]);
        }

        nmatches = vjoin_compare_float8_block(key_array, state->block_count,
                                              inner_val, match_indices);

        for (i = 0; i < nmatches; i++)
        {
            if (state->result_count >= state->result_capacity)
            {
                state->result_capacity *= 2;
                state->results = repalloc(state->results,
                    sizeof(VJoinMatch) * state->result_capacity);
                state->result_inner_tuples = repalloc(state->result_inner_tuples,
                    sizeof(MinimalTuple) * state->result_capacity);
            }
            state->results[state->result_count].outer_idx = match_indices[i];
            state->results[state->result_count].inner_idx = 0;
            state->result_inner_tuples[state->result_count] = inner_mt;
            state->result_count++;
        }

        pfree(key_array);
        pfree(match_indices);
        return;
    }

    /* Scalar fallback for multiple keys or unsupported types */
    for (i = 0; i < state->block_count; i++)
    {
        bool match = true;

        for (k = 0; k < state->num_keys; k++)
        {
            bool outer_null = state->block_nulls[i * state->num_keys + k];
            Datum outer_d,
                  inner_d;

            if (outer_null || inner_nulls[k])
            {
                match = false;
                break;
            }

            outer_d = state->block_keys[i * state->num_keys + k];
            inner_d = inner_keys[k];

            switch (state->key_types[k])
            {
                case INT4OID:
                    if (DatumGetInt32(outer_d) != DatumGetInt32(inner_d))
                        match = false;
                    break;
                case INT8OID:
                    if (DatumGetInt64(outer_d) != DatumGetInt64(inner_d))
                        match = false;
                    break;
                case FLOAT8OID:
                    if (DatumGetFloat8(outer_d) != DatumGetFloat8(inner_d))
                        match = false;
                    break;
                default:
                    if (outer_d != inner_d)
                        match = false;
                    break;
            }
            if (!match)
                break;
        }

        if (match)
        {
            if (state->result_count >= state->result_capacity)
            {
                state->result_capacity *= 2;
                state->results = repalloc(state->results,
                    sizeof(VJoinMatch) * state->result_capacity);
                state->result_inner_tuples = repalloc(state->result_inner_tuples,
                    sizeof(MinimalTuple) * state->result_capacity);
            }
            state->results[state->result_count].outer_idx = i;
            state->results[state->result_count].inner_idx = 0;
            state->result_inner_tuples[state->result_count] = inner_mt;
            state->result_count++;
        }
    }
}

/*
 * Scan inner relation against current outer block.
 * Accumulates all matches in result buffer, then returns.
 */
static void
nl_scan_inner(VJoinNestLoopState *state)
{
    TupleTableSlot *slot;
    MemoryContext oldctx = MemoryContextSwitchTo(state->block_ctx);

    state->result_count = 0;
    state->result_pos = 0;

    for (;;)
    {
        Datum  inner_keys[VJOIN_MAX_KEYS];
        bool   inner_nulls[VJOIN_MAX_KEYS];
        int    k;
        MinimalTuple inner_mt;

        MemoryContextSwitchTo(oldctx);

        if (state->inner_stored)
        {
            /* Read from tuplestore */
            if (!tuplestore_gettupleslot(state->inner_store, true, false,
                                         state->store_slot))
            {
                state->inner_exhausted = true;
                MemoryContextSwitchTo(state->block_ctx);
                MemoryContextSwitchTo(oldctx);
                return;
            }
            slot = state->store_slot;
        }
        else
        {
            /* First pass: pull from child and materialize */
            slot = ExecProcNode(state->inner_ps);
            if (TupIsNull(slot))
            {
                state->inner_exhausted = true;
                state->inner_stored = true;
                MemoryContextSwitchTo(state->block_ctx);
                MemoryContextSwitchTo(oldctx);
                return;
            }
            tuplestore_puttupleslot(state->inner_store, slot);
        }

        MemoryContextSwitchTo(state->block_ctx);

        /* Extract inner keys */
        for (k = 0; k < state->num_keys; k++)
        {
            inner_keys[k] = slot_getattr(slot, state->inner_keynos[k],
                                         &inner_nulls[k]);
        }

        /* Skip if any inner key is NULL */
        {
            bool has_null = false;
            for (k = 0; k < state->num_keys; k++)
            {
                if (inner_nulls[k])
                {
                    has_null = true;
                    break;
                }
            }
            if (has_null)
                continue;
        }

        /* Copy inner tuple for result construction */
        inner_mt = ExecCopySlotMinimalTuple(slot);

        /* Compare against outer block */
        nl_compare_block(state, inner_keys, inner_nulls, inner_mt);

        /*
         * If we have accumulated enough results, stop scanning inner
         * for now and emit them. We'll resume scanning on next call.
         */
        if (state->result_count > 0)
        {
            MemoryContextSwitchTo(oldctx);
            return;
        }
    }
}

/* ---- Public callbacks ---- */

void
vjoin_nestloop_begin(CustomScanState *node, EState *estate, int eflags)
{
    VJoinNestLoopState *state = (VJoinNestLoopState *) node;
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

    /* Deserialize key info */
    nl_deserialize_keys(cscan->custom_private,
                         &state->num_keys,
                         state->outer_keynos,
                         state->inner_keynos,
                         state->key_types);

    /* Block buffer */
    state->block_size = vjoin_batch_size;
    state->block_keys = palloc(sizeof(Datum) * state->block_size *
                               state->num_keys);
    state->block_nulls = palloc(sizeof(bool) * state->block_size *
                                state->num_keys);
    state->block_tuples = palloc(sizeof(MinimalTuple) * state->block_size);

    /* Result buffer */
    state->result_capacity = state->block_size * 4;
    state->results = palloc(sizeof(VJoinMatch) * state->result_capacity);
    state->result_inner_tuples = palloc(sizeof(MinimalTuple) *
                                        state->result_capacity);
    state->result_count = 0;
    state->result_pos = 0;

    /* Temp slots */
    state->outer_slot = MakeSingleTupleTableSlot(outer_desc,
                                                 &TTSOpsMinimalTuple);
    state->inner_slot = MakeSingleTupleTableSlot(inner_desc,
                                                 &TTSOpsMinimalTuple);

    /* Memory context for block-scoped allocations */
    state->block_ctx = AllocSetContextCreate(CurrentMemoryContext,
                                             "NestLoop block",
                                             ALLOCSET_DEFAULT_SIZES);

    state->use_simd = vjoin_simd_caps.has_avx2 || vjoin_simd_caps.has_sse2 ||
                      vjoin_simd_caps.has_neon;

    /* Inner materialization */
    state->inner_store = tuplestore_begin_heap(false, false, work_mem);
    state->inner_stored = false;
    state->store_slot = MakeSingleTupleTableSlot(inner_desc,
                                                 &TTSOpsMinimalTuple);

    state->inner_exhausted = false;
    state->outer_exhausted = false;
    state->phase = NL_LOAD_BLOCK;
}

TupleTableSlot *
vjoin_nestloop_exec(CustomScanState *node)
{
    VJoinNestLoopState *state = (VJoinNestLoopState *) node;
    ExprContext *econtext = node->ss.ps.ps_ExprContext;
    ExprState  *qual = node->ss.ps.qual;
    ProjectionInfo *projInfo = node->ss.ps.ps_ProjInfo;

    for (;;)
    {
        TupleTableSlot *result;

        switch (state->phase)
        {
            case NL_LOAD_BLOCK:
                if (state->outer_exhausted)
                {
                    state->phase = NL_DONE;
                    continue;
                }

                if (nl_load_outer_block(state) == 0)
                {
                    state->phase = NL_DONE;
                    continue;
                }

                /* Rescan inner: use tuplestore if already materialized */
                state->inner_exhausted = false;
                if (state->inner_stored)
                    tuplestore_rescan(state->inner_store);
                else
                    ExecReScan(state->inner_ps);
                state->phase = NL_SCAN_INNER;
                continue;

            case NL_SCAN_INNER:
                nl_scan_inner(state);
                if (state->result_count > 0)
                {
                    state->phase = NL_EMIT;
                    continue;
                }

                if (state->inner_exhausted)
                {
                    /* This block done, load next */
                    state->phase = NL_LOAD_BLOCK;
                    continue;
                }

                /* No matches yet but inner not exhausted — keep scanning */
                continue;

            case NL_EMIT:
                if (state->result_pos >= state->result_count)
                {
                    /* Buffer exhausted */
                    state->result_count = 0;
                    state->result_pos = 0;

                    if (state->inner_exhausted)
                        state->phase = NL_LOAD_BLOCK;
                    else
                        state->phase = NL_SCAN_INNER;
                    continue;
                }

                {
                    int oi = state->results[state->result_pos].outer_idx;
                    MinimalTuple inner_mt =
                        state->result_inner_tuples[state->result_pos];
                    state->result_pos++;

                    result = nl_form_result(state,
                                            state->block_tuples[oi],
                                            inner_mt);

                    ResetExprContext(econtext);
                    econtext->ecxt_scantuple = result;

                    if (qual && !ExecQual(qual, econtext))
                        continue;

                    if (projInfo)
                        return ExecProject(projInfo);
                    return result;
                }

            case NL_DONE:
                return NULL;
        }
    }
}

void
vjoin_nestloop_end(CustomScanState *node)
{
    VJoinNestLoopState *state = (VJoinNestLoopState *) node;

    ExecEndNode(state->outer_ps);
    ExecEndNode(state->inner_ps);

    ExecDropSingleTupleTableSlot(state->outer_slot);
    ExecDropSingleTupleTableSlot(state->inner_slot);
    ExecDropSingleTupleTableSlot(state->store_slot);

    if (state->inner_store)
        tuplestore_end(state->inner_store);

    if (state->block_ctx)
        MemoryContextDelete(state->block_ctx);
}

void
vjoin_nestloop_rescan(CustomScanState *node)
{
    VJoinNestLoopState *state = (VJoinNestLoopState *) node;

    ExecReScan(state->outer_ps);

    /* Reset inner materialization */
    tuplestore_clear(state->inner_store);
    state->inner_stored = false;
    ExecReScan(state->inner_ps);

    state->phase = NL_LOAD_BLOCK;
    state->block_count = 0;
    state->result_count = 0;
    state->result_pos = 0;
    state->inner_exhausted = false;
    state->outer_exhausted = false;
}

void
vjoin_nestloop_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
    VJoinNestLoopState *state = (VJoinNestLoopState *) node;

    ExplainPropertyInteger("Block Size", NULL, state->block_size, es);
    ExplainPropertyBool("SIMD", state->use_simd, es);
}
