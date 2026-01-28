#include "postgres.h"
#include "access/htup_details.h"
#include "executor/executor.h"
#include "executor/tuptable.h"
#include "nodes/extensible.h"
#include "nodes/value.h"
#include "utils/memutils.h"
#include "pg_vectorjoin.h"
#include "vjoin_state.h"

/*
 * Deserialize key info from custom_private.
 */
static void
bnl_deserialize_keys(List *private_data,
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
bnl_form_result(BlockNestLoopState *state, MinimalTuple outer_mt,
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
bnl_load_outer_block(BlockNestLoopState *state)
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
 * Compare one inner tuple against the entire outer block (scalar).
 * SIMD fast paths will be wired in a later commit.
 */
static void
bnl_compare_block(BlockNestLoopState *state,
                  Datum *inner_keys, bool *inner_nulls,
                  MinimalTuple inner_mt)
{
    int k, i;

    /* Scalar comparison for all key types */
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
bnl_scan_inner(BlockNestLoopState *state)
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

        /* Pull inner tuple in parent context */
        MemoryContextSwitchTo(oldctx);
        slot = ExecProcNode(state->inner_ps);
        MemoryContextSwitchTo(state->block_ctx);
        if (TupIsNull(slot))
        {
            state->inner_exhausted = true;
            MemoryContextSwitchTo(oldctx);
            return;
        }

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
        bnl_compare_block(state, inner_keys, inner_nulls, inner_mt);

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
vjoin_bnl_begin(CustomScanState *node, EState *estate, int eflags)
{
    BlockNestLoopState *state = (BlockNestLoopState *) node;
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
    bnl_deserialize_keys(cscan->custom_private,
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
                                             "BlockNestLoop block",
                                             ALLOCSET_DEFAULT_SIZES);

    /* SIMD not yet wired — will be enabled in later commit */
    state->use_simd = false;

    state->inner_exhausted = false;
    state->outer_exhausted = false;
    state->phase = BNL_LOAD_BLOCK;
}

TupleTableSlot *
vjoin_bnl_exec(CustomScanState *node)
{
    BlockNestLoopState *state = (BlockNestLoopState *) node;
    ExprContext *econtext = node->ss.ps.ps_ExprContext;
    ExprState  *qual = node->ss.ps.qual;
    ProjectionInfo *projInfo = node->ss.ps.ps_ProjInfo;

    for (;;)
    {
        TupleTableSlot *result;

        switch (state->phase)
        {
            case BNL_LOAD_BLOCK:
                if (state->outer_exhausted)
                {
                    state->phase = BNL_DONE;
                    continue;
                }

                if (bnl_load_outer_block(state) == 0)
                {
                    state->phase = BNL_DONE;
                    continue;
                }

                /* Rescan inner for new block */
                state->inner_exhausted = false;
                ExecReScan(state->inner_ps);
                state->phase = BNL_SCAN_INNER;
                continue;

            case BNL_SCAN_INNER:
                bnl_scan_inner(state);
                if (state->result_count > 0)
                {
                    state->phase = BNL_EMIT;
                    continue;
                }

                if (state->inner_exhausted)
                {
                    /* This block done, load next */
                    state->phase = BNL_LOAD_BLOCK;
                    continue;
                }

                /* No matches yet but inner not exhausted — keep scanning */
                continue;

            case BNL_EMIT:
                if (state->result_pos >= state->result_count)
                {
                    /* Buffer exhausted */
                    state->result_count = 0;
                    state->result_pos = 0;

                    if (state->inner_exhausted)
                        state->phase = BNL_LOAD_BLOCK;
                    else
                        state->phase = BNL_SCAN_INNER;
                    continue;
                }

                {
                    int oi = state->results[state->result_pos].outer_idx;
                    MinimalTuple inner_mt =
                        state->result_inner_tuples[state->result_pos];
                    state->result_pos++;

                    result = bnl_form_result(state,
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

            case BNL_DONE:
                return NULL;
        }
    }
}

void
vjoin_bnl_end(CustomScanState *node)
{
    BlockNestLoopState *state = (BlockNestLoopState *) node;

    ExecEndNode(state->outer_ps);
    ExecEndNode(state->inner_ps);

    ExecDropSingleTupleTableSlot(state->outer_slot);
    ExecDropSingleTupleTableSlot(state->inner_slot);

    if (state->block_ctx)
        MemoryContextDelete(state->block_ctx);
}

void
vjoin_bnl_rescan(CustomScanState *node)
{
    BlockNestLoopState *state = (BlockNestLoopState *) node;

    ExecReScan(state->outer_ps);
    ExecReScan(state->inner_ps);

    state->phase = BNL_LOAD_BLOCK;
    state->block_count = 0;
    state->result_count = 0;
    state->result_pos = 0;
    state->inner_exhausted = false;
    state->outer_exhausted = false;
}

void
vjoin_bnl_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
    BlockNestLoopState *state = (BlockNestLoopState *) node;

    ExplainPropertyInteger("Block Size", NULL, state->block_size, es);
    ExplainPropertyBool("SIMD", state->use_simd, es);
}
