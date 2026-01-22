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
vjoin_deserialize_keys(List *private_data,
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
 * Fill the scan slot from matched outer + inner tuples.
 */
static TupleTableSlot *
vjoin_form_result(VectorHashJoinState *state, MinimalTuple outer_mt,
                  MinimalTuple inner_mt)
{
    TupleTableSlot *scan_slot = state->css.ss.ss_ScanTupleSlot;
    TupleTableSlot *outer_slot = state->outer_slot;
    TupleTableSlot *inner_slot = state->inner_slot;

    ExecStoreMinimalTuple(outer_mt, outer_slot, false);
    slot_getallattrs(outer_slot);

    ExecStoreMinimalTuple(inner_mt, inner_slot, false);
    slot_getallattrs(inner_slot);

    ExecClearTuple(scan_slot);

    memcpy(scan_slot->tts_values,
           outer_slot->tts_values,
           state->num_outer_attrs * sizeof(Datum));
    memcpy(scan_slot->tts_isnull,
           outer_slot->tts_isnull,
           state->num_outer_attrs * sizeof(bool));

    memcpy(scan_slot->tts_values + state->num_outer_attrs,
           inner_slot->tts_values,
           state->num_inner_attrs * sizeof(Datum));
    memcpy(scan_slot->tts_isnull + state->num_outer_attrs,
           inner_slot->tts_isnull,
           state->num_inner_attrs * sizeof(bool));

    ExecStoreVirtualTuple(scan_slot);
    return scan_slot;
}

/*
 * Build phase: read all inner tuples into hash table.
 */
static void
vjoin_hash_build(VectorHashJoinState *state)
{
    TupleTableSlot *slot;
    MemoryContext oldctx;

    oldctx = MemoryContextSwitchTo(state->hash_ctx);

    for (;;)
    {
        bool    has_null;
        uint32  hashval;
        Datum   keyvals[VJOIN_MAX_KEYS];
        bool    keynulls[VJOIN_MAX_KEYS];
        int     i;

        /* Pull inner tuple in parent context, copy into hash_ctx */
        MemoryContextSwitchTo(oldctx);
        slot = ExecProcNode(state->inner_ps);
        MemoryContextSwitchTo(state->hash_ctx);
        if (TupIsNull(slot))
            break;

        /* Extract key values */
        has_null = false;
        for (i = 0; i < state->num_keys; i++)
        {
            keyvals[i] = slot_getattr(slot, state->inner_keynos[i],
                                      &keynulls[i]);
            if (keynulls[i])
                has_null = true;
        }

        /* Skip NULL keys — NULLs never match in equijoin */
        if (has_null)
            continue;

        /* Compute hash */
        hashval = 0;
        for (i = 0; i < state->num_keys; i++)
        {
            uint32 h = vjoin_hash_datum(keyvals[i], state->key_types[i]);
            hashval = (i == 0) ? h : vjoin_combine_hashes(hashval, h);
        }

        /* Insert into hash table */
        vjoin_ht_insert(state->hashtable, hashval,
                        ExecCopySlotMinimalTuple(slot),
                        keyvals, keynulls);
    }

    MemoryContextSwitchTo(oldctx);
    state->phase = VHJ_PROBE;
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
        int i;

        /* Switch to parent context for ExecProcNode */
        MemoryContextSwitchTo(oldctx);
        slot = ExecProcNode(state->outer_ps);
        MemoryContextSwitchTo(state->batch_ctx);
        if (TupIsNull(slot))
            break;

        state->batch_tuples[batch_idx] = ExecCopySlotMinimalTuple(slot);

        /* Extract keys and compute hash */
        for (i = 0; i < state->num_keys; i++)
        {
            bool isnull;
            Datum d = slot_getattr(slot, state->outer_keynos[i], &isnull);
            state->batch_keys[batch_idx * state->num_keys + i] = d;
            state->batch_nulls[batch_idx * state->num_keys + i] = isnull;
            if (isnull)
                has_null = true;
            else
            {
                uint32 h = vjoin_hash_datum(d, state->key_types[i]);
                hashval = (i == 0) ? h : vjoin_combine_hashes(hashval, h);
            }
        }

        state->batch_hashes[batch_idx] = has_null ? 0 : hashval;
        state->batch_count++;
    }

    /* Probe hash table for each outer tuple in batch */
    for (batch_idx = 0; batch_idx < state->batch_count; batch_idx++)
    {
        uint32  hashval = state->batch_hashes[batch_idx];
        int     pos;
        bool    any_null = false;
        int     k;

        /* Check for NULL in outer keys */
        for (k = 0; k < state->num_keys; k++)
        {
            if (state->batch_nulls[batch_idx * state->num_keys + k])
            {
                any_null = true;
                break;
            }
        }
        if (any_null)
            continue;

        /* Linear probe in hash table */
        pos = hashval & ht->mask;
        while (ht->hashvals[pos] != 0)
        {
            if (ht->hashvals[pos] == hashval)
            {
                /* Compare all keys */
                bool match = true;
                for (k = 0; k < state->num_keys; k++)
                {
                    Datum outer_key = state->batch_keys[
                        batch_idx * state->num_keys + k];
                    Datum inner_key = ht->keys[
                        pos * ht->num_keys + k];

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
                            if (outer_key != inner_key)
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
                }
            }
            pos = (pos + 1) & ht->mask;
        }
    }

    MemoryContextSwitchTo(oldctx);

    if (state->result_count > 0)
        state->phase = VHJ_EMIT;
    else if (state->batch_count < state->batch_size)
        state->phase = VHJ_DONE;
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
    vjoin_deserialize_keys(cscan->custom_private,
                           &state->num_keys,
                           state->outer_keynos,
                           state->inner_keynos,
                           state->key_types);

    /* Memory contexts */
    state->hash_ctx = AllocSetContextCreate(CurrentMemoryContext,
                                            "VectorHashJoin hash",
                                            ALLOCSET_DEFAULT_SIZES);
    state->batch_ctx = AllocSetContextCreate(CurrentMemoryContext,
                                             "VectorHashJoin batch",
                                             ALLOCSET_DEFAULT_SIZES);

    /* Create hash table (estimate based on inner plan rows) */
    {
        double inner_rows = inner_ps->plan->plan_rows;
        if (inner_rows < 64)
            inner_rows = 64;
        state->hashtable = vjoin_ht_create((int) inner_rows,
                                           state->num_keys,
                                           state->hash_ctx);
    }

    /* Allocate batch buffers */
    state->batch_size = vjoin_batch_size;
    state->batch_keys = palloc(sizeof(Datum) * state->batch_size *
                               state->num_keys);
    state->batch_nulls = palloc(sizeof(bool) * state->batch_size *
                                state->num_keys);
    state->batch_hashes = palloc(sizeof(uint32) * state->batch_size);
    state->batch_tuples = palloc(sizeof(MinimalTuple) * state->batch_size);

    /* Result buffer */
    state->result_capacity = state->batch_size * 4;
    state->results = palloc(sizeof(VJoinMatch) * state->result_capacity);
    state->result_count = 0;
    state->result_pos = 0;

    /* Temp slots for result construction */
    state->outer_slot = MakeSingleTupleTableSlot(outer_desc,
                                                 &TTSOpsMinimalTuple);
    state->inner_slot = MakeSingleTupleTableSlot(inner_desc,
                                                 &TTSOpsMinimalTuple);

    /* SIMD not yet wired — will be enabled in later commit */
    state->use_simd = false;

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
                    /* Buffer exhausted — get next batch */
                    if (state->batch_count < state->batch_size)
                        state->phase = VHJ_DONE;
                    else
                        state->phase = VHJ_PROBE;
                    continue;
                }

                {
                    int oi = state->results[state->result_pos].outer_idx;
                    int ii = state->results[state->result_pos].inner_idx;
                    state->result_pos++;

                    result = vjoin_form_result(state,
                                              state->batch_tuples[oi],
                                              state->hashtable->tuples[ii]);

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

    ExecDropSingleTupleTableSlot(state->outer_slot);
    ExecDropSingleTupleTableSlot(state->inner_slot);

    if (state->hashtable)
        vjoin_ht_destroy(state->hashtable);
    if (state->hash_ctx)
        MemoryContextDelete(state->hash_ctx);
    if (state->batch_ctx)
        MemoryContextDelete(state->batch_ctx);
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
}

void
vjoin_hash_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
    VectorHashJoinState *state = (VectorHashJoinState *) node;

    ExplainPropertyInteger("Hash Table Size", NULL,
                           state->hashtable ? state->hashtable->num_entries : 0,
                           es);
    ExplainPropertyInteger("Batch Size", NULL, state->batch_size, es);
    ExplainPropertyBool("SIMD", state->use_simd, es);
}
