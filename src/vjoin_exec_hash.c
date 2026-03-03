#include "postgres.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "executor/executor.h"
#include "executor/tuptable.h"
#include "nodes/extensible.h"
#include "nodes/value.h"
#include "utils/datum.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "pg_vectorjoin.h"
#include "vjoin_state.h"
#include "vjoin_simd.h"

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

        /* Deform all inner attributes for fast emit */
        slot_getallattrs(slot);

        /* Insert into hash table with full deformed values */
        vjoin_ht_insert(state->hashtable, hashval,
                        ExecCopySlotMinimalTuple(slot),
                        keyvals, keynulls,
                        slot->tts_values, slot->tts_isnull);
    }

    MemoryContextSwitchTo(oldctx);
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

        /* Compute hash from pre-extracted key values */
        off = batch_idx * state->num_outer_attrs;
        for (i = 0; i < state->num_keys; i++)
        {
            int keyoff = state->outer_keynos[i] - 1;  /* 1-based → 0-based */
            bool isnull = state->batch_isnull[off + keyoff];
            Datum d = state->batch_values[off + keyoff];

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

    /* Deserialize key info */
    vjoin_deserialize_keys(cscan->custom_private,
                           &state->num_keys,
                           state->outer_keynos,
                           state->inner_keynos,
                           state->key_types);

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

    /* Create hash table (estimate based on inner plan rows) */
    {
        double inner_rows = inner_ps->plan->plan_rows;
        if (inner_rows < 64)
            inner_rows = 64;
        state->hashtable = vjoin_ht_create((int) inner_rows,
                                           state->num_keys,
                                           state->num_inner_attrs,
                                           state->hash_ctx);
    }

    /* Allocate batch buffers */
    state->batch_size = vjoin_batch_size;
    state->batch_keys = palloc(sizeof(Datum) * state->batch_size *
                               state->num_keys);
    state->batch_nulls = palloc(sizeof(bool) * state->batch_size *
                                state->num_keys);
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
