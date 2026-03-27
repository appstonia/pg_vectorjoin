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
#include "port/atomics.h"
#include "utils/datum.h"
#include "utils/dsa.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "pg_vectorjoin.h"
#include "vjoin_state.h"
#include "vjoin_simd.h"

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
    int inner_off = (inner_ht_pos >= 0) ? inner_ht_pos * ht->num_all_attrs : -1;

    ExecClearTuple(scan_slot);

    memcpy(scan_slot->tts_values,
           &state->batch_values[outer_off],
           state->num_outer_attrs * sizeof(Datum));
    memcpy(scan_slot->tts_isnull,
           &state->batch_isnull[outer_off],
           state->num_outer_attrs * sizeof(bool));

        if (inner_ht_pos >= 0)
        {
         memcpy(scan_slot->tts_values + state->num_outer_attrs,
             &ht->all_values[inner_off],
             state->num_inner_attrs * sizeof(Datum));
         memcpy(scan_slot->tts_isnull + state->num_outer_attrs,
             &ht->all_isnull[inner_off],
             state->num_inner_attrs * sizeof(bool));
        }
        else
        {
         memset(scan_slot->tts_values + state->num_outer_attrs,
             0,
             state->num_inner_attrs * sizeof(Datum));
         memset(scan_slot->tts_isnull + state->num_outer_attrs,
             true,
             state->num_inner_attrs * sizeof(bool));
        }

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
 *
 * When the inner child is parallel-aware and all inner attrs are byval,
 * all participants (leader + workers) scan inner concurrently and insert
 * into the shared DSA hash table using CAS-based lock-free insertion.
 */
static void
vjoin_hash_build(VectorHashJoinState *state)
{
    TupleTableSlot *slot;
    MemoryContext oldctx;
    bool parallel_build;

    /*
     * Determine if all participants should build concurrently.
     * Requires: parallel mode + byval inner attrs + parallel-aware inner scan.
     */
    parallel_build = state->is_parallel &&
                     state->hashtable->all_attrs_byval &&
                     state->inner_ps->plan->parallel_aware;

    /*
     * All-participants parallel build (byval + parallel inner scan).
     * Leader and workers all scan inner chunks and CAS-insert into shared HT.
     */
    if (parallel_build)
    {
        VJoinParallelState *ps = state->pstate;
        dsa_area   *dsa = state->dsa;
        int         na = state->num_inner_attrs;

        /* Destroy the local HT created during begin — we'll use shared DSA */
        if (state->is_leader)
        {
            /* Leader: save type metadata, create shared HT */
            bool  *saved_byval  = palloc(sizeof(bool) * na);
            int16 *saved_typlen = palloc(sizeof(int16) * na);

            memcpy(saved_byval, state->hashtable->attr_byval, sizeof(bool) * na);
            memcpy(saved_typlen, state->hashtable->attr_typlen, sizeof(int16) * na);

            vjoin_ht_destroy(state->hashtable);

            state->hashtable = vjoin_ht_create_shared(
                ps, dsa, state->num_keys, na,
                state->hash_ctx, state->inner_keynos,
                saved_byval, saved_typlen);

            /* Signal that workers can read DSA pointers for parallel build */
            pg_write_barrier();
            ps->parallel_build = true;

            pfree(saved_byval);
            pfree(saved_typlen);
        }
        else
        {
            /* Worker: wait for leader to set up shared HT metadata */
            while (!ps->parallel_build)
                pg_spin_delay();
            pg_read_barrier();

            /* Destroy local empty HT */
            vjoin_ht_destroy(state->hashtable);
            state->hashtable = NULL;

            /* Create thin wrapper pointing at shared DSA arrays */
            {
                MemoryContext  htctx;
                VJoinHashTable *ht;

                htctx = AllocSetContextCreate(state->hash_ctx,
                                              "VJoinHashTable (parallel-build)",
                                              ALLOCSET_DEFAULT_SIZES);
                ht = (VJoinHashTable *)
                    MemoryContextAllocZero(htctx, sizeof(VJoinHashTable));

                ht->htctx          = htctx;
                ht->capacity       = ps->capacity;
                ht->mask           = ps->mask;
                ht->num_entries    = 0;
                ht->num_all_attrs  = ps->num_all_attrs;
                ht->num_keys       = ps->num_keys;
                ht->all_attrs_byval = true;
                ht->is_shared      = true;
                ht->dsa            = dsa;
                ht->pstate         = ps;

                ht->hashvals     = (uint32 *)    dsa_get_address(dsa, ps->hashvals_dp);
                ht->all_values   = (Datum *)     dsa_get_address(dsa, ps->all_values_dp);
                ht->all_isnull   = (bool *)      dsa_get_address(dsa, ps->all_isnull_dp);
                ht->inner_keynos = (AttrNumber *) dsa_get_address(dsa, ps->inner_keynos_dp);
                ht->attr_byval   = NULL;
                ht->attr_typlen  = NULL;

                state->hashtable = ht;
            }
        }

        /* All participants: scan inner and CAS-insert */
        {
        int local_count = 0;
        bool had_overflow = false;
        List *overflow_tuples = NIL;   /* List of MinimalTuple */
        List *overflow_hashes = NIL;   /* parallel List of uint32 (boxed) */

        for (;;)
        {
            bool    has_null = false;
            uint32  hashval;
            int     i;

            CHECK_FOR_INTERRUPTS();

            slot = ExecProcNode(state->inner_ps);
            if (TupIsNull(slot))
                break;

            slot_getallattrs(slot);

            /* Check for NULL keys */
            for (i = 0; i < state->num_keys; i++)
            {
                if (slot->tts_isnull[state->inner_keynos[i] - 1])
                {
                    has_null = true;
                    break;
                }
            }
            if (has_null)
                continue;  /* skip NULLs in parallel build (INNER/LEFT only) */

            /* Compute hash */
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

            if (had_overflow)
            {
                /* Table was full — buffer remaining tuples locally */
                overflow_tuples = lappend(overflow_tuples,
                                          ExecCopySlotMinimalTuple(slot));
                overflow_hashes = lappend_int(overflow_hashes, (int) hashval);
                continue;
            }

            if (!vjoin_ht_insert_cas(state->hashtable, hashval,
                                     slot->tts_values, slot->tts_isnull))
            {
                had_overflow = true;
                pg_atomic_write_u32(&ps->cas_resizing, 1);
                overflow_tuples = lappend(overflow_tuples,
                                          ExecCopySlotMinimalTuple(slot));
                overflow_hashes = lappend_int(overflow_hashes, (int) hashval);
                continue;
            }
            local_count++;
        }

        /* One atomic add per participant instead of per-insert */
        pg_atomic_fetch_add_u32(&ps->num_entries_atomic, local_count);

        /* Barrier: all participants done building */
        BarrierArriveAndWait(&ps->barrier, 0);

        /*
         * If any participant overflowed, leader resizes the shared HT
         * single-threaded (safe after barrier — all writes are visible),
         * then all participants re-insert their overflow tuples.
         */
        if (pg_atomic_read_u32(&ps->cas_resizing) != 0)
        {
            if (state->is_leader)
            {
                /* Leader: rehash into doubled DSA arrays */
                int old_cap = ps->capacity;
                int new_cap;
                int resize_na = ps->num_all_attrs;
                int new_mask;
                dsa_pointer old_hv_dp, old_val_dp, old_null_dp;
                dsa_pointer new_hv_dp, new_val_dp, new_null_dp;
                uint32 *old_hv;
                Datum  *old_val;
                bool   *old_null;
                uint32 *new_hv;
                Datum  *new_val;
                bool   *new_null;
                int j;

                if (old_cap > INT_MAX / 2)
                    ereport(ERROR,
                            (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                             errmsg("pg_vectorjoin: hash table capacity overflow")));
                new_cap = old_cap * 2;
                new_mask = new_cap - 1;

                old_hv_dp   = ps->hashvals_dp;
                old_val_dp  = ps->all_values_dp;
                old_null_dp = ps->all_isnull_dp;

                old_hv = (uint32 *) dsa_get_address(dsa, old_hv_dp);
                old_val = (Datum *)  dsa_get_address(dsa, old_val_dp);
                old_null = (bool *)  dsa_get_address(dsa, old_null_dp);

                new_hv_dp   = dsa_allocate0(dsa, (Size) sizeof(uint32) * new_cap);
                new_val_dp  = dsa_allocate0(dsa, (Size) sizeof(Datum) * new_cap * resize_na);
                new_null_dp = dsa_allocate0(dsa, (Size) sizeof(bool) * new_cap * resize_na);

                new_hv   = (uint32 *) dsa_get_address(dsa, new_hv_dp);
                new_val  = (Datum *)  dsa_get_address(dsa, new_val_dp);
                new_null = (bool *)   dsa_get_address(dsa, new_null_dp);

                for (j = 0; j < old_cap; j++)
                {
                    if (old_hv[j] != 0)
                    {
                        int p = old_hv[j] & new_mask;
                        while (new_hv[p] != 0)
                            p = (p + 1) & new_mask;
                        new_hv[p] = old_hv[j];
                        memcpy(&new_val[p * resize_na], &old_val[j * resize_na], sizeof(Datum) * resize_na);
                        memcpy(&new_null[p * resize_na], &old_null[j * resize_na], sizeof(bool) * resize_na);
                    }
                }

                ps->hashvals_dp   = new_hv_dp;
                ps->all_values_dp = new_val_dp;
                ps->all_isnull_dp = new_null_dp;
                ps->capacity      = new_cap;
                ps->mask          = new_mask;

                /* Free old DSA arrays now that rehash is complete */
                dsa_free(dsa, old_hv_dp);
                dsa_free(dsa, old_val_dp);
                dsa_free(dsa, old_null_dp);

                pg_write_barrier();
                pg_atomic_write_u32(&ps->cas_resizing, 0);
            }
            else
            {
                /* Workers: wait for leader to finish resize */
                while (pg_atomic_read_u32(&ps->cas_resizing) != 0)
                    pg_spin_delay();
                pg_read_barrier();
            }

            /* All participants: refresh local HT wrapper */
            state->hashtable->hashvals   = (uint32 *) dsa_get_address(dsa, ps->hashvals_dp);
            state->hashtable->all_values = (Datum *)  dsa_get_address(dsa, ps->all_values_dp);
            state->hashtable->all_isnull = (bool *)   dsa_get_address(dsa, ps->all_isnull_dp);
            state->hashtable->capacity   = ps->capacity;
            state->hashtable->mask       = ps->mask;

            /* Re-insert overflow tuples via CAS */
            if (overflow_tuples != NIL)
            {
                TupleTableSlot *tmpslot;
                ListCell *lc_tup, *lc_hash;
                int extra = 0;

                tmpslot = MakeSingleTupleTableSlot(
                    state->inner_ps->ps_ResultTupleDesc,
                    &TTSOpsMinimalTuple);

                forboth(lc_tup, overflow_tuples, lc_hash, overflow_hashes)
                {
                    MinimalTuple mt = (MinimalTuple) lfirst(lc_tup);
                    uint32 hv = (uint32) lfirst_int(lc_hash);

                    ExecStoreMinimalTuple(mt, tmpslot, false);
                    slot_getallattrs(tmpslot);

                    if (!vjoin_ht_insert_cas(state->hashtable, hv,
                                             tmpslot->tts_values,
                                             tmpslot->tts_isnull))
                        ereport(ERROR,
                                (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                                 errmsg("parallel vector hash table capacity "
                                        "exceeded after resize")));
                    extra++;
                }
                pg_atomic_fetch_add_u32(&ps->num_entries_atomic, extra);

                ExecDropSingleTupleTableSlot(tmpslot);
                list_free_deep(overflow_tuples);
                list_free(overflow_hashes);
            }

            /* Second barrier: all overflow inserts done */
            BarrierArriveAndWait(&ps->barrier, 0);
        }

        /* Read final entry count from atomic counter */
        {
            uint32 total = pg_atomic_read_u32(&ps->num_entries_atomic);
            state->hashtable->num_entries = (int) total;
            state->cached_ht_entries = (int) total;

            if (state->is_leader)
            {
                ps->num_entries     = (int) total;
                ps->all_attrs_byval = true;
                ps->built_in_dsa   = true;
            }
        }
        } /* end local_count block */

        state->phase = VHJ_PROBE;
        return;
    }

    /*
     * Parallel worker (leader-only build): skip build, wait at barrier,
     * then attach to the shared HT.
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

        if (state->pstate->built_in_dsa)
        {
            /*
             * Fast path: leader built directly in DSA (byval tables).
             * Create a thin read-only wrapper pointing at shared arrays.
             */
            VJoinParallelState *ps = state->pstate;
            dsa_area      *dsa = state->dsa;
            MemoryContext  htctx;
            VJoinHashTable *ht;

            htctx = AllocSetContextCreate(state->hash_ctx,
                                          "VJoinHashTable (shared-direct)",
                                          ALLOCSET_DEFAULT_SIZES);
            ht = (VJoinHashTable *)
                MemoryContextAllocZero(htctx, sizeof(VJoinHashTable));

            ht->htctx          = htctx;
            ht->capacity       = ps->capacity;
            ht->mask           = ps->mask;
            ht->num_entries    = ps->num_entries;
            ht->num_all_attrs  = ps->num_all_attrs;
            ht->num_keys       = ps->num_keys;
            ht->all_attrs_byval = ps->all_attrs_byval;
            ht->is_shared      = false;  /* worker doesn't write */

            /* Point directly at shared DSA arrays — zero memcpy */
            ht->hashvals     = (uint32 *)    dsa_get_address(dsa, ps->hashvals_dp);
            ht->all_values   = (Datum *)     dsa_get_address(dsa, ps->all_values_dp);
            ht->all_isnull   = (bool *)      dsa_get_address(dsa, ps->all_isnull_dp);
            ht->inner_keynos = (AttrNumber *) dsa_get_address(dsa, ps->inner_keynos_dp);
            ht->attr_byval   = NULL;
            ht->attr_typlen  = NULL;

            state->hashtable = ht;
        }
        else
        {
            /* Slow path: byref table — attach with offset→pointer fixup */
            state->hashtable = vjoin_ht_attach_from_dsa(state->pstate,
                                                         state->dsa,
                                                         state->hash_ctx);
        }

        state->cached_ht_entries = state->hashtable->num_entries;

        state->phase = VHJ_PROBE;
        return;
    }

    /*
     * Leader (parallel) or non-parallel: build the hash table.
     *
     * For parallel + all-byval, switch to a DSA-backed HT so that
     * the build writes directly into shared memory — no serialize step.
     */
    if (state->is_parallel && state->hashtable->all_attrs_byval)
    {
        /* Copy type metadata out of the local HT's context before destroying */
        int    na = state->num_inner_attrs;
        bool  *saved_byval  = palloc(sizeof(bool) * na);
        int16 *saved_typlen = palloc(sizeof(int16) * na);

        memcpy(saved_byval, state->hashtable->attr_byval, sizeof(bool) * na);
        memcpy(saved_typlen, state->hashtable->attr_typlen, sizeof(int16) * na);

        vjoin_ht_destroy(state->hashtable);

        /* Create a shared HT that writes directly into pre-allocated DSA arrays */
        state->hashtable = vjoin_ht_create_shared(
            state->pstate, state->dsa,
            state->num_keys, na,
            state->hash_ctx, state->inner_keynos,
            saved_byval, saved_typlen);

        pfree(saved_byval);
        pfree(saved_typlen);
    }

    oldctx = MemoryContextSwitchTo(state->hash_ctx);

    for (;;)
    {
        bool    has_null = false;
        uint32  hashval;
        int     i;

        CHECK_FOR_INTERRUPTS();

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

        /* Skip NULL keys — NULLs never match in equijoin. */
        if (has_null)
            continue;

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

    /* In parallel mode, share the HT */
    if (state->is_parallel)
    {
        if (state->hashtable->is_shared)
        {
            /*
             * DSA-direct path (byval): HT is already in shared memory.
             * Just publish metadata to pstate so workers can read it.
             */
            VJoinParallelState *ps = state->pstate;
            ps->num_entries    = state->hashtable->num_entries;
            ps->all_attrs_byval = true;
            ps->built_in_dsa  = true;
        }
        else
        {
            /* Byref path: serialize HT arrays into DSA (existing logic) */
            vjoin_ht_serialize_to_dsa(state->hashtable, state->dsa,
                                      state->pstate);
            state->pstate->built_in_dsa = false;
        }

        BarrierArriveAndWait(&state->pstate->barrier, 0);
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

    if (state->jointype == JOIN_LEFT && state->batch_count > 0)
        memset(state->batch_matched, 0, sizeof(bool) * state->batch_count);

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

                    if (state->jointype == JOIN_LEFT)
                        state->batch_matched[batch_idx] = true;

                }
            }
            pos = (pos + 1) & ht->mask;
        }
    }

    if (state->jointype == JOIN_LEFT)
    {
        for (batch_idx = 0; batch_idx < state->batch_count; batch_idx++)
        {
            if (state->batch_matched[batch_idx])
                continue;

            if (state->result_count >= state->result_capacity)
            {
                state->result_capacity *= 2;
                state->results = repalloc(state->results,
                    sizeof(VJoinMatch) * state->result_capacity);
            }

            state->results[state->result_count].outer_idx = batch_idx;
            state->results[state->result_count].inner_idx = -1;
            state->result_count++;
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

    if (state->jointype == JOIN_LEFT)
        state->batch_matched = palloc0(sizeof(bool) * state->batch_size);
    else
        state->batch_matched = NULL;
    state->inner_matched = NULL;
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
}

void
vjoin_hash_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
    VectorHashJoinState *state = (VectorHashJoinState *) node;
    ExplainPropertyText("Join Type",
                        state->jointype == JOIN_LEFT ? "Left" : "Inner",
                        es);

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
    double inner_rows;
    int    est_capacity;

    /* Create DSA in the DSM segment */
    state->dsa = dsa_create(LWTRANCHE_PARALLEL_HASH_JOIN);
    dsa_pin_mapping(state->dsa);

    /* Initialize the shared state */
    pstate->dsa_handle = dsa_get_handle(state->dsa);
    BarrierInit(&pstate->barrier, vjoin_pcxt_nworkers(pcxt) + 1);
    pstate->num_entries = 0;
    pstate->built_in_dsa = false;
    pstate->parallel_build = false;
    pg_atomic_init_u32(&pstate->num_entries_atomic, 0);
    pg_atomic_init_u32(&pstate->cas_resizing, 0);

    /*
     * Pre-allocate DSA arrays based on estimated inner rows.
     * This lets us build the HT directly in shared memory for byval tables,
     * eliminating the serialize step.  VJOIN_HT_LOAD_FACTOR (2×) gives a
     * ~38-50% load factor after power-of-2 rounding — good balance between
     * probe efficiency and memory footprint (fits in L3 cache).
     */
    inner_rows = state->inner_ps->plan->plan_rows;
    if (inner_rows < 64)
        inner_rows = 64;
    est_capacity = vjoin_next_power_of_2((int)(inner_rows * VJOIN_HT_LOAD_FACTOR * 2));
    pstate->est_inner_rows = (int) inner_rows;
    pstate->num_all_attrs = state->num_inner_attrs;
    pstate->num_keys = state->num_keys;
    pstate->capacity = est_capacity;
    pstate->mask = est_capacity - 1;

    /* Pre-allocate the main arrays in DSA shared memory */
    pstate->hashvals_dp = dsa_allocate0(state->dsa,
                                        sizeof(uint32) * est_capacity);
    pstate->all_values_dp = dsa_allocate0(state->dsa,
                                          sizeof(Datum) * est_capacity * state->num_inner_attrs);
    pstate->all_isnull_dp = dsa_allocate0(state->dsa,
                                          sizeof(bool) * est_capacity * state->num_inner_attrs);
    pstate->inner_keynos_dp = dsa_allocate(state->dsa,
                                           sizeof(AttrNumber) * state->num_keys);
    pstate->vardata_dp = InvalidDsaPointer;
    pstate->attr_byval_dp = InvalidDsaPointer;

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
