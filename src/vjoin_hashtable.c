#include "postgres.h"
#include "miscadmin.h"
#include "optimizer/cost.h"
#include "utils/datum.h"
#include "utils/dsa.h"
#include "utils/memutils.h"
#include "pg_vectorjoin.h"
#include "vjoin_state.h"
#include "vjoin_spill.h"

/*
 * Runtime guard: only enforce MaxAllocSize (palloc hard ceiling).
 * work_mem is a planner-level preference enforced in vjoin_path.c;
 * at runtime, row estimates may be off, so we allow the hash table
 * to exceed work_mem — same behavior as native PostgreSQL hash join.
 */
static void
vjoin_hash_check_array_sizes(int capacity, int num_all_attrs)
{
    if (capacity <= 0 || num_all_attrs <= 0)
        ereport(ERROR,
                (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                 errmsg("pg_vectorjoin: invalid hash table dimensions")));

    if ((Size) capacity > MaxAllocSize / sizeof(uint32) ||
        (Size) capacity > MaxAllocSize / ((Size) sizeof(Datum) * num_all_attrs) ||
        (Size) capacity > MaxAllocSize / ((Size) sizeof(bool) * num_all_attrs))
        ereport(ERROR,
                (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                 errmsg("pg_vectorjoin: vector hash table exceeds allocation limit")));
}

VJoinHashTable *
vjoin_ht_create(int estimated_rows, int num_keys, int num_all_attrs,
                MemoryContext parent, AttrNumber *inner_keynos,
                bool *attr_byval, int16 *attr_typlen)
{
    MemoryContext htctx;
    VJoinHashTable *ht;
    int capacity;
    int i;

    htctx = AllocSetContextCreate(parent,
                                  "VJoinHashTable",
                                  ALLOCSET_DEFAULT_SIZES);

    ht = (VJoinHashTable *) MemoryContextAllocZero(htctx,
                                                    sizeof(VJoinHashTable));
    ht->htctx = htctx;
    ht->num_keys = num_keys;
    ht->num_all_attrs = num_all_attrs;

    /* Store inner key attribute positions for probe-time comparison */
    ht->inner_keynos = (AttrNumber *)
        MemoryContextAlloc(htctx, sizeof(AttrNumber) * num_keys);
    memcpy(ht->inner_keynos, inner_keynos, sizeof(AttrNumber) * num_keys);

    /* Store inner attr type metadata for datumCopy of pass-by-ref values */
    ht->attr_byval = (bool *)
        MemoryContextAlloc(htctx, sizeof(bool) * num_all_attrs);
    ht->attr_typlen = (int16 *)
        MemoryContextAlloc(htctx, sizeof(int16) * num_all_attrs);
    memcpy(ht->attr_byval, attr_byval, sizeof(bool) * num_all_attrs);
    memcpy(ht->attr_typlen, attr_typlen, sizeof(int16) * num_all_attrs);

    /* Precompute all-byval flag for fast insert path */
    ht->all_attrs_byval = true;
    for (i = 0; i < num_all_attrs; i++)
    {
        if (!attr_byval[i])
        {
            ht->all_attrs_byval = false;
            break;
        }
    }

    /* Capacity = next power of 2 >= estimated_rows * load_factor */
    capacity = vjoin_next_power_of_2(Max(estimated_rows * VJOIN_HT_LOAD_FACTOR, 128));
    vjoin_hash_check_array_sizes(capacity, num_all_attrs);
    ht->capacity = capacity;
    ht->mask = capacity - 1;
    ht->num_entries = 0;

    ht->hashvals = (uint32 *)
        MemoryContextAllocZero(htctx, sizeof(uint32) * capacity);
    ht->all_values = (Datum *)
        MemoryContextAllocZero(htctx, sizeof(Datum) * capacity * num_all_attrs);
    ht->all_isnull = (bool *)
        MemoryContextAllocZero(htctx, sizeof(bool) * capacity * num_all_attrs);

    ht->is_shared = false;
    ht->dsa = NULL;
    ht->pstate = NULL;

    /*
     * Multi-batch spilling setup (serial path only).  Until the in-memory
     * footprint exceeds space_allowed the table behaves as a single batch.
     */
    ht->nbatch = 1;
    ht->curbatch = 0;
    ht->space_used = 0;
    ht->routing_frozen = false;
    ht->log2_nbuckets = 0;
    ht->valctx = htctx;
    ht->inner_batch_files = NULL;
    ht->outer_batch_files = NULL;

    ht->spill_enabled = vjoin_enable_hash_spill;
    if (ht->spill_enabled)
    {
        double  mult = hash_mem_multiplier;
        Size    budget;

        if (mult < 1.0)
            mult = 1.0;
        budget = (Size) ((double) work_mem * 1024.0 * mult);
        if (budget < 64 * 1024)
            budget = 64 * 1024;     /* floor: avoid pathological tiny batches */
        ht->space_allowed = budget;

        ht->max_nbatch = vjoin_next_power_of_2(Max(vjoin_hash_max_batches, 1));

        /*
         * Byref payloads live in a resettable child context so they can be
         * discarded between batches when reloading spilled inner tuples.
         */
        ht->valctx = AllocSetContextCreate(htctx, "VJoinHashBatchVals",
                                           ALLOCSET_DEFAULT_SIZES);
    }
    else
    {
        ht->space_allowed = (Size) -1;  /* never trigger */
        ht->max_nbatch = 1;
    }

    return ht;
}

/*
 * Create a hash table whose flat arrays live directly in DSA shared memory.
 * The leader inserts into these arrays during the build phase; workers
 * read them after the barrier — no serialize/attach step needed for byval.
 *
 * pstate must already have hashvals_dp/all_values_dp/all_isnull_dp/
 * inner_keynos_dp pre-allocated by vjoin_hash_initialize_dsm.
 */
VJoinHashTable *
vjoin_ht_create_shared(VJoinParallelState *pstate, dsa_area *dsa,
                        int num_keys, int num_all_attrs,
                        MemoryContext parent, AttrNumber *inner_keynos,
                        bool *attr_byval, int16 *attr_typlen)
{
    MemoryContext htctx;
    VJoinHashTable *ht;
    int i;

    htctx = AllocSetContextCreate(parent,
                                  "VJoinHashTable (DSA-direct)",
                                  ALLOCSET_DEFAULT_SIZES);

    ht = (VJoinHashTable *) MemoryContextAllocZero(htctx,
                                                    sizeof(VJoinHashTable));
    ht->htctx = htctx;
    ht->num_keys = num_keys;
    ht->num_all_attrs = num_all_attrs;

    /* Point arrays directly at pre-allocated DSA memory */
    ht->capacity = pstate->capacity;
    ht->mask = pstate->mask;
    ht->num_entries = 0;

    ht->hashvals   = (uint32 *)    dsa_get_address(dsa, pstate->hashvals_dp);
    ht->all_values = (Datum *)     dsa_get_address(dsa, pstate->all_values_dp);
    ht->all_isnull = (bool *)      dsa_get_address(dsa, pstate->all_isnull_dp);
    ht->inner_keynos = (AttrNumber *) dsa_get_address(dsa, pstate->inner_keynos_dp);

    /* Copy keynos into shared memory */
    memcpy(ht->inner_keynos, inner_keynos, sizeof(AttrNumber) * num_keys);

    /* Store type metadata locally (only leader needs this during build) */
    ht->attr_byval = (bool *)
        MemoryContextAlloc(htctx, sizeof(bool) * num_all_attrs);
    ht->attr_typlen = (int16 *)
        MemoryContextAlloc(htctx, sizeof(int16) * num_all_attrs);
    memcpy(ht->attr_byval, attr_byval, sizeof(bool) * num_all_attrs);
    memcpy(ht->attr_typlen, attr_typlen, sizeof(int16) * num_all_attrs);

    ht->all_attrs_byval = true;
    for (i = 0; i < num_all_attrs; i++)
    {
        if (!attr_byval[i])
        {
            ht->all_attrs_byval = false;
            break;
        }
    }

    ht->is_shared = true;
    ht->dsa = dsa;
    ht->pstate = pstate;

    /* Spilling is not supported for shared/parallel hash tables in v1. */
    ht->nbatch = 1;
    ht->curbatch = 0;
    ht->space_used = 0;
    ht->space_allowed = (Size) -1;
    ht->max_nbatch = 1;
    ht->spill_enabled = false;
    ht->routing_frozen = false;
    ht->log2_nbuckets = 0;
    ht->valctx = htctx;
    ht->inner_batch_files = NULL;
    ht->outer_batch_files = NULL;

    return ht;
}

/* log2 of a power-of-2 value. */
static int
vjoin_log2_int(int n)
{
    int l = 0;

    while ((1 << l) < n)
        l++;
    return l;
}

/* Logical in-memory byte cost of one stored entry (for the space budget). */
static inline Size
vjoin_entry_bytes(VJoinHashTable *ht, const Datum *values, const bool *isnull)
{
    int   na = ht->num_all_attrs;
    Size  b = sizeof(uint32) + (Size) na * (sizeof(Datum) + sizeof(bool));
    int   a;

    if (ht->all_attrs_byval)
        return b;

    for (a = 0; a < na; a++)
    {
        if (isnull[a] || ht->attr_byval[a])
            continue;
        b += (Size) datumGetSize(values[a], false, ht->attr_typlen[a]);
    }
    return b;
}

/* Ensure inner/outer spill-file pointer arrays exist and are sized [nbatch]. */
static void
vjoin_ht_ensure_file_arrays(VJoinHashTable *ht, int nbatch)
{
    MemoryContext old = MemoryContextSwitchTo(ht->htctx);

    if (ht->inner_batch_files == NULL)
    {
        ht->inner_batch_files = palloc0(sizeof(vjoin_spill_file *) * nbatch);
        ht->outer_batch_files = palloc0(sizeof(vjoin_spill_file *) * nbatch);
    }
    else
    {
        int old_nbatch = ht->nbatch;

        ht->inner_batch_files = repalloc(ht->inner_batch_files,
                                         sizeof(vjoin_spill_file *) * nbatch);
        ht->outer_batch_files = repalloc(ht->outer_batch_files,
                                         sizeof(vjoin_spill_file *) * nbatch);
        memset(&ht->inner_batch_files[old_nbatch], 0,
               sizeof(vjoin_spill_file *) * (nbatch - old_nbatch));
        memset(&ht->outer_batch_files[old_nbatch], 0,
               sizeof(vjoin_spill_file *) * (nbatch - old_nbatch));
    }

    MemoryContextSwitchTo(old);
}

/* Lazily create and return the inner spill file for the given batch. */
vjoin_spill_file *
vjoin_ht_inner_file(VJoinHashTable *ht, int batchno)
{
    MemoryContext old;

    if (ht->inner_batch_files[batchno] != NULL)
        return ht->inner_batch_files[batchno];

    old = MemoryContextSwitchTo(ht->htctx);
    ht->inner_batch_files[batchno] = vjoin_spill_create();
    MemoryContextSwitchTo(old);
    return ht->inner_batch_files[batchno];
}

/* Lazily create and return the outer spill file for the given batch. */
vjoin_spill_file *
vjoin_ht_outer_file(VJoinHashTable *ht, int batchno)
{
    MemoryContext old;

    if (ht->outer_batch_files[batchno] != NULL)
        return ht->outer_batch_files[batchno];

    old = MemoryContextSwitchTo(ht->htctx);
    ht->outer_batch_files[batchno] = vjoin_spill_create();
    MemoryContextSwitchTo(old);
    return ht->outer_batch_files[batchno];
}

/*
 * Double nbatch and evict every in-memory entry that no longer belongs to
 * curbatch to its inner spill file.  Called when the in-memory footprint
 * exceeds space_allowed.  Re-hashes survivors into fresh arrays (open
 * addressing cannot tolerate tombstones).
 */
static void
vjoin_ht_grow_batches(VJoinHashTable *ht)
{
    int         na = ht->num_all_attrs;
    int         old_cap = ht->capacity;
    uint32     *old_hv = ht->hashvals;
    Datum      *old_val = ht->all_values;
    bool       *old_null = ht->all_isnull;
    uint32     *new_hv;
    Datum      *new_val;
    bool       *new_null;
    int         new_nbatch;
    int         i;
    MemoryContext old;

    Assert(!ht->is_shared);

    if (ht->nbatch >= ht->max_nbatch)
    {
        /* Cannot split further: stop enforcing the budget. */
        ht->space_allowed = (Size) -1;
        return;
    }

    new_nbatch = ht->nbatch * 2;

    /* Freeze the bucket-index width used for batch routing at first split. */
    if (!ht->routing_frozen)
    {
        ht->log2_nbuckets = vjoin_log2_int(ht->capacity);
        ht->routing_frozen = true;
    }

    vjoin_ht_ensure_file_arrays(ht, new_nbatch);
    ht->nbatch = new_nbatch;

    /* Fresh arrays (same capacity); re-insert survivors, spill the rest. */
    old = MemoryContextSwitchTo(ht->htctx);
    new_hv = (uint32 *) palloc0(sizeof(uint32) * old_cap);
    new_val = (Datum *) palloc0(sizeof(Datum) * old_cap * na);
    new_null = (bool *) palloc0(sizeof(bool) * old_cap * na);
    MemoryContextSwitchTo(old);

    ht->hashvals = new_hv;
    ht->all_values = new_val;
    ht->all_isnull = new_null;
    ht->num_entries = 0;
    ht->space_used = 0;

    for (i = 0; i < old_cap; i++)
    {
        uint32  hv = old_hv[i];
        int     batchno;
        int     pos;
        int     sbase;

        if (hv == 0)
            continue;

        sbase = i * na;
        batchno = VJOIN_BATCH_OF(ht, hv);

        if (batchno != ht->curbatch)
        {
            vjoin_spill_write_tuple(vjoin_ht_inner_file(ht, batchno), hv, na,
                                    &old_val[sbase], &old_null[sbase],
                                    ht->attr_byval, ht->attr_typlen);
            continue;
        }

        /* Survivor: re-insert into the new arrays. */
        pos = hv & ht->mask;
        while (ht->hashvals[pos] != 0)
            pos = (pos + 1) & ht->mask;
        ht->hashvals[pos] = hv;
        memcpy(&ht->all_values[pos * na], &old_val[sbase], sizeof(Datum) * na);
        memcpy(&ht->all_isnull[pos * na], &old_null[sbase], sizeof(bool) * na);
        ht->num_entries++;
        ht->space_used += vjoin_entry_bytes(ht, &old_val[sbase],
                                            &old_null[sbase]);
    }

    pfree(old_hv);
    pfree(old_val);
    pfree(old_null);

    /*
     * If a single split was not enough to get back under budget, recurse.
     * (Bounded by max_nbatch.)
     */
    if (ht->space_used > ht->space_allowed)
        vjoin_ht_grow_batches(ht);
}

/*
 * Reset the in-memory portion of the table so a new batch can be loaded.
 * Clears occupancy (hashvals) and any byref payloads; keeps the flat
 * arrays and metadata.
 */
void
vjoin_ht_reset_for_batch(VJoinHashTable *ht)
{
    memset(ht->hashvals, 0, sizeof(uint32) * ht->capacity);
    ht->num_entries = 0;
    ht->space_used = 0;
    if (ht->valctx != ht->htctx)
        MemoryContextReset(ht->valctx);
}

/* Close and free all spill files (idempotent). */
void
vjoin_ht_close_spill_files(VJoinHashTable *ht)
{
    int i;

    if (ht->inner_batch_files != NULL)
    {
        for (i = 0; i < ht->nbatch; i++)
        {
            if (ht->inner_batch_files[i] != NULL)
            {
                vjoin_spill_close(ht->inner_batch_files[i]);
                ht->inner_batch_files[i] = NULL;
            }
        }
    }
    if (ht->outer_batch_files != NULL)
    {
        for (i = 0; i < ht->nbatch; i++)
        {
            if (ht->outer_batch_files[i] != NULL)
            {
                vjoin_spill_close(ht->outer_batch_files[i]);
                ht->outer_batch_files[i] = NULL;
            }
        }
    }
}

void
vjoin_ht_insert(VJoinHashTable *ht, uint32 hashval,
                Datum *all_values, bool *all_isnull)
{
    int pos;
    int na = ht->num_all_attrs;
    int base;
    MemoryContext old = NULL;

    /* Ensure hash is non-zero (0 = empty marker) */
    if (hashval == 0)
        hashval = 1;

    /*
     * Spilling: enforce the memory budget before inserting.  When the
     * in-memory footprint exceeds space_allowed we split into more batches,
     * evicting non-curbatch entries to disk.  After (possibly) splitting,
     * route this tuple: if it no longer belongs to curbatch, spill it.
     */
    if (ht->spill_enabled && !ht->is_shared)
    {
        int batchno;

        if (ht->space_used > ht->space_allowed && ht->nbatch < ht->max_nbatch)
            vjoin_ht_grow_batches(ht);

        batchno = VJOIN_BATCH_OF(ht, hashval);
        if (batchno != ht->curbatch)
        {
            vjoin_spill_write_tuple(vjoin_ht_inner_file(ht, batchno), hashval,
                                    na, all_values, all_isnull,
                                    ht->attr_byval, ht->attr_typlen);
            return;
        }
    }

    /* Check if we need to grow (load factor > 50%) */
    if (ht->num_entries * 2 >= ht->capacity)
    {
        /* Rehash: double capacity (with overflow guard) */
        int            old_cap = ht->capacity;
        uint32        *old_hashvals = ht->hashvals;
        Datum         *old_vals = ht->all_values;
        bool          *old_inull = ht->all_isnull;
        int            new_cap;
        int            i;
        dsa_pointer    old_hv_dp = InvalidDsaPointer;
        dsa_pointer    old_val_dp = InvalidDsaPointer;
        dsa_pointer    old_null_dp = InvalidDsaPointer;

        if (old_cap > INT_MAX / 2)
            ereport(ERROR,
                    (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                     errmsg("pg_vectorjoin: hash table capacity overflow")));
        new_cap = old_cap * 2;
        vjoin_hash_check_array_sizes(new_cap, na);

        ht->capacity = new_cap;
        ht->mask = new_cap - 1;

        if (ht->is_shared)
        {
            /* DSA-backed: allocate new arrays in shared memory */
            dsa_area   *dsa = ht->dsa;
            VJoinParallelState *ps = ht->pstate;
            old_hv_dp  = ps->hashvals_dp;
            old_val_dp = ps->all_values_dp;
            old_null_dp = ps->all_isnull_dp;

            ps->hashvals_dp = dsa_allocate0(dsa, (Size) sizeof(uint32) * new_cap);
            ps->all_values_dp = dsa_allocate0(dsa, (Size) sizeof(Datum) * new_cap * na);
            ps->all_isnull_dp = dsa_allocate0(dsa, (Size) sizeof(bool) * new_cap * na);

            ht->hashvals   = (uint32 *) dsa_get_address(dsa, ps->hashvals_dp);
            ht->all_values = (Datum *)  dsa_get_address(dsa, ps->all_values_dp);
            ht->all_isnull = (bool *)   dsa_get_address(dsa, ps->all_isnull_dp);

            ps->capacity = new_cap;
            ps->mask = new_cap - 1;
        }
        else
        {
            old = MemoryContextSwitchTo(ht->htctx);
            ht->hashvals = (uint32 *) palloc0(sizeof(uint32) * new_cap);
            ht->all_values = (Datum *) palloc0(sizeof(Datum) * new_cap * na);
            ht->all_isnull = (bool *) palloc0(sizeof(bool) * new_cap * na);
            MemoryContextSwitchTo(old);
        }

        /* Reinsert existing entries */
        for (i = 0; i < old_cap; i++)
        {
            if (old_hashvals[i] != 0)
            {
                pos = old_hashvals[i] & ht->mask;
                while (ht->hashvals[pos] != 0)
                    pos = (pos + 1) & ht->mask;

                ht->hashvals[pos] = old_hashvals[i];
                memcpy(&ht->all_values[pos * na],
                       &old_vals[i * na],
                       sizeof(Datum) * na);
                memcpy(&ht->all_isnull[pos * na],
                       &old_inull[i * na],
                       sizeof(bool) * na);
            }
        }

        if (ht->is_shared)
        {
            /* Free old DSA arrays now that rehash is complete */
            dsa_free(ht->dsa, old_hv_dp);
            dsa_free(ht->dsa, old_val_dp);
            dsa_free(ht->dsa, old_null_dp);
        }
        else
        {
            pfree(old_hashvals);
            pfree(old_vals);
            pfree(old_inull);
            MemoryContextSwitchTo(old);
        }
    }

    /* Insert into table */
    old = MemoryContextSwitchTo(ht->valctx);

    pos = hashval & ht->mask;
    while (ht->hashvals[pos] != 0)
        pos = (pos + 1) & ht->mask;

    ht->hashvals[pos] = hashval;
    base = pos * na;

    if (ht->all_attrs_byval)
    {
        /* Fast path: all pass-by-value, just memcpy */
        memcpy(&ht->all_values[base], all_values, sizeof(Datum) * na);
        memcpy(&ht->all_isnull[base], all_isnull, sizeof(bool) * na);
    }
    else
    {
        int a;
        memcpy(&ht->all_isnull[base], all_isnull, sizeof(bool) * na);
        for (a = 0; a < na; a++)
        {
            if (all_isnull[a] || ht->attr_byval[a])
                ht->all_values[base + a] = all_values[a];
            else
                ht->all_values[base + a] =
                    datumCopy(all_values[a], false, ht->attr_typlen[a]);
        }
    }

    ht->num_entries++;
    if (ht->spill_enabled && !ht->is_shared)
        ht->space_used += vjoin_entry_bytes(ht, all_values, all_isnull);

    MemoryContextSwitchTo(old);
}

/*
 * Lock-free insert for parallel build (byval tables only).
 * Uses CAS on hashvals[pos] to claim an empty slot.
 * No rehash — caller must ensure sufficient pre-allocated capacity.
 */
bool
vjoin_ht_insert_cas(VJoinHashTable *ht,
                     uint32 hashval,
                     Datum *all_values, bool *all_isnull)
{
    int    na = ht->num_all_attrs;
    int    pos;
    int    start_pos;
    uint32 expected;
    int    base;

    /* 0 = empty marker */
    if (hashval == 0)
        hashval = 1;

    pos = hashval & ht->mask;
    start_pos = pos;

    for (;;)
    {
        expected = 0;
        if (pg_atomic_compare_exchange_u32(
                (pg_atomic_uint32 *) &ht->hashvals[pos],
                &expected, hashval))
        {
            /* We claimed this slot — write values (no other writer touches it) */
            base = pos * na;
            memcpy(&ht->all_values[base], all_values, sizeof(Datum) * na);
            memcpy(&ht->all_isnull[base], all_isnull, sizeof(bool) * na);
            return true;
        }

        /* Slot taken — linear probe */
        pos = (pos + 1) & ht->mask;
        if (pos == start_pos)
            return false;  /* table full — caller coordinates resize */
    }
}

void
vjoin_ht_destroy(VJoinHashTable *ht)
{
    if (ht && ht->htctx)
        MemoryContextDelete(ht->htctx);
}

/*
 * Copy a locally-built hash table's flat arrays into DSA shared memory
 * and fill the parallel state metadata so workers can attach.
 *
 * For pass-by-reference Datums (text, varchar, etc.) the Datum is a
 * pointer into the leader's private heap — unusable by workers.  We
 * deep-copy all such data into a single flat DSA buffer and store
 * *offsets* (from the buffer start) in the shared all_values array.
 * Workers translate offsets back to valid local pointers after attach.
 */
void
vjoin_ht_serialize_to_dsa(VJoinHashTable *ht, dsa_area *dsa,
                           VJoinParallelState *pstate)
{
    Size hv_sz  = sizeof(uint32) * ht->capacity;
    Size val_sz = sizeof(Datum) * ht->capacity * ht->num_all_attrs;
    Size null_sz = sizeof(bool) * ht->capacity * ht->num_all_attrs;
    Size kn_sz  = sizeof(AttrNumber) * ht->num_keys;
    Size bv_sz  = sizeof(bool) * ht->num_all_attrs;
    int  na     = ht->num_all_attrs;
    int  cap    = ht->capacity;

    pstate->capacity       = cap;
    pstate->mask           = ht->mask;
    pstate->num_entries    = ht->num_entries;
    pstate->num_all_attrs  = na;
    pstate->num_keys       = ht->num_keys;
    pstate->all_attrs_byval = ht->all_attrs_byval;

    /* hashvals — no pointers, safe to memcpy */
    pstate->hashvals_dp = dsa_allocate(dsa, hv_sz);
    memcpy(dsa_get_address(dsa, pstate->hashvals_dp), ht->hashvals, hv_sz);

    /* all_isnull — no pointers */
    pstate->all_isnull_dp = dsa_allocate(dsa, null_sz);
    memcpy(dsa_get_address(dsa, pstate->all_isnull_dp), ht->all_isnull, null_sz);

    /* inner_keynos */
    pstate->inner_keynos_dp = dsa_allocate(dsa, kn_sz);
    memcpy(dsa_get_address(dsa, pstate->inner_keynos_dp), ht->inner_keynos, kn_sz);

    /* attr_byval (workers need this for pointer fixup) */
    pstate->attr_byval_dp = dsa_allocate(dsa, bv_sz);
    memcpy(dsa_get_address(dsa, pstate->attr_byval_dp), ht->attr_byval, bv_sz);

    /* all_values — deep-copy pass-by-ref data into a flat DSA buffer */
    pstate->all_values_dp = dsa_allocate(dsa, val_sz);
    {
        Datum *shared_vals = (Datum *) dsa_get_address(dsa, pstate->all_values_dp);
        memcpy(shared_vals, ht->all_values, val_sz);

        if (!ht->all_attrs_byval)
        {
            /* Pass 1: compute total size of all pass-by-ref data */
            Size total_vardata = 0;
            int  i, a;

            for (i = 0; i < cap; i++)
            {
                if (ht->hashvals[i] == 0)
                    continue;
                for (a = 0; a < na; a++)
                {
                    int idx = i * na + a;
                    if (!ht->attr_byval[a] && !ht->all_isnull[idx])
                        total_vardata += MAXALIGN(datumGetSize(
                            ht->all_values[idx], false, ht->attr_typlen[a]));
                }
            }

            if (total_vardata > 0)
            {
                /* Pass 2: allocate one buffer, copy data, store offsets */
                dsa_pointer var_dp;
                char       *var_buf;
                Size        offset = 0;

                var_dp  = dsa_allocate_extended(dsa, total_vardata,
                                               DSA_ALLOC_HUGE);
                var_buf = (char *) dsa_get_address(dsa, var_dp);
                pstate->vardata_dp = var_dp;

                for (i = 0; i < cap; i++)
                {
                    if (ht->hashvals[i] == 0)
                        continue;
                    for (a = 0; a < na; a++)
                    {
                        int idx = i * na + a;
                        if (!ht->attr_byval[a] && !ht->all_isnull[idx])
                        {
                            Size dsz = datumGetSize(
                                ht->all_values[idx], false, ht->attr_typlen[a]);
                            memcpy(var_buf + offset,
                                   DatumGetPointer(ht->all_values[idx]), dsz);
                            /* Store offset from buffer start */
                            shared_vals[idx] = (Datum) offset;
                            offset += MAXALIGN(dsz);
                        }
                    }
                }
            }
            else
            {
                pstate->vardata_dp = InvalidDsaPointer;
            }
        }
        else
        {
            pstate->vardata_dp = InvalidDsaPointer;
        }
    }
}

/*
 * Create a read-only VJoinHashTable wrapper that points at DSA shared
 * memory.  For tables with pass-by-ref columns the all_values array is
 * copied locally and offsets are translated back to valid pointers.
 */
VJoinHashTable *
vjoin_ht_attach_from_dsa(VJoinParallelState *pstate, dsa_area *dsa,
                          MemoryContext parent)
{
    MemoryContext htctx;
    VJoinHashTable *ht;
    int na  = pstate->num_all_attrs;
    int cap = pstate->capacity;

    htctx = AllocSetContextCreate(parent,
                                  "VJoinHashTable (shared)",
                                  ALLOCSET_DEFAULT_SIZES);

    ht = (VJoinHashTable *) MemoryContextAllocZero(htctx,
                                                    sizeof(VJoinHashTable));
    ht->htctx          = htctx;
    ht->capacity       = cap;
    ht->mask           = pstate->mask;
    ht->num_entries    = pstate->num_entries;
    ht->num_all_attrs  = na;
    ht->num_keys       = pstate->num_keys;
    ht->all_attrs_byval = pstate->all_attrs_byval;

    /* These arrays are pure values / booleans — shared read-only */
    ht->hashvals    = (uint32 *)    dsa_get_address(dsa, pstate->hashvals_dp);
    ht->all_isnull  = (bool *)      dsa_get_address(dsa, pstate->all_isnull_dp);
    ht->inner_keynos = (AttrNumber *) dsa_get_address(dsa, pstate->inner_keynos_dp);

    if (pstate->all_attrs_byval)
    {
        /* All byval: Datums are values, not pointers — share directly */
        ht->all_values = (Datum *) dsa_get_address(dsa, pstate->all_values_dp);
    }
    else
    {
        /*
         * Has pass-by-ref columns: the shared all_values array stores
         * *offsets* for byref Datums.  Make a local copy and convert
         * each offset to a valid local pointer via dsa_get_address.
         */
        Size   val_sz = sizeof(Datum) * cap * na;
        Datum *local_vals;

        local_vals = (Datum *) MemoryContextAlloc(htctx, val_sz);
        memcpy(local_vals,
               dsa_get_address(dsa, pstate->all_values_dp), val_sz);

        if (DsaPointerIsValid(pstate->vardata_dp))
        {
            bool *attr_byval = (bool *) dsa_get_address(dsa,
                                                         pstate->attr_byval_dp);
            char *var_base   = (char *) dsa_get_address(dsa,
                                                         pstate->vardata_dp);
            int   i, a;

            for (i = 0; i < cap; i++)
            {
                if (ht->hashvals[i] == 0)
                    continue;
                for (a = 0; a < na; a++)
                {
                    int idx = i * na + a;
                    if (!attr_byval[a] && !ht->all_isnull[idx])
                    {
                        /* offset → local pointer */
                        Size off = (Size) local_vals[idx];
                        local_vals[idx] = PointerGetDatum(var_base + off);
                    }
                }
            }
        }

        ht->all_values = local_vals;
    }

    ht->attr_byval  = NULL;
    ht->attr_typlen = NULL;

    return ht;
}
