#include "postgres.h"
#include "utils/datum.h"
#include "utils/dsa.h"
#include "utils/memutils.h"
#include "pg_vectorjoin.h"
#include "vjoin_state.h"

/* Round up to next power of 2 */
static int
next_power_of_2(int n)
{
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
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
    capacity = next_power_of_2(Max(estimated_rows * VJOIN_HT_LOAD_FACTOR, 128));
    ht->capacity = capacity;
    ht->mask = capacity - 1;
    ht->num_entries = 0;

    ht->hashvals = (uint32 *)
        MemoryContextAllocZero(htctx, sizeof(uint32) * capacity);
    ht->all_values = (Datum *)
        MemoryContextAllocZero(htctx, sizeof(Datum) * capacity * num_all_attrs);
    ht->all_isnull = (bool *)
        MemoryContextAllocZero(htctx, sizeof(bool) * capacity * num_all_attrs);

    return ht;
}

void
vjoin_ht_insert(VJoinHashTable *ht, uint32 hashval,
                Datum *all_values, bool *all_isnull)
{
    int pos;
    int na = ht->num_all_attrs;
    int base;
    MemoryContext old;

    /* Ensure hash is non-zero (0 = empty marker) */
    if (hashval == 0)
        hashval = 1;

    /* Check if we need to grow (load factor > 50%) */
    if (ht->num_entries * 2 >= ht->capacity)
    {
        /* Rehash: double capacity */
        int            old_cap = ht->capacity;
        uint32        *old_hashvals = ht->hashvals;
        Datum         *old_vals = ht->all_values;
        bool          *old_inull = ht->all_isnull;
        int            new_cap = old_cap * 2;
        int            i;

        old = MemoryContextSwitchTo(ht->htctx);

        ht->capacity = new_cap;
        ht->mask = new_cap - 1;
        ht->hashvals = (uint32 *) palloc0(sizeof(uint32) * new_cap);
        ht->all_values = (Datum *) palloc0(sizeof(Datum) * new_cap * na);
        ht->all_isnull = (bool *) palloc0(sizeof(bool) * new_cap * na);

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

        pfree(old_hashvals);
        pfree(old_vals);
        pfree(old_inull);

        MemoryContextSwitchTo(old);
    }

    /* Insert into table */
    old = MemoryContextSwitchTo(ht->htctx);

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

    MemoryContextSwitchTo(old);
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
