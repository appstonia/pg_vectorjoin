#include "postgres.h"
#include "utils/datum.h"
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
