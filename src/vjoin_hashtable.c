#include "postgres.h"
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
                MemoryContext parent)
{
    MemoryContext htctx;
    VJoinHashTable *ht;
    int capacity;

    htctx = AllocSetContextCreate(parent,
                                  "VJoinHashTable",
                                  ALLOCSET_DEFAULT_SIZES);

    ht = (VJoinHashTable *) MemoryContextAllocZero(htctx,
                                                    sizeof(VJoinHashTable));
    ht->htctx = htctx;
    ht->num_keys = num_keys;
    ht->num_all_attrs = num_all_attrs;

    /* Capacity = next power of 2 >= estimated_rows * load_factor */
    capacity = next_power_of_2(Max(estimated_rows * VJOIN_HT_LOAD_FACTOR, 128));
    ht->capacity = capacity;
    ht->mask = capacity - 1;
    ht->num_entries = 0;

    ht->hashvals = (uint32 *)
        MemoryContextAllocZero(htctx, sizeof(uint32) * capacity);
    ht->tuples = (MinimalTuple *)
        MemoryContextAllocZero(htctx, sizeof(MinimalTuple) * capacity);
    ht->keys = (Datum *)
        MemoryContextAllocZero(htctx, sizeof(Datum) * capacity * num_keys);
    ht->key_nulls = (bool *)
        MemoryContextAllocZero(htctx, sizeof(bool) * capacity * num_keys);
    ht->all_values = (Datum *)
        MemoryContextAllocZero(htctx, sizeof(Datum) * capacity * num_all_attrs);
    ht->all_isnull = (bool *)
        MemoryContextAllocZero(htctx, sizeof(bool) * capacity * num_all_attrs);

    return ht;
}

void
vjoin_ht_insert(VJoinHashTable *ht, uint32 hashval,
                MinimalTuple tuple, Datum *keyvals, bool *keynulls,
                Datum *all_values, bool *all_isnull)
{
    int pos;
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
        MinimalTuple  *old_tuples = ht->tuples;
        Datum         *old_keys = ht->keys;
        bool          *old_nulls = ht->key_nulls;
        Datum         *old_vals = ht->all_values;
        bool          *old_inull = ht->all_isnull;
        int            new_cap = old_cap * 2;
        int            na = ht->num_all_attrs;
        int            i;

        old = MemoryContextSwitchTo(ht->htctx);

        ht->capacity = new_cap;
        ht->mask = new_cap - 1;
        ht->hashvals = (uint32 *) palloc0(sizeof(uint32) * new_cap);
        ht->tuples = (MinimalTuple *) palloc0(sizeof(MinimalTuple) * new_cap);
        ht->keys = (Datum *) palloc0(sizeof(Datum) * new_cap * ht->num_keys);
        ht->key_nulls = (bool *) palloc0(sizeof(bool) * new_cap * ht->num_keys);
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
                ht->tuples[pos] = old_tuples[i];
                memcpy(&ht->keys[pos * ht->num_keys],
                       &old_keys[i * ht->num_keys],
                       sizeof(Datum) * ht->num_keys);
                memcpy(&ht->key_nulls[pos * ht->num_keys],
                       &old_nulls[i * ht->num_keys],
                       sizeof(bool) * ht->num_keys);
                memcpy(&ht->all_values[pos * na],
                       &old_vals[i * na],
                       sizeof(Datum) * na);
                memcpy(&ht->all_isnull[pos * na],
                       &old_inull[i * na],
                       sizeof(bool) * na);
            }
        }

        pfree(old_hashvals);
        pfree(old_tuples);
        pfree(old_keys);
        pfree(old_nulls);
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
    ht->tuples[pos] = tuple;
    memcpy(&ht->keys[pos * ht->num_keys], keyvals,
           sizeof(Datum) * ht->num_keys);
    memcpy(&ht->key_nulls[pos * ht->num_keys], keynulls,
           sizeof(bool) * ht->num_keys);
    memcpy(&ht->all_values[pos * ht->num_all_attrs], all_values,
           sizeof(Datum) * ht->num_all_attrs);
    memcpy(&ht->all_isnull[pos * ht->num_all_attrs], all_isnull,
           sizeof(bool) * ht->num_all_attrs);
    ht->num_entries++;

    MemoryContextSwitchTo(old);
}

void
vjoin_ht_destroy(VJoinHashTable *ht)
{
    if (ht && ht->htctx)
        MemoryContextDelete(ht->htctx);
}
