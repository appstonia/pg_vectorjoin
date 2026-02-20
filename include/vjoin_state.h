#ifndef VJOIN_STATE_H
#define VJOIN_STATE_H

#include "pg_vectorjoin.h"
#include "executor/tuptable.h"

/* ---------- Open-addressing hash table ---------- */
typedef struct VJoinHashTable
{
    uint32     *hashvals;       /* [capacity] — 0 means empty slot */
    MinimalTuple *tuples;       /* [capacity] */
    Datum      *keys;           /* [capacity * num_keys] */
    bool       *key_nulls;      /* [capacity * num_keys] */
    int         capacity;       /* power of 2 */
    int         mask;           /* capacity - 1 */
    int         num_entries;
    int         num_keys;
    MemoryContext htctx;
} VJoinHashTable;

/* ---------- Result buffer entry ---------- */
typedef struct VJoinMatch
{
    int outer_idx;
    int inner_idx;     /* index into hash table (for hash join) */
} VJoinMatch;

/* ---------- Vectorized Hash Join state ---------- */
typedef enum VHJPhase
{
    VHJ_BUILD,
    VHJ_PROBE,
    VHJ_EMIT,
    VHJ_DONE
} VHJPhase;

typedef struct VectorHashJoinState
{
    CustomScanState css;                /* must be first */

    VHJPhase    phase;

    /* Key info */
    int         num_keys;
    AttrNumber  outer_keynos[VJOIN_MAX_KEYS];
    AttrNumber  inner_keynos[VJOIN_MAX_KEYS];
    Oid         key_types[VJOIN_MAX_KEYS];

    /* Children */
    PlanState  *outer_ps;
    PlanState  *inner_ps;
    int         num_outer_attrs;
    int         num_inner_attrs;

    /* Hash table */
    VJoinHashTable *hashtable;

    /* Probe batch */
    int         batch_size;
    int         batch_count;       /* tuples in current batch */
    int         batch_pos;         /* next unprocessed in batch */
    Datum      *batch_keys;        /* [batch_size * num_keys] */
    bool       *batch_nulls;       /* [batch_size * num_keys] */
    uint32     *batch_hashes;      /* [batch_size] */
    MinimalTuple *batch_tuples;    /* [batch_size] */

    /* Result buffer */
    VJoinMatch *results;
    int         result_count;
    int         result_pos;
    int         result_capacity;

    /* Temp slots for result construction */
    TupleTableSlot *outer_slot;
    TupleTableSlot *inner_slot;

    /* Memory */
    MemoryContext hash_ctx;
    MemoryContext batch_ctx;

    bool        use_simd;
} VectorHashJoinState;

/* ---------- Block Nested Loop state ---------- */
typedef enum BNLPhase
{
    BNL_LOAD_BLOCK,
    BNL_SCAN_INNER,
    BNL_EMIT,
    BNL_DONE
} BNLPhase;

typedef struct BlockNestLoopState
{
    CustomScanState css;                /* must be first */

    BNLPhase    phase;

    /* Key info */
    int         num_keys;
    AttrNumber  outer_keynos[VJOIN_MAX_KEYS];
    AttrNumber  inner_keynos[VJOIN_MAX_KEYS];
    Oid         key_types[VJOIN_MAX_KEYS];

    /* Children */
    PlanState  *outer_ps;
    PlanState  *inner_ps;
    int         num_outer_attrs;
    int         num_inner_attrs;

    /* Outer block */
    int         block_size;
    int         block_count;        /* tuples in current block */
    Datum      *block_keys;         /* [block_size * num_keys] */
    bool       *block_nulls;        /* [block_size * num_keys] */
    MinimalTuple *block_tuples;     /* [block_size] */

    /* Result buffer */
    VJoinMatch *results;
    MinimalTuple *result_inner_tuples;  /* parallel with results */
    int         result_count;
    int         result_pos;
    int         result_capacity;

    /* Temp slots */
    TupleTableSlot *outer_slot;
    TupleTableSlot *inner_slot;

    bool        inner_exhausted;
    bool        outer_exhausted;

    MemoryContext block_ctx;
    bool        use_simd;
} BlockNestLoopState;

#endif /* VJOIN_STATE_H */
