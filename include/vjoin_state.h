#ifndef VJOIN_STATE_H
#define VJOIN_STATE_H

#include "pg_vectorjoin.h"
#include "executor/tuptable.h"
#include "fmgr.h"
#include "nodes/nodes.h"
#include "utils/tuplestore.h"

/* ---------- Parallel shared state ---------- */
/*
 * Minimal DSM-resident struct shared between leader and workers.
 * Each worker independently builds its own join resources (hash table,
 * tuplestore, etc.) and probes with its partial outer scan output.
 * This struct exists for proper DSM protocol compliance and can be
 * extended with barrier coordination or shared hash table in the future.
 */
typedef struct VJoinParallelState
{
    int     initialized;    /* set to 1 by leader */
} VJoinParallelState;

/* ---------- Open-addressing hash table ---------- */
typedef struct VJoinHashTable
{
    uint32     *hashvals;       /* [capacity] — 0 means empty slot */
    Datum      *all_values;     /* [capacity * num_all_attrs] pre-deformed */
    bool       *all_isnull;     /* [capacity * num_all_attrs] */
    int         num_all_attrs;  /* total inner attrs */
    int         capacity;       /* power of 2 */
    int         mask;           /* capacity - 1 */
    int         num_entries;
    int         num_keys;
    AttrNumber *inner_keynos;   /* [num_keys] — 1-based attr positions of keys */
    bool       *attr_byval;     /* [num_all_attrs] — pass-by-value per attr */
    int16      *attr_typlen;    /* [num_all_attrs] — type length per attr */
    bool        all_attrs_byval; /* true if all attrs are pass-by-value */
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
    VHJ_LEFT_EMIT,
    VHJ_RIGHT_EMIT,
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

    /* Generic hash/eq support (InvalidOid for INT4/INT8/FLOAT8 fast path) */
    Oid         hash_funcs[VJOIN_MAX_KEYS];
    Oid         eq_funcs[VJOIN_MAX_KEYS];
    Oid         key_collations[VJOIN_MAX_KEYS];
    FmgrInfo    hash_finfo[VJOIN_MAX_KEYS];
    FmgrInfo    eq_finfo[VJOIN_MAX_KEYS];
    bool        key_byval[VJOIN_MAX_KEYS];
    int16       key_typlen[VJOIN_MAX_KEYS];
    bool        keys_all_byval;     /* true if all key types pass-by-value */

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
    uint32     *batch_hashes;      /* [batch_size] — 0 means NULL key */
    Datum      *batch_values;      /* [batch_size * num_outer_attrs] deformed */
    bool       *batch_isnull;      /* [batch_size * num_outer_attrs] */

    /* Result buffer */
    VJoinMatch *results;
    int         result_count;
    int         result_pos;
    int         result_capacity;

    /* Type metadata for outer attrs (pass-by-ref handling) */
    TupleDesc   outer_desc;
    TupleDesc   inner_desc;
    bool       *outer_byval;       /* [num_outer_attrs] */
    int16      *outer_typlen;      /* [num_outer_attrs] */
    bool        batch_all_byval;   /* true if all outer attrs pass-by-value */

    /* Memory */
    MemoryContext hash_ctx;
    MemoryContext batch_ctx;

    bool        use_simd;

    /* Outer join support */
    JoinType    jointype;
    bool       *batch_matched;      /* [batch_size] — outer tuple matched? */
    int         left_emit_pos;      /* scan pos for LEFT/FULL unmatched outer */
    bool       *inner_matched;      /* [ht capacity] — inner tuple matched? */
    int         right_emit_pos;     /* scan pos for RIGHT/FULL unmatched inner */
} VectorHashJoinState;

/* ---------- Block Nested Loop state ---------- */
typedef enum NLPhase
{
    NL_LOAD_BLOCK,
    NL_SCAN_INNER,
    NL_EMIT,
    NL_THETA_SCAN,
    NL_LEFT_EMIT,
    NL_RIGHT_EMIT,
    NL_DONE
} NLPhase;

typedef struct VJoinNestLoopState
{
    CustomScanState css;                /* must be first */

    NLPhase    phase;

    /* Key info */
    int         num_keys;
    AttrNumber  outer_keynos[VJOIN_MAX_KEYS];
    AttrNumber  inner_keynos[VJOIN_MAX_KEYS];
    Oid         key_types[VJOIN_MAX_KEYS];

    /* Generic equality support */
    Oid         eq_funcs[VJOIN_MAX_KEYS];
    Oid         key_collations[VJOIN_MAX_KEYS];
    FmgrInfo    eq_finfo[VJOIN_MAX_KEYS];
    bool        key_byval[VJOIN_MAX_KEYS];
    int16       key_typlen[VJOIN_MAX_KEYS];

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

    /* Pre-deformed outer values for theta scan (num_keys == 0) */
    Datum      *block_values;       /* [block_size * num_outer_attrs] */
    bool       *block_isnull;       /* [block_size * num_outer_attrs] */

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

    /* Inner materialization (tuplestore) */
    Tuplestorestate *inner_store;
    bool        inner_stored;       /* true after first full scan */
    TupleTableSlot *store_slot;     /* slot for reading from tuplestore */

    MemoryContext block_ctx;
    bool        use_simd;

    /* Theta-join direct iteration state (used when num_keys == 0) */
    int         theta_outer_pos;
    bool        theta_has_inner;

    /* Theta SIMD support (single-key INT4/INT8/FLOAT8 theta-join) */
    int         theta_strategy;      /* 0=none, 1=LT,2=LE,4=GE,5=GT,6=NE */
    AttrNumber  theta_outer_keyno;
    AttrNumber  theta_inner_keyno;
    Oid         theta_keytype;
    void       *theta_typed_keys;    /* [block_size] typed key array */
    bool       *theta_block_nulls;   /* [block_size] */
    int        *theta_match_buf;     /* [block_size] match indices from SIMD */
    int         theta_match_count;
    int         theta_match_pos;

    /* Outer join support */
    JoinType    jointype;
    bool       *block_matched;      /* [block_size] — outer tuple matched? */
    bool       *inner_matched;      /* [inner_store count] — inner matched? */
    int         inner_match_count;  /* capacity of inner_matched */
    int         inner_scan_idx;     /* current inner tuple index during scan */
    int         right_emit_pos;     /* scan pos for RIGHT/FULL unmatched inner */
} VJoinNestLoopState;

/* ---------- Vectorized Merge Join state ---------- */
typedef enum VMJPhase
{
    VMJ_INIT,
    VMJ_ADVANCE,
    VMJ_MATCH_OUTER,
    VMJ_MATCH_INNER,
    VMJ_EMIT,
    VMJ_BATCH_FILL,
    VMJ_BATCH_MERGE,
    VMJ_BATCH_EMIT,
    VMJ_LEFT_EMIT,
    VMJ_RIGHT_EMIT,
    VMJ_DONE
} VMJPhase;

typedef struct VectorMergeJoinState
{
    CustomScanState css;                /* must be first */

    VMJPhase    phase;

    /* Key info (single-key for v1) */
    int         num_keys;
    AttrNumber  outer_keynos[VJOIN_MAX_KEYS];
    AttrNumber  inner_keynos[VJOIN_MAX_KEYS];
    Oid         key_types[VJOIN_MAX_KEYS];

    /* Generic comparison/equality support */
    Oid         eq_funcs[VJOIN_MAX_KEYS];
    Oid         key_collations[VJOIN_MAX_KEYS];
    FmgrInfo    cmp_finfo[VJOIN_MAX_KEYS];
    FmgrInfo    eq_finfo[VJOIN_MAX_KEYS];
    bool        key_byval[VJOIN_MAX_KEYS];
    int16       key_typlen[VJOIN_MAX_KEYS];

    /* Children */
    PlanState  *outer_ps;
    PlanState  *inner_ps;
    int         num_outer_attrs;
    int         num_inner_attrs;

    /* Current position in each sorted stream */
    MinimalTuple outer_tuple;           /* current outer tuple */
    MinimalTuple inner_tuple;           /* current inner tuple */
    TupleTableSlot *outer_cur_slot;     /* last slot from outer ExecProcNode */
    TupleTableSlot *inner_cur_slot;     /* last slot from inner ExecProcNode */
    Datum       outer_key[VJOIN_MAX_KEYS];
    Datum       inner_key[VJOIN_MAX_KEYS];
    bool        outer_null[VJOIN_MAX_KEYS];
    bool        inner_null[VJOIN_MAX_KEYS];
    bool        outer_done;
    bool        inner_done;

    /* Saved tuples for singleton detection */
    MinimalTuple saved_outer;
    MinimalTuple saved_inner;
    Datum       match_key[VJOIN_MAX_KEYS];
    bool        match_null[VJOIN_MAX_KEYS];
    bool        outer_multi;
    bool        inner_multi;

    /* Outer group: all tuples with same key */
    MinimalTuple *outer_group;
    int         outer_group_count;
    int         outer_group_capacity;

    /* Inner group: all tuples with same key */
    MinimalTuple *inner_group;
    int         inner_group_count;
    int         inner_group_capacity;

    /* Emit position in cross product */
    int         emit_outer_pos;
    int         emit_inner_pos;

    /* Temp slots for result construction */
    TupleTableSlot *outer_slot;
    TupleTableSlot *inner_slot;
    TupleDesc   outer_desc;             /* saved for MT reconstruction */
    TupleDesc   inner_desc;

    /* Memory */
    MemoryContext match_ctx;
    bool        use_simd;
    bool        all_byval;              /* all output columns pass-by-value */

    /* Block merge buffers (used when all_byval is true) */
    int         batch_size;

    /* Outer block — pre-deformed into columnar arrays */
    Datum      *ob_keys;                /* [batch_size] primary key values */
    Datum      *ob_values;              /* [batch_size * num_outer_attrs] */
    bool       *ob_isnull;              /* [batch_size * num_outer_attrs] */
    int         ob_count;               /* tuples in current block */
    int         ob_pos;                 /* current merge position */
    bool        ob_exhausted;           /* child returned NULL */

    /* Inner block */
    Datum      *ib_keys;                /* [batch_size] primary key values */
    Datum      *ib_values;              /* [batch_size * num_inner_attrs] */
    bool       *ib_isnull;              /* [batch_size * num_inner_attrs] */
    int         ib_count;
    int         ib_pos;
    bool        ib_exhausted;

    /* Batch result buffer */
    VJoinMatch *batch_results;
    int         batch_result_count;
    int         batch_result_pos;
    int         batch_result_capacity;

    /* Cross-product tracking for multi-match groups within a batch */
    int         batch_cp_oi;            /* -1 if not in cross product */
    int         batch_cp_ii;
    int         batch_cp_oe;            /* outer group end */
    int         batch_cp_ie;            /* inner group end */
    int         batch_cp_ii_start;      /* inner group start for reset */

    /* Outer join support */
    JoinType    jointype;
    bool        outer_matched;          /* current outer tuple has a match */
    bool        inner_matched;          /* current inner tuple has a match */
} VectorMergeJoinState;

#endif /* VJOIN_STATE_H */
