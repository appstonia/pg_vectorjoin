#ifndef VJOIN_STATE_H
#define VJOIN_STATE_H

#include "pg_vectorjoin.h"
#include "executor/tuptable.h"
#include "fmgr.h"
#include "nodes/nodes.h"
#include "port/atomics.h"
#include "storage/barrier.h"
#include "utils/dsa.h"
#include "utils/tuplestore.h"

/* ---------- Generic parallel state for NL (DSM-resident) ---------- */
typedef struct VJoinGenericParallelState
{
    int     initialized;    /* set to 1 by leader */
} VJoinGenericParallelState;

/* ---------- Parallel shared state for Merge Join (DSM-resident) ---------- */
/*
 * Leader scans the full sorted inner and materializes pre-deformed
 * Datum/isnull arrays + extracted keys into DSA shared memory.
 * Workers attach to DSA and read the shared inner directly —
 * zero redundant inner scans.
 *
 * Barrier phases:
 *   0 → materialize: leader scans inner, stores in DSA
 *   1 → merge:       all participants merge partial outer vs shared inner
 */
typedef struct VJoinMergeParallelState
{
    Barrier     barrier;            /* materialize / merge synchronization */
    dsa_handle  dsa_handle;         /* handle for DSA area */
    int         inner_count;        /* number of inner tuples materialized */
    int         num_inner_attrs;    /* inner tuple descriptor width */
    int         num_keys;           /* number of join keys */
    dsa_pointer inner_values_dp;    /* Datum[inner_count * num_inner_attrs] */
    dsa_pointer inner_isnull_dp;    /* bool[inner_count * num_inner_attrs] */
    dsa_pointer inner_keys_dp;      /* Datum[inner_count * num_keys] */
} VJoinMergeParallelState;

/* ---------- Parallel shared state for Hash Join (DSM-resident) ---------- */
/*
 * Leader builds the hash table locally, then copies the three flat arrays
 * (hashvals, all_values, all_isnull) into DSA shared memory.  Workers
 * attach to DSA and create a read-only VJoinHashTable wrapper that points
 * directly at the shared arrays — zero memcpy on the reader side.
 *
 * Barrier phases:
 *   0 → build:     leader builds HT and copies to DSA
 *   1 → probe:     all participants probe with partial outer
 */
typedef struct VJoinParallelState
{
    Barrier     barrier;        /* build/probe synchronization */
    dsa_handle  dsa_handle;     /* handle for DSA area */

    /* Shared hash table metadata */
    int         capacity;
    int         mask;
    int         num_entries;
    int         num_all_attrs;
    int         num_keys;

    bool        all_attrs_byval; /* true if every inner attr is pass-by-value */
    bool        built_in_dsa;    /* true if leader built HT directly in DSA */
    bool        parallel_build;  /* true if all participants build concurrently */

    pg_atomic_uint32  num_entries_atomic;  /* CAS counter for parallel build */
    pg_atomic_uint32  cas_resizing;        /* 1 = CAS resize in progress */

    /* DSA pointers to the flat arrays */
    dsa_pointer hashvals_dp;    /* uint32[capacity] */
    dsa_pointer all_values_dp;  /* Datum[capacity * num_all_attrs] */
    dsa_pointer all_isnull_dp;  /* bool[capacity * num_all_attrs] */
    dsa_pointer inner_keynos_dp;/* AttrNumber[num_keys] */
    dsa_pointer vardata_dp;     /* flat buffer for all pass-by-ref datum data */
    dsa_pointer attr_byval_dp;  /* bool[num_all_attrs] */

    /* Pre-allocation: estimated inner rows for DSA-direct build */
    int         est_inner_rows;
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

    /* DSA-direct build: arrays live in shared memory, not palloc'd */
    bool        is_shared;      /* true = arrays are DSA-backed */
    dsa_area   *dsa;            /* DSA area (only when is_shared) */
    struct VJoinParallelState *pstate; /* shared state (only when is_shared) */
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

    /* Parallel support (leader-builds, workers-probe) */
    bool        is_parallel;        /* true if running under Gather */
    bool        is_leader;          /* true if this is the leader process */
    VJoinParallelState *pstate;     /* pointer to DSM-resident shared state */
    dsa_area   *dsa;                /* attached DSA area (leader + workers) */
    int         cached_ht_entries;  /* snapshot for EXPLAIN after DSM detach */
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
    VMJ_BATCH_LEFT,
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
    int         total_attrs;            /* num_outer_attrs + num_inner_attrs */

    /* Pre-built result arrays for zero-copy emit */
    Datum      *batch_result_values;    /* [capacity * total_attrs] */
    bool       *batch_result_isnull;    /* [capacity * total_attrs] */
    Datum      *saved_tts_values;       /* original scan_slot->tts_values */
    bool       *saved_tts_isnull;       /* original scan_slot->tts_isnull */

    /* Cross-product tracking for multi-match groups within a batch */
    int         batch_cp_oi;            /* -1 if not in cross product */
    int         batch_cp_ii;
    int         batch_cp_oe;            /* outer group end */
    int         batch_cp_ie;            /* inner group end */
    int         batch_cp_ii_start;      /* inner group start for reset */

    /* Batch LEFT JOIN: track which outer tuples had at least one match */
    bool       *ob_matched;             /* [batch_size], NULL if INNER */
    int         batch_left_pos;         /* scan position for unmatched outers */

    /* Outer join support */
    JoinType    jointype;
    bool        outer_matched;          /* current outer tuple has a match */
    bool        inner_matched;          /* current inner tuple has a match */

    /* Parallel merge state */
    VJoinMergeParallelState *pstate;    /* shared DSM state (NULL if non-parallel) */
    dsa_area               *dsa;       /* DSA area (NULL if non-parallel) */
    bool        is_parallel;
    bool        is_leader;

    /* Shared inner materialized buffer (worker-side pointers) */
    Datum      *shared_inner_values;   /* points into DSA */
    bool       *shared_inner_isnull;   /* points into DSA */
    Datum      *shared_inner_keys;     /* points into DSA */
    int         shared_inner_count;    /* total inner tuples */
    int         shared_inner_pos;      /* current read position */
} VectorMergeJoinState;

#endif /* VJOIN_STATE_H */
