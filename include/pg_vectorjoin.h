#ifndef PG_VECTORJOIN_H
#define PG_VECTORJOIN_H

#include "postgres.h"
#include "catalog/pg_type_d.h"
#include "nodes/extensible.h"
#include "optimizer/paths.h"
#include "utils/dsa.h"

/* Constants */
#define VJOIN_MAX_KEYS          8
/*
 * Batch (= outer block) sizing for SIMD.  The batch is the outer block fed as
 * block_count to the SIMD compare kernels; it is re-scanned for every inner
 * tuple, so it must stay cache-hot and be a multiple of the widest SIMD lane
 * (AVX2 int32 = 8 lanes).  Values are powers of two:
 *   MIN 64    = 8x the widest lane -> SIMD always runs >=1 full vector pass.
 *   DEFAULT 128 = 16KB (int32) / 32KB (int64,float8) -> L1-resident.
 *   MAX 8192  = 32KB (int32, = L1) / 64KB (8-byte keys, L2) -> upper bound.
 */
#define VJOIN_DEFAULT_BATCH     128
#define VJOIN_MIN_BATCH         64
#define VJOIN_MAX_BATCH         8192
#define VJOIN_HT_LOAD_FACTOR    2       /* capacity = inner_rows * factor */

/*
 * GUC default / range macros.  Centralised so initializers and
 * DefineCustomXxxVariable() calls cannot drift out of sync.
 *
 * Development-phase policy (pre-1.0):
 *   - All physical operators enabled.
 *   - cost_factor / per-jointype factors are NEUTRAL (1.0); we no longer
 *     hand-bias the model down to make vjoin look cheaper.
 *   - auto_tune is ON by default: when a native path of the same physical
 *     type is cheaper, we clamp our cost to native * auto_tune_margin.
 *     This makes vjoin selectable without manual tuning while still
 *     deferring to honest model wins.
 */
#define VJOIN_DEFAULT_ENABLE              true
#define VJOIN_DEFAULT_ENABLE_HASHJOIN     true
#define VJOIN_DEFAULT_ENABLE_NESTLOOP     true
#define VJOIN_DEFAULT_ENABLE_MERGEJOIN    true

#define VJOIN_DEFAULT_COST_FACTOR         1.0
#define VJOIN_MIN_COST_FACTOR             0.01
#define VJOIN_MAX_COST_FACTOR             10.0

#define VJOIN_DEFAULT_HASH_COST_FACTOR     1.0
#define VJOIN_DEFAULT_MERGE_COST_FACTOR    1.0
#define VJOIN_DEFAULT_NESTLOOP_COST_FACTOR 1.0

#define VJOIN_DEFAULT_AUTO_TUNE           true
#define VJOIN_DEFAULT_AUTO_TUNE_MARGIN    0.95
#define VJOIN_MIN_AUTO_TUNE_MARGIN        0.10
#define VJOIN_MAX_AUTO_TUNE_MARGIN        1.50

/*
 * Hash join multi-batch spilling.  When enabled (default), an oversized
 * inner relation is partitioned into 2^k batches via the high bits of the
 * 32-bit hashvalue and spilled to BufFiles instead of being rejected by
 * the planner.  hash_max_batches is a hard upper bound on nbatch to guard
 * against pathological recursive splits.
 */
#define VJOIN_DEFAULT_ENABLE_HASH_SPILL   true
#define VJOIN_DEFAULT_HASH_MAX_BATCHES    1024
#define VJOIN_MIN_HASH_MAX_BATCHES        1
#define VJOIN_MAX_HASH_MAX_BATCHES        1048576

/* Approximate spilled tuples per temp-file page, used by the spill I/O cost
 * estimate in vjoin_path.c. */
#define VJOIN_SPILL_TUPLES_PER_PAGE       100.0

/* GUC variables */
extern bool vjoin_enable;
extern bool vjoin_enable_hashjoin;
extern bool vjoin_enable_nestloop;
extern bool vjoin_enable_mergejoin;
extern int  vjoin_batch_size;
extern double vjoin_cost_factor;
extern double vjoin_hash_cost_factor;
extern double vjoin_merge_cost_factor;
extern double vjoin_nestloop_cost_factor;
extern bool   vjoin_auto_tune;
extern double vjoin_auto_tune_margin;
extern bool   vjoin_enable_hash_spill;
extern int    vjoin_hash_max_batches;

/* Saved previous hooks (needed across translation units) */
extern set_join_pathlist_hook_type prev_join_pathlist_hook;

/* CustomScanMethods for registration */
extern CustomScanMethods vjoin_hash_scan_methods;
extern CustomScanMethods vjoin_nestloop_scan_methods;
extern CustomScanMethods vjoin_merge_scan_methods;

/* CustomPathMethods */
extern CustomPathMethods vjoin_hash_path_methods;
extern CustomPathMethods vjoin_nestloop_path_methods;
extern CustomPathMethods vjoin_merge_path_methods;

/* CustomExecMethods */
extern CustomExecMethods vjoin_hash_exec_methods;
extern CustomExecMethods vjoin_nestloop_exec_methods;
extern CustomExecMethods vjoin_merge_exec_methods;

/* Path generation hook */
void vjoin_pathlist_hook(PlannerInfo *root,
                         RelOptInfo *joinrel,
                         RelOptInfo *outerrel,
                         RelOptInfo *innerrel,
                         JoinType jointype,
                         JoinPathExtraData *extra);

/* Shared deserialization (vjoin_plan.c) */
int vjoin_deserialize_keys(List *private_data,
                           JoinType *jointype,
                           int *num_keys,
                           AttrNumber *outer_keynos,
                           AttrNumber *inner_keynos,
                           Oid *key_types,
                           Oid *hash_funcs,
                           Oid *eq_funcs,
                           Oid *key_collations);

/* PlanCustomPath callbacks */
Plan *vjoin_hash_plan(PlannerInfo *root, RelOptInfo *rel,
                      CustomPath *best_path, List *tlist,
                      List *clauses, List *custom_plans);
Plan *vjoin_nestloop_plan(PlannerInfo *root, RelOptInfo *rel,
                     CustomPath *best_path, List *tlist,
                     List *clauses, List *custom_plans);
Plan *vjoin_merge_plan(PlannerInfo *root, RelOptInfo *rel,
                       CustomPath *best_path, List *tlist,
                       List *clauses, List *custom_plans);

/* CreateCustomScanState callbacks */
Node *vjoin_hash_create_state(CustomScan *cscan);
Node *vjoin_nestloop_create_state(CustomScan *cscan);
Node *vjoin_merge_create_state(CustomScan *cscan);

/* Hash join executor callbacks */
void vjoin_hash_begin(CustomScanState *node, EState *estate, int eflags);
TupleTableSlot *vjoin_hash_exec(CustomScanState *node);
void vjoin_hash_end(CustomScanState *node);
void vjoin_hash_rescan(CustomScanState *node);
void vjoin_hash_explain(CustomScanState *node, List *ancestors, ExplainState *es);

/* Hash join parallel callbacks */
Size vjoin_hash_estimate_dsm(CustomScanState *node, ParallelContext *pcxt);
void vjoin_hash_initialize_dsm(CustomScanState *node, ParallelContext *pcxt,
                                void *coordinate);
void vjoin_hash_reinitialize_dsm(CustomScanState *node, ParallelContext *pcxt,
                                  void *coordinate);
void vjoin_hash_initialize_worker(CustomScanState *node, shm_toc *toc,
                                   void *coordinate);
void vjoin_hash_shutdown(CustomScanState *node);

/* NL executor callbacks */
void vjoin_nestloop_begin(CustomScanState *node, EState *estate, int eflags);
TupleTableSlot *vjoin_nestloop_exec(CustomScanState *node);
void vjoin_nestloop_end(CustomScanState *node);
void vjoin_nestloop_rescan(CustomScanState *node);
void vjoin_nestloop_explain(CustomScanState *node, List *ancestors, ExplainState *es);

/* NL parallel callbacks */
Size vjoin_nestloop_estimate_dsm(CustomScanState *node, ParallelContext *pcxt);
void vjoin_nestloop_initialize_dsm(CustomScanState *node, ParallelContext *pcxt,
                                    void *coordinate);
void vjoin_nestloop_reinitialize_dsm(CustomScanState *node, ParallelContext *pcxt,
                                      void *coordinate);
void vjoin_nestloop_initialize_worker(CustomScanState *node, shm_toc *toc,
                                       void *coordinate);
void vjoin_nestloop_shutdown(CustomScanState *node);

/* Merge join executor callbacks */
void vjoin_merge_begin(CustomScanState *node, EState *estate, int eflags);
TupleTableSlot *vjoin_merge_exec(CustomScanState *node);
void vjoin_merge_end(CustomScanState *node);
void vjoin_merge_rescan(CustomScanState *node);
void vjoin_merge_explain(CustomScanState *node, List *ancestors, ExplainState *es);

/* Merge join parallel callbacks */
Size vjoin_merge_estimate_dsm(CustomScanState *node, ParallelContext *pcxt);
void vjoin_merge_initialize_dsm(CustomScanState *node, ParallelContext *pcxt,
                                 void *coordinate);
void vjoin_merge_reinitialize_dsm(CustomScanState *node, ParallelContext *pcxt,
                                   void *coordinate);
void vjoin_merge_initialize_worker(CustomScanState *node, shm_toc *toc,
                                    void *coordinate);
void vjoin_merge_shutdown(CustomScanState *node);

/* Hash table functions (vjoin_hashtable.c) */
struct VJoinHashTable;
struct VJoinParallelState;
typedef struct VJoinHashTable VJoinHashTable;
VJoinHashTable *vjoin_ht_create(int estimated_rows, int num_keys,
                                int num_all_attrs, MemoryContext parent,
                                AttrNumber *inner_keynos,
                                bool *attr_byval, int16 *attr_typlen);
VJoinHashTable *vjoin_ht_create_shared(struct VJoinParallelState *pstate,
                                        dsa_area *dsa,
                                        int num_keys, int num_all_attrs,
                                        MemoryContext parent,
                                        AttrNumber *inner_keynos,
                                        bool *attr_byval, int16 *attr_typlen);
void vjoin_ht_insert(VJoinHashTable *ht, uint32 hashval,
                     Datum *all_values, bool *all_isnull);
bool vjoin_ht_insert_cas(VJoinHashTable *ht,
                         uint32 hashval,
                         Datum *all_values, bool *all_isnull);
void vjoin_ht_destroy(VJoinHashTable *ht);
/* Multi-batch spilling (vjoin_hashtable.c) */
struct vjoin_spill_file;
void vjoin_ht_reset_for_batch(VJoinHashTable *ht);
void vjoin_ht_close_spill_files(VJoinHashTable *ht);
struct vjoin_spill_file *vjoin_ht_inner_file(VJoinHashTable *ht, int batchno);
struct vjoin_spill_file *vjoin_ht_outer_file(VJoinHashTable *ht, int batchno);
void vjoin_ht_serialize_to_dsa(VJoinHashTable *ht, dsa_area *dsa,
                                struct VJoinParallelState *pstate);
VJoinHashTable *vjoin_ht_attach_from_dsa(struct VJoinParallelState *pstate,
                                          dsa_area *dsa,
                                          MemoryContext parent);

#include "fmgr.h"

/* Inline murmurhash32 finalizer */
static inline uint32
vjoin_murmurhash32(uint32 h)
{
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return h;
}

/* Fast-path hash for INT4/INT8/FLOAT8 */
static inline uint32
vjoin_hash_datum(Datum d, Oid keytype)
{
    switch (keytype)
    {
        case INT4OID:
            return vjoin_murmurhash32((uint32) DatumGetInt32(d));
        case INT8OID:
        {
            uint64 v = (uint64) DatumGetInt64(d);
            return vjoin_murmurhash32((uint32)(v ^ (v >> 32)));
        }
        case FLOAT8OID:
        {
            union { double d; uint64 u; } conv;
            conv.d = DatumGetFloat8(d);
            /* Normalize -0.0 to +0.0 */
            if (conv.d == 0.0) conv.u = 0;
            return vjoin_murmurhash32((uint32)(conv.u ^ (conv.u >> 32)));
        }
        default:
            return vjoin_murmurhash32((uint32) d);
    }
}

/* Generic hash via PG's type-specific hash function (collation-aware) */
static inline uint32
vjoin_hash_datum_generic(Datum d, FmgrInfo *hashfn, Oid collation)
{
    return DatumGetUInt32(FunctionCall1Coll(hashfn, collation, d));
}

/*
 * Map a type OID to its SIMD-compatible base type.
 *
 * Only pass-by-value types whose Datum representation matches the target
 * integer/float width are eligible.  In particular FLOAT4 is excluded
 * because DatumGetFloat8() reads 8 bytes but FLOAT4 stores only 4 — the
 * remaining bits are undefined, producing wrong hashes and comparisons.
 *
 *   32-bit integer path (DatumGetInt32):
 *     INT4, INT2, DATE, OID, REGCLASS, REGTYPE, REGPROC
 *   64-bit integer path (DatumGetInt64):
 *     INT8, TIMESTAMP, TIMESTAMPTZ, TIME
 *   64-bit float path (DatumGetFloat8):
 *     FLOAT8
 */
static inline Oid
vjoin_map_fast_type(Oid typid)
{
    switch (typid)
    {
        case INT4OID:
        case INT2OID:
        case DATEOID:
        case OIDOID:
        case REGCLASSOID:
        case REGTYPEOID:
        case REGPROCOID:
            return INT4OID;
        case INT8OID:
        case TIMESTAMPOID:
        case TIMESTAMPTZOID:
        case TIMEOID:
            return INT8OID;
        case FLOAT8OID:
            return FLOAT8OID;
        default:
            return InvalidOid;
    }
}

/* Returns true if type uses inline fast path, false if generic FmgrInfo needed */
static inline bool
vjoin_is_fast_type(Oid typid)
{
    return OidIsValid(vjoin_map_fast_type(typid));
}

static inline uint32
vjoin_combine_hashes(uint32 h1, uint32 h2)
{
    h1 = ((h1 << 1) | (h1 >> 31));  /* rotate left 1 */
    return h1 ^ h2;
}

/* Round up to next power of 2 (with overflow guard) */
static inline int
vjoin_next_power_of_2(int n)
{
    int p = 1;
    while (p < n)
    {
        if (p > INT_MAX / 2)
            return INT_MAX / 2 + 1;
        p <<= 1;
    }
    return p;
}

#endif /* PG_VECTORJOIN_H */
