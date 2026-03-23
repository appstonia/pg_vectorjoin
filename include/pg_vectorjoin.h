#ifndef PG_VECTORJOIN_H
#define PG_VECTORJOIN_H

#include "postgres.h"
#include "nodes/extensible.h"
#include "optimizer/paths.h"
#include "utils/dsa.h"

/* Constants */
#define VJOIN_MAX_KEYS          8
#define VJOIN_DEFAULT_BATCH     2000
#define VJOIN_MIN_BATCH         100
#define VJOIN_MAX_BATCH         10000
#define VJOIN_HT_LOAD_FACTOR   2       /* capacity = inner_rows * factor */

/* GUC variables */
extern bool vjoin_enable;
extern bool vjoin_enable_hashjoin;
extern bool vjoin_enable_nestloop;
extern bool vjoin_enable_mergejoin;
extern int  vjoin_batch_size;
extern double vjoin_cost_factor;

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
void vjoin_ht_insert(VJoinHashTable *ht, uint32 hashval,
                     Datum *all_values, bool *all_isnull);
void vjoin_ht_destroy(VJoinHashTable *ht);
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

/* Returns true if type uses inline fast path, false if generic FmgrInfo needed */
static inline bool
vjoin_is_fast_type(Oid typid)
{
    return typid == INT4OID || typid == INT8OID || typid == FLOAT8OID;
}

static inline uint32
vjoin_combine_hashes(uint32 h1, uint32 h2)
{
    h1 = ((h1 << 1) | (h1 >> 31));  /* rotate left 1 */
    return h1 ^ h2;
}

#endif /* PG_VECTORJOIN_H */
