#ifndef PG_VECTORJOIN_H
#define PG_VECTORJOIN_H

#include "postgres.h"
#include "optimizer/paths.h"

/* Constants */
#define VJOIN_MAX_KEYS          8
#define VJOIN_DEFAULT_BATCH     1024
#define VJOIN_MIN_BATCH         64
#define VJOIN_MAX_BATCH         8192
#define VJOIN_HT_LOAD_FACTOR   2       /* capacity = inner_rows * factor */

/* GUC variables */
extern bool vjoin_enable;
extern bool vjoin_enable_hashjoin;
extern bool vjoin_enable_bnl;
extern int  vjoin_batch_size;
extern double vjoin_cost_factor;

/* Saved previous hooks (needed across translation units) */
extern set_join_pathlist_hook_type prev_join_pathlist_hook;

/* Path generation hook */
void vjoin_pathlist_hook(PlannerInfo *root,
                         RelOptInfo *joinrel,
                         RelOptInfo *outerrel,
                         RelOptInfo *innerrel,
                         JoinType jointype,
                         JoinPathExtraData *extra);

#endif /* PG_VECTORJOIN_H */
