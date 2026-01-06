#ifndef PG_VECTORJOIN_H
#define PG_VECTORJOIN_H

#include "postgres.h"

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

#endif /* PG_VECTORJOIN_H */
