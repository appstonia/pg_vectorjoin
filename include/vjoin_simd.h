#ifndef VJOIN_SIMD_H
#define VJOIN_SIMD_H

#include "postgres.h"

/* Runtime SIMD capability detection */
typedef struct VJoinSIMDCaps
{
    bool has_sse2;
    bool has_avx2;
    bool has_neon;
} VJoinSIMDCaps;

extern VJoinSIMDCaps vjoin_simd_caps;

void vjoin_detect_simd(void);

/*
 * Block comparison: compare block_count values against a single value.
 * Returns number of matches, fills match_indices[] with positions.
 * match_indices must be at least block_count elements.
 */
int vjoin_compare_int4_block(const int32 *block, int block_count,
                             int32 value, int *match_indices);
int vjoin_compare_int8_block(const int64 *block, int block_count,
                             int64 value, int *match_indices);
int vjoin_compare_float8_block(const double *block, int block_count,
                               double value, int *match_indices);

/*
 * Batch hash: compute hashes for an array of keys.
 */
void vjoin_hash_int4_batch(const int32 *values, int count, uint32 *hashes);
void vjoin_hash_int8_batch(const int64 *values, int count, uint32 *hashes);

#endif /* VJOIN_SIMD_H */
