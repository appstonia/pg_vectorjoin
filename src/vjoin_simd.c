#include "postgres.h"
#include "vjoin_simd.h"

VJoinSIMDCaps vjoin_simd_caps = {false, false, false};

/* ---- CPU Feature Detection ---- */

void
vjoin_detect_simd(void)
{
#if defined(__x86_64__) || defined(_M_X64)
    vjoin_simd_caps.has_sse2 = true;  /* baseline on x86_64 */

#ifdef __GNUC__
    __builtin_cpu_init();
    vjoin_simd_caps.has_avx2 = __builtin_cpu_supports("avx2");
#elif defined(_MSC_VER)
    {
        int cpuinfo[4];
        __cpuidex(cpuinfo, 7, 0);
        vjoin_simd_caps.has_avx2 = (cpuinfo[1] & (1 << 5)) != 0;
    }
#endif

#elif defined(__aarch64__) || defined(_M_ARM64)
    vjoin_simd_caps.has_neon = true;  /* baseline on ARMv8 */
#endif
}

/* ================================================================
 * SIMD comparison kernels
 * ================================================================ */

/* ---- AVX2 int32: 8 comparisons per instruction ---- */
#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#ifdef __GNUC__
__attribute__((target("avx2")))
#endif
static int
compare_int4_avx2(const int32 *block, int block_count,
                  int32 value, int *match_indices)
{
    int count = 0;
    int i;
    __m256i vval = _mm256_set1_epi32(value);

    for (i = 0; i + 8 <= block_count; i += 8)
    {
        __m256i vblock = _mm256_loadu_si256((const __m256i *)(block + i));
        __m256i vcmp = _mm256_cmpeq_epi32(vblock, vval);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(vcmp));

        while (mask)
        {
            int bit = __builtin_ctz(mask);
            match_indices[count++] = i + bit;
            mask &= mask - 1;
        }
    }

    /* Scalar tail */
    for (; i < block_count; i++)
    {
        if (block[i] == value)
            match_indices[count++] = i;
    }

    return count;
}

#ifdef __GNUC__
__attribute__((target("avx2")))
#endif
static int
compare_int8_avx2(const int64 *block, int block_count,
                  int64 value, int *match_indices)
{
    int count = 0;
    int i;
    __m256i vval = _mm256_set1_epi64x(value);

    for (i = 0; i + 4 <= block_count; i += 4)
    {
        __m256i vblock = _mm256_loadu_si256((const __m256i *)(block + i));
        __m256i vcmp = _mm256_cmpeq_epi64(vblock, vval);
        int mask = _mm256_movemask_pd(_mm256_castsi256_pd(vcmp));

        while (mask)
        {
            int bit = __builtin_ctz(mask);
            match_indices[count++] = i + bit;
            mask &= mask - 1;
        }
    }

    for (; i < block_count; i++)
    {
        if (block[i] == value)
            match_indices[count++] = i;
    }

    return count;
}

#ifdef __GNUC__
__attribute__((target("avx2")))
#endif
static int
compare_float8_avx2(const double *block, int block_count,
                    double value, int *match_indices)
{
    int count = 0;
    int i;
    __m256d vval = _mm256_set1_pd(value);

    for (i = 0; i + 4 <= block_count; i += 4)
    {
        __m256d vblock = _mm256_loadu_pd(block + i);
        __m256d vcmp = _mm256_cmp_pd(vblock, vval, _CMP_EQ_OQ);
        int mask = _mm256_movemask_pd(vcmp);

        while (mask)
        {
            int bit = __builtin_ctz(mask);
            match_indices[count++] = i + bit;
            mask &= mask - 1;
        }
    }

    for (; i < block_count; i++)
    {
        if (block[i] == value)
            match_indices[count++] = i;
    }

    return count;
}

/* ---- SSE2 int32: 4 comparisons per instruction ---- */
static int
compare_int4_sse2(const int32 *block, int block_count,
                  int32 value, int *match_indices)
{
    int count = 0;
    int i;
    __m128i vval = _mm_set1_epi32(value);

    for (i = 0; i + 4 <= block_count; i += 4)
    {
        __m128i vblock = _mm_loadu_si128((const __m128i *)(block + i));
        __m128i vcmp = _mm_cmpeq_epi32(vblock, vval);
        int mask = _mm_movemask_ps(_mm_castsi128_ps(vcmp));

        while (mask)
        {
            int bit = __builtin_ctz(mask);
            match_indices[count++] = i + bit;
            mask &= mask - 1;
        }
    }

    for (; i < block_count; i++)
    {
        if (block[i] == value)
            match_indices[count++] = i;
    }

    return count;
}

static int
compare_int8_sse2(const int64 *block, int block_count,
                  int64 value, int *match_indices)
{
    /* SSE2 has no native _mm_cmpeq_epi64; use scalar */
    int count = 0;
    int i;

    for (i = 0; i < block_count; i++)
    {
        if (block[i] == value)
            match_indices[count++] = i;
    }

    return count;
}

static int
compare_float8_sse2(const double *block, int block_count,
                    double value, int *match_indices)
{
    int count = 0;
    int i;
    __m128d vval = _mm_set1_pd(value);

    for (i = 0; i + 2 <= block_count; i += 2)
    {
        __m128d vblock = _mm_loadu_pd(block + i);
        __m128d vcmp = _mm_cmpeq_pd(vblock, vval);
        int mask = _mm_movemask_pd(vcmp);

        while (mask)
        {
            int bit = __builtin_ctz(mask);
            match_indices[count++] = i + bit;
            mask &= mask - 1;
        }
    }

    for (; i < block_count; i++)
    {
        if (block[i] == value)
            match_indices[count++] = i;
    }

    return count;
}

#endif /* x86_64 */

/* ---- ARM NEON int32: 4 comparisons per instruction ---- */
#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

static int
compare_int4_neon(const int32 *block, int block_count,
                  int32 value, int *match_indices)
{
    int count = 0;
    int i;
    int32x4_t vval = vdupq_n_s32(value);

    for (i = 0; i + 4 <= block_count; i += 4)
    {
        int32x4_t vblock = vld1q_s32(block + i);
        uint32x4_t vcmp = vceqq_s32(vblock, vval);

        /* Extract mask from NEON result */
        uint32_t lane0 = vgetq_lane_u32(vcmp, 0);
        uint32_t lane1 = vgetq_lane_u32(vcmp, 1);
        uint32_t lane2 = vgetq_lane_u32(vcmp, 2);
        uint32_t lane3 = vgetq_lane_u32(vcmp, 3);

        if (lane0) match_indices[count++] = i;
        if (lane1) match_indices[count++] = i + 1;
        if (lane2) match_indices[count++] = i + 2;
        if (lane3) match_indices[count++] = i + 3;
    }

    for (; i < block_count; i++)
    {
        if (block[i] == value)
            match_indices[count++] = i;
    }

    return count;
}

static int
compare_int8_neon(const int64 *block, int block_count,
                  int64 value, int *match_indices)
{
    int count = 0;
    int i;
    int64x2_t vval = vdupq_n_s64(value);

    for (i = 0; i + 2 <= block_count; i += 2)
    {
        int64x2_t vblock = vld1q_s64((const int64_t *)(block + i));
        uint64x2_t vcmp = vceqq_s64(vblock, vval);

        if (vgetq_lane_u64(vcmp, 0)) match_indices[count++] = i;
        if (vgetq_lane_u64(vcmp, 1)) match_indices[count++] = i + 1;
    }

    for (; i < block_count; i++)
    {
        if (block[i] == value)
            match_indices[count++] = i;
    }

    return count;
}

static int
compare_float8_neon(const double *block, int block_count,
                    double value, int *match_indices)
{
    int count = 0;
    int i;
    float64x2_t vval = vdupq_n_f64(value);

    for (i = 0; i + 2 <= block_count; i += 2)
    {
        float64x2_t vblock = vld1q_f64(block + i);
        uint64x2_t vcmp = vceqq_f64(vblock, vval);

        if (vgetq_lane_u64(vcmp, 0)) match_indices[count++] = i;
        if (vgetq_lane_u64(vcmp, 1)) match_indices[count++] = i + 1;
    }

    for (; i < block_count; i++)
    {
        if (block[i] == value)
            match_indices[count++] = i;
    }

    return count;
}

#endif /* aarch64 */

/* ---- Scalar fallback ---- */
static int
compare_int4_scalar(const int32 *block, int block_count,
                    int32 value, int *match_indices)
{
    int count = 0, i;
    for (i = 0; i < block_count; i++)
        if (block[i] == value)
            match_indices[count++] = i;
    return count;
}

static int
compare_int8_scalar(const int64 *block, int block_count,
                    int64 value, int *match_indices)
{
    int count = 0, i;
    for (i = 0; i < block_count; i++)
        if (block[i] == value)
            match_indices[count++] = i;
    return count;
}

static int
compare_float8_scalar(const double *block, int block_count,
                      double value, int *match_indices)
{
    int count = 0, i;
    for (i = 0; i < block_count; i++)
        if (block[i] == value)
            match_indices[count++] = i;
    return count;
}

/* ================================================================
 * Public dispatch functions
 * ================================================================ */

int
vjoin_compare_int4_block(const int32 *block, int block_count,
                         int32 value, int *match_indices)
{
#if defined(__x86_64__) || defined(_M_X64)
    if (vjoin_simd_caps.has_avx2)
        return compare_int4_avx2(block, block_count, value, match_indices);
    if (vjoin_simd_caps.has_sse2)
        return compare_int4_sse2(block, block_count, value, match_indices);
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (vjoin_simd_caps.has_neon)
        return compare_int4_neon(block, block_count, value, match_indices);
#endif
    return compare_int4_scalar(block, block_count, value, match_indices);
}

int
vjoin_compare_int8_block(const int64 *block, int block_count,
                         int64 value, int *match_indices)
{
#if defined(__x86_64__) || defined(_M_X64)
    if (vjoin_simd_caps.has_avx2)
        return compare_int8_avx2(block, block_count, value, match_indices);
    if (vjoin_simd_caps.has_sse2)
        return compare_int8_sse2(block, block_count, value, match_indices);
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (vjoin_simd_caps.has_neon)
        return compare_int8_neon(block, block_count, value, match_indices);
#endif
    return compare_int8_scalar(block, block_count, value, match_indices);
}

int
vjoin_compare_float8_block(const double *block, int block_count,
                           double value, int *match_indices)
{
#if defined(__x86_64__) || defined(_M_X64)
    if (vjoin_simd_caps.has_avx2)
        return compare_float8_avx2(block, block_count, value, match_indices);
    if (vjoin_simd_caps.has_sse2)
        return compare_float8_sse2(block, block_count, value, match_indices);
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (vjoin_simd_caps.has_neon)
        return compare_float8_neon(block, block_count, value, match_indices);
#endif
    return compare_float8_scalar(block, block_count, value, match_indices);
}

/* ================================================================
 * Theta (inequality) comparison kernels
 * Strategy: 1=LT, 2=LE, 3=EQ, 4=GE, 5=GT, 6=NE
 * ================================================================ */

/* ---- Theta scalar fallbacks ---- */
static int
compare_int4_theta_scalar(const int32 *block, int block_count,
                          int32 value, int strategy, int *match_indices)
{
    int count = 0, i;
    for (i = 0; i < block_count; i++)
    {
        bool match;
        switch (strategy)
        {
            case 1: match = block[i] < value; break;
            case 2: match = block[i] <= value; break;
            case 3: match = block[i] == value; break;
            case 4: match = block[i] >= value; break;
            case 5: match = block[i] > value; break;
            case 6: match = block[i] != value; break;
            default: match = false; break;
        }
        if (match)
            match_indices[count++] = i;
    }
    return count;
}

static int
compare_int8_theta_scalar(const int64 *block, int block_count,
                          int64 value, int strategy, int *match_indices)
{
    int count = 0, i;
    for (i = 0; i < block_count; i++)
    {
        bool match;
        switch (strategy)
        {
            case 1: match = block[i] < value; break;
            case 2: match = block[i] <= value; break;
            case 3: match = block[i] == value; break;
            case 4: match = block[i] >= value; break;
            case 5: match = block[i] > value; break;
            case 6: match = block[i] != value; break;
            default: match = false; break;
        }
        if (match)
            match_indices[count++] = i;
    }
    return count;
}

static int
compare_float8_theta_scalar(const double *block, int block_count,
                            double value, int strategy, int *match_indices)
{
    int count = 0, i;
    for (i = 0; i < block_count; i++)
    {
        bool match;
        switch (strategy)
        {
            case 1: match = block[i] < value; break;
            case 2: match = block[i] <= value; break;
            case 3: match = block[i] == value; break;
            case 4: match = block[i] >= value; break;
            case 5: match = block[i] > value; break;
            case 6: match = block[i] != value; break;
            default: match = false; break;
        }
        if (match)
            match_indices[count++] = i;
    }
    return count;
}

/* ---- AVX2 theta kernels ---- */
#if defined(__x86_64__) || defined(_M_X64)

#ifdef __GNUC__
__attribute__((target("avx2")))
#endif
static int
compare_int4_theta_avx2(const int32 *block, int block_count,
                        int32 value, int strategy, int *match_indices)
{
    int count = 0, i;
    __m256i vval = _mm256_set1_epi32(value);

    for (i = 0; i + 8 <= block_count; i += 8)
    {
        __m256i vblock = _mm256_loadu_si256((const __m256i *)(block + i));
        __m256i vcmp;
        int mask;

        switch (strategy)
        {
            case 1: /* LT: block < val  →  val > block */
                vcmp = _mm256_cmpgt_epi32(vval, vblock);
                break;
            case 2: /* LE: block <= val  →  NOT(block > val) */
                vcmp = _mm256_andnot_si256(
                    _mm256_cmpgt_epi32(vblock, vval),
                    _mm256_set1_epi32(-1));
                break;
            case 3: /* EQ */
                vcmp = _mm256_cmpeq_epi32(vblock, vval);
                break;
            case 4: /* GE: block >= val  →  NOT(val > block) */
                vcmp = _mm256_andnot_si256(
                    _mm256_cmpgt_epi32(vval, vblock),
                    _mm256_set1_epi32(-1));
                break;
            case 5: /* GT: block > val */
                vcmp = _mm256_cmpgt_epi32(vblock, vval);
                break;
            case 6: /* NE: NOT(block == val) */
                vcmp = _mm256_andnot_si256(
                    _mm256_cmpeq_epi32(vblock, vval),
                    _mm256_set1_epi32(-1));
                break;
            default:
                vcmp = _mm256_setzero_si256();
                break;
        }

        mask = _mm256_movemask_ps(_mm256_castsi256_ps(vcmp));
        while (mask)
        {
            int bit = __builtin_ctz(mask);
            match_indices[count++] = i + bit;
            mask &= mask - 1;
        }
    }

    /* Scalar tail */
    for (; i < block_count; i++)
    {
        bool match;
        switch (strategy)
        {
            case 1: match = block[i] < value; break;
            case 2: match = block[i] <= value; break;
            case 3: match = block[i] == value; break;
            case 4: match = block[i] >= value; break;
            case 5: match = block[i] > value; break;
            case 6: match = block[i] != value; break;
            default: match = false; break;
        }
        if (match)
            match_indices[count++] = i;
    }

    return count;
}

#ifdef __GNUC__
__attribute__((target("avx2")))
#endif
static int
compare_int8_theta_avx2(const int64 *block, int block_count,
                        int64 value, int strategy, int *match_indices)
{
    int count = 0, i;
    __m256i vval = _mm256_set1_epi64x(value);

    for (i = 0; i + 4 <= block_count; i += 4)
    {
        __m256i vblock = _mm256_loadu_si256((const __m256i *)(block + i));
        __m256i vcmp;
        int mask;

        switch (strategy)
        {
            case 1:
                vcmp = _mm256_cmpgt_epi64(vval, vblock);
                break;
            case 2:
                vcmp = _mm256_andnot_si256(
                    _mm256_cmpgt_epi64(vblock, vval),
                    _mm256_set1_epi64x(-1));
                break;
            case 3:
                vcmp = _mm256_cmpeq_epi64(vblock, vval);
                break;
            case 4:
                vcmp = _mm256_andnot_si256(
                    _mm256_cmpgt_epi64(vval, vblock),
                    _mm256_set1_epi64x(-1));
                break;
            case 5:
                vcmp = _mm256_cmpgt_epi64(vblock, vval);
                break;
            case 6:
                vcmp = _mm256_andnot_si256(
                    _mm256_cmpeq_epi64(vblock, vval),
                    _mm256_set1_epi64x(-1));
                break;
            default:
                vcmp = _mm256_setzero_si256();
                break;
        }

        mask = _mm256_movemask_pd(_mm256_castsi256_pd(vcmp));
        while (mask)
        {
            int bit = __builtin_ctz(mask);
            match_indices[count++] = i + bit;
            mask &= mask - 1;
        }
    }

    for (; i < block_count; i++)
    {
        bool match;
        switch (strategy)
        {
            case 1: match = block[i] < value; break;
            case 2: match = block[i] <= value; break;
            case 3: match = block[i] == value; break;
            case 4: match = block[i] >= value; break;
            case 5: match = block[i] > value; break;
            case 6: match = block[i] != value; break;
            default: match = false; break;
        }
        if (match)
            match_indices[count++] = i;
    }

    return count;
}

#ifdef __GNUC__
__attribute__((target("avx2")))
#endif
static int
compare_float8_theta_avx2(const double *block, int block_count,
                          double value, int strategy, int *match_indices)
{
    int count = 0, i;
    __m256d vval = _mm256_set1_pd(value);

    for (i = 0; i + 4 <= block_count; i += 4)
    {
        __m256d vblock = _mm256_loadu_pd(block + i);
        __m256d vcmp;
        int mask;

        switch (strategy)
        {
            case 1: vcmp = _mm256_cmp_pd(vblock, vval, _CMP_LT_OQ); break;
            case 2: vcmp = _mm256_cmp_pd(vblock, vval, _CMP_LE_OQ); break;
            case 3: vcmp = _mm256_cmp_pd(vblock, vval, _CMP_EQ_OQ); break;
            case 4: vcmp = _mm256_cmp_pd(vblock, vval, _CMP_GE_OQ); break;
            case 5: vcmp = _mm256_cmp_pd(vblock, vval, _CMP_GT_OQ); break;
            case 6: vcmp = _mm256_cmp_pd(vblock, vval, _CMP_NEQ_OQ); break;
            default: vcmp = _mm256_setzero_pd(); break;
        }

        mask = _mm256_movemask_pd(vcmp);
        while (mask)
        {
            int bit = __builtin_ctz(mask);
            match_indices[count++] = i + bit;
            mask &= mask - 1;
        }
    }

    for (; i < block_count; i++)
    {
        bool match;
        switch (strategy)
        {
            case 1: match = block[i] < value; break;
            case 2: match = block[i] <= value; break;
            case 3: match = block[i] == value; break;
            case 4: match = block[i] >= value; break;
            case 5: match = block[i] > value; break;
            case 6: match = block[i] != value; break;
            default: match = false; break;
        }
        if (match)
            match_indices[count++] = i;
    }

    return count;
}

/* ---- SSE2 theta kernels (int32 only; int8/float8 use scalar) ---- */
static int
compare_int4_theta_sse2(const int32 *block, int block_count,
                        int32 value, int strategy, int *match_indices)
{
    int count = 0, i;
    __m128i vval = _mm_set1_epi32(value);

    for (i = 0; i + 4 <= block_count; i += 4)
    {
        __m128i vblock = _mm_loadu_si128((const __m128i *)(block + i));
        __m128i vcmp;
        int mask;

        switch (strategy)
        {
            case 1:
                vcmp = _mm_cmpgt_epi32(vval, vblock);
                break;
            case 2:
                vcmp = _mm_andnot_si128(
                    _mm_cmpgt_epi32(vblock, vval),
                    _mm_set1_epi32(-1));
                break;
            case 3:
                vcmp = _mm_cmpeq_epi32(vblock, vval);
                break;
            case 4:
                vcmp = _mm_andnot_si128(
                    _mm_cmpgt_epi32(vval, vblock),
                    _mm_set1_epi32(-1));
                break;
            case 5:
                vcmp = _mm_cmpgt_epi32(vblock, vval);
                break;
            case 6:
                vcmp = _mm_andnot_si128(
                    _mm_cmpeq_epi32(vblock, vval),
                    _mm_set1_epi32(-1));
                break;
            default:
                vcmp = _mm_setzero_si128();
                break;
        }

        mask = _mm_movemask_ps(_mm_castsi128_ps(vcmp));
        while (mask)
        {
            int bit = __builtin_ctz(mask);
            match_indices[count++] = i + bit;
            mask &= mask - 1;
        }
    }

    for (; i < block_count; i++)
    {
        bool match;
        switch (strategy)
        {
            case 1: match = block[i] < value; break;
            case 2: match = block[i] <= value; break;
            case 3: match = block[i] == value; break;
            case 4: match = block[i] >= value; break;
            case 5: match = block[i] > value; break;
            case 6: match = block[i] != value; break;
            default: match = false; break;
        }
        if (match)
            match_indices[count++] = i;
    }

    return count;
}

static int
compare_float8_theta_sse2(const double *block, int block_count,
                          double value, int strategy, int *match_indices)
{
    int count = 0, i;
    __m128d vval = _mm_set1_pd(value);

    for (i = 0; i + 2 <= block_count; i += 2)
    {
        __m128d vblock = _mm_loadu_pd(block + i);
        __m128d vcmp;
        int mask;

        switch (strategy)
        {
            case 1: vcmp = _mm_cmplt_pd(vblock, vval); break;
            case 2: vcmp = _mm_cmple_pd(vblock, vval); break;
            case 3: vcmp = _mm_cmpeq_pd(vblock, vval); break;
            case 4: vcmp = _mm_cmpge_pd(vblock, vval); break;
            case 5: vcmp = _mm_cmpgt_pd(vblock, vval); break;
            case 6: vcmp = _mm_cmpneq_pd(vblock, vval); break;
            default: vcmp = _mm_setzero_pd(); break;
        }

        mask = _mm_movemask_pd(vcmp);
        while (mask)
        {
            int bit = __builtin_ctz(mask);
            match_indices[count++] = i + bit;
            mask &= mask - 1;
        }
    }

    for (; i < block_count; i++)
    {
        bool match;
        switch (strategy)
        {
            case 1: match = block[i] < value; break;
            case 2: match = block[i] <= value; break;
            case 3: match = block[i] == value; break;
            case 4: match = block[i] >= value; break;
            case 5: match = block[i] > value; break;
            case 6: match = block[i] != value; break;
            default: match = false; break;
        }
        if (match)
            match_indices[count++] = i;
    }

    return count;
}

#endif /* x86_64 */

/* ---- NEON theta kernels ---- */
#if defined(__aarch64__) || defined(_M_ARM64)

static int
compare_int4_theta_neon(const int32 *block, int block_count,
                        int32 value, int strategy, int *match_indices)
{
    int count = 0, i;
    int32x4_t vval = vdupq_n_s32(value);

    for (i = 0; i + 4 <= block_count; i += 4)
    {
        int32x4_t vblock = vld1q_s32(block + i);
        uint32x4_t vcmp;

        switch (strategy)
        {
            case 1: vcmp = vcltq_s32(vblock, vval); break;
            case 2: vcmp = vcleq_s32(vblock, vval); break;
            case 3: vcmp = vceqq_s32(vblock, vval); break;
            case 4: vcmp = vcgeq_s32(vblock, vval); break;
            case 5: vcmp = vcgtq_s32(vblock, vval); break;
            case 6: vcmp = vmvnq_u32(vceqq_s32(vblock, vval)); break;
            default: vcmp = vdupq_n_u32(0); break;
        }

        if (vgetq_lane_u32(vcmp, 0)) match_indices[count++] = i;
        if (vgetq_lane_u32(vcmp, 1)) match_indices[count++] = i + 1;
        if (vgetq_lane_u32(vcmp, 2)) match_indices[count++] = i + 2;
        if (vgetq_lane_u32(vcmp, 3)) match_indices[count++] = i + 3;
    }

    for (; i < block_count; i++)
    {
        bool match;
        switch (strategy)
        {
            case 1: match = block[i] < value; break;
            case 2: match = block[i] <= value; break;
            case 3: match = block[i] == value; break;
            case 4: match = block[i] >= value; break;
            case 5: match = block[i] > value; break;
            case 6: match = block[i] != value; break;
            default: match = false; break;
        }
        if (match)
            match_indices[count++] = i;
    }

    return count;
}

static int
compare_int8_theta_neon(const int64 *block, int block_count,
                        int64 value, int strategy, int *match_indices)
{
    int count = 0, i;
    int64x2_t vval = vdupq_n_s64(value);

    for (i = 0; i + 2 <= block_count; i += 2)
    {
        int64x2_t vblock = vld1q_s64((const int64_t *)(block + i));
        uint64x2_t vcmp;

        switch (strategy)
        {
            case 1: vcmp = vcltq_s64(vblock, vval); break;
            case 2: vcmp = vcleq_s64(vblock, vval); break;
            case 3: vcmp = vceqq_s64(vblock, vval); break;
            case 4: vcmp = vcgeq_s64(vblock, vval); break;
            case 5: vcmp = vcgtq_s64(vblock, vval); break;
            case 6: vcmp = vreinterpretq_u64_u32(
                        vmvnq_u32(vreinterpretq_u32_u64(
                            vceqq_s64(vblock, vval)))); break;
            default: vcmp = vdupq_n_u64(0); break;
        }

        if (vgetq_lane_u64(vcmp, 0)) match_indices[count++] = i;
        if (vgetq_lane_u64(vcmp, 1)) match_indices[count++] = i + 1;
    }

    for (; i < block_count; i++)
    {
        bool match;
        switch (strategy)
        {
            case 1: match = block[i] < value; break;
            case 2: match = block[i] <= value; break;
            case 3: match = block[i] == value; break;
            case 4: match = block[i] >= value; break;
            case 5: match = block[i] > value; break;
            case 6: match = block[i] != value; break;
            default: match = false; break;
        }
        if (match)
            match_indices[count++] = i;
    }

    return count;
}

static int
compare_float8_theta_neon(const double *block, int block_count,
                          double value, int strategy, int *match_indices)
{
    int count = 0, i;
    float64x2_t vval = vdupq_n_f64(value);

    for (i = 0; i + 2 <= block_count; i += 2)
    {
        float64x2_t vblock = vld1q_f64(block + i);
        uint64x2_t vcmp;

        switch (strategy)
        {
            case 1: vcmp = vcltq_f64(vblock, vval); break;
            case 2: vcmp = vcleq_f64(vblock, vval); break;
            case 3: vcmp = vceqq_f64(vblock, vval); break;
            case 4: vcmp = vcgeq_f64(vblock, vval); break;
            case 5: vcmp = vcgtq_f64(vblock, vval); break;
            case 6: vcmp = vreinterpretq_u64_u32(
                        vmvnq_u32(vreinterpretq_u32_u64(
                            vceqq_f64(vblock, vval)))); break;
            default: vcmp = vdupq_n_u64(0); break;
        }

        if (vgetq_lane_u64(vcmp, 0)) match_indices[count++] = i;
        if (vgetq_lane_u64(vcmp, 1)) match_indices[count++] = i + 1;
    }

    for (; i < block_count; i++)
    {
        bool match;
        switch (strategy)
        {
            case 1: match = block[i] < value; break;
            case 2: match = block[i] <= value; break;
            case 3: match = block[i] == value; break;
            case 4: match = block[i] >= value; break;
            case 5: match = block[i] > value; break;
            case 6: match = block[i] != value; break;
            default: match = false; break;
        }
        if (match)
            match_indices[count++] = i;
    }

    return count;
}

#endif /* aarch64 */

/* ---- Theta dispatch functions ---- */

int
vjoin_compare_int4_block_theta(const int32 *block, int block_count,
                               int32 value, int strategy, int *match_indices)
{
#if defined(__x86_64__) || defined(_M_X64)
    if (vjoin_simd_caps.has_avx2)
        return compare_int4_theta_avx2(block, block_count, value, strategy,
                                       match_indices);
    if (vjoin_simd_caps.has_sse2)
        return compare_int4_theta_sse2(block, block_count, value, strategy,
                                       match_indices);
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (vjoin_simd_caps.has_neon)
        return compare_int4_theta_neon(block, block_count, value, strategy,
                                       match_indices);
#endif
    return compare_int4_theta_scalar(block, block_count, value, strategy,
                                     match_indices);
}

int
vjoin_compare_int8_block_theta(const int64 *block, int block_count,
                               int64 value, int strategy, int *match_indices)
{
#if defined(__x86_64__) || defined(_M_X64)
    if (vjoin_simd_caps.has_avx2)
        return compare_int8_theta_avx2(block, block_count, value, strategy,
                                       match_indices);
    /* SSE2: no native 64-bit compare, fall through to scalar */
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (vjoin_simd_caps.has_neon)
        return compare_int8_theta_neon(block, block_count, value, strategy,
                                       match_indices);
#endif
    return compare_int8_theta_scalar(block, block_count, value, strategy,
                                     match_indices);
}

int
vjoin_compare_float8_block_theta(const double *block, int block_count,
                                 double value, int strategy,
                                 int *match_indices)
{
#if defined(__x86_64__) || defined(_M_X64)
    if (vjoin_simd_caps.has_avx2)
        return compare_float8_theta_avx2(block, block_count, value, strategy,
                                         match_indices);
    if (vjoin_simd_caps.has_sse2)
        return compare_float8_theta_sse2(block, block_count, value, strategy,
                                         match_indices);
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (vjoin_simd_caps.has_neon)
        return compare_float8_theta_neon(block, block_count, value, strategy,
                                         match_indices);
#endif
    return compare_float8_theta_scalar(block, block_count, value, strategy,
                                       match_indices);
}

/* ================================================================
 * Batch hash functions
 * ================================================================ */

void
vjoin_hash_int4_batch(const int32 *values, int count, uint32 *hashes)
{
    int i;
    for (i = 0; i < count; i++)
    {
        uint32 h = (uint32) values[i];
        h ^= h >> 16;
        h *= 0x45d9f3b;
        h ^= h >> 16;
        h *= 0x45d9f3b;
        h ^= h >> 16;
        hashes[i] = h;
    }
}

void
vjoin_hash_int8_batch(const int64 *values, int count, uint32 *hashes)
{
    int i;
    for (i = 0; i < count; i++)
    {
        uint64 v = (uint64) values[i];
        uint32 h = (uint32)(v ^ (v >> 32));
        h ^= h >> 16;
        h *= 0x45d9f3b;
        h ^= h >> 16;
        h *= 0x45d9f3b;
        h ^= h >> 16;
        hashes[i] = h;
    }
}
