/*******************************************************************************
* Copyright 2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "include/batch_headers/generic_vector_ops.cl"
#include "include/batch_headers/sdpa_utils.cl"
#include "include/batch_headers/tile_ops.cl"

/* The quantization parameter may be unique for each token/element */
#define QUANTIZE_2D 2

/* The quantization parameter shares the same value across the work-group */
#define QUANTIZE_COMMON 3

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define DIV_UP(x, y) (((x) + (y)-1) / (y))

#define sg_per_wg (ugemm_kq_sg_per_wg_m * ugemm_kq_sg_per_wg_n)
#define q_tile_sg_n DIV_UP(ugemm_kq_wg_tile_n, sg_per_wg)

/* Instantiate tile types and operations */
typedef ugemm_kq_c_type s_tile_type;
typedef ugemm_vs_c_type a_tile_type;



#define DECLARE_2D_MASK_FILL(tile_type, element_type, sg, br, bc, nbr, nbc) \
__attribute__((overloadable)) void fill_mask_t(tile_type *t, \
        int m, int n, int ld, \
        int offset_r, int offset_c) { \
    if (offset_c + bc * nbc - 1 <= offset_r) { \
        tile_fill((*t), 0); \
        return; \
    } else if (offset_r + br * nbr <= offset_c) { \
        tile_fill((*t), -HALF_MAX); \
        return; \
    } \
    _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
        int i = i0 + get_sub_group_local_id(); \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            tile_access(*t, i0, j, sg, br, bc, nbr) = (offset_c + j > offset_r + i) ? -HALF_MAX : 0.0f; \
        } \
    } \
}

DECLARE_2D_TILE(q_tile_type, uint, SUBGROUP_SIZE, D_MAX / 2, 1, 1, q_tile_sg_n)

#ifdef BLOCK_Q
DECLARE_2D_TILE_BLOCK_OPS(
        q_tile_type, uint, SUBGROUP_SIZE, D_MAX / 2, 1, 1, q_tile_sg_n)
#elif Q_ALIGN < 4
DECLARE_2D_TILE_LOAD_PACKED_HALF(
        q_tile_type, SUBGROUP_SIZE, D_MAX / 2, 1, 1, q_tile_sg_n)
#endif

#ifdef BLOCK_A
DECLARE_2D_TILE(a_tile_type_half, half, SUBGROUP_SIZE, ugemm_vs_sg_tile_m, 1, 1,
        ugemm_vs_sg_tile_n)
#else
DECLARE_2D_TILE(a_tile_type_half, half, SUBGROUP_SIZE, ugemm_vs_sg_tile_m, 8, 1,
        ugemm_vs_sg_tile_n / 8)
#endif

DECLARE_2D_TILE(s_tile_type_half2, uint, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1 / 2, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1)

DECLARE_2D_TILE(
        s_sum_tile_type, float, SUBGROUP_SIZE, ugemm_kq_sg_tile_n, 1, 1, 1)

DECLARE_2D_TILE(
        a_scale_tile_type, float, SUBGROUP_SIZE, ugemm_vs_sg_tile_n, 1, 1, 1)


DECLARE_2D_TILE(mask_tile_type, half, SUBGROUP_SIZE, ugemm_kq_c_type_block0, ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0, ugemm_kq_c_type_nblock1)
DECLARE_2D_TILE(mask_tile_type_float, float, SUBGROUP_SIZE, ugemm_kq_c_type_block0, ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0, ugemm_kq_c_type_nblock1)
DECLARE_2D_MASK_FILL(mask_tile_type_float, float, SUBGROUP_SIZE, ugemm_kq_c_type_block0, ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0, ugemm_kq_c_type_nblock1)

#ifdef BLOCK_A
DECLARE_2D_TILE_BLOCK_OPS(a_tile_type_half, half, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 1, 1, ugemm_vs_sg_tile_n)
#endif
#ifdef BLOCK_2D_A
DECLARE_2D_TILE_BLOCK2D_OPS(a_tile_type_half, half, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 8, 1, ugemm_vs_sg_tile_n / 8)
#endif

#ifdef BLOCK_A
DECLARE_2D_TILE_COPY_REBLOCK(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_tile_type_half, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 1, 1, ugemm_vs_sg_tile_n)
#else
DECLARE_2D_TILE_COPY_REBLOCK(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_tile_type_half, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 8, 1, ugemm_vs_sg_tile_n / 8)
#endif

DECLARE_2D_TILE_VREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_sum_tile_type, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_n, 1, 1, 1)

DECLARE_2D_TILE_HREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, mask_tile_type_float, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_m, 1, 1, 1)

DECLARE_2D_TILE_HREDUCE(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_scale_tile_type, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, 1, 1)

#if ugemm_kq_wg_tile_n == ugemm_vs_wg_tile_n \
        && (ugemm_kq_sg_tile_n % ugemm_vs_sg_tile_n) == 0
DECLARE_2D_TILE_RSELECT(a_scale_tile_type, SUBGROUP_SIZE, ugemm_vs_sg_tile_n, 1,
        1, 1, s_sum_tile_type, SUBGROUP_SIZE, ugemm_kq_sg_tile_n, 1, 1, 1)
#endif

#if PREFETCH_REMAINDER
#define cooperative_prefetch_2d_maybe_rem cooperative_prefetch_2d_rem
#else
#define cooperative_prefetch_2d_maybe_rem( \
        ptr, r, c, rmax, cmax, ld, sg_id, n_sg, sg_size, caching) \
    cooperative_prefetch_2d(ptr, rmax, cmax, ld, sg_id, n_sg, sg_size, caching)
#endif

#if TRANSPOSE_K
#define cooperative_prefetch_2d_k( \
        ptr, r, c, rmax, cmax, ld, sg_id, n_sg, sg_size, caching) \
    cooperative_prefetch_2d_maybe_rem( \
            ptr, c, r, cmax, rmax, ld, sg_id, n_sg, sg_size, caching)
#else
#define cooperative_prefetch_2d_k cooperative_prefetch_2d_maybe_rem
#endif

#if REMAINDER_Q
#define tile_load_block_rem_q tile_load_block
#define tile_store_block_rem_q tile_store_block
#else
#define tile_load_block_rem_q(t, ptr, n, ld, off_r, off_c) \
    tile_load_block(t, ptr, ld, off_r, off_c)
#define tile_store_block_rem_q(t, ptr, n, ld, off_r, off_c) \
    tile_store_block(t, ptr, ld, off_r, off_c)
#endif

#define binary_add(x, y) ((x) + (y))

#define DEBUG_PRINT 0

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(micro_sdpa)(OPTIONAL_SHAPE_INFO_ARG
        const global KEY_DATA_T *K,
        const global QRY_DATA_T *Q,
        const global VAL_DATA_T *V,
        global half *A,
#if IS_PAGED_ATTENTION
    const __global INPUT3_TYPE* subsequence_begins,
#endif
#if WITH_ATTN_MASK
        const global half *msk,
#endif
#if WITH_SCALE
        global SCALE_DATA_T *scale_ptr,
#endif
#if IS_PAGED_ATTENTION
        const __global int* blocked_indexes_start_and_gws_mapping
#else
        int d,
        int k,
        int q
#endif
#ifdef KV_COMPRESSED
        , const global KEY_ATTR_SCALES_DATA_T *K_scales
        , const global KEY_ATTR_ZP_DATA_T *K_zp
        , const global VAL_ATTR_SCALES_DATA_T *V_scales
        , const global VAL_ATTR_ZP_DATA_T *V_zp
#endif
        ) {
#if IS_PAGED_ATTENTION
    const uint q_tile_idx = get_group_id(0) * 2;
    const uint block_start_pos = blocked_indexes_start_and_gws_mapping[q_tile_idx];
    const uint gws_mapping = blocked_indexes_start_and_gws_mapping[q_tile_idx + 1];
    const uint subsequence_begin = subsequence_begins[gws_mapping];
    const uint subsequence_end = subsequence_begins[gws_mapping + 1];
    const uint subsequence_q_tile_idx = block_start_pos - subsequence_begin;
    const int k = subsequence_end - subsequence_begin;
    const int q = k;
    const int d = HEAD_SIZE;
//     content: 0, 0, 16, 0, 32, 0, 48, 0, 64, 0, 80, 0, 96, 0, 112, 0, 128, 0, 144, 0, 160, 0, 176, 0, 192, 0, 208, 0, 224, 0, 240, 0,
//              256, 1, 272, 1, 288, 1, 304, 1,
//              305, 2, 321, 2, 337, 2,
#endif
    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);
    uint b0 = get_group_id(1);
    uint b1 = get_group_id(2);
    uint b0_kv = b0 / KV_GROUP_SIZE;

#if IS_PAGED_ATTENTION
    uint wg_j0 = subsequence_q_tile_idx;
#else
    uint wg_j0 = get_group_id(0) * ugemm_kq_wg_tile_n;
#endif

    /* Leading dimension for matrices */
#if IS_PAGED_ATTENTION
    uint ldk = HEAD_SIZE * KV_HEADS_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM + INPUT1_PAD_AFTER_FEATURE_NUM;
    uint ldq = HEAD_SIZE * HEADS_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    uint ldv = HEAD_SIZE * KV_HEADS_NUM + INPUT2_PAD_BEFORE_FEATURE_NUM + INPUT2_PAD_AFTER_FEATURE_NUM;
    uint lda = HEAD_SIZE * HEADS_NUM;
#else
    uint ldk = TRANSPOSE_K ? KEY_S3 : KEY_S2;
    uint ldq = QRY_S2;
    uint ldv = VAL_S2;
    uint lda = DST_S2;
#endif

#if KEY_SCALES || KEY_ZERO_POINTS
    uint ldkq = DIV_UP(d, KEY_GROUP_SIZE);
#endif
#if VAL_SCALES || VAL_ZERO_POINTS
    uint ldvq = DIV_UP(d, VAL_GROUP_SIZE);
#endif

    /* Subgroup IDs for each GEMM */
    uint sg_i_kq = sg_ij % ugemm_kq_sg_per_wg_m;
    uint sg_j_kq = sg_ij / ugemm_kq_sg_per_wg_m;

    uint sg_i_vs = sg_ij % ugemm_vs_sg_per_wg_m;
    uint sg_j_vs = sg_ij / ugemm_vs_sg_per_wg_m;

    /* SLM allocations -- place in one array to work around compiler bug */
#define Q_slm_size (D_MAX * ugemm_kq_wg_tile_n * sizeof(half))
#define S_slm_size (ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n * sizeof(half))
#define S_sum_slm_size \
    (ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m * sizeof(float))
#define S_max_slm_size (ugemm_kq_wg_tile_n * sizeof(float))
#define ugemm_slm_size MAX(ugemm_kq_slm_size, ugemm_vs_slm_size)

    local char slm[Q_slm_size + S_slm_size + S_sum_slm_size + S_max_slm_size
            + ugemm_slm_size];

    local half *Q_slm = (local half *)&slm[0];
    local half *S_slm = (local half *)&slm[Q_slm_size];
    local float *S_sum_slm = (local float *)&slm[Q_slm_size + S_slm_size];
    local float *S_max_slm
            = (local float *)&slm[Q_slm_size + S_slm_size + S_sum_slm_size];
    local uint *ugemm_slm = (local uint *)&slm[Q_slm_size + S_slm_size
            + S_sum_slm_size + S_max_slm_size];

    const bool need_sum_barrier = (ugemm_vs_barrier_count == 0);

    /* Locate K/Q/V/A matrices within batch */
#if IS_PAGED_ATTENTION
    K += subsequence_begin * (HEAD_SIZE * KV_HEADS_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM + INPUT1_PAD_AFTER_FEATURE_NUM)
       + b0_kv * HEAD_SIZE + INPUT1_PAD_BEFORE_FEATURE_NUM;
    Q += subsequence_begin * (HEAD_SIZE * HEADS_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM)
       + b0 * HEAD_SIZE + INPUT0_PAD_BEFORE_FEATURE_NUM;
    V += subsequence_begin * (HEAD_SIZE * KV_HEADS_NUM + INPUT2_PAD_BEFORE_FEATURE_NUM + INPUT2_PAD_AFTER_FEATURE_NUM)
       + b0_kv * HEAD_SIZE + INPUT2_PAD_BEFORE_FEATURE_NUM;
    A += subsequence_begin * (HEAD_SIZE * HEADS_NUM)
       + b0 * HEAD_SIZE;
#else
    K += (KEY_OFF(b1, b0_kv, 0, 0) + INPUT1_OFFSET) / KEY_ELEMENTS_PER_BYTE;
    Q += (QRY_OFF(b1, b0, 0, 0) + INPUT0_OFFSET);
    V += (VAL_OFF(b1, b0_kv, 0, 0) + INPUT2_OFFSET) / VAL_ELEMENTS_PER_BYTE;
    A += DST_OFF(b1, b0, 0, 0, 0);
#endif

#if KEY_SCALES
    K_scales += KEY_COMP_OFF(b1, b0_kv, 0, 0);
#endif
#if KEY_SCALES == QUANTIZE_COMMON
    float k_scale = convert_float(*K_scales);
#endif
#if KEY_ZERO_POINTS
    K_zp += KEY_COMP_OFF(b1, b0_kv, 0, 0) / KEY_ZP_ELEMENTS_PER_BYTE;
#endif
#if VAL_SCALES
    V_scales += VAL_COMP_OFF(b1, b0_kv, 0, 0);
#endif
#if VAL_SCALES == QUANTIZE_COMMON
    float v_scale = convert_float(*V_scales);
#endif
#if VAL_ZERO_POINTS
    V_zp += VAL_COMP_OFF(b1, b0_kv, 0, 0) / VAL_ZP_ELEMENTS_PER_BYTE;
#endif

    __builtin_assume_aligned(K, K_ALIGN);
    __builtin_assume_aligned(Q, Q_ALIGN);
    __builtin_assume_aligned(V, V_ALIGN);
    __builtin_assume_aligned(A, A_ALIGN);

    /* Load Q tile, destined for SLM */
    q_tile_type Q_tile;
    uint q0_copy = q_tile_sg_n * sg_ij;
#ifdef BLOCK_Q
    tile_load_block_rem_q(
            &Q_tile, (global uint *)Q, q, ldq >> 1, 0, wg_j0 + q0_copy);
#elif Q_ALIGN >= 4
    tile_load(&Q_tile, (global uint *)Q, (d + 1) >> 1, q, ldq >> 1, 0,
            wg_j0 + q0_copy);
#else
    tile_load_packed_half(&Q_tile, Q, d, q, ldq, 0, wg_j0 + q0_copy);
#endif

#if WITH_SCALE
        /* Load scale */
    #if INVERT_SCALE
        float iscale = convert_float(*scale_ptr);
        float scale = native_recip(iscale);
    #else
        float scale = convert_float(*scale_ptr);
        float iscale = native_recip(scale);
    #endif
#else
    float iscale = sqrt(convert_float(HEAD_SIZE));
    float scale = native_recip(iscale);
#endif
    scale *= 1.442695f; // log2(e)

#ifdef PREFETCH_K0
    /* Prefetch first K tile. */
    cooperative_prefetch_2d_k(K, d, k, ugemm_kq_wg_tile_m, PREFETCH_D_MAX, ldk,
            sg_ij, sg_per_wg, SUBGROUP_SIZE, LSC_LDCC_L1C_L3C);
#endif

    /* Initialize S column sums in SLM to -inf */
    const uint n_col_sg = DIV_UP(ugemm_kq_wg_tile_n, SUBGROUP_SIZE * sg_per_wg);
    const float neg_inf = -INFINITY;

#pragma unroll
    for (int q = 0; q < n_col_sg; q++)
        intel_sub_group_block_write(
                (local uint *)&S_max_slm[(q + sg_ij * n_col_sg)
                        * SUBGROUP_SIZE],
                as_uint(neg_inf));

    /* Clear accumulator */
    a_tile_type A_tile;
    tile_fill(A_tile, 0.0f);

    /* Store Q tile to SLM */
    tile_store_t_sys_src1(
            Q_tile, (local uint *)&Q_slm[0], D_MAX / 2, q0_copy, 0);

    /* Clear S column sums/maxes */
    s_sum_tile_type S_sum_tile;
    s_sum_tile_type S_max_tile, S_max_tile_old;
    tile_fill(S_sum_tile, 0.0f);
    tile_fill(S_max_tile, -INFINITY);

    /* Wait for Q data to reach SLM */
    barrier(CLK_LOCAL_MEM_FENCE);

#if DEBUG_PRINT
    if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
        printf("Q SLM (slm_size=%d)\n", Q_slm_size / sizeof(half));
        // for (int i = 0; i < D_MAX; i++) {
        // {
        //     int i = 0;
        //     int slm_idx = i * ugemm_kq_wg_tile_n;
        //     printf("%d. %f %f %f %f  %f %f %f %f  %f %f %f %f  %f %f %f %f", i, Q_slm[slm_idx + 0], Q_slm[slm_idx + 1], Q_slm[slm_idx + 2], Q_slm[slm_idx + 3],
        //                                                                         Q_slm[slm_idx + 4], Q_slm[slm_idx + 5], Q_slm[slm_idx + 6], Q_slm[slm_idx + 7],
        //                                                                         Q_slm[slm_idx + 8], Q_slm[slm_idx + 9], Q_slm[slm_idx + 10]), Q_slm[slm_idx + 11],
        //                                                                         Q_slm[slm_idx + 12], Q_slm[slm_idx + 13], Q_slm[slm_idx + 14], Q_slm[slm_idx + 15];
        // }
        // {
        //     int i = 1;
        //     int slm_idx = i * ugemm_kq_wg_tile_n;
        //     printf("i=1. %f %f %f %f  %f %f %f %f  %f %f %f %f  %f %f %f %f\n", Q_slm[slm_idx + 0], Q_slm[slm_idx + 1], Q_slm[slm_idx + 2], Q_slm[slm_idx + 3],
        //                                                                         Q_slm[slm_idx + 4], Q_slm[slm_idx + 5], Q_slm[slm_idx + 6], Q_slm[slm_idx + 7],
        //                                                                         Q_slm[slm_idx + 8], Q_slm[slm_idx + 9], Q_slm[slm_idx + 10]), Q_slm[slm_idx + 11],
        //                                                                         Q_slm[slm_idx + 12], Q_slm[slm_idx + 13], Q_slm[slm_idx + 14], Q_slm[slm_idx + 15];
        // }
        // {
        //     int i = 2;
        //     int slm_idx = i * ugemm_kq_wg_tile_n;
        //     printf("i=2. %f %f %f %f  %f %f %f %f  %f %f %f %f  %f %f %f %f\n", Q_slm[slm_idx + 0], Q_slm[slm_idx + 1], Q_slm[slm_idx + 2], Q_slm[slm_idx + 3],
        //                                                                         Q_slm[slm_idx + 4], Q_slm[slm_idx + 5], Q_slm[slm_idx + 6], Q_slm[slm_idx + 7],
        //                                                                         Q_slm[slm_idx + 8], Q_slm[slm_idx + 9], Q_slm[slm_idx + 10]), Q_slm[slm_idx + 11],
        //                                                                         Q_slm[slm_idx + 12], Q_slm[slm_idx + 13], Q_slm[slm_idx + 14], Q_slm[slm_idx + 15];
        // }

        for (int i = 0; i < D_MAX * ugemm_kq_wg_tile_n; i++) {
            printf("%d. %f\n", i, Q_slm[i]);
        }
    }
#endif

    /* Main loop over k blocks */
    for (int k0 = 0; k0 < k; k0 += ugemm_kq_wg_tile_m) {
        bool first = (k0 == 0);
        bool last = (k0 + ugemm_kq_wg_tile_m >= k);

        uint sg_i0_kq = sg_i_kq * ugemm_kq_sg_tile_m;
        uint sg_j0_kq = sg_j_kq * ugemm_kq_sg_tile_n;

#if WITH_ATTN_MASK
        mask_tile_type mask_tile;
        tile_load_t(&mask_tile, msk, q, k, q, sg_j0_kq + wg_j0, k0 + sg_i0_kq);
#endif

#if REMAINDER_K
        /* Prepare k mask: NaN in bounds, -inf out of bounds */
        mask_tile_type_float k_mask;
#pragma unroll
        for (int ii = 0; ii < ugemm_kq_sg_tile_m / SUBGROUP_SIZE; ii++)
            k_mask.x[0][ii] = (k0 + sg_i0_kq + ii * SUBGROUP_SIZE
                                              + get_sub_group_local_id()
                                      < k)
                    ? nan(0u)
                    : -INFINITY;
#endif

        /* Calculate S = (K^T) * Q */
        s_tile_type S_tile
                = ugemm_kq(K, ldk, Q_slm, D_MAX, k, ugemm_kq_wg_tile_n, d, k0,
                        0, 0, sg_i_kq, sg_j_kq, (local char *)ugemm_slm
#if KEY_SCALES == QUANTIZE_2D
                        ,
                        K_scales
#endif
#if KEY_ZERO_POINTS
                        ,
                        K_zp
#endif
#if (KEY_SCALES == QUANTIZE_2D) || KEY_ZERO_POINTS
                        ,
                        ldkq
#endif
                );

#if KEY_SCALES == QUANTIZE_COMMON
#define k_scale_op(x) ((x)*k_scale)
        tile_elementwise(S_tile, k_scale_op);
#endif

#if DEBUG_PRINT
        if (b0 == 0 && b1 == 0 && k0 == 0 && sg_i_kq == 0 && sg_j_kq == 0 && get_sub_group_local_id() == 0) {
            printf("S_tile before:\n");
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++)
                    printf("\t%f", xlane_tile_access(S_tile, i, j, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
                                                     ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0));
                printf("\n");
            }
        }
#endif

        /* Apply attention mask */
#if WITH_ATTN_MASK
#define unscale(x) ((x)*iscale)
        mask_tile_type_float mask_tile_float;
        tile_copy(mask_tile, mask_tile_float);
        tile_elementwise(mask_tile_float, unscale);
        tile_binary(S_tile, mask_tile_float, binary_add);
#endif

#if IS_PAGED_ATTENTION
#define unscale(x) ((x)*iscale)
        mask_tile_type_float mask_tile_float;
        // tile_fill(mask_tile_float, -66.0f);
#if DEBUG_PRINT
        if (b0 == 0 && b1 == 0 && k0 == 0 && sg_i_kq == 0 && sg_j_kq == 0 && get_sub_group_local_id() == 0) {
            printf("mask after init:\n");
            printf("fill_mask_t(q=%d, k=%d, col_r=%d, col_c=%d; br=%d bc=%d nbr=%d nbc=%d)\n", q, k, sg_j0_kq + wg_j0, k0 + sg_i0_kq, ugemm_kq_c_type_block0, ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0, ugemm_kq_c_type_nblock1);
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++)
                    printf("\t%f", xlane_tile_access(mask_tile_float, i, j, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
                                                     ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0));
                printf("\n");
            }
        }
#endif

        fill_mask_t(&mask_tile_float, q, k, q, sg_j0_kq + wg_j0, k0 + sg_i0_kq);

#if DEBUG_PRINT
        if (b0 == 0 && b1 == 0 && k0 == 0 && sg_i_kq == 0 && sg_j_kq == 0 && (get_sub_group_local_id() < 4)) {
            printf("mask values for %d sglid: %f %f %f %f  %f %f %f %f\n", get_sub_group_local_id(), mask_tile_float.x[0].s0, mask_tile_float.x[0].s1,
            mask_tile_float.x[0].s2, mask_tile_float.x[0].s3,
            mask_tile_float.x[0].s4, mask_tile_float.x[0].s5,
            mask_tile_float.x[0].s6, mask_tile_float.x[0].s7);
        }
#endif

#if DEBUG_PRINT
        if (b0 == 0 && b1 == 0 && k0 == 0 && sg_i_kq == 0 && sg_j_kq == 0 && get_sub_group_local_id() == 0) {
            printf("updated mask before scale:\n");
            printf("fill_mask_t(q=%d, k=%d, col_r=%d, col_c=%d; br=%d bc=%d nbr=%d nbc=%d)\n", q, k, sg_j0_kq + wg_j0, k0 + sg_i0_kq, ugemm_kq_c_type_block0, ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0, ugemm_kq_c_type_nblock1);
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++)
                    printf("\t%f", xlane_tile_access(mask_tile_float, i, j, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
                                                     ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0));
                printf("\n");
            }
        }
#endif

        tile_elementwise(mask_tile_float, unscale);
        tile_binary(S_tile, mask_tile_float, binary_add);
#endif

#if DEBUG_PRINT
        if (b0 == 0 && b1 == 0 && k0 == 0 && sg_i_kq == 0 && sg_j_kq == 0 && get_sub_group_local_id() == 0) {
            printf("mask after scale:\n");
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++)
                    printf("\t%f", xlane_tile_access(mask_tile_float, i, j, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
                                                     ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0));
                printf("\n");
            }

            printf("S_tile after:\n");
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++)
                    printf("\t%f", xlane_tile_access(S_tile, i, j, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
                                                     ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0));
                printf("\n");
            }
        }
#endif

        /* Apply k mask */
#if REMAINDER_K
        tile_hbroadcast_min(&S_tile, k_mask);
#endif

        /* Before softmax, we will need to scale columns by maximum values to avoid overflow. */

        /* Compute our maxima and reduce across SLM */
        tile_vreduce_max(S_tile, &S_max_tile);
        tile_atomic_max_full(
                S_max_tile, S_max_slm, ugemm_kq_wg_tile_n, sg_j0_kq, 0);
        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

#ifdef PREFETCH_V
        /* Prefetch V tile. */
        cooperative_prefetch_2d_maybe_rem(V, d, k - k0, D_MAX,
                (ugemm_kq_wg_tile_m * PREFETCH_D_MAX) / D_MAX, ldv, sg_ij,
                sg_per_wg, SUBGROUP_SIZE, LSC_LDCC_L1C_L3C);
#endif

#ifndef ALT_MAX
        /* Read back WG-wide maxima */
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
        tile_load_full(&S_max_tile, S_max_slm, ugemm_kq_wg_tile_n, sg_j0_kq, 0);
#endif

        tile_vbroadcast_sub(&S_tile, S_max_tile);

/* Scale + exponentiate */
#define scaled_exp(x) native_vexp2(x *scale)
        tile_elementwise(S_tile, scaled_exp);

#ifdef ALT_MAX
        /* Read back WG-wide maxima and adjust S to match */
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
        s_sum_tile_type S_max_tile1;
        tile_copy(S_max_tile, S_max_tile1);
        tile_load_full(&S_max_tile, S_max_slm, ugemm_kq_wg_tile_n, sg_j0_kq, 0);

#define binary_exp_neg(x, y) native_vexp2(scale *((x) - (y)))
        tile_binary(S_max_tile1, S_max_tile, binary_exp_neg);
        tile_vbroadcast_mul(&S_tile, S_max_tile1);
#endif

        /* Accumulate sums. S tile is transposed for easy summation. */
        s_sum_tile_type S_sum_tile1;
        tile_fill(S_sum_tile1, 0.0f);
        tile_vreduce_add(S_tile, &S_sum_tile1);

        /* Convert to half, VNNI format */
        s_tile_type_half2 S_tile_half2;
        tile_copy_to_half2(S_tile, S_tile_half2);

        /* Store to SLM, in packed format */
        tile_store_t_sys_src2(S_tile_half2, (local uint *)S_slm,
                ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_m / 2, sg_i0_kq / 2,
                sg_j0_kq);
        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

        /* Rescale existing accumulator and sums to match new maxima */
        if (!first) {
#define binary_exp_sub(x, y) native_vexp2(scale *((x) - (y)))
#define binary_mul(x, y) ((x) * (y))
            tile_binary(S_max_tile_old, S_max_tile, binary_exp_sub);
            tile_binary(S_sum_tile, S_max_tile_old, binary_mul);

            /* Find the subset of sums that applies to the accumulation tile */
            a_scale_tile_type A_scale_tile;
#if ugemm_kq_wg_tile_n == ugemm_vs_wg_tile_n \
        && ugemm_kq_sg_tile_n == ugemm_vs_sg_tile_n
            tile_copy(S_max_tile_old, A_scale_tile);
#elif ugemm_kq_wg_tile_n == ugemm_vs_wg_tile_n \
        && (ugemm_kq_sg_tile_n % ugemm_vs_sg_tile_n) == 0
            tile_rselect(&A_scale_tile, S_max_tile_old,
                    sg_j_vs % (ugemm_kq_sg_tile_n / ugemm_vs_sg_tile_n));
#else
#error unimplemented
#endif
            tile_hbroadcast_mul(&A_tile, A_scale_tile);
        }

/* Accumulate sums */
        tile_binary(S_sum_tile, S_sum_tile1, binary_add);

        /* Save maxima */
        tile_copy(S_max_tile, S_max_tile_old);

        /* Last iteration: store column sums in SLM */
        if (last) {
            tile_store_full(S_sum_tile, S_sum_slm, ugemm_kq_wg_tile_n, sg_j0_kq,
                    sg_i_kq);
        }

#ifdef PREFETCH_K
        /* Prefetch next K tile. */
        if (!last) {
#if TRANSPOSE_K
            const uint stride_k = ldk;
#else
            const uint stride_k = 1;
#endif
            cooperative_prefetch_2d_k(K + (k0 + ugemm_kq_wg_tile_m) * stride_k,
                    k - k0 - ugemm_kq_wg_tile_m, d, ugemm_kq_wg_tile_m,
                    PREFETCH_D_MAX, ldk, sg_ij, sg_per_wg, SUBGROUP_SIZE,
                    LSC_LDCC_L1C_L3C);
        }
#endif
#if WITH_ATTN_MASK && defined(PREFETCH_MASK)
        /* Prefetch next mask tile. */
        if (!last) {
            cooperative_prefetch_2d(msk + k0 + ugemm_kq_wg_tile_m + sg_i0_kq + (sg_j0_kq + wg_j0) * q,
                    ugemm_kq_sg_tile_m, ugemm_kq_sg_tile_n, 0, 0, 1, SUBGROUP_SIZE,
                    LSC_LDCC_L1UC_L3C);
        }
#endif

        /* Wait for S stores */
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

        /* Last iteration: signal column sums are ready */
        if (last && need_sum_barrier)
            intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

        /* Accumulate A += V * S */
        int k_chunk = min(k - k0, ugemm_kq_wg_tile_m);

        a_tile_type A_tile1 = ugemm_vs(
                V, ldv, S_slm, ugemm_kq_wg_tile_m, d, ugemm_kq_wg_tile_n,
                k_chunk, 0, 0, 0, sg_i_vs, sg_j_vs, (local char *)ugemm_slm
#if VAL_SCALES == QUANTIZE_2D
                ,
                V_scales
#endif
#if VAL_ZERO_POINTS
                ,
                V_zp
#endif
#if (VAL_SCALES == QUANTIZE_2D) || VAL_ZERO_POINTS
                ,
                ldvq
#endif
        );

        V += ldv * ugemm_kq_wg_tile_m / VAL_ELEMENTS_PER_BYTE;
#if VAL_SCALES == QUANTIZE_2D
        V_scales += ldvq * ugemm_kq_wg_tile_m;
#endif
#if VAL_ZERO_POINTS == QUANTIZE_2D
        V_zp += ldvq * ugemm_kq_wg_tile_m / VAL_ZP_ELEMENTS_PER_BYTE;
#endif
        tile_binary(A_tile, A_tile1, binary_add);
    }

    /* Wait for column sums to be ready */
    if (need_sum_barrier) intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

    /* Load column sums from SLM + reduce in registers */
    a_scale_tile_type A_scale_tile, A_scale_tile_load;
    tile_fill(A_scale_tile, 0.0f);

#pragma unroll
    for (uint sg1 = 0; sg1 < ugemm_kq_sg_per_wg_m; sg1++) {
        tile_load_full(&A_scale_tile_load, S_sum_slm, ugemm_kq_wg_tile_n,
                ugemm_vs_sg_tile_n * sg_j_vs, sg1);
        tile_binary(A_scale_tile, A_scale_tile_load, binary_add);
    }

#if VAL_SCALES == QUANTIZE_COMMON
#define v_scale_op(x) ((x)*v_scale)
    tile_elementwise(A_tile, v_scale_op);
#endif

    /* Rescale by 1 / (column sums) */
    tile_elementwise(A_scale_tile, native_vrecip);
    tile_hbroadcast_mul(&A_tile, A_scale_tile);

    /* Convert to half precision and store */
    a_tile_type_half A_tile_half;
    tile_copy_reblock(A_tile, &A_tile_half);

    uint sg_i0_vs = sg_i_vs * ugemm_vs_sg_tile_m;
    uint sg_j0_vs = sg_j_vs * ugemm_vs_sg_tile_n + wg_j0;

#ifdef BLOCK_2D_A
    tile_store_block2d(A_tile_half, A, d, q, lda, sg_i0_vs, sg_j0_vs);
#elif defined(BLOCK_A)
    tile_store_block_rem_q(A_tile_half, A, q, lda, sg_i0_vs, sg_j0_vs);
#else
    tile_store(A_tile_half, A, d, q, lda, sg_i0_vs, sg_j0_vs);
#endif
}
