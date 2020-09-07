// Copyright (C) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/common.cl"
#include "include/data_types.cl"
#include "include/include_all.cl"

inline uint FUNC(get_input_index)(uint b, uint f, uint z, uint y, uint x)
{
#if INPUT0_DIMS < 5
    return INPUT0_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
    return INPUT0_GET_INDEX(b, f, z, y, x);
#else
#error [clDNN resample_ref.cl]: input format - not supported
#endif
}

inline uint FUNC(get_output_index)(uint b, uint f, uint z, uint y, uint x)
{
#if OUTPUT_DIMS < 5
    return OUTPUT_GET_INDEX(b, f, y, x);
#elif OUTPUT_DIMS == 5
    return OUTPUT_GET_INDEX(b, f, z, y, x);
#else
#error [clDNN resample_ref.cl]: output format - not supported
#endif
}

inline int FUNC(get_nearest_val)(float num, bool is_downsample)
{
#if defined(NEAREST_ROUND_PREFER_FLOOR)
    return (num == (int)num + 0.5f) ? (int)floor(num) : (int)round(num);
#elif defined(NEAREST_ROUND_PREFER_CEIL)
    return (int)round(num);
#elif defined(NEAREST_FLOOR)
    return (int)floor(num);
#elif defined(NEAREST_CEIL)
    return (int)ceil(num);
#elif defined(NEAREST_SIMPLE)
    return is_downsample ? (int)ceil(num) : (int)num;
#else
#error [clDNN resample_ref.cl]: nearest mode - not supported
#endif
}

// 2.5 * (7/10) - 0.5 = 1.5
// 2.5 / (10/7) - 0.5 = 
// const ACCUMULATOR_TYPE ix = FUNC_CALL(get_original_coordinate)(ox, X_RATIO, OUTPUT_SIZE_X, INPUT0_SIZE_X);
inline float FUNC(get_original_coordinate)(float num, float scale, int length_resized, int length_original)
{
#if defined(COORD_TRANS_MODE_HALF_PIXEL)
    return (num + 0.5f) * scale - 0.5f;
#elif defined(COORD_TRANS_MODE_PYTORCH_HALF_PIXEL)
    return (length_resized > 1) ? (num + 0.5f) * scale - 0.5f : 0.f;
#elif defined(COORD_TRANS_MODE_ASYMMETRIC)
    return num * scale;
#elif defined(COORD_TRANS_MODE_TF_HALF_PIXEL_FOR_NN)
    return (num + 0.5f) * scale;
#elif defined(COORD_TRANS_MODE_ALIGN_CORNERS)
    return (length_resized != 1) ? num * ((float)(length_original - 1) / (float)(length_resized - 1)) : 0.f;
#else
#error [clDNN resample_ref.cl]: coordinate transformation mode - not supported
#endif
}

inline void FUNC(get_cubic_coeff)(float* cubic_coef, float coord, float coef)
{
    float abs_num = fabs(coord);
    cubic_coef[0] = coef * (abs_num - 1.0) * (abs_num - 1.0) * abs_num;
    cubic_coef[1] = ((coef + 2.0) * abs_num - (coef + 3.0)) * abs_num * abs_num + 1.0;
    cubic_coef[2] = (((-coef - 2.0) * abs_num + (2.0 * coef + 3.0)) * abs_num - coef) * abs_num;
    cubic_coef[3] = -coef * abs_num * abs_num * (abs_num - 1.0);
}

#define TRIANGLE_COEFF(x) (ACCUMULATOR_MAX_FUNC(ACCUMULATOR_VAL_ZERO, ACCUMULATOR_VAL_ONE - ACCUMULATOR_ABS_FUNC(x)))
#define unroll_for __attribute__((opencl_unroll_hint)) for

KERNEL (resample_gpu_ref)(__global INPUT0_TYPE* input,
                          __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                          , FUSED_OPS_DECLS
#endif
)
{
    const float scale[5] = { B_RATIO, F_RATIO, Z_RATIO, Y_RATIO, X_RATIO };
    const int in_size[5] = { INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X };
    const int out_size[5] = { OUTPUT_BATCH_NUM, OUTPUT_FEATURE_NUM, OUTPUT_SIZE_Z, OUTPUT_SIZE_Y, OUTPUT_SIZE_X };
#if defined(SAMPLE_TYPE_NEAREST) && FEATURE_PACKED_MODE
    typedef MAKE_VECTOR_TYPE(INPUT0_TYPE, PACK_SIZE) in_pack_t;
    typedef MAKE_VECTOR_TYPE(OUTPUT_TYPE, PACK_SIZE) out_pack_t;

    int out_coords[5];
    out_coords[4] = get_global_id(0);
#if OUTPUT_DIMS <= 4
    out_coords[3] = get_global_id(1);
    out_coords[2] = 0;
#else // OUTPUT_DIMS <= 4
    out_coords[3] = (int)get_global_id(1) % OUTPUT_SIZE_Y;
    out_coords[2] = (int)get_global_id(1) / OUTPUT_SIZE_Y;
#endif //  OUTPUT_DIMS <= 4
    out_coords[1] = ((int)get_global_id(2) * PACK_SIZE) % OUTPUT_FEATURE_NUM;
    out_coords[0] = ((int)get_global_id(2) * PACK_SIZE) / OUTPUT_FEATURE_NUM;
    int in_coords[5];
    unroll_for (int i = 0; i < 5; ++i) {
        const float orig_coord = FUNC_CALL(get_original_coordinate)(out_coords[i], scale[i], out_size[i], in_size[i]);
        const int nearest_pixel = FUNC_CALL(get_nearest_val)(orig_coord, scale[i] > 1);
        in_coords[i] = max(0, min(nearest_pixel, in_size[i] - 1));
    }

    uint input_idx = FUNC_CALL(get_input_index)(in_coords[0], in_coords[1], in_coords[2], in_coords[3], in_coords[4]);
    uint output_idx = FUNC_CALL(get_output_index)(out_coords[0], out_coords[1], out_coords[2], out_coords[3], out_coords[4]);

    in_pack_t interp_val_pack = ((const __global in_pack_t*)(input + input_idx))[0];
    out_pack_t res;
    unroll_for (uint pi = 0; pi < PACK_SIZE; ++pi) {
        INPUT0_TYPE interp_val = interp_val_pack[pi];
    #if HAS_FUSED_OPS
        #define OF_ID (out_coords[1] + pi)
        FUSED_OPS;
        res[pi] = FUSED_OPS_RESULT;
    #else // HAS_FUSED_OPS
        res[pi] = ACTIVATION(interp_val, ACTIVATION_PARAMS);
    #endif // HAS_FUSED_OPS
    }
    ((__global out_pack_t*)(output + output_idx))[0] = res;

#elif defined(SAMPLE_TYPE_NEAREST) // defined(SAMPLE_TYPE_NEAREST) && FEATURE_PACKED_MODE
    int out_coords[5];
    out_coords[4] = get_global_id(0);
#if OUTPUT_DIMS <= 4
    out_coords[3] = get_global_id(1);
    out_coords[2] = 0;
#else // OUTPUT_DIMS <= 4
    out_coords[3] = (int)get_global_id(1) % OUTPUT_SIZE_Y;
    out_coords[2] = (int)get_global_id(1) / OUTPUT_SIZE_Y;
#endif // OUTPUT_DIMS <= 4
    out_coords[1] = (int)get_global_id(2) % OUTPUT_FEATURE_NUM;
    out_coords[0] = (int)get_global_id(2) / OUTPUT_FEATURE_NUM;
    int in_coords[5];
    bool zero_val = false;
    int nearest_coords[5];
    float orig_coords[5];
    unroll_for (int i = 0; i < 5; ++i) {
        const float orig_coord = FUNC_CALL(get_original_coordinate)(out_coords[i], scale[i], out_size[i], in_size[i]) -PADS_BEGIN[i];
        orig_coords[i] = orig_coord;
        const int nearest_pixel = FUNC_CALL(get_nearest_val)(orig_coord, scale[i] >= 1);
        nearest_coords[i] = nearest_pixel;
        in_coords[i] = max(-PADS_BEGIN[i], min(nearest_pixel, in_size[i] + PADS_END[i] - 1));
        if (in_coords[i] < 0 || in_coords[i] >= in_size[i])
            zero_val |= true;
    }

    if (out_coords[4] < 3 && out_coords[3] == 0 && out_coords[2] == 0 && out_coords[1] == 0 && out_coords[0] == 0)
        printf("Out: %d %d In: %d %d %d %d %d Orig_coord: %f %f %f %f %f Nearest %d %d %d %d %d Scales: %f %f %f %f %f\n", out_coords[3], out_coords[4], 
        in_coords[0], in_coords[1], in_coords[2], in_coords[3], in_coords[4],
        orig_coords[0], orig_coords[1], orig_coords[2], orig_coords[3], orig_coords[4],
        nearest_coords[0], nearest_coords[1], nearest_coords[2], nearest_coords[3], nearest_coords[4],
        scale[0], scale[1], scale[2], scale[3], scale[4]);

// Nearest [0 1 1 1] [0.5 0.5 0.3 0.3] [1. 1. 1. 1.] [1.         1.         1.66666667 1.66666667]

// #elif defined(NEAREST_ROUND_PREFER_CEIL) return (int)round(num);

    INPUT0_TYPE interp_val = INPUT0_VAL_ZERO;
    if (!zero_val)
        interp_val = input[FUNC_CALL(get_input_index)(in_coords[0], in_coords[1], in_coords[2], in_coords[3], in_coords[4])];

#if HAS_FUSED_OPS
    #define OF_ID (out_coords[1])
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;
#else // HAS_FUSED_OPS
    OUTPUT_TYPE res = ACTIVATION(interp_val, ACTIVATION_PARAMS);
#endif // HAS_FUSED_OPS
    output[FUNC_CALL(get_output_index)(out_coords[0], out_coords[1], out_coords[2], out_coords[3], out_coords[4])] = res;
#elif defined(SAMPLE_TYPE_CUBIC) // defined(SAMPLE_TYPE_NEAREST) && FEATURE_PACKED_MODE
    int out_coords[5]; // BFZYX
    out_coords[4] = get_global_id(0);
#if OUTPUT_DIMS <= 4
    out_coords[3] = get_global_id(1);
    out_coords[2] = 0;
#else // OUTPUT_DIMS <= 4
    out_coords[3] = (int)get_global_id(1) % OUTPUT_SIZE_Y;
    out_coords[2] = (int)get_global_id(1) / OUTPUT_SIZE_Y;
#endif // OUTPUT_DIMS <= 4
    out_coords[1] = (int)get_global_id(2) % OUTPUT_FEATURE_NUM;
    out_coords[0] = (int)get_global_id(2) / OUTPUT_FEATURE_NUM;
    int in_coords[5];
    float orig_coords[5];
    float cubic_coeff[5][CUBIC_COEF_COUNT];
    unroll_for (int i = 0; i < 5; ++i) {
        float orig_coord = FUNC_CALL(get_original_coordinate)(out_coords[i], scale[i], out_size[i], PADS_BEGIN[i] + in_size[i] + PADS_END[i]) - PADS_BEGIN[i];
        orig_coords[i] = orig_coord;
        in_coords[i] = floor(orig_coord);
        FUNC_CALL(get_cubic_coeff)(cubic_coeff[i], orig_coord - in_coords[i], CUBE_COEFF);
    }

    if (out_coords[4] == 0 && out_coords[3] == 0 && out_coords[1] == 1 && out_coords[0] == 0)
        printf("%d %d. orig: %f %f %f %f %f, in_coords: %d %d %d %d %d, cubic: %f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n \
               YX_END %d %d F_END %d\n", out_coords[3], out_coords[4], 
        orig_coords[0], orig_coords[1], orig_coords[2], orig_coords[3], orig_coords[4],
        in_coords[0], in_coords[1], in_coords[2], in_coords[3], in_coords[4],
        cubic_coeff[0][0], cubic_coeff[0][1], cubic_coeff[0][2], cubic_coeff[0][3],
        cubic_coeff[1][0], cubic_coeff[1][1], cubic_coeff[1][2], cubic_coeff[1][3],
        cubic_coeff[2][0], cubic_coeff[2][1], cubic_coeff[2][2], cubic_coeff[2][3],
        cubic_coeff[3][0], cubic_coeff[3][1], cubic_coeff[3][2], cubic_coeff[3][3],
        cubic_coeff[4][0], cubic_coeff[4][1], cubic_coeff[4][2], cubic_coeff[4][3], INDICES_Y_END, INDICES_X_END, INDICES_F_END);

    INPUT0_TYPE interp_val = INPUT0_VAL_ZERO;
    int index[5];
    unroll_for (index[0] = INDICES_B_START; index[0] < INDICES_B_END; ++index[0]) {
        unroll_for (index[1] = INDICES_F_START; index[1] < INDICES_F_END; ++index[1]) {
            unroll_for (index[2] = INDICES_Z_START; index[2] < INDICES_Z_END; ++index[2]) {
                unroll_for (index[3] = INDICES_Y_START; index[3] < INDICES_Y_END; ++index[3]) {
                    unroll_for (index[4] = INDICES_X_START; index[4] < INDICES_X_END; ++index[4]) {
                        int coords_sum[5] =  { in_coords[0], in_coords[1], in_coords[2], in_coords[3], in_coords[4] };
                        float coeff_prod = 1.0f;
                        bool zero_val = false;
                        unroll_for (int i = 0; i < 5; ++i) {
                            int should_scale = (scale[i] != 1);
                            coords_sum[i] = max(-PADS_BEGIN[i], min(in_coords[i] + index[i] - should_scale, in_size[i] - 1 + PADS_END[i]));
                            float coeff = (scale[i] == 1) ? 1.f : cubic_coeff[i][index[i]];
                            coeff_prod *= coeff;
                            if (coords_sum[i] < 0 || coords_sum[i] >= in_size[i])
                                zero_val |= true;
                        }
                        INPUT0_TYPE in = INPUT0_VAL_ZERO;
                        if (!zero_val)
                            in = input[FUNC_CALL(get_input_index)(coords_sum[0], coords_sum[1], coords_sum[2], coords_sum[3], coords_sum[4])];
                        // if (out_coords[4] < 3 && out_coords[3] < 3 && out_coords[1] == 0 && out_coords[0] == 0 ||
                        //     out_coords[4] == 7  && out_coords[3] == 1 && out_coords[1] == 0 && out_coords[0] == 0)
                        //     printf("%d %d. %d %d %d: %f\n",  out_coords[3], out_coords[4], index[1], index[3], index[4], coeff_prod);
                        interp_val += coeff_prod * in;
                        if (out_coords[4] == 0 && out_coords[3] == 0 && out_coords[1] == 1 && out_coords[0] == 0)
                            printf("%d %d %d %d %d: coeff_prod %f * %f (%d %d %d %d %d) = %f\n", index[0], index[1], index[2], index[3], index[4], 
                            coeff_prod, in, coords_sum[0], coords_sum[1], coords_sum[2], coords_sum[3], coords_sum[4], interp_val);
                    }
                }
            }
        }
    }    
    if (out_coords[4] == 0 && out_coords[3] == 0 && out_coords[1] == 1 && out_coords[0] == 0) {
        bool test = isinf(as_float(0x1a));
        printf("Hi %f %d\n", interp_val, test);
    }

// #if HAS_FUSED_OPS
//     #define OF_ID (out_coords[1])
//     FUSED_OPS;
//     OUTPUT_TYPE res = FUSED_OPS_RESULT;
// #else // HAS_FUSED_OPS
//     OUTPUT_TYPE res = ACTIVATION(interp_val, ACTIVATION_PARAMS);
// #endif // HAS_FUSED_OPS
    output[FUNC_CALL(get_output_index)(out_coords[0], out_coords[1], out_coords[2], out_coords[3], out_coords[4])] = interp_val;
#elif defined(SAMPLE_TYPE_INTERP) // defined(SAMPLE_TYPE_NEAREST) && FEATURE_PACKED_MODE
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    const int feature = 0;
    const int batch = get_global_id(2);
    const float ix = FUNC_CALL(get_original_coordinate)(ox, X_RATIO, OUTPUT_SIZE_X, INPUT0_SIZE_X);
    const float iy = FUNC_CALL(get_original_coordinate)(oy, Y_RATIO, OUTPUT_SIZE_Y, INPUT0_SIZE_Y);

#ifdef LEFTOVERS
    if (ox >= OUTPUT_SIZE_X)
        return;
#endif

    const int top_y_index    = (int)(floor(iy));
    const int bottom_y_index = min((int)ceil(iy), INPUT0_SIZE_Y - 1);
    const int left_x_index   = (int)(floor(ix));
    const int right_x_index  = min((int)ceil(ix), INPUT0_SIZE_X - 1);

    const ACCUMULATOR_TYPE dx = TO_ACCUMULATOR_TYPE(ix - left_x_index);
    const ACCUMULATOR_TYPE dy = TO_ACCUMULATOR_TYPE(iy - top_y_index);

    unroll_for(int in_f = 0; in_f < OUTPUT_FEATURE_NUM; in_f++) {
        INPUT0_TYPE top_left = input[INPUT0_GET_INDEX(batch, in_f, top_y_index, left_x_index)];
        INPUT0_TYPE top_right = input[INPUT0_GET_INDEX(batch, in_f, top_y_index, right_x_index)];
        INPUT0_TYPE bottom_left = input[INPUT0_GET_INDEX(batch, in_f, bottom_y_index, left_x_index)];
        INPUT0_TYPE bottom_right = input[INPUT0_GET_INDEX(batch, in_f, bottom_y_index, right_x_index)];

        ACCUMULATOR_TYPE top = TO_ACCUMULATOR_TYPE(top_left) + (TO_ACCUMULATOR_TYPE(top_right) - TO_ACCUMULATOR_TYPE(top_left)) * dx;
        ACCUMULATOR_TYPE bottom = TO_ACCUMULATOR_TYPE(bottom_left) + (TO_ACCUMULATOR_TYPE(bottom_right) - TO_ACCUMULATOR_TYPE(bottom_left)) * dx;

        ACCUMULATOR_TYPE interp_val = top + (bottom - top) * dy;

#if HAS_FUSED_OPS
        #define OF_ID (in_f)
        FUSED_OPS;
        OUTPUT_TYPE res = FUSED_OPS_RESULT;
#else
        OUTPUT_TYPE res = TO_OUTPUT_TYPE(ACTIVATION(interp_val, ACTIVATION_PARAMS));
#endif
        output[OUTPUT_GET_INDEX(batch, in_f, oy, ox)] = res;
    }
#elif defined(SAMPLE_TYPE_CAFFE_INTERP) // defined(SAMPLE_TYPE_NEAREST) && FEATURE_PACKED_MODE
    const int ox = (int)get_global_id(0) % OUTPUT_SIZE_X;
    const int oy = (int)get_global_id(0) / OUTPUT_SIZE_X;
    const int feature_block_nun = get_global_id(1);
    const int feature = feature_block_nun * FEATURE_BLOCK_SIZE;
#if OUTPUT_DIMS <= 4
    const int batch = get_global_id(2);
    const int oz = 0;
#else
    const int batch = (int)get_global_id(2) % OUTPUT_BATCH_NUM;
    const int oz    = (int)get_global_id(2) / OUTPUT_BATCH_NUM;
#endif

    const ACCUMULATOR_TYPE ix = FUNC_CALL(get_original_coordinate)(ox, X_RATIO, OUTPUT_SIZE_X, PADS_BEGIN[4] + INPUT0_SIZE_X + PADS_END[4]) - PADS_BEGIN[4];
    const ACCUMULATOR_TYPE iy = FUNC_CALL(get_original_coordinate)(oy, Y_RATIO, OUTPUT_SIZE_Y, PADS_BEGIN[3] + INPUT0_SIZE_Y + PADS_END[3]) - PADS_BEGIN[3];
    const ACCUMULATOR_TYPE iz = FUNC_CALL(get_original_coordinate)(oz, Z_RATIO, OUTPUT_SIZE_Z, PADS_BEGIN[2] + INPUT0_SIZE_Z + PADS_END[2]) - PADS_BEGIN[2];
    const ACCUMULATOR_TYPE ic = FUNC_CALL(get_original_coordinate)(feature, F_RATIO, OUTPUT_FEATURE_NUM, PADS_BEGIN[1] + INPUT0_FEATURE_NUM + PADS_END[1]) - PADS_BEGIN[1];
    const ACCUMULATOR_TYPE ib = FUNC_CALL(get_original_coordinate)(batch, B_RATIO, OUTPUT_BATCH_NUM, PADS_BEGIN[0] + INPUT0_BATCH_NUM + PADS_END[0]) - PADS_BEGIN[0];

    const int ix_r = (int)ix;
    const int iy_r = (int)iy;
    const int iz_r = (int)iz;
    const int ic_r = (int)ic;
    const int ib_r = (int)ib;

#if ANTIALIAS == 1
    const ACCUMULATOR_TYPE ax = 1.0f / X_RATIO;
    const ACCUMULATOR_TYPE ay = 1.0f / Y_RATIO;
    const ACCUMULATOR_TYPE az = 1.0f / Z_RATIO;
    const ACCUMULATOR_TYPE ac = 1.0f / F_RATIO;
    const ACCUMULATOR_TYPE ab = 1.0f / B_RATIO;
#else
    const ACCUMULATOR_TYPE ax = 1.0f;
    const ACCUMULATOR_TYPE ay = 1.0f;
    const ACCUMULATOR_TYPE az = 1.0f;
    const ACCUMULATOR_TYPE ac = 1.0f;
    const ACCUMULATOR_TYPE ab = 1.0f;
#endif
    const int rx = (X_RATIO < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / ax);
    const int ry = (Y_RATIO < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / ay);
    const int rz = (Z_RATIO < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / az);
    const int rc = (F_RATIO < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / ac);
    const int rb = (B_RATIO < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / ab);

    ACCUMULATOR_TYPE sum[FEATURE_BLOCK_SIZE];
    for (int i = 0; i < FEATURE_BLOCK_SIZE; i++)
        sum[i] = 0;

    ACCUMULATOR_TYPE wsum = 0;

    int const y_init = max(-PADS_BEGIN[3], iy_r - ry);
    int const x_init = max(-PADS_BEGIN[4], ix_r - rx);
    int const z_init = max(-PADS_BEGIN[2], iz_r - rz);
    int const f_init = max(-PADS_BEGIN[1], ic_r - rc);
    int const b_init = max(-PADS_BEGIN[0], ib_r - rb);
    int const y_max = min(INPUT0_SIZE_Y + PADS_END[3], iy_r + ry + 1);
    int const x_max = min(INPUT0_SIZE_X + PADS_END[4], ix_r + rx + 1);
    int const z_max = min(INPUT0_SIZE_Z + PADS_END[2], iz_r + rz + 1);
    int const f_max = min(INPUT0_FEATURE_NUM + PADS_END[1], ic_r + rc + 1);
    int const b_max = min(INPUT0_BATCH_NUM + PADS_END[0], ib_r + rb + 1);
if (ox == 2 && oy == 1 && feature_block_nun == 0 && batch == 0) {
        printf("iz,iy,ix =[%f %f %f], %d, ix_r %d, rx %d, x_init %d, x_max %d. X_RATIO %f, ax %f\n", iz, iy, ix, X_PAD_BEGIN, ix_r, rx, x_init, x_max, X_RATIO, ax);
        printf("Z_init %d Z_max %d\n", z_init, z_max);
        printf("Y_init %d Y_max %d\n", y_init, y_max);
        printf("X_init %d X_max %d\n", x_init, x_max);
    }

#ifdef COORD_TRANS_MODE_TF_HALF_PIXEL_FOR_NN
    unroll_for(int b = b_init; b < b_max; b++) {
    unroll_for(int f = f_init; f < f_max; f++) {
#else
    const int f = 0;
#endif
    unroll_for(int z = z_init; z < z_max; z++) {
    unroll_for(int y = y_init; y < y_max; y++) {
    unroll_for(int x = x_init; x < x_max; x++) {
        ACCUMULATOR_TYPE dx = ix - x;
        ACCUMULATOR_TYPE dy = iy - y;
        ACCUMULATOR_TYPE dz = iz - z;
#ifdef COORD_TRANS_MODE_TF_HALF_PIXEL_FOR_NN
        ACCUMULATOR_TYPE df = ic - f;
        ACCUMULATOR_TYPE db = ib - b;
#else
        ACCUMULATOR_TYPE df = ACCUMULATOR_VAL_ZERO;
        ACCUMULATOR_TYPE db = ACCUMULATOR_VAL_ZERO;
#endif
#if ANTIALIAS == 1
        ACCUMULATOR_TYPE w = ax * TRIANGLE_COEFF(ax * dx) * ay * TRIANGLE_COEFF(ay * dy) * az * triangleCoeff(az * dz);
#else
        ACCUMULATOR_TYPE w = TRIANGLE_COEFF(dx) * TRIANGLE_COEFF(dy) * TRIANGLE_COEFF(dz) * TRIANGLE_COEFF(df) * TRIANGLE_COEFF(db);
#endif
        if (ox == 2 && oy == 1 && feature_block_nun == 0 && batch == 0) {
            printf("%f %f %f %f %f, w = %f\n", db, df, dz, dy, dx, w);
        }


#ifndef LEFTOVERS
        unroll_for(int f_b = 0; f_b < FEATURE_BLOCK_SIZE; f_b++) {
#else
        const int f_max = min(FEATURE_BLOCK_SIZE, FEATURE_LEFTOVER);
        unroll_for(int f_b = 0; f_b < f_max; f_b++) {
#endif
#if IS_DATA_PADDED
        if (w != 0 && x >= 0 && x < INPUT0_SIZE_X && y >= 0 && y < INPUT0_SIZE_Y && z >= 0 && z < INPUT0_SIZE_Z)
#else
        if (w != 0)
#endif
            sum[f_b] += w * TO_ACCUMULATOR_TYPE(input[FUNC_CALL(get_input_index)(batch, feature + f_b + f, z, y, x)]);
        }
        wsum += w;
    }
    }
    }
#ifdef COORD_TRANS_MODE_TF_HALF_PIXEL_FOR_NN
    }
    }
#endif
#ifndef LEFTOVERS
    unroll_for (int f = 0; f < FEATURE_BLOCK_SIZE; f++) {
#else
    const int f_max = min(FEATURE_BLOCK_SIZE, FEATURE_LEFTOVER);
    unroll_for (int f = 0; f < f_max; f++) {
#endif

        ACCUMULATOR_TYPE interp_val = (wsum == 0) ? 0 : (sum[f] / wsum);
#if HAS_FUSED_OPS
        #define OF_ID (feature + f)
        FUSED_OPS;
        OUTPUT_TYPE res = FUSED_OPS_RESULT;
#else
        OUTPUT_TYPE res = TO_OUTPUT_TYPE(ACTIVATION(interp_val, ACTIVATION_PARAMS));
#endif
        output[FUNC_CALL(get_output_index)(batch, feature + f, oz, oy, ox)] = res;
    }
#endif // defined(SAMPLE_TYPE_NEAREST) && FEATURE_PACKED_MODE
}

#undef unroll_for
#undef TRIANGLE_COEFF
