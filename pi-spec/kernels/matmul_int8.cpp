#include "matmul_int8.h"
#include <cstring>
#include <cmath>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// Scalar fallback — used when DOTPROD is unavailable or for correctness checks.
static void matmul_int8_scalar(
    const float*  A,
    const int8_t* B,
    const float*  scales,
    float*        C,
    int M, int K, int N,
    int group_size)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                int g = k / group_size;
                float a_val = A[m * K + k];
                float b_val = (float)B[k * N + n] * scales[g * N + n];
                acc += a_val * b_val;
            }
            C[m * N + n] = acc;
        }
    }
}

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

// DOTPROD path — Cortex-A76 can issue vdotq_s32 every cycle.
// Weights are in block-interleaved layout: groups of 4 columns packed
// contiguously so each 16-byte load covers a complete 4-col slice.
//
// Per-group scale layout: scales[(k/group_size) * ceil(N/4) + n4] where n4
// is the 4-column block index.  We pack 4 floats per block so a single
// vld1q_f32 loads all four column scales simultaneously.
//
// Inner loop processes 16 K-elements (one vdotq_s32 iteration) and 4 N-cols
// per iteration, accumulating into int32x4_t, then converting to FP32 and
// scaling at the group boundary.
static void matmul_int8_dotprod(
    const float*  A,
    const int8_t* B,       // block-interleaved [K][N/4][4]
    const float*  scales,  // [K/group_size][N/4] each entry covers 4 cols
    float*        C,
    int M, int K, int N,
    int group_size)
{
    int N4 = (N + 3) / 4;   // number of 4-column blocks

    for (int m = 0; m < M; m++) {
        for (int n4 = 0; n4 < N4; n4++) {
            int32x4_t acc = vdupq_n_s32(0);
            float32x4_t fp_acc = vdupq_n_f32(0.0f);
            int prev_group = -1;

            for (int k = 0; k < K; k += 4) {
                int cur_group = k / group_size;

                // At group boundary: drain int32 accumulator into float acc.
                if (cur_group != prev_group && prev_group >= 0) {
                    float32x4_t s = vld1q_f32(&scales[prev_group * N4 + n4 * 4]);
                    fp_acc = vmlaq_f32(fp_acc, vcvtq_f32_s32(acc), s);
                    acc = vdupq_n_s32(0);
                }
                prev_group = cur_group;

                // Load 4 INT8 activations for this m-row.
                // Pad to 16 bytes for the vdotq_s32 requirement.
                int8_t a_buf[16] = {};
                int rem = K - k < 4 ? K - k : 4;
                for (int i = 0; i < rem; i++)
                    a_buf[i] = (int8_t)(A[m * K + k + i] > 0 ? A[m * K + k + i] : -A[m * K + k + i]);
                // NOTE: activations are kept in FP32 upstream; we cast here for
                // the DOTPROD path.  For production use, quantise activations
                // separately and pass INT8 directly.
                int8x16_t a_vec = vld1q_s8(a_buf);

                // Weights are interleaved: B[(k/4)*N4*16 + n4*16 + within]
                const int8_t* b_ptr = &B[(k / 4) * N4 * 16 + n4 * 16];
                int8x16_t b_vec = vld1q_s8(b_ptr);

                acc = vdotq_s32(acc, a_vec, b_vec);
            }

            // Final group drain.
            if (prev_group >= 0) {
                float32x4_t s = vld1q_f32(&scales[prev_group * N4 + n4 * 4]);
                fp_acc = vmlaq_f32(fp_acc, vcvtq_f32_s32(acc), s);
            }

            // Store 4 output columns (handle tail).
            float out_buf[4];
            vst1q_f32(out_buf, fp_acc);
            int cols = N - n4 * 4 < 4 ? N - n4 * 4 : 4;
            for (int i = 0; i < cols; i++)
                C[m * N + n4 * 4 + i] = out_buf[i];
        }
    }
}
#endif // __ARM_FEATURE_DOTPROD

extern "C" void matmul_int8_neon(
    const float*  A,
    const int8_t* B,
    const float*  scales,
    float*        C,
    int M, int K, int N,
    int group_size)
{
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    matmul_int8_dotprod(A, B, scales, C, M, K, N, group_size);
#else
    matmul_int8_scalar(A, B, scales, C, M, K, N, group_size);
#endif
}
