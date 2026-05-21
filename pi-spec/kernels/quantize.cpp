#include "quantize.h"
#include <cmath>
#include <cstring>
#include <algorithm>

extern "C" void quantize_weights(
    const float* src,
    int8_t*      dst,
    float*       scales,
    int          n_weights,
    int          group_size)
{
    int n_groups = (n_weights + group_size - 1) / group_size;
    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        int end   = start + group_size < n_weights ? start + group_size : n_weights;

        float max_abs = 0.0f;
        for (int i = start; i < end; i++)
            max_abs = std::max(max_abs, std::abs(src[i]));

        // Cactus's exact scale formula.
        float scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
        scales[g] = scale;

        float inv_scale = 1.0f / scale;
        for (int i = start; i < end; i++) {
            float q = std::round(src[i] * inv_scale);
            // Clamp to [-127, 127] (not -128: avoids asymmetry issues).
            q = q < -127.0f ? -127.0f : (q > 127.0f ? 127.0f : q);
            dst[i] = (int8_t)q;
        }
    }
}

extern "C" void dequantize_weights(
    const int8_t* src,
    const float*  scales,
    float*        dst,
    int           n_weights,
    int           group_size)
{
    int n_groups = (n_weights + group_size - 1) / group_size;
    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        int end   = start + group_size < n_weights ? start + group_size : n_weights;
        float scale = scales[g];
        for (int i = start; i < end; i++)
            dst[i] = (float)src[i] * scale;
    }
}

// Reorders weights from row-major [K x N] into a block-interleaved layout
// where groups of 4 columns are packed contiguously.
//
// Output layout: [K/4][N/4][4][4]
//   – outer K stride: 4 K-elements at a time
//   – outer N stride: 4-column blocks
//   – inner 4x4: 4 K-elements × 4 N-cols packed as 16 INT8 values
//
// This matches the access pattern in matmul_int8_dotprod so every 16-byte
// weight load is contiguous in memory.
extern "C" void interleave_weights_4col(
    const int8_t* src,   // [K x N] row-major
    int8_t*       dst,   // [K/4][N/4][16]
    int K, int N)
{
    int N4 = (N + 3) / 4;
    int K4 = (K + 3) / 4;

    std::memset(dst, 0, (size_t)K4 * N4 * 16);

    for (int k4 = 0; k4 < K4; k4++) {
        for (int n4 = 0; n4 < N4; n4++) {
            int8_t* block = &dst[(k4 * N4 + n4) * 16];
            for (int ki = 0; ki < 4; ki++) {
                int k = k4 * 4 + ki;
                for (int ni = 0; ni < 4; ni++) {
                    int n = n4 * 4 + ni;
                    int8_t val = (k < K && n < N) ? src[k * N + n] : 0;
                    block[ki * 4 + ni] = val;
                }
            }
        }
    }
}
