#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void matmul_int8_neon(
    const float*   A,       // [M x K] FP32 activations
    const int8_t*  B,       // [K x N] INT8 weights (block-interleaved)
    const float*   scales,  // per-group scale factors
    float*         C,       // [M x N] FP32 output
    int M, int K, int N,
    int group_size          // should be 32
);

#ifdef __cplusplus
}
#endif
