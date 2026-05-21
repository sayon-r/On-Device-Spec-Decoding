#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void quantize_weights(
    const float* src,
    int8_t*      dst,
    float*       scales,
    int          n_weights,
    int          group_size
);

void dequantize_weights(
    const int8_t* src,
    const float*  scales,
    float*        dst,
    int           n_weights,
    int           group_size
);

void interleave_weights_4col(
    const int8_t* src,
    int8_t*       dst,
    int K, int N
);

#ifdef __cplusplus
}
#endif
