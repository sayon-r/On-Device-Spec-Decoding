#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void rmsnorm_neon(
    const float* x,
    const float* weight,
    float*       out,
    int          n,
    float        eps
);

#ifdef __cplusplus
}
#endif
