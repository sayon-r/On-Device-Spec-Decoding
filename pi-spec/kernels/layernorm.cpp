#include "layernorm.h"
#include <cmath>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

extern "C" void rmsnorm_neon(
    const float* x,
    const float* weight,
    float*       out,
    int          n,
    float        eps)
{
#ifdef __ARM_NEON
    // Compute sum of squares using NEON.
    float32x4_t vsum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= n - 4; i += 4) {
        float32x4_t v = vld1q_f32(&x[i]);
        vsum = vmlaq_f32(vsum, v, v);
    }
    // Horizontal reduction.
    float32x2_t lo = vget_low_f32(vsum);
    float32x2_t hi = vget_high_f32(vsum);
    float32x2_t s2 = vadd_f32(lo, hi);
    float ss = vget_lane_f32(vpadd_f32(s2, s2), 0);
    // Scalar tail.
    for (; i < n; i++)
        ss += x[i] * x[i];

    float scale = 1.0f / sqrtf(ss / (float)n + eps);

    // Apply scale * weight using NEON.
    float32x4_t vscale = vdupq_n_f32(scale);
    i = 0;
    for (; i <= n - 4; i += 4) {
        float32x4_t vx = vld1q_f32(&x[i]);
        float32x4_t vw = vld1q_f32(&weight[i]);
        vst1q_f32(&out[i], vmulq_f32(vmulq_f32(vx, vscale), vw));
    }
    for (; i < n; i++)
        out[i] = x[i] * scale * weight[i];

#else
    // Scalar fallback.
    float ss = 0.0f;
    for (int i = 0; i < n; i++)
        ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * scale * weight[i];
#endif
}
