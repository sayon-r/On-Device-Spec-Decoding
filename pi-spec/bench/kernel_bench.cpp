/**
 * Kernel microbenchmark — naive vs NEON vs reference.
 *
 * Measures wall-clock time for matmul_int8_neon, rmsnorm_neon, and
 * quantize_weights against plain scalar implementations.
 *
 * Build:
 *   g++ -O3 -march=armv8-a+dotprod -I../kernels \
 *       kernel_bench.cpp ../kernels/matmul_int8.cpp \
 *       ../kernels/quantize.cpp ../kernels/layernorm.cpp \
 *       -o kernel_bench
 *
 * Run:
 *   ./kernel_bench --matrix-size 512 --runs 100
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include "matmul_int8.h"
#include "quantize.h"
#include "layernorm.h"

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               Clock::now().time_since_epoch())
        .count();
}

// ---------------------------------------------------------------------------
// Scalar baselines (independent of kernels/ files)

static void matmul_scalar(
    const float*  A,
    const int8_t* B,
    const float*  scales,
    float*        C,
    int M, int K, int N,
    int group_size)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) {
                int g = k / group_size;
                acc += A[m * K + k] * (float)B[k * N + n] * scales[g];
            }
            C[m * N + n] = acc;
        }
}

static void rmsnorm_scalar(
    const float* x, const float* w, float* out, int n, float eps)
{
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

static void quantize_scalar(
    const float* src, int8_t* dst, float* scales, int n, int gs)
{
    int ng = (n + gs - 1) / gs;
    for (int g = 0; g < ng; g++) {
        int s = g * gs, e = std::min(s + gs, n);
        float mx = 0.f;
        for (int i = s; i < e; i++) mx = std::max(mx, std::abs(src[i]));
        float sc = mx == 0.f ? 1.f : mx / 127.f;
        scales[g] = sc;
        for (int i = s; i < e; i++)
            dst[i] = (int8_t)std::round(src[i] / sc);
    }
}

// ---------------------------------------------------------------------------

static void fill_rand(float* p, int n) {
    for (int i = 0; i < n; i++)
        p[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
}

// ---------------------------------------------------------------------------

struct BenchResult {
    std::string name;
    double mean_ms;
    double min_ms;
    double max_ms;
};

static BenchResult bench(
    const std::string& name,
    int runs,
    std::function<void()> fn)
{
    std::vector<double> times(runs);
    for (int r = 0; r < runs; r++) {
        double t0 = now_ms();
        fn();
        times[r] = now_ms() - t0;
    }
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / runs;
    double mn   = *std::min_element(times.begin(), times.end());
    double mx   = *std::max_element(times.begin(), times.end());
    return {name, mean, mn, mx};
}

static void print_result(const BenchResult& r) {
    printf("  %-30s  mean=%7.3f ms  min=%7.3f ms  max=%7.3f ms\n",
           r.name.c_str(), r.mean_ms, r.min_ms, r.max_ms);
}

// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    int matrix_size = 512;
    int runs        = 50;
    int group_size  = 32;

    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);
        if (a == "--matrix-size" && i + 1 < argc) matrix_size = atoi(argv[++i]);
        if (a == "--runs"        && i + 1 < argc) runs        = atoi(argv[++i]);
    }

    printf("=== pi-spec kernel microbenchmark ===\n");
    printf("  Matrix size : %d x %d\n", matrix_size, matrix_size);
    printf("  Runs        : %d\n\n", runs);

    srand(42);

    // --- matmul INT8 --------------------------------------------------------
    {
        int M = 1, K = matrix_size, N = matrix_size;
        int ng = (K + group_size - 1) / group_size;
        int ng_n = ng * ((N + 3) / 4); // scale layout for NEON path

        std::vector<float>  A(M * K), C_scalar(M * N), C_neon(M * N);
        std::vector<float>  fp_weights(K * N);
        std::vector<int8_t> B_raw(K * N), B_interleaved(K * N);
        std::vector<float>  scales_flat(ng), scales_neon(ng_n * 4, 1.f);

        fill_rand(A.data(), M * K);
        fill_rand(fp_weights.data(), K * N);

        quantize_weights(fp_weights.data(), B_raw.data(),
                         scales_flat.data(), K * N, group_size);
        interleave_weights_4col(B_raw.data(), B_interleaved.data(), K, N);

        printf("matmul INT8 (M=1, K=%d, N=%d):\n", K, N);

        auto r_scalar = bench("scalar", runs, [&]() {
            matmul_scalar(A.data(), B_raw.data(), scales_flat.data(),
                          C_scalar.data(), M, K, N, group_size);
        });

        auto r_neon = bench("NEON/DOTPROD", runs, [&]() {
            matmul_int8_neon(A.data(), B_interleaved.data(), scales_neon.data(),
                             C_neon.data(), M, K, N, group_size);
        });

        print_result(r_scalar);
        print_result(r_neon);
        printf("  Speedup: %.2fx\n\n",
               r_scalar.mean_ms / (r_neon.mean_ms + 1e-9));
    }

    // --- RMSNorm ------------------------------------------------------------
    {
        int n = matrix_size * 4;
        std::vector<float> x(n), w(n), out_s(n), out_n(n);
        fill_rand(x.data(), n);
        fill_rand(w.data(), n);

        printf("RMSNorm (n=%d):\n", n);

        auto r_s = bench("scalar", runs, [&]() {
            rmsnorm_scalar(x.data(), w.data(), out_s.data(), n, 1e-5f);
        });
        auto r_n = bench("NEON", runs, [&]() {
            rmsnorm_neon(x.data(), w.data(), out_n.data(), n, 1e-5f);
        });

        print_result(r_s);
        print_result(r_n);
        printf("  Speedup: %.2fx\n\n",
               r_s.mean_ms / (r_n.mean_ms + 1e-9));
    }

    // --- Quantize -----------------------------------------------------------
    {
        int n = matrix_size * matrix_size;
        int ng = (n + group_size - 1) / group_size;
        std::vector<float>  src(n), scales(ng);
        std::vector<int8_t> dst(n);
        fill_rand(src.data(), n);

        printf("Quantize (n=%d, group=%d):\n", n, group_size);

        auto r_s = bench("scalar", runs, [&]() {
            quantize_scalar(src.data(), dst.data(), scales.data(), n, group_size);
        });
        auto r_k = bench("kernel", runs, [&]() {
            quantize_weights(src.data(), dst.data(), scales.data(), n, group_size);
        });

        print_result(r_s);
        print_result(r_k);
        printf("  Speedup: %.2fx\n\n",
               r_s.mean_ms / (r_k.mean_ms + 1e-9));
    }

    return 0;
}
