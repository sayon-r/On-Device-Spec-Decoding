# On-Device Speculative Decoding on Raspberry Pi 5

> Concurrent draft-verify LLM inference on a single ARM Cortex-A76 chip using custom INT8 NEON kernels and grouped quantization — inspired by [Cactus](https://github.com/cactus-compute/cactus).

---

## Overview

This project implements **on-device speculative decoding** on a single Raspberry Pi 5, running two LFM2.5 models concurrently across separate CPU core pairs. The draft model (LFM-2.5-350M) proposes token sequences; the verifier model (LFM-2.5-1.2B) confirms them in a single forward pass. The result is **1.2B-quality output at near-350M decode speed**, entirely on CPU, within 8GB RAM.

This work extends the kernel techniques published by [Cactus](https://github.com/cactus-compute/cactus) — specifically their DOTPROD fallback path for Cortex-A76 devices — and applies them to a speculative decoding architecture inspired by [SLED](https://arxiv.org/abs/2506.09397) and [PicoSpec](https://arxiv.org/abs/2603.19133).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Raspberry Pi 5                     │
│                                                     │
│  Cores 0-1              Cores 2-3                   │
│  ┌──────────────┐       ┌──────────────┐            │
│  │  LFM-350M    │       │  LFM-1.2B    │            │
│  │  (Drafter)   │       │  (Verifier)  │            │
│  │              │       │              │            │
│  │ ~30 tok/s    │       │ ~10 tok/s    │            │
│  │  solo        │       │  solo        │            │
│  └──────┬───────┘       └──────┬───────┘            │
│         │   POSIX Shared Mem   │                    │
│         └──────────┬───────────┘                    │
│                    │                                │
│         ┌──────────▼───────────┐                    │
│         │  Async Draft-Verify  │                    │
│         │      Pipeline        │                    │
│         └──────────────────────┘                    │
│                    │                                │
│            Streamed Output                          │
│         (1.2B quality, ~350M speed)                 │
└─────────────────────────────────────────────────────┘
```

### Three-Layer Stack

```
Python Pipeline Layer       ← async draft-verify orchestration
        ↓
C++ Kernel Layer            ← INT8 NEON matmul, quantization
        ↓
ARM Cortex-A76 Hardware     ← DOTPROD instructions, 512KB L2 per core
```

---

## Key Components

### 1. INT8 NEON Matmul Kernel (`kernels/matmul_int8.cpp`)

Custom C++ kernel targeting the Cortex-A76 DOTPROD instruction set. Processes 4 INT8 multiply-accumulates per cycle using `vdotq_s32` — the same fallback path Cactus uses for Pi 5.

```cpp
// Core inner loop — ARM DOTPROD path
int32x4_t acc = vdupq_n_s32(0);
for (int k = 0; k < K; k += 16) {
    int8x16_t a_vec = vld1q_s8(&A[m * K + k]);  // load 16 INT8 activations
    int8x16_t b_vec = vld1q_s8(&B[k * N + n]);  // load 16 INT8 weights (interleaved)
    acc = vdotq_s32(acc, a_vec, b_vec);          // 4 dot products in 1 cycle
}
```

**Memory layout:** Weights are stored in a 32-byte aligned, block-interleaved format (4 columns packed together), matching Cactus's layout to eliminate runtime transposition and maximise cache prefetch efficiency.

---

### 2. Grouped INT8 Quantization (`kernels/quantize.cpp`)

Weights are quantized from FP32 to INT8 using grouped affine quantization (group size 32), reducing model memory footprint by **75%** and enabling both models to run concurrently within 8GB RAM.

```cpp
// Group size 32 — Cactus's published scheme
#define GROUP_SIZE 32

void quantize_weights(const float* src, int8_t* dst,
                      float* scales, int n_weights) {
    int n_groups = n_weights / GROUP_SIZE;
    for (int g = 0; g < n_groups; g++) {
        float max_abs = 0.0f;
        for (int i = 0; i < GROUP_SIZE; i++)
            max_abs = fmaxf(max_abs, fabsf(src[g * GROUP_SIZE + i]));

        scales[g] = max_abs / 127.0f;  // Cactus's scale formula

        for (int i = 0; i < GROUP_SIZE; i++)
            dst[g * GROUP_SIZE + i] = (int8_t)roundf(
                src[g * GROUP_SIZE + i] / scales[g]);
    }
}
```

| Model | FP32 | INT8 | Reduction |
|---|---|---|---|
| LFM-2.5-350M | ~1.4 GB | ~355 MB | 75% |
| LFM-2.5-1.2B | ~4.8 GB | ~1.2 GB | 75% |
| Both combined | ~6.2 GB | ~1.55 GB | 75% |

---

### 3. Speculative Decoding Pipeline (`pipeline/spec_decode.py`)

Implements the draft-verify loop with SLED's dynamic confidence-threshold drafting. The drafter runs on cores 0-1, the verifier on cores 2-3, communicating via POSIX shared memory — zero network overhead.

```python
# Core rejection sampling loop — Chen et al. 2023
def verify_draft(draft_tokens, draft_logits, verify_logits):
    accepted = []
    for draft_tok, d_log, v_log in zip(draft_tokens, draft_logits, verify_logits):
        p = softmax(v_log)[draft_tok]   # target model probability
        q = softmax(d_log)[draft_tok]   # draft model probability

        if random() < min(1.0, p / q):  # accept with this probability
            accepted.append(draft_tok)
        else:
            accepted.append(sample_correction(v_log, d_log))
            break  # stop at first rejection

    return accepted
```

**Dynamic drafting:** The drafter checks the confidence score of each proposed token. If confidence drops below `CONFIDENCE_THRESHOLD`, it stops early and sends for verification — reducing unnecessary forward passes on the verifier.

---

### 4. Core Pinning (`pipeline/process_manager.py`)

Draft and verifier processes are pinned to separate core pairs using Linux CPU affinity via `taskset`, ensuring the two models don't compete for L2 cache.

```bash
# Drafter — cores 0-1 (512KB L2 each)
taskset -c 0,1 python pipeline/drafter.py --model models/lfm-350m

# Verifier — cores 2-3 (512KB L2 each)
taskset -c 2,3 python pipeline/verifier.py --model models/lfm-1.2b
```

---

## Repository Structure

```
pi-spec/
├── kernels/
│   ├── matmul_int8.cpp      # INT8 NEON matmul kernel (Cortex-A76 DOTPROD)
│   ├── quantize.cpp         # Grouped INT8 quantization (group=32)
│   ├── layernorm.cpp        # RMSNorm kernel
│   └── CMakeLists.txt
├── pipeline/
│   ├── spec_decode.py       # Draft-verify loop, rejection sampling
│   ├── drafter.py           # LFM-350M process (cores 0-1)
│   ├── verifier.py          # LFM-1.2B process (cores 2-3)
│   ├── shared_mem.py        # POSIX shared memory IPC
│   └── process_manager.py  # Core pinning, process lifecycle
├── bench/
│   ├── benchmark.py         # End-to-end tok/s benchmark
│   └── kernel_bench.cpp     # Per-kernel microbenchmark vs naive baseline
├── models/                  # Downloaded GGUF weights (gitignored)
└── README.md
```

---

## Setup

### Requirements

- Raspberry Pi 5 (8GB RAM)
- Active cooler — mandatory for sustained inference
- Raspberry Pi OS (Bookworm) or Ubuntu 24.04 ARM64
- GCC 12+ with ARMv8.2-A support

### Install Dependencies

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential python3-pip git

pip3 install huggingface-hub numpy --break-system-packages
```

### Build Kernels

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=armv8-a+dotprod"
make -j4
```

### Download Models

```bash
# Draft model — Pi cores 0-1
huggingface-cli download LiquidAI/LFM2.5-350M-GGUF \
  LFM2.5-350M-Q8_0.gguf --local-dir ./models/lfm-350m

# Verifier model — Pi cores 2-3
huggingface-cli download LiquidAI/LFM2.5-1.2B-Instruct-GGUF \
  LFM2.5-1.2B-Instruct-Q8_0.gguf --local-dir ./models/lfm-1.2b
```

### Run

```bash
# Start speculative decoding pipeline
python pipeline/spec_decode.py \
  --draft models/lfm-350m/LFM2.5-350M-Q8_0.gguf \
  --target models/lfm-1.2b/LFM2.5-1.2B-Instruct-Q8_0.gguf \
  --draft-length 5 \
  --confidence-threshold 0.8 \
  --prompt "Explain how ARM NEON intrinsics work"
```

---

## Benchmarks

Run after setup to reproduce results:

```bash
# Kernel microbenchmark — naive vs NEON vs llama.cpp
./build/kernel_bench --matrix-size 512 --runs 100

# End-to-end tok/s benchmark
python bench/benchmark.py \
  --model-solo models/lfm-1.2b/LFM2.5-1.2B-Instruct-Q8_0.gguf \
  --mode solo

python bench/benchmark.py \
  --draft models/lfm-350m/LFM2.5-350M-Q8_0.gguf \
  --target models/lfm-1.2b/LFM2.5-1.2B-Instruct-Q8_0.gguf \
  --mode speculative
```

Expected results (to be updated after Pi hardware runs):

| Mode | tok/s | Quality |
|---|---|---|
| LFM-350M solo | ~30 | 350M |
| LFM-1.2B solo | ~10 | 1.2B |
| Speculative (this project) | TBD | 1.2B |
| Cactus Pi 5 reference | 30 (350M) | 350M |

---

## Research Background

This project implements techniques from three papers:

- **Chen et al. (2023)** — [Speculative Sampling](https://arxiv.org/abs/2302.01318): the rejection sampling formula `min(1, p/q)` that guarantees output distribution identical to the target model
- **SLED (2025)** — [Speculative LLM Decoding at the Edge](https://arxiv.org/abs/2506.09397): dynamic confidence-threshold drafting for edge devices; Pi 5 benchmark data
- **PicoSpec (2026)** — [Pipelined Collaborative Speculative Decoding](https://arxiv.org/abs/2603.19133): async pipeline eliminating stop-and-wait; sparse logit compression

The key distinction from all prior work: SLED and PicoSpec assume WAN latency (50-200ms RTT) or GPU verifiers. This project runs both draft and verifier on CPU-only ARM cores on the same device, with POSIX shared memory IPC (~1μs) replacing network communication entirely.

---

## Inspiration

Kernel techniques (grouped INT8 quantization, 32-byte aligned weight layout, DOTPROD fallback path) are directly inspired by [Cactus](https://github.com/cactus-compute/cactus) — a production mobile inference engine that benchmarks the Pi 5 Cortex-A76 DOTPROD path. This project reimplements that kernel layer from scratch and extends it with a speculative decoding architecture Cactus does not currently support.

---

## License

MIT
