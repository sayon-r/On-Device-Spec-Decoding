#!/usr/bin/env python3
"""
End-to-end tok/s benchmark.

Modes:
  solo        — run a single model (baseline, no speculation)
  speculative — run the full draft-verify pipeline

Usage examples:
  # Solo baseline — 1.2B model
  python bench/benchmark.py \\
    --model-solo models/lfm-1.2b/LFM2.5-1.2B-Instruct-Q8_0.gguf \\
    --mode solo

  # Solo baseline — 350M model
  python bench/benchmark.py \\
    --model-solo models/lfm-350m/LFM2.5-350M-Q8_0.gguf \\
    --mode solo

  # Speculative decoding
  python bench/benchmark.py \\
    --draft  models/lfm-350m/LFM2.5-350M-Q8_0.gguf \\
    --target models/lfm-1.2b/LFM2.5-1.2B-Instruct-Q8_0.gguf \\
    --mode speculative
"""

import argparse
import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

BENCH_PROMPTS = [
    "Explain how ARM NEON intrinsics work in detail.",
    "What is speculative decoding and why does it improve throughput?",
    "Describe the architecture of the Raspberry Pi 5 CPU.",
    "Write a Python function that computes matrix multiplication using NumPy.",
    "Explain the difference between INT8 and FP32 quantization.",
]


def run_solo(model_path: str, prompts: list[str], max_tokens: int, runs: int):
    from llama_cpp import Llama

    print(f"[solo] Loading model: {model_path}")
    model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        logits_all=False,
        verbose=False,
    )
    print("[solo] Model loaded.")

    all_tps: list[float] = []

    for run_i in range(runs):
        prompt = prompts[run_i % len(prompts)]
        print(f"  Run {run_i + 1}/{runs}: prompt='{prompt[:50]}...'")

        t0 = time.monotonic()
        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            echo=False,
        )
        elapsed = time.monotonic() - t0

        n_tok = output["usage"]["completion_tokens"]
        tps = n_tok / elapsed if elapsed > 0 else 0.0
        all_tps.append(tps)
        print(f"    {n_tok} tokens in {elapsed:.2f}s = {tps:.2f} tok/s")

    return all_tps


def run_speculative(
    draft_path: str,
    target_path: str,
    prompts: list[str],
    max_tokens: int,
    runs: int,
    draft_length: int,
    confidence_threshold: float,
):
    from process_manager import launch_speculative_pipeline

    all_tps: list[float] = []

    for run_i in range(runs):
        prompt = prompts[run_i % len(prompts)]
        print(f"  Run {run_i + 1}/{runs}: prompt='{prompt[:50]}...'")

        text, n_tok, elapsed = launch_speculative_pipeline(
            draft_model_path=draft_path,
            target_model_path=target_path,
            prompt=prompt,
            max_tokens=max_tokens,
            draft_length=draft_length,
            confidence_threshold=confidence_threshold,
        )

        tps = n_tok / elapsed if elapsed > 0 else 0.0
        all_tps.append(tps)
        print(f"    {n_tok} tokens in {elapsed:.2f}s = {tps:.2f} tok/s")

    return all_tps


def print_stats(label: str, tps_list: list[float]):
    if not tps_list:
        return
    mean = statistics.mean(tps_list)
    med  = statistics.median(tps_list)
    mn   = min(tps_list)
    mx   = max(tps_list)
    std  = statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0
    print(f"\n=== {label} ===")
    print(f"  Runs   : {len(tps_list)}")
    print(f"  Mean   : {mean:.2f} tok/s")
    print(f"  Median : {med:.2f} tok/s")
    print(f"  Min    : {mn:.2f} tok/s")
    print(f"  Max    : {mx:.2f} tok/s")
    print(f"  StdDev : {std:.2f} tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",                 required=True,
                    choices=["solo", "speculative"])
    ap.add_argument("--model-solo",           help="Solo mode: path to GGUF")
    ap.add_argument("--draft",                help="Speculative mode: draft GGUF")
    ap.add_argument("--target",               help="Speculative mode: target GGUF")
    ap.add_argument("--max-tokens",           type=int,   default=100)
    ap.add_argument("--runs",                 type=int,   default=3)
    ap.add_argument("--draft-length",         type=int,   default=5)
    ap.add_argument("--confidence-threshold", type=float, default=0.8)
    args = ap.parse_args()

    if args.mode == "solo":
        if not args.model_solo:
            ap.error("--model-solo required for solo mode")
        tps = run_solo(
            args.model_solo,
            BENCH_PROMPTS,
            args.max_tokens,
            args.runs,
        )
        print_stats("Solo mode", tps)

    elif args.mode == "speculative":
        if not args.draft or not args.target:
            ap.error("--draft and --target required for speculative mode")
        tps = run_speculative(
            args.draft,
            args.target,
            BENCH_PROMPTS,
            args.max_tokens,
            args.runs,
            args.draft_length,
            args.confidence_threshold,
        )
        print_stats("Speculative decoding", tps)


if __name__ == "__main__":
    main()
