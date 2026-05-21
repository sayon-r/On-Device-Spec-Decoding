#!/usr/bin/env python3
"""
On-Device Speculative Decoding — main entry point.

Usage:
  python pipeline/spec_decode.py \\
    --draft  models/lfm-350m/LFM2.5-350M-Q8_0.gguf \\
    --target models/lfm-1.2b/LFM2.5-1.2B-Instruct-Q8_0.gguf \\
    --prompt "Explain how ARM NEON intrinsics work" \\
    --max-tokens 200 \\
    --draft-length 5 \\
    --confidence-threshold 0.8
"""

import argparse
import sys
import time

from process_manager import launch_speculative_pipeline


def main():
    ap = argparse.ArgumentParser(
        description="On-device speculative decoding on Raspberry Pi 5"
    )
    ap.add_argument("--draft",                required=True,
                    help="Path to draft model GGUF (LFM-2.5-350M)")
    ap.add_argument("--target",               required=True,
                    help="Path to target model GGUF (LFM-2.5-1.2B)")
    ap.add_argument("--prompt",               required=True,
                    help="Input prompt text")
    ap.add_argument("--max-tokens",           type=int,   default=200)
    ap.add_argument("--draft-length",         type=int,   default=5,
                    help="Max draft tokens per round (default 5)")
    ap.add_argument("--confidence-threshold", type=float, default=0.8,
                    help="Stop drafting early below this confidence (default 0.8)")
    ap.add_argument("--top-k",                type=int,   default=32,
                    help="Top-k logits transmitted per draft token (default 32)")
    ap.add_argument("--shm-name",             default="pispec_draft",
                    help="POSIX shared memory segment name")
    args = ap.parse_args()

    print(f"\n=== On-Device Speculative Decoding ===")
    print(f"  Draft model  : {args.draft}")
    print(f"  Target model : {args.target}")
    print(f"  Draft length : {args.draft_length}")
    print(f"  Conf thresh  : {args.confidence_threshold}")
    print(f"  Max tokens   : {args.max_tokens}")
    print(f"\nPrompt: {args.prompt}\n")
    print("--- Output ---")
    sys.stdout.flush()

    t0 = time.monotonic()

    text, n_tokens, elapsed = launch_speculative_pipeline(
        draft_model_path=args.draft,
        target_model_path=args.target,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        draft_length=args.draft_length,
        confidence_threshold=args.confidence_threshold,
        top_k=args.top_k,
        shm_name=args.shm_name,
    )

    print(text)
    print("\n--- Stats ---")
    print(f"  Tokens generated : {n_tokens}")
    print(f"  Elapsed          : {elapsed:.2f}s")
    if elapsed > 0:
        print(f"  Throughput       : {n_tokens / elapsed:.2f} tok/s")


if __name__ == "__main__":
    main()
