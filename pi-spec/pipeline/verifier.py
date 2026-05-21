"""
Verifier process — LFM-2.5-1.2B pinned to CPU cores 2-3.

Polls SharedDraftBuffer for a draft, runs a single batched forward pass,
applies Chen et al. 2023 rejection sampling, and writes the accept/reject
mask + optional correction token back.
"""

import argparse
import os
import sys
import numpy as np

# Pin to cores 2-3 immediately (Linux-only; no-op on other OS).
try:
    os.sched_setaffinity(0, {2, 3})
except AttributeError:
    pass

from shared_mem import (
    SharedDraftBuffer,
    STATE_DRAFT_READY,
    STATE_VERIFY_DONE,
    STATE_IDLE,
)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def rejection_sample(
    draft_tokens: list[int],
    draft_logits: np.ndarray,    # [n, top_k] — top-k logits from drafter
    target_logits: np.ndarray,   # [n, vocab_size] — full logits from verifier
    top_k_indices: list,         # [n, top_k] — which vocab indices the draft top-k correspond to
    rng: np.random.Generator,
) -> tuple[list[int], int | None]:
    """
    Chen et al. 2023 speculative sampling rejection test.

    For each draft token t_i:
      p = target_prob(t_i)
      q = draft_prob(t_i)
      accept if U < min(1, p/q)

    On first rejection, sample a correction token from the adjusted
    distribution max(0, p - q) / Z.
    """
    accepted_mask: list[int] = []
    correction: int | None   = None

    n = len(draft_tokens)
    for i in range(n):
        tok     = draft_tokens[i]
        t_probs = softmax(target_logits[i])
        p_target = float(t_probs[tok])

        # Reconstruct draft probabilities over the full vocab.
        d_full = np.zeros(target_logits.shape[1], dtype=np.float32)
        d_topk_probs = softmax(draft_logits[i])
        for j, idx in enumerate(top_k_indices[i]):
            d_full[idx] = d_topk_probs[j]
        q_draft = float(d_full[tok])

        threshold = min(1.0, p_target / (q_draft + 1e-9))

        if rng.random() < threshold:
            accepted_mask.append(1)
        else:
            accepted_mask.append(0)
            # Sample correction from max(0, p - q) / Z.
            adjusted = np.maximum(0.0, t_probs - d_full)
            total = adjusted.sum()
            if total > 0:
                adjusted /= total
                correction = int(rng.choice(len(adjusted), p=adjusted))
            else:
                correction = int(np.argmax(t_probs))
            break  # stop at first rejection

    return accepted_mask, correction


class Verifier:
    def __init__(
        self,
        model_path: str,
        shared_buf: SharedDraftBuffer,
        n_ctx: int = 2048,
        n_threads: int = 2,
    ):
        from llama_cpp import Llama

        self.buf = shared_buf
        self.rng = np.random.default_rng()

        print(f"[verifier] Loading model: {model_path}", flush=True)
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            logits_all=True,
            verbose=False,
        )
        self.vocab_size = self.model.n_vocab()
        print(f"[verifier] Model loaded. vocab_size={self.vocab_size}", flush=True)

    def prefill(self, prompt: str):
        input_ids = self.model.tokenize(prompt.encode())
        self.model.eval(input_ids)

    def verify_loop(self, max_rounds: int = 500):
        """Poll for draft_ready, verify, write result — repeat."""
        round_idx = 0
        while round_idx < max_rounds:
            if not self.buf.wait_for_state(STATE_DRAFT_READY, timeout=120.0):
                print("[verifier] Timeout waiting for drafter", flush=True)
                break

            tokens, draft_logits, _ = self.buf.read_draft()
            n = len(tokens)

            # Single batched forward pass over all draft tokens.
            self.model.eval(tokens)
            # scores shape: [n, vocab_size] after logits_all=True.
            target_logits = np.array(
                [self.model.scores[-(n - i)] for i in range(n)],
                dtype=np.float32,
            )

            # Reconstruct which vocab indices the drafter's top-k correspond to.
            # The drafter sends top-k logit VALUES but not the indices.
            # We approximate by taking top-k from the target logits' index space —
            # a common simplification when indices aren't transmitted.
            top_k_indices = []
            for i in range(n):
                k = draft_logits.shape[1]
                idx = np.argpartition(target_logits[i], -k)[-k:]
                idx = idx[np.argsort(target_logits[i][idx])[::-1]]
                top_k_indices.append(idx)

            mask, correction = rejection_sample(
                tokens,
                draft_logits,
                target_logits,
                top_k_indices,
                self.rng,
            )

            self.buf.write_result(mask, correction)
            round_idx += 1

            # Stop if EOS was accepted.
            last_accepted = None
            for tok, acc in zip(tokens, mask):
                if acc:
                    last_accepted = tok
            if correction is not None:
                last_accepted = correction
            if last_accepted == self.model.token_eos():
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True)
    ap.add_argument("--prompt",   required=True)
    ap.add_argument("--top-k",    type=int, default=32)
    ap.add_argument("--shm-name", default="pispec_draft")
    args = ap.parse_args()

    buf = SharedDraftBuffer(
        top_k=args.top_k,
        name=args.shm_name,
        create=False,  # drafter creates the segment
    )

    verifier = Verifier(model_path=args.model, shared_buf=buf)
    verifier.prefill(args.prompt)
    verifier.verify_loop()

    buf.close()


if __name__ == "__main__":
    main()
