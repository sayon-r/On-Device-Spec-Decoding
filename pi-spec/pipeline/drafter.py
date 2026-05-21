"""
Drafter process — LFM-2.5-350M pinned to CPU cores 0-1.

Generates draft token sequences using dynamic confidence-threshold
stopping, then signals the verifier via SharedDraftBuffer and waits
for the accept/reject result before advancing.
"""

import argparse
import os
import sys
import time
import numpy as np

# Pin to cores 0-1 immediately (Linux-only; no-op on other OS).
try:
    os.sched_setaffinity(0, {0, 1})
except AttributeError:
    pass

from shared_mem import SharedDraftBuffer, STATE_VERIFY_DONE, STATE_IDLE


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def top_k_logits(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (top_k_indices, top_k_logits) sorted descending."""
    idx = np.argpartition(logits, -k)[-k:]
    idx = idx[np.argsort(logits[idx])[::-1]]
    return idx, logits[idx]


class Drafter:
    def __init__(
        self,
        model_path: str,
        shared_buf: SharedDraftBuffer,
        max_draft: int = 5,
        confidence_threshold: float = 0.8,
        top_k: int = 32,
        n_ctx: int = 2048,
        n_threads: int = 2,
    ):
        from llama_cpp import Llama

        self.buf   = shared_buf
        self.max_draft = max_draft
        self.conf_thresh = confidence_threshold
        self.top_k = top_k

        print(f"[drafter] Loading model: {model_path}", flush=True)
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            logits_all=True,
            verbose=False,
        )
        self.vocab_size = self.model.n_vocab()
        print(f"[drafter] Model loaded. vocab_size={self.vocab_size}", flush=True)

    def _eval_token(self, token_id: int):
        """Run a single token through the model and return raw logits."""
        self.model.eval([token_id])
        logits = np.array(self.model.scores[-1], dtype=np.float32)
        return logits

    def generate(self, prompt: str, max_tokens: int = 200) -> list[int]:
        """
        Main generation loop.  Returns the full list of accepted token IDs.
        """
        # Encode and prefill prompt.
        input_ids = self.model.tokenize(prompt.encode())
        self.model.eval(input_ids)

        all_tokens: list[int] = []
        step = 0

        while len(all_tokens) < max_tokens:
            # --- Draft phase --------------------------------------------------
            draft_tokens: list[int]    = []
            draft_logits: list[np.ndarray] = []
            draft_confs:  list[float]  = []

            # Seed the first draft token from the last accepted logits.
            logits = np.array(self.model.scores[-1], dtype=np.float32)

            for _ in range(self.max_draft):
                probs     = softmax(logits)
                token_id  = int(np.argmax(probs))   # greedy sample
                confidence = float(probs[token_id])

                _, topk_vals = top_k_logits(logits, self.top_k)
                draft_tokens.append(token_id)
                draft_logits.append(topk_vals)
                draft_confs.append(confidence)

                if confidence < self.conf_thresh:
                    break  # early exit — verifier will correct low-confidence tail

                if len(draft_tokens) >= self.max_draft:
                    break

                # Eval next step.
                logits = self._eval_token(token_id)

            # --- Send to verifier --------------------------------------------
            logits_arr = np.stack(draft_logits, axis=0)  # [n, top_k]
            self.buf.write_draft(draft_tokens, logits_arr, draft_confs)

            # --- Wait for verification ----------------------------------------
            if not self.buf.wait_for_state(STATE_VERIFY_DONE, timeout=60.0):
                print("[drafter] Timeout waiting for verifier", flush=True)
                break

            mask, correction = self.buf.read_result()
            self.buf.reset()

            # --- Accept confirmed tokens --------------------------------------
            accepted_ids: list[int] = []
            for tok, accepted in zip(draft_tokens, mask):
                if accepted:
                    accepted_ids.append(tok)
                else:
                    break

            if correction is not None:
                accepted_ids.append(correction)

            # Feed accepted tokens back into the model state.
            for tok in accepted_ids:
                self.model.eval([tok])
                all_tokens.append(tok)

            step += 1

            # Check for EOS.
            if all_tokens and all_tokens[-1] == self.model.token_eos():
                break

        return all_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",               required=True)
    ap.add_argument("--prompt",              required=True)
    ap.add_argument("--max-tokens",          type=int,   default=200)
    ap.add_argument("--draft-length",        type=int,   default=5)
    ap.add_argument("--confidence-threshold",type=float, default=0.8)
    ap.add_argument("--top-k",               type=int,   default=32)
    ap.add_argument("--shm-name",            default="pispec_draft")
    args = ap.parse_args()

    buf = SharedDraftBuffer(
        max_draft=args.draft_length,
        top_k=args.top_k,
        name=args.shm_name,
        create=True,
    )

    drafter = Drafter(
        model_path=args.model,
        shared_buf=buf,
        max_draft=args.draft_length,
        confidence_threshold=args.confidence_threshold,
        top_k=args.top_k,
    )

    tokens = drafter.generate(args.prompt, args.max_tokens)

    text = drafter.model.detokenize(tokens).decode("utf-8", errors="replace")
    print(text, flush=True)

    buf.close()
    buf.unlink()


if __name__ == "__main__":
    main()
