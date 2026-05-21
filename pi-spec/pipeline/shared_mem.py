"""
POSIX shared memory IPC between drafter and verifier processes.

Layout of the shared buffer (all little-endian):
  Offset 0       : 1 byte  — state flag
                     0 = idle
                     1 = draft_ready   (drafter → verifier)
                     2 = verify_done   (verifier → drafter)
  Offset 1       : 1 byte  — n_tokens written by drafter (0-MAX_DRAFT)
  Offset 2       : 1 byte  — n_accepted written by verifier
  Offset 3       : 1 byte  — correction_token_valid flag
  Offset 4       : 4 bytes — correction_token (int32)
  Offset 8       : MAX_DRAFT * 4 bytes — draft token IDs (int32)
  Offset 8+D*4   : MAX_DRAFT * TOP_K * 4 bytes — draft top-k logits (float32)
  Offset above   : MAX_DRAFT * 4 bytes — confidence scores (float32)
  Offset above   : MAX_DRAFT bytes — accept/reject mask (uint8)
"""

import struct
import time
import numpy as np
from multiprocessing.shared_memory import SharedMemory

STATE_IDLE        = 0
STATE_DRAFT_READY = 1
STATE_VERIFY_DONE = 2

_HEADER = 8   # bytes: state(1) + n_tok(1) + n_acc(1) + corr_valid(1) + corr_tok(4)


def _buffer_size(max_draft: int, top_k: int) -> int:
    return (
        _HEADER
        + max_draft * 4          # token IDs
        + max_draft * top_k * 4  # logits
        + max_draft * 4          # confidences
        + max_draft              # accept mask
    )


class SharedDraftBuffer:
    """
    Lock-free shared memory buffer for draft/verify coordination.

    The protocol relies on a single state byte that is written last
    (after all payload bytes are committed), so the reader never sees
    a partially-written payload.  On ARMv8 this is sufficient because
    stores are observable in program order within a single thread, and
    the Python GIL is released during mmap operations.  For production
    use, add dmb ish fences if running on weakly-ordered hardware.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        max_draft: int = 8,
        top_k: int = 32,
        name: str = "pispec_draft",
        create: bool = True,
    ):
        self.vocab_size = vocab_size
        self.max_draft  = max_draft
        self.top_k      = top_k
        self.name       = name

        self._size = _buffer_size(max_draft, top_k)

        if create:
            try:
                # Clean up stale segment from a previous run.
                old = SharedMemory(name=name)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
            self._shm = SharedMemory(name=name, create=True, size=self._size)
        else:
            self._shm = SharedMemory(name=name, create=False)

        self._buf = self._shm.buf

        # Pre-compute byte offsets.
        self._off_tokens   = _HEADER
        self._off_logits   = self._off_tokens + max_draft * 4
        self._off_conf     = self._off_logits + max_draft * top_k * 4
        self._off_mask     = self._off_conf   + max_draft * 4

    # ------------------------------------------------------------------
    # State helpers

    def _set_state(self, s: int):
        self._buf[0] = s

    def _get_state(self) -> int:
        return self._buf[0]

    # ------------------------------------------------------------------
    # Drafter → verifier

    def write_draft(
        self,
        tokens:      list,          # int32, up to max_draft
        logits:      np.ndarray,    # [n_tokens, top_k] float32
        confidences: list,          # float, one per token
    ):
        n = len(tokens)
        assert 1 <= n <= self.max_draft

        # Header fields.
        self._buf[1] = n  # n_tokens

        # Token IDs.
        tok_arr = np.array(tokens, dtype=np.int32)
        tok_bytes = tok_arr.tobytes()
        self._buf[self._off_tokens: self._off_tokens + n * 4] = tok_bytes

        # Logits (only the n actual tokens).
        log_flat = logits[:n].astype(np.float32).flatten()
        log_bytes = log_flat.tobytes()
        self._buf[self._off_logits: self._off_logits + n * self.top_k * 4] = log_bytes

        # Confidence scores.
        conf_arr = np.array(confidences, dtype=np.float32)
        conf_bytes = conf_arr.tobytes()
        self._buf[self._off_conf: self._off_conf + n * 4] = conf_bytes

        # Signal verifier — written last so payload is visible first.
        self._set_state(STATE_DRAFT_READY)

    def read_draft(self):
        """Returns (tokens, logits, confidences)."""
        n = self._buf[1]
        tokens = np.frombuffer(
            bytes(self._buf[self._off_tokens: self._off_tokens + n * 4]),
            dtype=np.int32,
        ).tolist()
        logits = np.frombuffer(
            bytes(self._buf[self._off_logits: self._off_logits + n * self.top_k * 4]),
            dtype=np.float32,
        ).reshape(n, self.top_k).copy()
        confidences = np.frombuffer(
            bytes(self._buf[self._off_conf: self._off_conf + n * 4]),
            dtype=np.float32,
        ).tolist()
        return tokens, logits, confidences

    # ------------------------------------------------------------------
    # Verifier → drafter

    def write_result(self, accepted_mask: list, correction_token: int | None):
        n = len(accepted_mask)
        n_accepted = sum(accepted_mask)

        self._buf[2] = n_accepted

        has_corr = correction_token is not None
        self._buf[3] = 1 if has_corr else 0
        if has_corr:
            struct.pack_into("<i", self._buf, 4, correction_token)

        mask_arr = np.array(accepted_mask, dtype=np.uint8)
        self._buf[self._off_mask: self._off_mask + n] = mask_arr.tobytes()

        self._set_state(STATE_VERIFY_DONE)

    def read_result(self):
        """Returns (accepted_mask, correction_token_or_None)."""
        n_accepted = self._buf[2]
        has_corr   = self._buf[3]
        corr_tok   = struct.unpack_from("<i", self._buf, 4)[0] if has_corr else None

        n = self._buf[1]
        mask = list(
            np.frombuffer(
                bytes(self._buf[self._off_mask: self._off_mask + n]),
                dtype=np.uint8,
            )
        )
        return mask, corr_tok

    # ------------------------------------------------------------------
    # Polling helpers

    def wait_for_state(self, target: int, timeout: float = 30.0, poll_us: float = 50):
        """Busy-poll until state matches target or timeout expires."""
        deadline = time.monotonic() + timeout
        sleep_s = poll_us / 1_000_000
        while True:
            if self._get_state() == target:
                return True
            if time.monotonic() > deadline:
                return False
            time.sleep(sleep_s)

    def reset(self):
        self._set_state(STATE_IDLE)

    # ------------------------------------------------------------------

    def close(self):
        self._buf.release()
        self._shm.close()

    def unlink(self):
        self._shm.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
