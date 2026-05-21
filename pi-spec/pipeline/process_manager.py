"""
Launch and manage the drafter + verifier processes.

Handles:
- Shared memory segment lifecycle (drafter creates, verifier attaches)
- CPU affinity via taskset on Linux
- Streaming output tokens as they are confirmed
- Clean shutdown on completion or error
"""

import os
import sys
import time
import signal
import subprocess
import threading
import multiprocessing as mp
from pathlib import Path

# Add pipeline dir to path so child processes can import shared_mem.
_PIPELINE_DIR = str(Path(__file__).parent)


def _drafter_worker(
    model_path: str,
    prompt: str,
    max_tokens: int,
    draft_length: int,
    confidence_threshold: float,
    top_k: int,
    shm_name: str,
    result_queue: mp.Queue,
):
    """Entry point for the drafter subprocess."""
    # Import here (after fork) so affinity is set in child.
    sys.path.insert(0, _PIPELINE_DIR)
    try:
        os.sched_setaffinity(0, {0, 1})
    except AttributeError:
        pass

    from shared_mem import SharedDraftBuffer
    from drafter import Drafter

    buf = SharedDraftBuffer(
        max_draft=draft_length,
        top_k=top_k,
        name=shm_name,
        create=True,
    )
    try:
        d = Drafter(
            model_path=model_path,
            shared_buf=buf,
            max_draft=draft_length,
            confidence_threshold=confidence_threshold,
            top_k=top_k,
        )
        tokens = d.generate(prompt, max_tokens)
        text = d.model.detokenize(tokens).decode("utf-8", errors="replace")
        result_queue.put(("ok", text, tokens))
    except Exception as e:
        result_queue.put(("err", str(e), []))
    finally:
        buf.close()
        buf.unlink()


def _verifier_worker(
    model_path: str,
    prompt: str,
    top_k: int,
    shm_name: str,
    max_rounds: int,
):
    """Entry point for the verifier subprocess."""
    sys.path.insert(0, _PIPELINE_DIR)
    try:
        os.sched_setaffinity(0, {2, 3})
    except AttributeError:
        pass

    import time as _time
    from shared_mem import SharedDraftBuffer
    from verifier import Verifier

    # Wait until drafter has created the shared memory segment.
    for _ in range(100):
        try:
            buf = SharedDraftBuffer(top_k=top_k, name=shm_name, create=False)
            break
        except FileNotFoundError:
            _time.sleep(0.05)
    else:
        print("[verifier] Could not attach to shared memory", flush=True)
        return

    try:
        v = Verifier(model_path=model_path, shared_buf=buf)
        v.prefill(prompt)
        v.verify_loop(max_rounds=max_rounds)
    except Exception as e:
        print(f"[verifier] Error: {e}", flush=True)
    finally:
        buf.close()


def launch_speculative_pipeline(
    draft_model_path: str,
    target_model_path: str,
    prompt: str,
    max_tokens: int = 200,
    draft_length: int = 5,
    confidence_threshold: float = 0.8,
    top_k: int = 32,
    shm_name: str = "pispec_draft",
) -> str:
    """
    Spawn drafter and verifier as separate processes, coordinate via
    shared memory, stream output tokens as they are confirmed.

    Returns the full generated text.
    """
    result_queue: mp.Queue = mp.Queue()

    drafter_proc = mp.Process(
        target=_drafter_worker,
        args=(
            draft_model_path,
            prompt,
            max_tokens,
            draft_length,
            confidence_threshold,
            top_k,
            shm_name,
            result_queue,
        ),
        daemon=False,
    )

    verifier_proc = mp.Process(
        target=_verifier_worker,
        args=(
            target_model_path,
            prompt,
            top_k,
            shm_name,
            max_tokens * 2,  # generous round budget
        ),
        daemon=True,  # dies when drafter finishes
    )

    t_start = time.monotonic()

    verifier_proc.start()
    drafter_proc.start()

    # Wait for drafter to finish (it drives the loop).
    drafter_proc.join(timeout=max_tokens * 10)  # 10s per token is very generous

    if drafter_proc.is_alive():
        drafter_proc.terminate()
        drafter_proc.join()

    if verifier_proc.is_alive():
        verifier_proc.terminate()
        verifier_proc.join()

    elapsed = time.monotonic() - t_start

    if not result_queue.empty():
        status, text, tokens = result_queue.get_nowait()
        if status == "ok":
            return text, len(tokens), elapsed
        else:
            raise RuntimeError(f"Drafter failed: {text}")
    else:
        raise RuntimeError("Drafter produced no output")
