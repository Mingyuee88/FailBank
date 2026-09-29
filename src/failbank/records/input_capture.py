"""Capture exact model inputs and nominal action chunks around client.infer.

Ordering note: in run_episode, client.infer() fires BEFORE runtime_env_step() for
the same control step (the policy produces the action, then the runtime executes
it). So capture must be armed from process start, keyed by a GLOBAL monotonic
infer index -- NOT gated on an episode-start call that only happens inside the
first runtime_env_step (which would drop infer #0). At replan_steps=1 there is
exactly one infer per executed step, so infer index k corresponds to executed
step k. Consumers (the recorder) pop by that global index. One episode per
process (--trials 1) is assumed for Phase-1 collection.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np


def _snapshot(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {str(key): _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return copy.deepcopy(value)


@dataclass(frozen=True)
class CapturedInfer:
    infer_call_index: int
    element: dict[str, Any]
    action_chunk: np.ndarray


class InferCapture:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_index = 0
        self._records: dict[int, CapturedInfer] = {}
        self._original_infer: Optional[Callable[..., Any]] = None
        self._patched_cls: Any = None

    def wrap_class(self, client_cls: Any) -> None:
        """Patch `client_cls.infer` at the class level (the client instance is
        created deep in the evaluator and not visible to run.py)."""
        if self._original_infer is not None:
            raise RuntimeError("InferCapture is already mounted")
        original_infer = client_cls.infer
        self._original_infer = original_infer
        self._patched_cls = client_cls
        capture = self

        def captured_infer(client_self: Any, element: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
            result = original_infer(client_self, element, *args, **kwargs)
            try:
                with capture._lock:
                    call_index = capture._next_index
                    capture._next_index += 1
                    action_chunk = np.asarray(result["actions"]).copy()
                    capture._records[call_index] = CapturedInfer(
                        infer_call_index=call_index,
                        element=_snapshot(dict(element)),
                        action_chunk=action_chunk,
                    )
            except Exception:
                # Capture must never break inference / rollout.
                pass
            return result

        client_cls.infer = captured_infer

    def pop(self, infer_call_index: int) -> Optional[CapturedInfer]:
        with self._lock:
            return self._records.pop(infer_call_index, None)
