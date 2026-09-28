"""Isolated, fail-closed OpenPI policy-server connection management."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable
import traceback


def port_for_attempt(base_port: int, array_task_id: int, attempt: int) -> int:
    """Allocate disjoint task and retry namespaces without probing shared ports."""
    base_port = int(base_port)
    array_task_id = int(array_task_id)
    attempt = int(attempt)
    if not 1 <= array_task_id <= 999:
        raise ValueError(f"array_task_id out of range: {array_task_id}")
    if attempt < 0:
        raise ValueError(f"attempt must be nonnegative: {attempt}")
    port = base_port + array_task_id + 1000 * attempt
    if not 1024 <= port <= 65535:
        raise ValueError(f"allocated port out of range: {port}")
    return port


def connect_policy_server(
    cfg: Any,
    *,
    base_port: int,
    array_task_id: int,
    attempts: int,
    is_port_open: Callable[..., bool],
    is_ready: Callable[..., bool],
    create_client: Callable[[Any], tuple],
    stop_process: Callable[[Any], None] | None = None,
):
    """Start only on an unused owned port and return only after readiness passes."""
    evidence: dict[str, Any] = {
        "base_port": int(base_port),
        "array_task_id": int(array_task_id),
        "max_attempts": int(attempts),
        "attempts": [],
    }
    for attempt in range(int(attempts)):
        port = port_for_attempt(base_port, array_task_id, attempt)
        row: dict[str, Any] = {"attempt": attempt + 1, "port": port}
        if is_port_open(cfg.host, port, timeout_sec=0.2):
            row.update({"status": "rejected", "reason": "port_preoccupied"})
            evidence["attempts"].append(row)
            continue
        attempt_cfg = replace(cfg, port=port)
        managed_process = None
        try:
            client, source, selected_config, managed_process = create_client(attempt_cfg)
            if not is_ready(attempt_cfg.host, port, timeout_sec=1.0):
                raise RuntimeError(f"policy server did not pass post-start readiness on port {port}")
            row.update({"status": "ready", "reason": "started_and_polled"})
            evidence["attempts"].append(row)
            evidence.update({"status": "pass", "selected_port": port,
                             "readiness_confirmed": True})
            return client, source, selected_config, managed_process, attempt_cfg, evidence
        except Exception as exc:
            if managed_process is not None and stop_process is not None:
                stop_process(managed_process)
            row.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
            evidence["attempts"].append(row)
    raise RuntimeError(
        f"policy server failed after {attempts} attempts: "
        + "; ".join(f"port {row['port']}: {row['reason']}" for row in evidence["attempts"])
    )


def assert_episode_transport_clean(log_text: str, inference_errors: list[str]) -> None:
    """Reject swallowed evaluator transport errors before another episode starts."""
    if "Episode error:" in str(log_text):
        raise ConnectionError(
            next((line for line in str(log_text).splitlines() if "Episode error:" in line),
                 "Episode error")
        )
    if inference_errors:
        raise ConnectionError(str(inference_errors[-1]))


def fail_cell_atomically(report: dict[str, Any], exc: Exception) -> dict[str, Any]:
    """Keep audit evidence but discard every partial scientific product."""
    for key in ("successes", "cost", "metric_decomposition", "retrieval", "adapter"):
        report.pop(key, None)
    report.update({
        "status": "fail",
        "episodes": [],
        "episodes_completed": 0,
        "partial_products_discarded": True,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    })
    return report


def install_evaluator_patches(evaluator: Any, *, timeout_floor_sec: float, log_dir) -> None:
    """Make the auto-started OpenPI policy server deterministic, logged and patient.

    * the readiness wait gets a floor (``timeout_floor_sec``): first JAX compile of a
      pi0.5/pi0 server can exceed the evaluator's default;
    * the server child runs with ``--xla_gpu_deterministic_ops=true`` and a fixed cuBLAS
      workspace; with these, a cell is bit-reproducible on a fixed GPU model;
    * the server does not preallocate the whole GPU (a second server on the same GPU
      would otherwise hang if teardown is slow);
    * server stdout/stderr go to ``<log_dir>/server_port<port>.log``.
    """
    import os
    import subprocess
    from pathlib import Path

    floor = float(timeout_floor_sec)
    original_wait = evaluator._wait_for_policy_server_ready

    def wait_with_floor(host, port, timeout_sec, poll_interval_sec, process):
        return original_wait(host, port, max(float(timeout_sec), floor), poll_interval_sec, process)

    def start_with_capture(cmd):
        port = cmd[cmd.index("--port") + 1] if "--port" in cmd else "unknown"
        path = Path(log_dir) / f"server_port{port}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        child_env = os.environ.copy()
        flag = "--xla_gpu_deterministic_ops=true"
        child_env["XLA_FLAGS"] = (child_env.get("XLA_FLAGS", "") + " " + flag).strip()
        child_env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        child_env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        child_env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
        log = path.open("ab", buffering=0)
        return subprocess.Popen(cmd, start_new_session=True, env=child_env, stdout=log,
                                stderr=subprocess.STDOUT)

    evaluator._wait_for_policy_server_ready = wait_with_floor
    evaluator._start_policy_server_process = start_with_capture
