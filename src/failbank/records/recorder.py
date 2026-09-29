"""Atomic Phase-1 episode recorder.

Rollout behavior is fail-open: append/finalize errors disable recording and are
returned as diagnostics. Dataset consumption is fail-closed: readers require a
valid COMPLETE marker and matching checksums.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .blobs import dump_npy
from .schema_v1 import (
    EpisodeRaw,
    RawStep,
    canonical_json_bytes,
    sha256_bytes,
    to_json_dict,
    to_json_line,
    validate_raw_step,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class EpisodeRecorder:
    def __init__(
        self,
        *,
        dataset_root: Path,
        episode: EpisodeRaw,
        enabled: bool,
    ) -> None:
        self.enabled = bool(enabled)
        self.dataset_root = Path(dataset_root)
        self.episode = episode
        self.disabled_reason: Optional[str] = None
        self.step_count = 0
        self._closed = False
        self._step_stream: Any = None

        shard = episode.episode_id[:2] or "00"
        self.final_dir = (
            self.dataset_root / "episodes" / shard / episode.episode_id
        )
        self.partial_dir = (
            self.dataset_root / ".partial" / shard / episode.episode_id
        )
        self.blob_root = self.dataset_root / "blobs"

        if self.enabled:
            try:
                if self.final_dir.exists():
                    raise FileExistsError(f"Episode already exists: {self.final_dir}")
                if self.partial_dir.exists():
                    shutil.rmtree(self.partial_dir)
                self.partial_dir.mkdir(parents=True)
                self.blob_root.mkdir(parents=True, exist_ok=True)
                self._step_stream = (
                    self.partial_dir / "raw_steps.jsonl"
                ).open("xb", buffering=0)
            except Exception as exc:
                self._disable(exc)

    def _disable(self, exc: BaseException) -> None:
        self.disabled_reason = (
            f"{type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc(limit=8)}"
        )
        self.enabled = False
        if self._step_stream is not None:
            try:
                self._step_stream.close()
            except Exception:
                pass
            self._step_stream = None

    def dump_array(self, array: np.ndarray) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Recorder is disabled")
        return dump_npy(np.asarray(array), self.blob_root)

    def append(self, step: RawStep) -> bool:
        if not self.enabled or self._closed:
            return False
        try:
            validate_raw_step(step)
            self._step_stream.write(to_json_line(step))
            self._step_stream.flush()
            os.fsync(self._step_stream.fileno())
            self.step_count += 1
            return True
        except Exception as exc:
            self._disable(exc)
            return False

    def finalize(
        self,
        *,
        result_path: Optional[Path] = None,
        result_value: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        if self._closed:
            return self.final_dir.exists()
        self._closed = True
        if not self.enabled:
            return False

        try:
            self._step_stream.flush()
            os.fsync(self._step_stream.fileno())
            self._step_stream.close()
            self._step_stream = None

            result_ref: Optional[dict[str, Any]] = None
            if result_path is not None and Path(result_path).exists():
                payload = Path(result_path).read_bytes()
                result_ref = {
                    "path": str(Path(result_path)),
                    "sha256": sha256_bytes(payload),
                    "source": "result.json",
                }
            elif result_value is not None:
                payload = canonical_json_bytes(result_value)
                result_file = self.partial_dir / "result.json"
                _atomic_write(result_file, payload + b"\n")
                result_ref = {
                    "path": "result.json",
                    "sha256": sha256_bytes(payload + b"\n"),
                    "source": "inline",
                }

            manifest = to_json_dict(self.episode)
            manifest["step_count"] = self.step_count
            manifest["result_ref"] = result_ref
            manifest["finalized_at_utc"] = _utc_now()
            _atomic_write(
                self.partial_dir / "episode.json",
                canonical_json_bytes(manifest) + b"\n",
            )

            checksums = {}
            for name in ("episode.json", "raw_steps.jsonl"):
                path = self.partial_dir / name
                checksums[name] = {
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            inline_result = self.partial_dir / "result.json"
            if inline_result.exists():
                checksums["result.json"] = {
                    "sha256": _file_sha256(inline_result),
                    "size_bytes": inline_result.stat().st_size,
                }

            complete = {
                "schema_version": self.episode.schema_version,
                "episode_id": self.episode.episode_id,
                "step_count": self.step_count,
                "checksums": checksums,
                "completed_at_utc": _utc_now(),
            }
            _atomic_write(
                self.partial_dir / "COMPLETE",
                canonical_json_bytes(complete) + b"\n",
            )

            self.final_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(self.partial_dir, self.final_dir)
            return True
        except Exception as exc:
            self._disable(exc)
            return False
