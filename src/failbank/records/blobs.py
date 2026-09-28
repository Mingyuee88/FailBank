"""Lossless, content-addressed ndarray storage."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _atomic_write_if_absent(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return

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
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def dump_npy(array: np.ndarray, blob_root: Path) -> dict[str, Any]:
    value = np.asarray(array)
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    relative = Path(digest[:2]) / f"{digest}.npy"
    _atomic_write_if_absent(blob_root / relative, payload)
    return {
        "kind": "npy",
        "sha256": digest,
        "path": relative.as_posix(),
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "nbytes": int(value.nbytes),
    }


def dump_npz(arrays: Mapping[str, np.ndarray], blob_root: Path) -> dict[str, Any]:
    normalized = {
        str(name): np.asarray(value)
        for name, value in sorted(arrays.items())
    }
    buffer = io.BytesIO()
    np.savez(buffer, **normalized)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    relative = Path(digest[:2]) / f"{digest}.npz"
    _atomic_write_if_absent(blob_root / relative, payload)
    return {
        "kind": "npz",
        "sha256": digest,
        "path": relative.as_posix(),
        "members": {
            name: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "nbytes": int(value.nbytes),
            }
            for name, value in normalized.items()
        },
    }


def verify_blob(ref: Mapping[str, Any], blob_root: Path) -> None:
    path = blob_root / str(ref["path"])
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != ref["sha256"]:
        raise ValueError(f"Blob checksum mismatch: {path}")
