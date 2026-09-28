"""Locate the OpenPI code that ships inside VLA-Arena.

VLA-Arena vendors OpenPI under ``vla_arena/models/openpi``. Its training entry point
``scripts/train.py`` is a script, not a module, so we load it by path under a private
name; FailBank only borrows ``init_train_state`` from it and never runs its ``main``.
Nothing here requires patching ``train.py``.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from functools import lru_cache


def openpi_root() -> pathlib.Path:
    import vla_arena

    root = pathlib.Path(vla_arena.__file__).resolve().parent / "models" / "openpi"
    if not (root / "scripts" / "train.py").is_file():
        raise FileNotFoundError(
            f"OpenPI training script not found under {root}; is VLA-Arena at commit 2ddcb00 "
            "on PYTHONPATH (see README, 'Install')?")
    return root


@lru_cache(maxsize=1)
def upstream_train_module():
    path = openpi_root() / "scripts" / "train.py"
    name = "_vla_arena_openpi_train"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
