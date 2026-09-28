# Derived from VLA-Arena's OpenPI data loader (vla_arena/models/openpi/src/openpi/training/,
# Apache-2.0); loads FailBank learning records instead of a LeRobot dataset.
#
# Stage 4 record dataset/loader (weighted, positive-anchored batches).
#
# Reuses the SAME OpenPI transform pipeline the base model was finetuned with:
#   repack_transforms -> data_transforms(LiberoInputs) -> Normalize(norm_stats,
#   use_quantiles=True) -> model_transforms.
# Recorded nominal/executed actions are in environment/dataset action space
# (= Unnormalize(model_out)), so applying the standard Normalize is CORRECT.
# The raw dataset therefore emits the PRE-repack LeRobot column keys
#   image, wrist_image, state, actions, prompt
# which RepackTransform maps to observation/image, observation/wrist_image,
# observation/state, actions, prompt for LiberoInputs.
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
from collections.abc import Iterator, Sequence
from typing import Any

import flax.nnx as nnx  # noqa: F401  (kept for downstream typing parity)
import jax
import numpy as np
import vla_arena.models.openpi.src.openpi.models.model as _model
import vla_arena.models.openpi.src.openpi.training.data_loader as _data_loader
import torch
from etils import epath
from torch.utils.data import BatchSampler, DataLoader, Dataset


_REQUIRED_REAL_IMAGE_KEYS = frozenset({"base_0_rgb", "left_wrist_0_rgb"})
_EXPECTED_IMAGE_KEYS = frozenset(
    {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
)


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object in {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _blob_path(records_root: pathlib.Path, reference: dict[str, Any]) -> pathlib.Path:
    relative_path = reference.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"Invalid blob reference: {reference!r}")
    path = records_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_blob(
    records_root: pathlib.Path,
    reference: dict[str, Any],
    *,
    verify_sha256: bool,
) -> np.ndarray:
    path = _blob_path(records_root, reference)
    if verify_sha256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_digest = reference.get("sha256") or path.stem
        if digest != expected_digest:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {expected_digest}, got {digest}"
            )
    return np.load(path, allow_pickle=False)


def create_derived_data_config(config, *, assets_dirs, asset_id: str):
    """Build the normalizing DataConfig from the finetuned checkpoint assets.

    Verified on cluster: overriding assets.asset_id and passing assets_dirs as
    the positional base loads norm_stats from assets_dirs/asset_id/norm_stats.json.
    """
    data_factory = config.data
    if not hasattr(data_factory, "assets"):
        raise AttributeError(
            "config.data has no 'assets' field; confirm the DataConfigFactory API"
        )
    asset_config = dataclasses.replace(data_factory.assets, asset_id=asset_id)
    data_factory = dataclasses.replace(data_factory, assets=asset_config)
    data_config = data_factory.create(epath.Path(assets_dirs), config.model)
    if data_config.norm_stats is None:
        raise ValueError(
            "Derived LoRA data requires norm_stats from finetuned assets "
            f"{assets_dirs!s} asset_id {asset_id!r}"
        )
    return data_config


class _RawDerivedRecordDataset(Dataset):
    """Emits PRE-repack LeRobot-style keys for one derived record."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        records_root: pathlib.Path,
        quiet_weight: float,
        verify_sha256: bool,
        prompt: str | None = None,
        action_horizon: int = 10,
    ):
        self._records = list(records)
        self._records_root = records_root
        self._quiet_weight = float(quiet_weight)
        self._verify_sha256 = bool(verify_sha256)
        # Derived records do not store the language instruction (build_derived
        # dropped it). It is fixed per task and recovered from raw_steps; the
        # caller injects it here when the record lacks prompt/instruction.
        self._prompt = prompt.strip() if isinstance(prompt, str) else None
        # Pi0.5 emits 10-step chunks, Pi0 emits 50. The chunk length must match the
        # model that produced the records; a mismatch means records and config were
        # paired wrongly, which is exactly what this check is for.
        self._action_horizon = int(action_horizon)
        if self._quiet_weight < 0:
            raise ValueError("quiet_weight must be non-negative")

    def __len__(self) -> int:
        return len(self._records)

    def weight(self, index: int) -> float:
        record = self._records[index]
        if bool(record.get("triggered", False)):
            teacher = record.get("teacher")
            if not isinstance(teacher, dict):
                raise ValueError(f"Triggered record {index} has no teacher object")
            weight = float(teacher["weight"])
        elif bool(record.get("quiet", False)):
            weight = self._quiet_weight
        else:
            weight = 0.0
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid record weight at index {index}: {weight}")
        return weight

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._records[index]
        observation_refs = record.get("observation_refs")
        if not isinstance(observation_refs, dict):
            raise ValueError(f"Record {index} has no observation_refs object")

        image = _load_blob(
            self._records_root, observation_refs["observation/image"],
            verify_sha256=self._verify_sha256,
        )
        wrist_image = _load_blob(
            self._records_root, observation_refs["observation/wrist_image"],
            verify_sha256=self._verify_sha256,
        )
        state = _load_blob(
            self._records_root, observation_refs["observation/state"],
            verify_sha256=self._verify_sha256,
        )

        action_chunk = np.asarray(
            _load_blob(
                self._records_root, record["nominal_action_chunk_ref"],
                verify_sha256=self._verify_sha256,
            ),
            dtype=np.float32,
        ).copy()
        expected_chunk = (self._action_horizon, 7)
        if action_chunk.shape != expected_chunk:
            raise ValueError(
                f"Record {index} action chunk shape {action_chunk.shape}, "
                f"expected {expected_chunk}"
            )

        if bool(record.get("triggered", False)):
            # `executed_action` is the TEACHER TARGET ~a_t (historical field name), not the
            # action the environment executed. It replaces only the chunk's first action;
            # the objective supervises that first action alone.
            executed_action = np.asarray(record["executed_action"], dtype=np.float32)
            if executed_action.shape != (7,):
                raise ValueError(
                    f"Record {index} executed_action shape {executed_action.shape}, expected (7,)"
                )
            action_chunk[0] = executed_action

        prompt = record.get("prompt") or record.get("instruction") or self._prompt
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Record {index} has no prompt/instruction and no --prompt was provided"
            )

        # PRE-repack LeRobot column keys. RepackTransform renames these to the
        # observation/* keys LiberoInputs consumes; do NOT prefix them here.
        return {
            "image": image,
            "wrist_image": wrist_image,
            "state": np.asarray(state, dtype=np.float32),
            # environment/dataset-space action chunk; transform_dataset applies
            # the standard Normalize -- do not clip or special-case any dim.
            "actions": action_chunk,
            "prompt": prompt,
        }


class DerivedRecordDataset(Dataset):
    """Derived records passed through the standard OpenPI transform pipeline."""

    def __init__(
        self,
        manifest_path,
        *,
        records_root,
        data_config,
        quiet_weight: float = 0.0,
        verify_sha256: bool = False,
        prompt: str | None = None,
        action_horizon: int = 10,
    ):
        self.manifest_path = pathlib.Path(manifest_path)
        self.records_root = pathlib.Path(records_root)
        records = _load_jsonl(self.manifest_path)
        self._raw_dataset = _RawDerivedRecordDataset(
            records,
            records_root=self.records_root,
            quiet_weight=quiet_weight,
            verify_sha256=verify_sha256,
            prompt=prompt,
            action_horizon=action_horizon,
        )
        self._transformed_dataset = _data_loader.transform_dataset(
            self._raw_dataset, data_config, skip_norm_stats=False
        )

    def __len__(self) -> int:
        return len(self._raw_dataset)

    def weight(self, index: int) -> float:
        return self._raw_dataset.weight(index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self._transformed_dataset[index])
        weight = self.weight(index)
        item["include_mask"] = np.float32(weight > 0)
        item["per_record_weight"] = np.float32(weight)
        return item


class PositiveBatchSampler(BatchSampler):
    """Finite-epoch sampler with >=1 positive-weight record per batch."""

    def __init__(self, dataset, *, batch_size: int, seed: int = 0, drop_last: bool = False):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0
        self._all_indices = np.arange(len(dataset), dtype=np.int64)
        self._positive_indices = np.asarray(
            [i for i in range(len(dataset)) if dataset.weight(i) > 0], dtype=np.int64
        )
        if self._positive_indices.size == 0:
            raise ValueError("Dataset has no included record with weight > 0")

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        num_batches = len(self)
        if num_batches == 0:
            return
        positive_order = rng.permutation(self._positive_indices)
        all_order = rng.permutation(self._all_indices)
        all_cursor = 0
        for batch_number in range(num_batches):
            anchor = int(positive_order[batch_number % len(positive_order)])
            batch = [anchor]
            while len(batch) < self.batch_size:
                if all_cursor >= len(all_order):
                    all_order = rng.permutation(self._all_indices)
                    all_cursor = 0
                batch.append(int(all_order[all_cursor]))
                all_cursor += 1
            rng.shuffle(batch)
            yield batch


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (np.number, int, float, bool)):
        return np.asarray(value)
    return value


class DerivedDataLoader:
    """Yields sharded (Observation, actions, include_mask, per_record_weight)."""

    def __init__(
        self,
        dataset: DerivedRecordDataset,
        *,
        batch_size: int,
        sharding,
        seed: int = 0,
        num_workers: int = 0,
        drop_last: bool = False,
        primary_image_key: str = "base_0_rgb",
        wrist_image_key: str = "left_wrist_0_rgb",
    ):
        self.dataset = dataset
        self.sharding = sharding
        self.primary_image_key = primary_image_key
        self.wrist_image_key = wrist_image_key
        sampler = PositiveBatchSampler(
            dataset, batch_size=batch_size, seed=seed, drop_last=drop_last
        )
        self._loader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=False
        )

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        for torch_batch in self._loader:
            local_batch = jax.tree.map(_to_numpy, torch_batch)

            include_mask = np.asarray(local_batch.pop("include_mask"), dtype=np.float32)
            per_record_weight = np.asarray(
                local_batch.pop("per_record_weight"), dtype=np.float32
            )
            positive_mass = float(np.sum(include_mask * per_record_weight))
            if positive_mass <= 0:
                raise ValueError(
                    "PositiveBatchSampler produced a batch with no weight>0 record"
                )

            actions = np.asarray(local_batch.pop("actions"), dtype=np.float32)
            observation = _model.Observation.from_dict(local_batch)

            image_keys = set(observation.images)
            if not _REQUIRED_REAL_IMAGE_KEYS.issubset(image_keys):
                raise ValueError(
                    "Transformed observation lacks required image keys: "
                    f"required={sorted(_REQUIRED_REAL_IMAGE_KEYS)}, actual={sorted(image_keys)}"
                )
            if image_keys != _EXPECTED_IMAGE_KEYS:
                raise ValueError(
                    f"Expected pi0 image keys {sorted(_EXPECTED_IMAGE_KEYS)}, got {sorted(image_keys)}"
                )
            if self.primary_image_key not in image_keys:
                raise ValueError(f"Missing primary image key {self.primary_image_key!r}")
            if self.wrist_image_key not in image_keys:
                raise ValueError(f"Missing wrist image key {self.wrist_image_key!r}")

            observation = jax.tree.map(
                lambda value: jax.make_array_from_process_local_data(self.sharding, value),
                observation,
            )
            actions = jax.make_array_from_process_local_data(self.sharding, actions)
            include_mask = jax.make_array_from_process_local_data(self.sharding, include_mask)
            per_record_weight = jax.make_array_from_process_local_data(
                self.sharding, per_record_weight
            )
            yield (observation, actions, include_mask, per_record_weight)
