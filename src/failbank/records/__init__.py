"""Stage 1 learning records: on-policy capture, schema, and content-addressed blobs.

One episode is written atomically under ``<root>/episodes/<shard>/<episode_id>/``
(``episode.json``, ``raw_steps.jsonl``, ``COMPLETE``); arrays go to ``<root>/blobs/`` keyed
by SHA-256. Readers must require the COMPLETE marker.

Field-name trap: ``RawStep.executed_action`` holds the teacher's proposal ~a_t, which is the
learning target. It is NOT what the environment executed during observe-only collection;
that is ``nominal_action``. The name is kept so that existing record banks stay readable.
"""

from .candidate_qp import CandidateSpec, evaluate_projection_candidates
from .input_capture import InferCapture
from .progress_capture import GoalExtractionError, ProgressCapture
from .recorder import EpisodeRecorder
from .schema_v1 import (
    EpisodeRaw,
    ProjectionCandidateRaw,
    RawStep,
    SCHEMA_VERSION,
)

__all__ = [
    "CandidateSpec",
    "EpisodeRaw",
    "EpisodeRecorder",
    "GoalExtractionError",
    "InferCapture",
    "ProgressCapture",
    "ProjectionCandidateRaw",
    "RawStep",
    "SCHEMA_VERSION",
    "evaluate_projection_candidates",
]
