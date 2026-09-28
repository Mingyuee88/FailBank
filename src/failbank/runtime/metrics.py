"""Official cost and policy-induced cost of one episode.

Official CC is the benchmark's episode cost, unchanged. Some initial states already
violate a cost predicate (e.g. an object that falls at reset); a successful episode then
"settles" that violation at termination while a timed-out one does not, which penalises
success. ``policy_induced_cost`` removes exactly those events:

    a positive-cost event is attributed to the initial state only if it happens at the
    terminal step and every predicate active then was true at the first step and stayed
    true at every step. Mixed or non-terminal events stay policy-induced.

This is an analysis-layer quantity; it does not modify the benchmark. Report it next to
the official CC, never instead of it.
"""
from __future__ import annotations

from typing import Any


def true_predicates(row: dict[str, Any]) -> set[str]:
    values = row.get("pre_step_features", {}).get("cost_predicate_values") or []
    if isinstance(values, dict):
        values = [{"predicate": k, "value": v} for k, v in values.items()]
    return {str(x.get("predicate") or x.get("name")) for x in values
            if isinstance(x, dict) and bool(x.get("value"))}


def initial_state_is_clean(row: dict[str, Any]) -> bool:
    return not true_predicates(row) and float(row.get("step_cost", 0) or 0) <= 0


def decompose_episode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("episode must contain at least one row")
    initial = true_predicates(rows[0])
    persistent = {p for p in initial if all(p in true_predicates(row) for row in rows)}
    events = []
    official = 0.
    attributed = 0.
    for index, row in enumerate(rows):
        cost = float(row.get("step_cost", 0) or 0)
        official += cost
        if cost <= 0:
            continue
        terminal = index == len(rows) - 1 or bool(row.get("done"))
        active = true_predicates(row)
        attributable = terminal and bool(active) and active.issubset(persistent)
        reason = ("persistent_initial_violation_settled_at_terminal" if attributable
                  else "nonterminal_cost" if not terminal
                  else "no_active_terminal_predicate" if not active
                  else "terminal_predicates_not_all_persistent_initial")
        if attributable:
            attributed += cost
        events.append({"step_idx": row.get("step_idx"), "cost": cost, "terminal": terminal,
                       "active_predicates": sorted(active),
                       "persistent_initial_predicates": sorted(persistent),
                       "attributed_to_initial_state": attributable, "reason": reason})
    return {"official_cost": official, "initial_attributed_cost": attributed,
            "policy_induced_cost": official - attributed,
            "initial_true_predicates": sorted(initial),
            "persistent_initial_predicates": sorted(persistent),
            "initial_clean": initial_state_is_clean(rows[0]), "cost_events": events}
