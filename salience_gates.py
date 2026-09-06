#!/usr/bin/env python3
"""S5-S8 falsification gates: novelty, relevance, and ping economy."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from salience_gate import (
    run_protocol,
    relevance_ablation_after_learning,
)


def _mean(xs):
    return float(np.mean(xs))


def _grid_mean(runs: list[dict], key: str) -> float:
    return _mean([r["grid"][key] for r in runs])


def gate_s5(n_seeds: int = 24) -> dict:
    runs = [run_protocol(seed, mode="dual") for seed in range(n_seeds)]
    initial_ping = _mean([r["familiarization"]["first_12_ping_rate"] for r in runs])
    late_ping = _mean([r["familiarization"]["last_80_ping_rate"] for r in runs])
    initial_surprise = _mean([r["familiarization"]["first_12_surprise"] for r in runs])
    late_surprise = _mean([r["familiarization"]["last_80_surprise"] for r in runs])
    return {
        "gate": "S5_NOVELTY_HABITUATION",
        "question": "Can a fast expectation suppress repeated harmless pings without changing the event itself?",
        "n_seeds": n_seeds,
        "initial_ping_rate": initial_ping,
        "late_ping_rate": late_ping,
        "initial_surprise": initial_surprise,
        "late_surprise": late_surprise,
        "pass": bool(initial_ping >= 0.70 and late_ping <= 0.05 and late_surprise < initial_surprise * 0.20),
        "interpretation": "The event becomes expected, so novelty-driven launch demand collapses.",
    }


def gate_s6(n_seeds: int = 24) -> dict:
    dual = [run_protocol(seed, mode="dual") for seed in range(n_seeds)]
    surprise_only = [run_protocol(seed, mode="surprise") for seed in range(n_seeds)]
    ablated = [relevance_ablation_after_learning(seed) for seed in range(n_seeds)]

    target = _grid_mean(dual, "c0_e0")
    wrong = _grid_mean(dual, "c1_e0")
    other0 = _grid_mean(dual, "c0_e1")
    other1 = _grid_mean(dual, "c1_e1")
    surprise_target = _grid_mean(surprise_only, "c0_e0")
    ablated_target = _mean([r["after"]["c0_e0"] for r in ablated])
    dual_ping_rate = float(np.mean([target, wrong, other0, other1]))
    dual_precision = target / max(target + wrong + other0 + other1, 1e-12)

    return {
        "gate": "S6_FAMILIAR_BUT_RELEVANT",
        "question": "When a fully familiar event later becomes consequential, can a separate relevance state reactivate sparse pings after surprise is gone?",
        "n_seeds": n_seeds,
        "dual_gate": {
            "relevant_pair_ping_rate": target,
            "same_event_wrong_context_ping_rate": wrong,
            "other_event_c0_ping_rate": other0,
            "other_event_c1_ping_rate": other1,
            "uniform_ping_rate": dual_ping_rate,
            "ping_precision": dual_precision,
        },
        "surprise_only_attacker": {"relevant_pair_ping_rate": surprise_target},
        "relevance_ablation_attacker": {"relevant_pair_ping_rate_after_ablation": ablated_target},
        "always_ping_baseline": {"uniform_ping_rate": 1.0, "ping_precision": 0.25},
        "pass": bool(target >= 0.95 and wrong <= 0.05 and other0 <= 0.05 and other1 <= 0.05 and surprise_target <= 0.05 and ablated_target <= 0.05 and dual_ping_rate <= 0.30 and dual_precision >= 0.95),
        "interpretation": "Expected does not mean irrelevant. Surprise-only gating misses the familiar important event; learned relevance restores launch priority without restoring indiscriminate pinging.",
    }


def gate_s7(n_seeds: int = 24) -> dict:
    local = [run_protocol(seed, mode="dual", context_specific_relevance=True) for seed in range(n_seeds)]
    global_attacker = [run_protocol(seed, mode="dual", context_specific_relevance=False) for seed in range(n_seeds)]
    local_target = _grid_mean(local, "c0_e0")
    local_wrong = _grid_mean(local, "c1_e0")
    global_target = _grid_mean(global_attacker, "c0_e0")
    global_wrong = _grid_mean(global_attacker, "c1_e0")
    return {
        "gate": "S7_CONTEXTUAL_RELEVANCE",
        "question": "Is relevance tied to event identity alone, or must it be keyed to the current context?",
        "n_seeds": n_seeds,
        "context_specific": {"target_ping_rate": local_target, "same_event_wrong_context_ping_rate": local_wrong},
        "global_relevance_attacker": {"target_ping_rate": global_target, "same_event_wrong_context_ping_rate": global_wrong},
        "pass": bool(local_target >= 0.95 and local_wrong <= 0.05 and global_target >= 0.80 and global_wrong >= 0.80),
        "interpretation": "A global event-level salience scalar overgeneralizes. The same familiar event must be launch-worthy in one context and quiet in another.",
    }


def gate_s8(n_seeds: int = 24) -> dict:
    fast_dual = [run_protocol(seed, mode="dual", prediction_rate=0.80, context_specific_relevance=True) for seed in range(n_seeds)]
    fast_surprise = [run_protocol(seed, mode="surprise", prediction_rate=0.80, context_specific_relevance=True) for seed in range(n_seeds)]
    dual_target = _grid_mean(fast_dual, "c0_e0")
    dual_wrong = _grid_mean(fast_dual, "c1_e0")
    surprise_target = _grid_mean(fast_surprise, "c0_e0")
    return {
        "gate": "S8_FAST_EXPECTATION_WITH_SEPARATE_RELEVANCE",
        "question": "Does fast prediction still destroy useful gating when relevance is stored separately from prediction error?",
        "n_seeds": n_seeds,
        "prediction_rate": 0.80,
        "dual_gate_relevant_ping_rate": dual_target,
        "dual_gate_wrong_context_ping_rate": dual_wrong,
        "surprise_only_relevant_ping_rate": surprise_target,
        "pass": bool(dual_target >= 0.95 and dual_wrong <= 0.05 and surprise_target <= 0.05),
        "interpretation": "S4's failure was specific to a surprise-only launcher. A fast expectation can coexist with persistent useful output if learned relevance is a different variable. The stronger hypothesis is separation of roles, not simply 'prediction must always be slow.'",
    }


def run_all_gates(n_seeds: int = 24) -> dict:
    gates = [gate_s5(n_seeds), gate_s6(n_seeds), gate_s7(n_seeds), gate_s8(n_seeds)]
    return {
        "schema": "jellobrain/salience-launch-gates-v1",
        "claim_boundary": "S5-S8 are a minimal launch-boundary isolator, not a biological chandelier-cell model and not a full spatial communication task. They test the computational distinction between surprise, contextual relevance, and sparse outward pings.",
        "gates": gates,
        "all_pass": bool(all(g["pass"] for g in gates)),
    }


def write_receipt(path: str | Path = "results/salience_gates.json", n_seeds: int = 24) -> dict:
    receipt = run_all_gates(n_seeds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    print(json.dumps(write_receipt(), indent=2))
