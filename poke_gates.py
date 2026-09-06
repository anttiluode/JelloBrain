#!/usr/bin/env python3
"""S9 falsification gate: can transient POKE bootstrap a permanent spatial road?"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from poke_launcher import train_mismatch_poke


def _summary(runs: list[dict]) -> dict:
    keys = [
        "final_raw_greedy_after_alarm_erased",
        "first_100_reward",
        "last_100_reward",
        "first_100_poke_rate",
        "last_100_poke_rate",
        "total_poke_rate",
    ]
    out = {}
    for key in keys:
        out[key + "_mean"] = float(np.mean([r[key] for r in runs]))
        if key == "final_raw_greedy_after_alarm_erased":
            out[key + "_all"] = [r[key] for r in runs]
    return out


def gate_s9(n_seeds: int = 12) -> dict:
    repaired = [train_mismatch_poke(seed) for seed in range(n_seeds)]
    no_write = [
        train_mismatch_poke(seed, write_material=False)
        for seed in range(n_seeds)
    ]
    no_consequence_alarm = [
        train_mismatch_poke(seed, update_alarm_from_consequence=False)
        for seed in range(n_seeds)
    ]
    always = [
        train_mismatch_poke(seed, always_explore=True)
        for seed in range(n_seeds)
    ]

    a = _summary(repaired)
    b = _summary(no_write)
    c = _summary(no_consequence_alarm)
    d = _summary(always)

    return {
        "gate": "S9_TRANSIENT_POKE_BOOTSTRAPS_SPATIAL_ROUTE",
        "question": (
            "Can a consequence-triggered exploratory POKE temporarily open alternatives, "
            "write the successful route into JelloWorld, and then disappear without storing the policy?"
        ),
        "n_seeds": n_seeds,
        "mismatch_poke_plus_write": a,
        "no_material_write_attacker": b,
        "no_consequence_alarm_attacker": c,
        "always_explore_baseline": d,
        "pass": bool(
            a["final_raw_greedy_after_alarm_erased_mean"] >= 0.95
            and a["last_100_reward_mean"] >= 0.90
            and a["last_100_poke_rate_mean"] <= 0.15
            and b["final_raw_greedy_after_alarm_erased_mean"] <= 0.05
            and c["final_raw_greedy_after_alarm_erased_mean"] <= 0.05
            and d["final_raw_greedy_after_alarm_erased_mean"] >= 0.95
            and d["last_100_poke_rate_mean"] >= 0.60
        ),
        "interpretation": (
            "The transient controller stores only 'my current launch failed', not the correct output identity. "
            "If S9 passes, consequence-triggered exploration can bootstrap real spatial learning; after the alarm is erased, "
            "the repaired policy remains in the body's material. Always-explore is an efficiency control rather than a stronger learner."
        ),
    }


def run_all_gates(n_seeds: int = 12) -> dict:
    gates = [gate_s9(n_seeds)]
    return {
        "schema": "jellobrain/poke-bootstrap-gates-v1",
        "claim_boundary": (
            "S9 tests a transient mismatch-triggered exploration controller on one spatial JelloWorld. "
            "It is not a biological chandelier-cell model, does not establish cognition, and does not yet implement two spatial communicators."
        ),
        "gates": gates,
        "all_pass": bool(all(g["pass"] for g in gates)),
    }


def write_receipt(path: str | Path = "results/poke_gates.json", n_seeds: int = 12) -> dict:
    receipt = run_all_gates(n_seeds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    print(json.dumps(write_receipt(), indent=2))
