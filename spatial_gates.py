#!/usr/bin/env python3
"""S0-S4 falsification gates for the spatial JelloBrain launch boundary."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from spatial_launcher import (
    NORMAL_CHANNEL,
    SWAPPED_CHANNEL,
    SpatialLaunchConfig,
    make_inherited_world,
    raw_greedy_accuracy,
    train_after_swap,
)


def gate_s0(n_seeds: int = 12) -> dict:
    scores = [raw_greedy_accuracy(make_inherited_world(seed), NORMAL_CHANNEL) for seed in range(n_seeds)]
    mean = float(np.mean(scores))
    return {
        "gate": "S0_INHERITED_GEOMETRIC_SHORTCUT",
        "question": "Does virgin spatial geometry already solve the unswapped pulse task?",
        "n_seeds": n_seeds,
        "accuracies": scores,
        "mean_accuracy": mean,
        "chance": 0.5,
        # S0 is an audit PASS when the confound is successfully demonstrated.
        "pass": bool(mean >= 0.95),
        "interpretation": "A perfect virgin score is a confound, not evidence of learning.",
    }


def gate_s1(n_seeds: int = 12) -> dict:
    normal = []
    swapped = []
    for seed in range(n_seeds):
        world = make_inherited_world(seed)
        normal.append(raw_greedy_accuracy(world, NORMAL_CHANNEL))
        swapped.append(raw_greedy_accuracy(world, SWAPPED_CHANNEL))
    mn = float(np.mean(normal)); ms = float(np.mean(swapped))
    return {
        "gate": "S1_CHANNEL_CAUSALITY",
        "question": "Does swapping the two physical channel wires destroy the inherited solution?",
        "normal_mean": mn,
        "swapped_mean": ms,
        "delta": mn - ms,
        "pass": bool(mn >= 0.95 and ms <= 0.05),
    }


def _summary(runs: list[dict]) -> dict:
    return {
        "greedy_mean": float(np.mean([r["greedy_accuracy"] for r in runs])),
        "greedy_all": [r["greedy_accuracy"] for r in runs],
        "first_50_reward_mean": float(np.mean([r["first_50_reward"] for r in runs])),
        "last_50_reward_mean": float(np.mean([r["last_50_reward"] for r in runs])),
        "last_100_reward_mean": float(np.mean([r["last_100_reward"] for r in runs])),
    }


def gate_s2(n_seeds: int = 12) -> dict:
    runs = [train_after_swap(seed, residual_gate=False) for seed in range(n_seeds)]
    s = _summary(runs)
    return {
        "gate": "S2_NAIVE_RAW_REPLAY_FAILS",
        "question": "Can raw-excitation choice plus rewarded spatial replay escape the inherited route after wire swap?",
        "n_seeds": n_seeds,
        **s,
        # This gate intentionally passes by reproducing the failure we need S3 to beat.
        "pass": bool(s["greedy_mean"] <= 0.05 and s["last_100_reward_mean"] <= 0.10),
        "interpretation": "Inherited excitation monopolizes the launcher, so rewarded counterfactual traffic is almost never sampled.",
    }


def gate_s3(n_seeds: int = 12) -> dict:
    repaired = [train_after_swap(seed, residual_gate=True) for seed in range(n_seeds)]
    no_write = [
        train_after_swap(seed, residual_gate=True, write_material=False)
        for seed in range(n_seeds)
    ]
    global_norm = [
        train_after_swap(seed, residual_gate=True, global_prediction_context=True)
        for seed in range(n_seeds)
    ]
    scrambled = [
        train_after_swap(seed, residual_gate=True, scramble_prediction_context=True)
        for seed in range(n_seeds)
    ]
    a = _summary(repaired); b = _summary(no_write); c = _summary(global_norm); d = _summary(scrambled)
    return {
        "gate": "S3_NEGATIVE_IMAGE_REPAIR",
        "question": "Does context-keyed observed-minus-predicted launch activity enable spatial repair after wire swap?",
        "n_seeds": n_seeds,
        "residual_plus_write": a,
        "residual_no_write_attacker": b,
        "global_prediction_attacker": c,
        "scrambled_prediction_context_attacker": d,
        "pass": bool(
            a["greedy_mean"] >= 0.95
            and a["last_100_reward_mean"] >= 0.90
            and b["last_100_reward_mean"] <= 0.60
            and c["greedy_mean"] <= 0.10
            and d["greedy_mean"] <= 0.10
        ),
        "interpretation": (
            "Subtraction alone is not the solution: the no-write attacker stays near chance. "
            "A global baseline also fails: cancellation must be keyed to the current corollary/context signal. "
            "The residual opens exploration/credit, and actual JelloWorld material writing builds the alternate road."
        ),
    }


def gate_s4(n_seeds: int = 12) -> dict:
    slow = [
        train_after_swap(seed, residual_gate=True, predictor_track_rate=0.0002)
        for seed in range(n_seeds)
    ]
    fast = [
        train_after_swap(seed, residual_gate=True, predictor_track_rate=0.20)
        for seed in range(n_seeds)
    ]
    a = _summary(slow); b = _summary(fast)
    return {
        "gate": "S4_PREDICTOR_TIMESCALE",
        "question": "Does an expectation model that tracks too fast erase the very innovation needed for repair?",
        "n_seeds": n_seeds,
        "slow_predictor": a,
        "fast_predictor_attacker": b,
        "pass": bool(a["greedy_mean"] >= 0.95 and b["greedy_mean"] <= 0.60),
        "interpretation": "The useful split depends on relative timescale: prediction must not instantly absorb newly relevant route changes.",
    }


def run_all_gates(n_seeds: int = 12) -> dict:
    gates = [gate_s0(n_seeds), gate_s1(n_seeds), gate_s2(n_seeds), gate_s3(n_seeds), gate_s4(n_seeds)]
    return {
        "schema": "jellobrain/spatial-launch-gates-v1",
        "claim_boundary": (
            "These gates isolate an axon-like launch boundary on one full spatial JelloWorld. "
            "They do not yet establish two full spatial communicators, biological corollary discharge, "
            "or an axon initial segment model."
        ),
        "gates": gates,
        "all_pass": bool(all(g["pass"] for g in gates)),
    }


def write_receipt(path: str | Path = "results/spatial_gates.json", n_seeds: int = 12) -> dict:
    receipt = run_all_gates(n_seeds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    print(json.dumps(write_receipt(), indent=2))
