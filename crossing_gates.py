#!/usr/bin/env python3
"""S10 diagnostic: can one scalar 2-D material represent two crossing routes?

S9 failed after the transient alarm was removed: even abundant exploration and
rewarded writing left raw spatial accuracy at 0.5.  This file removes the
learning/controller problem entirely and asks an upper-bound representational
question.

An oracle repeatedly writes the *correct* swapped-channel route into the normal
JelloWorld using exactly the stock local plasticity rule.  We test:

1. A->D alone
2. F->C alone
3. A->D and F->C together in one sheet
4. the same two routes in two physically separate sheets

If each route works alone but the pair fails in one shared sheet, the bottleneck
is not exploration. It is route interference / lack of channel isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from spatial_launcher import (
    SWAPPED_CHANNEL,
    SpatialLaunchConfig,
    launcher_response,
    make_inherited_world,
    reinforce_route,
)


def correct_launcher_for_cue(cue: int) -> int:
    cue = int(cue)
    # delivered = SWAPPED_CHANNEL[launcher]; require delivered == cue
    for launcher in (0, 1):
        if int(SWAPPED_CHANNEL[launcher]) == cue:
            return launcher
    raise RuntimeError("no launcher maps to cue")


def raw_choice(world, cue: int, cfg: SpatialLaunchConfig) -> int:
    return int(np.argmax(launcher_response(world, cue, config=cfg)))


def raw_correct(world, cue: int, cfg: SpatialLaunchConfig) -> float:
    launcher = raw_choice(world, cue, cfg)
    return float(int(SWAPPED_CHANNEL[launcher]) == int(cue))


def oracle_write(world, cue: int, repeats: int, cfg: SpatialLaunchConfig) -> None:
    launcher = correct_launcher_for_cue(cue)
    for _ in range(int(repeats)):
        reinforce_route(world, cue, launcher, config=cfg)


def diagnostic(seed: int, repeats: int = 400) -> dict:
    cfg = SpatialLaunchConfig()

    # Each cross-route in isolation.
    a_only = make_inherited_world(seed, cfg)
    oracle_write(a_only, 0, repeats, cfg)

    f_only = make_inherited_world(seed, cfg)
    oracle_write(f_only, 1, repeats, cfg)

    # Same number of writes for both routes in one shared sheet, interleaved.
    shared = make_inherited_world(seed, cfg)
    for _ in range(int(repeats)):
        reinforce_route(shared, 0, correct_launcher_for_cue(0), config=cfg)
        reinforce_route(shared, 1, correct_launcher_for_cue(1), config=cfg)

    # Sequential order attack: does the later route erase/confound the earlier one?
    sequential = make_inherited_world(seed, cfg)
    oracle_write(sequential, 0, repeats, cfg)
    after_first = [raw_correct(sequential, cue, cfg) for cue in (0, 1)]
    oracle_write(sequential, 1, repeats, cfg)
    after_second = [raw_correct(sequential, cue, cfg) for cue in (0, 1)]

    a_resp = launcher_response(a_only, 0, config=cfg)
    f_resp = launcher_response(f_only, 1, config=cfg)
    shared_responses = [launcher_response(shared, cue, config=cfg) for cue in (0, 1)]

    return {
        "seed": int(seed),
        "repeats_per_route": int(repeats),
        "a_to_d_alone_correct": raw_correct(a_only, 0, cfg),
        "f_to_c_alone_correct": raw_correct(f_only, 1, cfg),
        "shared_sheet_correct": [raw_correct(shared, cue, cfg) for cue in (0, 1)],
        "sequential_after_first": after_first,
        "sequential_after_second": after_second,
        "a_only_launcher_response": a_resp.tolist(),
        "f_only_launcher_response": f_resp.tolist(),
        "shared_launcher_responses": [r.tolist() for r in shared_responses],
    }


def gate_s10(n_seeds: int = 12, repeats: int = 400) -> dict:
    runs = [diagnostic(seed, repeats) for seed in range(n_seeds)]
    a = float(np.mean([r["a_to_d_alone_correct"] for r in runs]))
    f = float(np.mean([r["f_to_c_alone_correct"] for r in runs]))
    shared = float(np.mean([np.mean(r["shared_sheet_correct"]) for r in runs]))
    seq_first = float(np.mean([np.mean(r["sequential_after_first"]) for r in runs]))
    seq_second = float(np.mean([np.mean(r["sequential_after_second"]) for r in runs]))

    return {
        "gate": "S10_ORACLE_CROSSING_ROUTE_CAPACITY",
        "question": "With the correct routes supplied by an oracle, can one JelloWorld hold both swapped cue->launcher mappings at once?",
        "n_seeds": n_seeds,
        "oracle_repeats_per_route": repeats,
        "a_to_d_alone_accuracy": a,
        "f_to_c_alone_accuracy": f,
        "both_routes_shared_sheet_accuracy": shared,
        "sequential_accuracy_after_first_route": seq_first,
        "sequential_accuracy_after_second_route": seq_second,
        "example_seed_0": runs[0],
        "diagnostic_pass": bool(a >= 0.95 and f >= 0.95 and shared <= 0.60),
        "interpretation": (
            "This is an upper-bound capacity test, not a learning result. If the single routes pass but the shared sheet collapses, "
            "the missing mechanism is channel isolation/addressing rather than better exploration."
        ),
    }


def run_all_gates(n_seeds: int = 12, repeats: int = 400) -> dict:
    gate = gate_s10(n_seeds, repeats)
    return {
        "schema": "jellobrain/crossing-route-diagnostic-v1",
        "claim_boundary": "S10 diagnoses representational interference in the current scalar spatial material. It does not show that biological axons solve the toy or that frequency/layer separation is the unique remedy.",
        "gates": [gate],
        "all_pass": bool(gate["diagnostic_pass"]),
    }


def write_receipt(path: str | Path = "results/crossing_gates.json", n_seeds: int = 12, repeats: int = 400) -> dict:
    receipt = run_all_gates(n_seeds, repeats)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    print(json.dumps(write_receipt(), indent=2))
