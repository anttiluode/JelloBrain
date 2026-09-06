#!/usr/bin/env python3
"""S11: robustness audit for the balanced two-route spatial solution.

S10 falsified the simplest capacity story: if the *correct* crossing routes are
written in balanced/interleaved fashion, one scalar JelloWorld can formally
choose both correct launchers.

But the S10 launcher responses were almost exact ties.  S11 asks whether that
solution has usable margin or whether it exists only on a numerical knife-edge.

The diagnostic trains:
  - A->D alone
  - F->C alone
  - both crossing routes interleaved in one sheet

Then it measures the correct-vs-wrong launcher margin and adds tiny independent
readout noise before argmax.  This is an upper-bound representational audit,
not a learning model and not a biological-noise claim.
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
from crossing_gates import correct_launcher_for_cue


def oracle_write(world, cue: int, repeats: int, cfg: SpatialLaunchConfig) -> None:
    launcher = correct_launcher_for_cue(cue)
    for _ in range(int(repeats)):
        reinforce_route(world, cue, launcher, config=cfg)


def correct_margin(response: np.ndarray, cue: int) -> float:
    correct = correct_launcher_for_cue(cue)
    wrong = 1 - correct
    return float(response[correct] - response[wrong])


def noisy_accuracy(
    response: np.ndarray,
    cue: int,
    rng: np.random.Generator,
    *,
    sigma: float,
    samples: int,
) -> float:
    response = np.asarray(response, dtype=np.float64)
    noisy = response[None, :] + rng.normal(0.0, float(sigma), size=(int(samples), 2))
    chosen = np.argmax(noisy, axis=1)
    delivered = np.asarray([SWAPPED_CHANNEL[int(k)] for k in chosen], dtype=np.int64)
    return float(np.mean(delivered == int(cue)))


def diagnostic(
    seed: int,
    *,
    repeats: int = 400,
    noise_sigma: float = 1e-3,
    noise_samples: int = 4000,
) -> dict:
    cfg = SpatialLaunchConfig()
    rng = np.random.default_rng(1_700_003 + int(seed))

    a_only = make_inherited_world(seed, cfg)
    oracle_write(a_only, 0, repeats, cfg)

    f_only = make_inherited_world(seed, cfg)
    oracle_write(f_only, 1, repeats, cfg)

    shared = make_inherited_world(seed, cfg)
    for _ in range(int(repeats)):
        reinforce_route(shared, 0, correct_launcher_for_cue(0), config=cfg)
        reinforce_route(shared, 1, correct_launcher_for_cue(1), config=cfg)

    a_resp = launcher_response(a_only, 0, config=cfg)
    f_resp = launcher_response(f_only, 1, config=cfg)
    shared_resp = [launcher_response(shared, cue, config=cfg) for cue in (0, 1)]

    single_margins = [correct_margin(a_resp, 0), correct_margin(f_resp, 1)]
    shared_margins = [correct_margin(shared_resp[cue], cue) for cue in (0, 1)]

    single_noisy = [
        noisy_accuracy(a_resp, 0, rng, sigma=noise_sigma, samples=noise_samples),
        noisy_accuracy(f_resp, 1, rng, sigma=noise_sigma, samples=noise_samples),
    ]
    shared_noisy = [
        noisy_accuracy(shared_resp[cue], cue, rng, sigma=noise_sigma, samples=noise_samples)
        for cue in (0, 1)
    ]

    return {
        "seed": int(seed),
        "oracle_repeats_per_route": int(repeats),
        "noise_sigma": float(noise_sigma),
        "noise_samples_per_cue": int(noise_samples),
        "single_route_margins": single_margins,
        "shared_route_margins": shared_margins,
        "single_route_noisy_accuracies": single_noisy,
        "shared_route_noisy_accuracies": shared_noisy,
        "shared_launcher_responses": [r.tolist() for r in shared_resp],
    }


def gate_s11(
    n_seeds: int = 12,
    *,
    repeats: int = 400,
    noise_sigma: float = 1e-3,
    noise_samples: int = 4000,
) -> dict:
    runs = [
        diagnostic(
            seed,
            repeats=repeats,
            noise_sigma=noise_sigma,
            noise_samples=noise_samples,
        )
        for seed in range(int(n_seeds))
    ]

    single_margins = np.asarray([m for r in runs for m in r["single_route_margins"]])
    shared_margins = np.asarray([m for r in runs for m in r["shared_route_margins"]])
    single_noisy = np.asarray([a for r in runs for a in r["single_route_noisy_accuracies"]])
    shared_noisy = np.asarray([a for r in runs for a in r["shared_route_noisy_accuracies"]])

    mean_response = float(
        np.mean([v for r in runs for pair in r["shared_launcher_responses"] for v in pair])
    )
    relative_noise = float(noise_sigma / mean_response)

    gate = {
        "gate": "S11_SHARED_ROUTE_MARGIN_ROBUSTNESS",
        "question": (
            "Does the formally correct two-route shared-sheet solution retain a robust launcher margin, "
            "or does tiny readout perturbation collapse it toward chance?"
        ),
        "n_seeds": int(n_seeds),
        "oracle_repeats_per_route": int(repeats),
        "readout_noise_sigma": float(noise_sigma),
        "readout_noise_fraction_of_mean_response": relative_noise,
        "single_route": {
            "mean_correct_margin": float(np.mean(single_margins)),
            "min_correct_margin": float(np.min(single_margins)),
            "mean_noisy_accuracy": float(np.mean(single_noisy)),
        },
        "shared_two_route": {
            "mean_correct_margin": float(np.mean(shared_margins)),
            "min_correct_margin": float(np.min(shared_margins)),
            "max_correct_margin": float(np.max(shared_margins)),
            "mean_noisy_accuracy": float(np.mean(shared_noisy)),
        },
        "margin_ratio_shared_over_single": float(
            np.mean(np.abs(shared_margins)) / max(np.mean(np.abs(single_margins)), 1e-15)
        ),
        "example_seed_0": runs[0],
    }

    # This audit passes if it demonstrates a large single-route margin but a
    # knife-edge shared solution whose tiny perturbation behaves near chance.
    gate["diagnostic_pass"] = bool(
        gate["single_route"]["min_correct_margin"] >= 0.10
        and gate["single_route"]["mean_noisy_accuracy"] >= 0.99
        and abs(gate["shared_two_route"]["mean_correct_margin"]) <= 1e-3
        and 0.45 <= gate["shared_two_route"]["mean_noisy_accuracy"] <= 0.55
        and gate["margin_ratio_shared_over_single"] <= 0.01
    )
    gate["interpretation"] = (
        "S10's 1.0 shared-sheet argmax is not yet a robust representation if its margin collapses by orders of magnitude. "
        "A tiny perturbation can then erase the apparent two-route solution even though each route is strong in isolation. "
        "That points toward interference / insufficient separation at the launch boundary rather than simple lack of exploration."
    )
    return gate


def run_all_gates(n_seeds: int = 12) -> dict:
    gate = gate_s11(n_seeds)
    return {
        "schema": "jellobrain/shared-route-robustness-v1",
        "claim_boundary": (
            "S11 probes numerical/functional margin in the current toy readout. It does not establish biological noise levels, "
            "nor does it prove that a particular mechanism such as frequency channels, inhibition, or axonal compartmentation is required."
        ),
        "gates": [gate],
        "all_pass": bool(gate["diagnostic_pass"]),
    }


def write_receipt(path: str | Path = "results/robustness_gates.json", n_seeds: int = 12) -> dict:
    receipt = run_all_gates(n_seeds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    print(json.dumps(write_receipt(), indent=2))
