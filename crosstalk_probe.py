#!/usr/bin/env python3
"""S16X: measure the write-to-other-context crosstalk directly.

The JelloBrain reversal failures look like a credit problem from the boundary,
but S15 says explicit slow-state separation removes them without smarter credit.
This diagnostic asks for the smallest mathematical witness of that interference.

For cue i define its signed launcher margin under the SWAPPED channel as

    m_i = response(correct launcher) - response(wrong launcher).

After constructing a two-route solution, apply ONE additional rewarded material
write for cue j and measure every margin again:

    C[i,j] = m_i(after write j) - m_i(before).

C is the local cross-write Jacobian in finite-difference form.  Its diagonal is
self-reinforcement.  Its off-diagonal entries are collateral effects on another
context.

Two conditions use identical route-writing primitives:

- SHARED: both mappings inhabit one JelloWorld sheet.
- SEPARATED: each cue inhabits an independent anonymous slow band; this is the
  S15 upper-bound architecture, so cross-band off-diagonals should vanish by
  construction.

This is a representational-interference diagnostic, not a biological Jacobian
or a claim that real brains use separate bands.
"""

from __future__ import annotations

import copy
import json
import numpy as np

from spatial_launcher import (
    SWAPPED_CHANNEL,
    SpatialLaunchConfig,
    launcher_response,
    make_inherited_world,
    reinforce_route,
)


def correct_launcher(cue: int) -> int:
    # SWAPPED_CHANNEL[launcher] == cue
    return int(SWAPPED_CHANNEL.index(int(cue)))


def margin(world, cue: int, scfg: SpatialLaunchConfig) -> float:
    r = launcher_response(world, cue, config=scfg)
    c = correct_launcher(cue)
    w = 1 - c
    return float(r[c] - r[w])


def shared_world(seed: int, repeats: int, scfg: SpatialLaunchConfig):
    world = make_inherited_world(seed, scfg)
    # Alternate writes so neither route receives a recency advantage.
    for _ in range(repeats):
        reinforce_route(world, 0, correct_launcher(0), config=scfg)
        reinforce_route(world, 1, correct_launcher(1), config=scfg)
    return world


def separated_worlds(seed: int, repeats: int, scfg: SpatialLaunchConfig):
    # Start from exact copies so separation, not different initial geometry, is
    # the only representational difference.
    base = make_inherited_world(seed, scfg)
    bands = [copy.deepcopy(base), copy.deepcopy(base)]
    for cue in (0, 1):
        for _ in range(repeats):
            reinforce_route(bands[cue], cue, correct_launcher(cue), config=scfg)
    return bands


def shared_matrix(seed: int, repeats: int, scfg: SpatialLaunchConfig) -> dict:
    world = shared_world(seed, repeats, scfg)
    before = np.array([margin(world, cue, scfg) for cue in (0, 1)])
    C = np.zeros((2, 2), dtype=np.float64)
    afters = []
    for write_cue in (0, 1):
        probe = copy.deepcopy(world)
        reinforce_route(
            probe, write_cue, correct_launcher(write_cue), config=scfg
        )
        after = np.array([margin(probe, cue, scfg) for cue in (0, 1)])
        C[:, write_cue] = after - before
        afters.append(after.tolist())
    return {
        "before_margins": before.tolist(),
        "after_one_write": afters,
        "cross_write_matrix": C.tolist(),
    }


def separated_matrix(seed: int, repeats: int, scfg: SpatialLaunchConfig) -> dict:
    bands = separated_worlds(seed, repeats, scfg)
    before = np.array([margin(bands[cue], cue, scfg) for cue in (0, 1)])
    C = np.zeros((2, 2), dtype=np.float64)
    afters = []
    for write_cue in (0, 1):
        probe = [copy.deepcopy(b) for b in bands]
        reinforce_route(
            probe[write_cue],
            write_cue,
            correct_launcher(write_cue),
            config=scfg,
        )
        after = np.array([margin(probe[cue], cue, scfg) for cue in (0, 1)])
        C[:, write_cue] = after - before
        afters.append(after.tolist())
    return {
        "before_margins": before.tolist(),
        "after_one_write": afters,
        "cross_write_matrix": C.tolist(),
    }


def summarize(runs: list[dict]) -> dict:
    mats = np.asarray([r["cross_write_matrix"] for r in runs], dtype=np.float64)
    before = np.asarray([r["before_margins"] for r in runs], dtype=np.float64)
    diag = np.abs(np.stack([mats[:, 0, 0], mats[:, 1, 1]], axis=1))
    off = np.abs(np.stack([mats[:, 1, 0], mats[:, 0, 1]], axis=1))
    diag_mean = float(np.mean(diag))
    off_mean = float(np.mean(off))
    return {
        "mean_before_margin": float(np.mean(before)),
        "min_before_margin": float(np.min(before)),
        "mean_cross_write_matrix": np.mean(mats, axis=0).tolist(),
        "mean_abs_diagonal_self_effect": diag_mean,
        "mean_abs_off_diagonal_collateral_effect": off_mean,
        "off_over_diag": float(off_mean / max(diag_mean, 1e-30)),
        "per_seed_matrices": mats.tolist(),
    }


def main(n_seeds: int = 12, repeats: int = 400) -> dict:
    scfg = SpatialLaunchConfig()
    shared = [shared_matrix(seed, repeats, scfg) for seed in range(n_seeds)]
    separated = [separated_matrix(seed, repeats, scfg) for seed in range(n_seeds)]
    return {
        "schema": "jellobrain/cross-write-jacobian-v1",
        "question": (
            "Does one extra rewarded local write for one cue substantially perturb the other cue when both mappings share the same slow sheet?"
        ),
        "n_seeds": int(n_seeds),
        "oracle_repeats_per_route": int(repeats),
        "shared_sheet": summarize(shared),
        "separated_bands": summarize(separated),
        "claim_boundary": (
            "This finite-difference matrix quantifies collateral write effects in the current toy. It does not establish a universal credit-assignment law or a biological compartmentation mechanism."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
