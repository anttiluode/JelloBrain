#!/usr/bin/env python3
"""S15 upper-bound diagnostic: is the missing ingredient addressable separation?

S13-S14 show that a single scalar material sheet is poor at repeated meaning
reversal. Extra assurance does not help, and local negative replay produces a
rate-dependent switch between attractors rather than robust reversibility.

Before inventing a more elaborate controller, this probe asks a deliberately
simple capacity question. Give the same spatial body two *slow material bands*.
The fast cue identity selects a band, but the band contains no target launcher
and no policy table. Within a band, routing is still ordinary JelloWorld
physics and rewarded local writing.

This is intentionally an upper-bound representation test. Two bands are
computationally equivalent to two isolated slow substrates sharing the same
port geometry. A positive result would not prove frequency channels, cortical
layers, axonal compartments, or any biological mechanism. It would only show
that the failure class can be removed by addressable slow-state isolation.

We compare:
- COLLAPSED: both cues write/read one shared JelloWorld (current failure class)
- SEPARATED: cue 0 and cue 1 use distinct slow bands
- SEPARATED + COMMITTED RETIREMENT: same, plus S14's local weakening only when
  the body's own non-POKE winner produces a bad outward consequence

All conditions must traverse SWAPPED -> NORMAL -> SWAPPED.
"""

from __future__ import annotations

import json
import numpy as np

from retirement_sweep import RetirementConfig, retire_failed_route
from spatial_launcher import (
    NORMAL_CHANNEL,
    SWAPPED_CHANNEL,
    SpatialLaunchConfig,
    launcher_response,
    make_inherited_world,
    reinforce_route,
)


def band_accuracy(bands, channel_map, scfg):
    correct = 0
    for cue in (0, 1):
        world = bands[cue]
        chosen = int(np.argmax(launcher_response(world, cue, config=scfg)))
        correct += int(int(channel_map[chosen]) == cue)
    return correct / 2.0


def run_phase(
    bands,
    alarm,
    rng,
    channel_map,
    *,
    cfg,
    scfg,
    retire_rate=0.0,
):
    n = cfg.episodes_per_phase
    rewards = np.zeros(n)
    pokes = np.zeros(n)
    writes = np.zeros(n)
    retires = np.zeros(n)

    for ep in range(n):
        cue = int(rng.integers(2))
        world = bands[cue]
        raw_choice = int(np.argmax(launcher_response(world, cue, config=scfg)))
        poke = bool(rng.random() < cfg.explore_at_full_alarm * float(alarm[cue]))
        chosen = int(rng.integers(2)) if poke else raw_choice
        reward = int(int(channel_map[chosen]) == cue)

        rewards[ep] = reward
        pokes[ep] = poke

        if reward:
            if alarm[cue] >= cfg.plasticity_alarm_threshold:
                reinforce_route(world, cue, chosen, config=scfg)
                writes[ep] = 1.0
            alarm[cue] *= cfg.success_decay
        else:
            alarm[cue] = 1.0
            if (not poke) and retire_rate > 0:
                retire_failed_route(
                    world,
                    cue,
                    raw_choice,
                    rate=retire_rate,
                    config=cfg,
                )
                retires[ep] = 1.0

    return {
        "raw_accuracy": float(band_accuracy(bands, channel_map, scfg)),
        "last_200_reward": float(np.mean(rewards[-200:])),
        "last_200_poke_rate": float(np.mean(pokes[-200:])),
        "last_200_write_rate": float(np.mean(writes[-200:])),
        "last_200_retire_rate": float(np.mean(retires[-200:])),
    }


def make_bands(seed, separated, scfg):
    if separated:
        # Same inherited geometry distribution, independently individualized slow
        # material. The cue chooses the band; no output mapping is stored here.
        return [
            make_inherited_world(10_000 + 2 * seed, scfg),
            make_inherited_world(10_001 + 2 * seed, scfg),
        ]
    shared = make_inherited_world(seed, scfg)
    return [shared, shared]


def run_cycle(seed, *, separated, retire_rate):
    cfg = RetirementConfig()
    scfg = SpatialLaunchConfig()
    bands = make_bands(seed, separated, scfg)
    alarm = np.zeros(2)
    rng = np.random.default_rng(5_000_093 + int(seed))
    phases = []

    for name, channel in (
        ("learn_swapped", SWAPPED_CHANNEL),
        ("reverse_normal", NORMAL_CHANNEL),
        ("reverse_swapped_again", SWAPPED_CHANNEL),
    ):
        immediate = float(band_accuracy(bands, channel, scfg))
        phase = run_phase(
            bands,
            alarm,
            rng,
            channel,
            cfg=cfg,
            scfg=scfg,
            retire_rate=retire_rate,
        )
        phases.append({"phase": name, "immediate_raw_accuracy": immediate, **phase})

    alarm[:] = 0.0
    return {
        "seed": int(seed),
        "separated": bool(separated),
        "retire_rate": float(retire_rate),
        "phases": phases,
        "final_swapped_after_controller_erased": float(
            band_accuracy(bands, SWAPPED_CHANNEL, scfg)
        ),
    }


def summarize(label, *, separated, retire_rate, n_seeds=4):
    runs = [
        run_cycle(seed, separated=separated, retire_rate=retire_rate)
        for seed in range(n_seeds)
    ]

    def pm(i, key):
        return float(np.mean([r["phases"][i][key] for r in runs]))

    return {
        "condition": label,
        "n_seeds": int(n_seeds),
        "raw_accuracy": {
            "learn_swapped": pm(0, "raw_accuracy"),
            "reverse_normal": pm(1, "raw_accuracy"),
            "reverse_swapped_again": pm(2, "raw_accuracy"),
        },
        "last_200_reward": {
            "learn_swapped": pm(0, "last_200_reward"),
            "reverse_normal": pm(1, "last_200_reward"),
            "reverse_swapped_again": pm(2, "last_200_reward"),
        },
        "last_200_poke": {
            "learn_swapped": pm(0, "last_200_poke_rate"),
            "reverse_normal": pm(1, "last_200_poke_rate"),
            "reverse_swapped_again": pm(2, "last_200_poke_rate"),
        },
        "last_200_write": {
            "learn_swapped": pm(0, "last_200_write_rate"),
            "reverse_normal": pm(1, "last_200_write_rate"),
            "reverse_swapped_again": pm(2, "last_200_write_rate"),
        },
        "last_200_retire": {
            "learn_swapped": pm(0, "last_200_retire_rate"),
            "reverse_normal": pm(1, "last_200_retire_rate"),
            "reverse_swapped_again": pm(2, "last_200_retire_rate"),
        },
        "final_swapped_all": [r["final_swapped_after_controller_erased"] for r in runs],
    }


def main():
    return {
        "schema": "jellobrain/addressed-separation-probe-v1",
        "claim_boundary": (
            "This is an upper-bound representational capacity diagnostic. Separate slow bands are not proposed as a biological mechanism and cue-to-band addressing does not encode the correct output."
        ),
        "conditions": [
            summarize("collapsed_additive", separated=False, retire_rate=0.0),
            summarize("separated_additive", separated=True, retire_rate=0.0),
            summarize("separated_committed_retirement_0.005", separated=True, retire_rate=0.005),
        ],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
