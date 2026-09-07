#!/usr/bin/env python3
"""S14b: retire only a failed *committed* route, not failed exploratory POKEs.

The first negative-replay sweep weakened every failed chosen launcher. That
includes random exploratory POKEs, which can punish an alternative merely
because exploration sampled it at the wrong time. This variant makes the
credit rule stricter:

- raw/committed launch fails -> locally retire that embodied route;
- exploratory POKE fails -> keep attention high, but do not edit material;
- rewarded alternative while attention is high -> normal positive write.

No correct launcher identity is supplied. We again demand a full
SWAPPED -> NORMAL -> SWAPPED cycle so simply uncovering the inherited route is
not enough.
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
    raw_greedy_accuracy,
    reinforce_route,
)


def run_phase(world, alarm, rng, channel_map, *, retirement_rate, cfg, scfg):
    n = cfg.episodes_per_phase
    rewards = np.zeros(n)
    pokes = np.zeros(n)
    writes = np.zeros(n)
    retires = np.zeros(n)

    for ep in range(n):
        cue = int(rng.integers(2))
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
            # Key change: only the body's own committed winner gets a negative
            # write. A random probe is evidence-gathering, not yet a road to erase.
            if (not poke) and retirement_rate > 0:
                retire_failed_route(
                    world,
                    cue,
                    raw_choice,
                    rate=retirement_rate,
                    config=cfg,
                )
                retires[ep] = 1.0

    return {
        "raw_accuracy": float(raw_greedy_accuracy(world, channel_map, config=scfg)),
        "last_200_reward": float(np.mean(rewards[-200:])),
        "last_200_poke_rate": float(np.mean(pokes[-200:])),
        "last_200_write_rate": float(np.mean(writes[-200:])),
        "last_200_retire_rate": float(np.mean(retires[-200:])),
        "total_writes": int(np.sum(writes)),
        "total_retires": int(np.sum(retires)),
    }


def run_cycle(seed: int, rate: float) -> dict:
    cfg = RetirementConfig()
    scfg = SpatialLaunchConfig()
    world = make_inherited_world(seed, scfg)
    alarm = np.zeros(2)
    rng = np.random.default_rng(4_500_071 + int(seed))
    phases = []

    for name, channel in (
        ("learn_swapped", SWAPPED_CHANNEL),
        ("reverse_normal", NORMAL_CHANNEL),
        ("reverse_swapped_again", SWAPPED_CHANNEL),
    ):
        immediate = float(raw_greedy_accuracy(world, channel, config=scfg))
        phase = run_phase(
            world,
            alarm,
            rng,
            channel,
            retirement_rate=rate,
            cfg=cfg,
            scfg=scfg,
        )
        phases.append({"phase": name, "immediate_raw_accuracy": immediate, **phase})

    alarm[:] = 0.0
    return {
        "seed": int(seed),
        "rate": float(rate),
        "phases": phases,
        "final_swapped_after_controller_erased": float(
            raw_greedy_accuracy(world, SWAPPED_CHANNEL, config=scfg)
        ),
    }


def summarize(rate: float, n_seeds: int = 4) -> dict:
    runs = [run_cycle(seed, rate) for seed in range(n_seeds)]

    def pm(i, key):
        return float(np.mean([r["phases"][i][key] for r in runs]))

    return {
        "rate": float(rate),
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


def main() -> dict:
    rates = (0.0025, 0.005, 0.010, 0.020, 0.050)
    return {
        "schema": "jellobrain/committed-route-retirement-sweep-v1",
        "claim_boundary": (
            "This is a toy credit-assignment diagnostic. It tests whether negative plasticity should be attached to failed committed behavior rather than exploratory probes; it is not a biological plasticity model."
        ),
        "rates": [summarize(rate) for rate in rates],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
