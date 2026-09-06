#!/usr/bin/env python3
"""S14 diagnostic: can a failed embodied route be *retired* locally?

S12 learned a stable two-route mapping when successful familiar behavior stopped
rewriting itself. S13 then changed the outward meaning of the same familiar
cues. Failure re-opened POKEs and writes, but the additive-only material could
not replace both mappings; an explicit assurance loop made interference worse.

The current JelloWorld material is monotone under ordinary learning:
`material_decay == 1` and rewarded co-activity only adds non-negative material.
That suggests a concrete bottleneck: a boundary can discover that its present
launch was wrong, yet the body has no local operation meaning "trust the route
I just used less".

This diagnostic adds exactly that operation and nothing more. On a failed
outward consequence, the body replays the *actually chosen* cue->launcher pair
with writing disabled. Local eligibility/current-activity co-activity then
subtracts a small amount from the same directed material variables that normal
JelloBrain learning would strengthen. The controller knows which action it
just emitted and whether it failed; it is never told the correct launcher.

We sweep the depression rate through a three-phase meaning cycle:

    SWAPPED -> NORMAL -> SWAPPED

A genuine route-replacement mechanism should not merely reveal the virgin
mapping once. It should be able to return to the learned swapped mapping after
having retired it during the middle phase.

This is a computational forgetting/reversal diagnostic, not a biological LTD,
chandelier-cell, AIS, or synaptic-plasticity model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from spatial_launcher import (
    CUE_PORTS,
    LAUNCH_PORTS,
    NORMAL_CHANNEL,
    SWAPPED_CHANNEL,
    SpatialLaunchConfig,
    launcher_response,
    make_inherited_world,
    raw_greedy_accuracy,
    reinforce_route,
)


@dataclass(frozen=True)
class RetirementConfig:
    episodes_per_phase: int = 1600
    explore_at_full_alarm: float = 0.80
    success_decay: float = 0.55
    plasticity_alarm_threshold: float = 0.10
    retirement_replays: int = 1


def _depress_active_edges(world, rate: float) -> float:
    """Locally weaken directed edges active in the current eligibility trace.

    Returns total material removed. This mirrors the locality of JelloWorld's
    positive write, but with a negative sign. No geometric path is supplied.
    """
    rate = float(rate)
    if rate <= 0:
        return 0.0

    q = world.eligibility
    x = world.x
    before = float(
        world.east.sum() + world.west.sum() + world.south.sum() + world.north.sum()
    )

    world.east = np.clip(
        world.east - rate * q[:, :-1] * x[:, 1:], 0.0, 1.0
    )
    world.west = np.clip(
        world.west - rate * q[:, 1:] * x[:, :-1], 0.0, 1.0
    )
    world.south = np.clip(
        world.south - rate * q[:-1, :] * x[1:, :], 0.0, 1.0
    )
    world.north = np.clip(
        world.north - rate * q[1:, :] * x[:-1, :], 0.0, 1.0
    )

    after = float(
        world.east.sum() + world.west.sum() + world.south.sum() + world.north.sum()
    )
    return max(0.0, before - after)


def retire_failed_route(
    world,
    cue: int,
    chosen_launcher: int,
    *,
    rate: float,
    config: RetirementConfig,
) -> float:
    """Activity-coupled negative replay of the route that just failed.

    The replay is defined by the private cue and the launcher that was actually
    emitted. It contains no target/correct-output information.
    """
    if rate <= 0:
        return 0.0

    total_removed = 0.0
    word = (CUE_PORTS[int(cue)], LAUNCH_PORTS[int(chosen_launcher)])
    for _ in range(config.retirement_replays):
        world.wipe_fast()
        for symbol in word:
            world.inject(symbol)
            for _ in range(world.config.steps_per_symbol):
                world.step(write=False)
                total_removed += _depress_active_edges(world, rate)
    world.wipe_fast()
    return float(total_removed)


def run_phase(
    world,
    alarm: np.ndarray,
    rng: np.random.Generator,
    channel_map: tuple[int, int],
    *,
    retirement_rate: float,
    write_rewarded_routes: bool,
    config: RetirementConfig,
    spatial_config: SpatialLaunchConfig,
) -> dict:
    n = config.episodes_per_phase
    rewards = np.zeros(n, dtype=np.float64)
    pokes = np.zeros(n, dtype=np.float64)
    writes = np.zeros(n, dtype=np.float64)
    retires = np.zeros(n, dtype=np.float64)
    removed = np.zeros(n, dtype=np.float64)

    for ep in range(n):
        cue = int(rng.integers(2))
        observed = launcher_response(world, cue, config=spatial_config)
        raw_choice = int(np.argmax(observed))

        p_explore = config.explore_at_full_alarm * float(alarm[cue])
        poke = bool(rng.random() < p_explore)
        chosen = int(rng.integers(2)) if poke else raw_choice
        delivered = int(channel_map[chosen])
        reward = int(delivered == cue)

        rewards[ep] = reward
        pokes[ep] = float(poke)

        if reward:
            if (
                write_rewarded_routes
                and alarm[cue] >= config.plasticity_alarm_threshold
            ):
                reinforce_route(world, cue, chosen, config=spatial_config)
                writes[ep] = 1.0
            alarm[cue] *= config.success_decay
        else:
            # Failure contains no target identity, but it does identify the
            # just-used route as a bad bet under the current contingency.
            alarm[cue] = 1.0
            if retirement_rate > 0:
                removed[ep] = retire_failed_route(
                    world,
                    cue,
                    chosen,
                    rate=retirement_rate,
                    config=config,
                )
                retires[ep] = 1.0

    return {
        "raw_accuracy": float(raw_greedy_accuracy(world, channel_map, config=spatial_config)),
        "first_200_reward": float(np.mean(rewards[:200])),
        "last_200_reward": float(np.mean(rewards[-200:])),
        "first_200_poke_rate": float(np.mean(pokes[:200])),
        "last_200_poke_rate": float(np.mean(pokes[-200:])),
        "last_200_write_rate": float(np.mean(writes[-200:])),
        "last_200_retire_rate": float(np.mean(retires[-200:])),
        "total_writes": int(np.sum(writes)),
        "total_retire_events": int(np.sum(retires)),
        "total_material_removed": float(np.sum(removed)),
        "alarm_end": alarm.tolist(),
    }


def run_cycle(
    seed: int,
    retirement_rate: float,
    *,
    write_rewarded_routes: bool = True,
    config: RetirementConfig | None = None,
    spatial_config: SpatialLaunchConfig | None = None,
) -> dict:
    cfg = config or RetirementConfig()
    scfg = spatial_config or SpatialLaunchConfig()
    world = make_inherited_world(seed, scfg)
    alarm = np.zeros(2, dtype=np.float64)
    rng = np.random.default_rng(4_000_037 + int(seed))

    phases = []
    for name, channel in (
        ("learn_swapped", SWAPPED_CHANNEL),
        ("reverse_normal", NORMAL_CHANNEL),
        ("reverse_swapped_again", SWAPPED_CHANNEL),
    ):
        immediate = float(raw_greedy_accuracy(world, channel, config=scfg))
        result = run_phase(
            world,
            alarm,
            rng,
            channel,
            retirement_rate=retirement_rate,
            write_rewarded_routes=write_rewarded_routes,
            config=cfg,
            spatial_config=scfg,
        )
        phases.append({"phase": name, "immediate_raw_accuracy": immediate, **result})

    # Controller erase: only the body remains. raw_greedy_accuracy never reads alarm,
    # so this explicit erase is documentary rather than computationally necessary.
    alarm[:] = 0.0
    final_swapped = float(raw_greedy_accuracy(world, SWAPPED_CHANNEL, config=scfg))
    final_normal = float(raw_greedy_accuracy(world, NORMAL_CHANNEL, config=scfg))

    return {
        "seed": int(seed),
        "retirement_rate": float(retirement_rate),
        "write_rewarded_routes": bool(write_rewarded_routes),
        "phases": phases,
        "final_swapped_after_controller_erased": final_swapped,
        "final_normal_after_controller_erased": final_normal,
        "material_stats": world.material_stats(),
    }


def summarize_rate(rate: float, n_seeds: int = 4) -> dict:
    runs = [run_cycle(seed, rate) for seed in range(n_seeds)]

    def phase_mean(index: int, key: str) -> float:
        return float(np.mean([r["phases"][index][key] for r in runs]))

    return {
        "retirement_rate": float(rate),
        "n_seeds": int(n_seeds),
        "phase_raw_accuracy_means": {
            "learn_swapped": phase_mean(0, "raw_accuracy"),
            "reverse_normal": phase_mean(1, "raw_accuracy"),
            "reverse_swapped_again": phase_mean(2, "raw_accuracy"),
        },
        "phase_last_200_reward_means": {
            "learn_swapped": phase_mean(0, "last_200_reward"),
            "reverse_normal": phase_mean(1, "last_200_reward"),
            "reverse_swapped_again": phase_mean(2, "last_200_reward"),
        },
        "phase_last_200_poke_means": {
            "learn_swapped": phase_mean(0, "last_200_poke_rate"),
            "reverse_normal": phase_mean(1, "last_200_poke_rate"),
            "reverse_swapped_again": phase_mean(2, "last_200_poke_rate"),
        },
        "phase_last_200_write_means": {
            "learn_swapped": phase_mean(0, "last_200_write_rate"),
            "reverse_normal": phase_mean(1, "last_200_write_rate"),
            "reverse_swapped_again": phase_mean(2, "last_200_write_rate"),
        },
        "phase_last_200_retire_means": {
            "learn_swapped": phase_mean(0, "last_200_retire_rate"),
            "reverse_normal": phase_mean(1, "last_200_retire_rate"),
            "reverse_swapped_again": phase_mean(2, "last_200_retire_rate"),
        },
        "mean_total_material_removed": float(
            np.mean([
                sum(p["total_material_removed"] for p in r["phases"])
                for r in runs
            ])
        ),
        "final_swapped_after_controller_erased_mean": float(
            np.mean([r["final_swapped_after_controller_erased"] for r in runs])
        ),
        "final_swapped_all": [r["final_swapped_after_controller_erased"] for r in runs],
    }


def run_sweep(n_seeds: int = 4) -> dict:
    rates = (0.0, 0.005, 0.02, 0.08)
    summaries = [summarize_rate(rate, n_seeds=n_seeds) for rate in rates]
    return {
        "schema": "jellobrain/route-retirement-sweep-v1",
        "claim_boundary": (
            "This diagnostic tests whether bidirectional material adaptation is needed for repeated contingency reversal in the current toy. "
            "It does not model biological LTD, chandelier cells, AIS plasticity, or synaptic physiology."
        ),
        "question": (
            "When a familiar launch becomes wrong, is local weakening of the actually failed route enough to convert endless POKE/write interference into repeatable relearning?"
        ),
        "rates": summaries,
    }


def write_receipt(path: str | Path = "results/retirement_sweep.json", n_seeds: int = 4) -> dict:
    receipt = run_sweep(n_seeds=n_seeds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    print(json.dumps(write_receipt(), indent=2))
