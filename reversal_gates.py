#!/usr/bin/env python3
"""S13: a familiar successful boundary must wake up when consequences change.

S12 found that the shared spatial sheet works much better when reward is not a
permanent license to keep rewriting already-good routes.  Plasticity becomes
eligible after failure, then shuts down again after repeated success.

S13 asks the obvious next question: does that quiet state become brittle?

A body first adapts to the SWAPPED physical pulse channel until behavior is
reliable and POKE/write activity has habituated.  Without resetting the body,
the external channel is then changed to NORMAL.  The same familiar private
cues now have different outward consequences.

The controller is intentionally weak.  It stores only one scalar
"needs attention" value per cue.  It never receives the correct launcher
identity.  A failed outward consequence re-opens attention, attention permits
exploratory POKEs and successful material writes, and success lets attention
decay again.  Final performance is measured after the controller is erased.

Attackers:
- NO REACTIVATION: failures after the contingency change are not allowed to
  re-open attention.
- NO MATERIAL WRITE: reactivation and POKEs occur, but the body cannot change.

A pass therefore means neither a permanently active critic nor a side-table
policy is sufficient: changed consequences must transiently re-open plasticity,
and the new solution must end up in JelloWorld itself.

This is a toy continual-adaptation / launch-boundary audit, not a biological
model of chandelier cells, the AIS, acquired salience, or cortical learning.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from spatial_launcher import (
    NORMAL_CHANNEL,
    SWAPPED_CHANNEL,
    SpatialLaunchConfig,
    launcher_response,
    raw_greedy_accuracy,
    reinforce_route,
    make_inherited_world,
)


@dataclass(frozen=True)
class ReversalConfig:
    phase1_episodes: int = 1600
    phase2_episodes: int = 1600
    explore_at_full_alarm: float = 0.80
    success_decay: float = 0.55
    plasticity_alarm_threshold: float = 0.10


def _run_phase(
    world,
    alarm: np.ndarray,
    rng: np.random.Generator,
    channel_map: tuple[int, int],
    *,
    episodes: int,
    reactivate_on_failure: bool,
    write_material: bool,
    config: ReversalConfig,
    spatial_config: SpatialLaunchConfig,
) -> dict:
    cfg = config
    scfg = spatial_config

    rewards = np.zeros(episodes, dtype=np.float64)
    pokes = np.zeros(episodes, dtype=np.float64)
    writes = np.zeros(episodes, dtype=np.float64)

    for ep in range(episodes):
        cue = int(rng.integers(2))
        observed = launcher_response(world, cue, config=scfg)
        raw_choice = int(np.argmax(observed))

        p_explore = cfg.explore_at_full_alarm * float(alarm[cue])
        poke = bool(rng.random() < p_explore)
        chosen = int(rng.integers(2)) if poke else raw_choice
        delivered = int(channel_map[chosen])
        reward = int(delivered == cue)

        rewards[ep] = reward
        pokes[ep] = float(poke)

        write_allowed = bool(
            write_material
            and reward
            and alarm[cue] >= cfg.plasticity_alarm_threshold
        )
        if write_allowed:
            reinforce_route(world, cue, chosen, config=scfg)
            writes[ep] = 1.0

        if reward:
            alarm[cue] *= cfg.success_decay
        elif reactivate_on_failure:
            alarm[cue] = 1.0

    def m(a: np.ndarray, start: int, stop: int | None = None) -> float:
        return float(np.mean(a[slice(start, stop)]))

    return {
        "first_200_reward": m(rewards, 0, 200),
        "last_200_reward": m(rewards, -200, None),
        "first_200_poke_rate": m(pokes, 0, 200),
        "last_200_poke_rate": m(pokes, -200, None),
        "first_200_write_rate": m(writes, 0, 200),
        "last_200_write_rate": m(writes, -200, None),
        "total_writes": int(np.sum(writes)),
        "alarm_after_phase": alarm.copy().tolist(),
    }


def _condition_after_reversal(
    base_world,
    base_alarm: np.ndarray,
    seed: int,
    *,
    reactivate_on_failure: bool,
    write_material: bool,
    config: ReversalConfig,
    spatial_config: SpatialLaunchConfig,
) -> dict:
    world = copy.deepcopy(base_world)
    alarm = np.asarray(base_alarm, dtype=np.float64).copy()
    if not reactivate_on_failure:
        # Make the attack explicit: the familiar/quiet controller cannot wake.
        alarm[:] = 0.0

    rng = np.random.default_rng(3_100_019 + int(seed))
    immediate_new_accuracy = raw_greedy_accuracy(world, NORMAL_CHANNEL, config=spatial_config)

    phase = _run_phase(
        world,
        alarm,
        rng,
        NORMAL_CHANNEL,
        episodes=config.phase2_episodes,
        reactivate_on_failure=reactivate_on_failure,
        write_material=write_material,
        config=config,
        spatial_config=spatial_config,
    )

    pre_erase_alarm = alarm.copy()
    alarm[:] = 0.0
    final_new = raw_greedy_accuracy(world, NORMAL_CHANNEL, config=spatial_config)
    final_old = raw_greedy_accuracy(world, SWAPPED_CHANNEL, config=spatial_config)

    return {
        "immediate_new_channel_raw_accuracy": float(immediate_new_accuracy),
        "final_new_channel_raw_accuracy_after_controller_erased": float(final_new),
        "final_old_channel_raw_accuracy_after_relearning": float(final_old),
        "alarm_before_erase": pre_erase_alarm.tolist(),
        **phase,
    }


def run_seed(
    seed: int,
    *,
    config: ReversalConfig | None = None,
    spatial_config: SpatialLaunchConfig | None = None,
) -> dict:
    cfg = config or ReversalConfig()
    scfg = spatial_config or SpatialLaunchConfig()

    world = make_inherited_world(seed, scfg)
    alarm = np.zeros(2, dtype=np.float64)
    rng = np.random.default_rng(3_000_017 + int(seed))

    phase1 = _run_phase(
        world,
        alarm,
        rng,
        SWAPPED_CHANNEL,
        episodes=cfg.phase1_episodes,
        reactivate_on_failure=True,
        write_material=True,
        config=cfg,
        spatial_config=scfg,
    )

    phase1_raw = raw_greedy_accuracy(world, SWAPPED_CHANNEL, config=scfg)
    base_alarm = alarm.copy()

    reactive = _condition_after_reversal(
        world,
        base_alarm,
        seed,
        reactivate_on_failure=True,
        write_material=True,
        config=cfg,
        spatial_config=scfg,
    )
    frozen = _condition_after_reversal(
        world,
        base_alarm,
        seed,
        reactivate_on_failure=False,
        write_material=True,
        config=cfg,
        spatial_config=scfg,
    )
    no_write = _condition_after_reversal(
        world,
        base_alarm,
        seed,
        reactivate_on_failure=True,
        write_material=False,
        config=cfg,
        spatial_config=scfg,
    )

    return {
        "seed": int(seed),
        "phase1_swapped": {
            "raw_accuracy_before_reversal": float(phase1_raw),
            **phase1,
        },
        "reactivate_plus_write": reactive,
        "no_reactivation_attacker": frozen,
        "no_material_write_attacker": no_write,
    }


def _mean(runs: list[dict], branch: str, key: str) -> float:
    return float(np.mean([r[branch][key] for r in runs]))


def gate_s13(n_seeds: int = 12) -> dict:
    runs = [run_seed(seed) for seed in range(n_seeds)]

    phase1 = {
        "raw_accuracy_before_reversal_mean": _mean(runs, "phase1_swapped", "raw_accuracy_before_reversal"),
        "last_200_reward_mean": _mean(runs, "phase1_swapped", "last_200_reward"),
        "last_200_poke_rate_mean": _mean(runs, "phase1_swapped", "last_200_poke_rate"),
        "last_200_write_rate_mean": _mean(runs, "phase1_swapped", "last_200_write_rate"),
    }

    def summarize(branch: str) -> dict:
        keys = [
            "immediate_new_channel_raw_accuracy",
            "final_new_channel_raw_accuracy_after_controller_erased",
            "final_old_channel_raw_accuracy_after_relearning",
            "first_200_reward",
            "last_200_reward",
            "first_200_poke_rate",
            "last_200_poke_rate",
            "first_200_write_rate",
            "last_200_write_rate",
            "total_writes",
        ]
        out = {key + "_mean": _mean(runs, branch, key) for key in keys}
        out["final_new_raw_all"] = [
            r[branch]["final_new_channel_raw_accuracy_after_controller_erased"]
            for r in runs
        ]
        return out

    reactive = summarize("reactivate_plus_write")
    frozen = summarize("no_reactivation_attacker")
    no_write = summarize("no_material_write_attacker")

    gate = {
        "gate": "S13_FAMILIAR_BOUNDARY_REACTIVATES_WHEN_MEANING_CHANGES",
        "question": (
            "After a familiar successful spatial launcher has gone quiet, can changed outward consequences transiently re-open POKEs/plasticity, "
            "rewrite the body, and then habituate again without storing the new policy in the controller?"
        ),
        "n_seeds": int(n_seeds),
        "phase1_habituated_swapped_channel": phase1,
        "reactivate_plus_write": reactive,
        "no_reactivation_attacker": frozen,
        "no_material_write_attacker": no_write,
        "interpretation": (
            "The same familiar cues are used before and after the contingency change. The controller knows only whether the last outward consequence failed; "
            "it never receives the correct launcher identity. A positive result means quietness is conditional rather than permanent: failure can wake the boundary, "
            "successful exploratory traffic can rewrite JelloWorld, and both POKE and plasticity can shut down again after the new mapping is embodied."
        ),
    }

    gate["pass"] = bool(
        phase1["raw_accuracy_before_reversal_mean"] >= 0.95
        and phase1["last_200_reward_mean"] >= 0.90
        and phase1["last_200_poke_rate_mean"] <= 0.10
        and phase1["last_200_write_rate_mean"] <= 0.10
        and reactive["immediate_new_channel_raw_accuracy_mean"] <= 0.10
        and reactive["final_new_channel_raw_accuracy_after_controller_erased_mean"] >= 0.95
        and reactive["last_200_reward_mean"] >= 0.90
        and reactive["last_200_poke_rate_mean"] <= 0.10
        and reactive["last_200_write_rate_mean"] <= 0.10
        and reactive["first_200_poke_rate_mean"] >= 0.10
        and reactive["total_writes_mean"] >= 1.0
        and frozen["final_new_channel_raw_accuracy_after_controller_erased_mean"] <= 0.50
        and no_write["final_new_channel_raw_accuracy_after_controller_erased_mean"] <= 0.50
    )
    return gate


def run_all_gates(n_seeds: int = 12) -> dict:
    gate = gate_s13(n_seeds)
    return {
        "schema": "jellobrain/reversal-reactivation-gates-v1",
        "claim_boundary": (
            "S13 is a toy consequence-change/reactivation test. It does not establish acquired salience in cortex, "
            "a chandelier-cell mechanism, an AIS learning rule, or a biological observer."
        ),
        "gates": [gate],
        "all_pass": bool(gate["pass"]),
    }


def write_receipt(path: str | Path = "results/reversal_gates.json", n_seeds: int = 12) -> dict:
    receipt = run_all_gates(n_seeds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    print(json.dumps(write_receipt(), indent=2))
