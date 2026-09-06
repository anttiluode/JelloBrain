#!/usr/bin/env python3
"""S12: stop rehearsing what is already working.

S9 used a transient failure alarm to trigger exploratory POKEs, but it still
reinforced *every* rewarded route.  That creates a subtle positive-feedback
problem in a shared sheet: the first route to become reliable keeps receiving
writes simply because it is reliable, while the still-wrong route receives
successful writes only on exploratory trials.

This diagnostic asks whether a very small change fixes that imbalance:

    failure -> alarm / POKE / plasticity eligible
    success -> alarm decays
    familiar success -> stop writing

The controller still stores no desired launcher identity.  It stores one scalar
"needs attention" value per cue.  The final test erases those scalars and uses
raw JelloWorld launcher activity only.

An optional ASSURANCE condition re-probes both private contexts after a material
write and re-opens the alarm of any context whose raw output has been broken by
that write.  The probe does not edit material and does not supply the correct
launcher; it only asks whether the current outward consequence is still correct.

This is a toy continual-learning/plasticity-gating experiment, not a biological
model of chandelier cells, basket cells, the AIS, or replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from spatial_launcher import (
    SWAPPED_CHANNEL,
    SpatialLaunchConfig,
    launcher_response,
    make_inherited_world,
    raw_greedy_accuracy,
    reinforce_route,
)


@dataclass(frozen=True)
class AssuranceConfig:
    episodes: int = 1600
    explore_at_full_alarm: float = 0.80
    success_decay: float = 0.55
    plasticity_alarm_threshold: float = 0.10


def raw_outcome_correct(world, cue: int, scfg: SpatialLaunchConfig) -> bool:
    observed = launcher_response(world, cue, config=scfg)
    chosen = int(np.argmax(observed))
    delivered = int(SWAPPED_CHANNEL[chosen])
    return bool(delivered == int(cue))


def train(
    seed: int,
    *,
    gated_plasticity: bool,
    active_assurance: bool,
    config: AssuranceConfig | None = None,
    spatial_config: SpatialLaunchConfig | None = None,
) -> dict:
    cfg = config or AssuranceConfig()
    scfg = spatial_config or SpatialLaunchConfig()
    world = make_inherited_world(seed, scfg)
    rng = np.random.default_rng(2_300_011 + int(seed))

    alarm = np.zeros(2, dtype=np.float64)
    rewards = np.zeros(cfg.episodes, dtype=np.float64)
    pokes = np.zeros(cfg.episodes, dtype=np.float64)
    writes = np.zeros(cfg.episodes, dtype=np.float64)
    assurance_reopens = np.zeros(cfg.episodes, dtype=np.float64)

    for ep in range(cfg.episodes):
        cue = int(rng.integers(2))
        observed = launcher_response(world, cue, config=scfg)
        raw_choice = int(np.argmax(observed))

        p_explore = cfg.explore_at_full_alarm * alarm[cue]
        poke = bool(rng.random() < p_explore)
        chosen = int(rng.integers(2)) if poke else raw_choice
        delivered = int(SWAPPED_CHANNEL[chosen])
        reward = int(delivered == cue)

        rewards[ep] = reward
        pokes[ep] = float(poke)

        # Crucial distinction from S9. Reward is not by itself a license to
        # keep changing the body forever. Only a context still marked as
        # needing attention remains plastic.
        write_allowed = bool(
            reward
            and (
                not gated_plasticity
                or alarm[cue] >= cfg.plasticity_alarm_threshold
            )
        )
        if write_allowed:
            reinforce_route(world, cue, chosen, config=scfg)
            writes[ep] = 1.0

        if reward:
            alarm[cue] *= cfg.success_decay
        else:
            alarm[cue] = 1.0

        if active_assurance and write_allowed:
            reopened = 0
            for probe_cue in (0, 1):
                if not raw_outcome_correct(world, probe_cue, scfg):
                    if alarm[probe_cue] < 1.0:
                        reopened += 1
                    alarm[probe_cue] = 1.0
            assurance_reopens[ep] = reopened

    pre_erase_alarm = alarm.copy()
    alarm[:] = 0.0
    final_raw = raw_greedy_accuracy(world, SWAPPED_CHANNEL, config=scfg)
    final_per_cue = [float(raw_outcome_correct(world, cue, scfg)) for cue in (0, 1)]

    def mean_slice(a: np.ndarray, sl: slice) -> float:
        return float(np.mean(a[sl]))

    return {
        "seed": int(seed),
        "gated_plasticity": bool(gated_plasticity),
        "active_assurance": bool(active_assurance),
        "final_raw_greedy_after_controller_erased": float(final_raw),
        "final_raw_per_cue": final_per_cue,
        "first_200_reward": mean_slice(rewards, slice(0, 200)),
        "last_200_reward": mean_slice(rewards, slice(-200, None)),
        "first_200_poke_rate": mean_slice(pokes, slice(0, 200)),
        "last_200_poke_rate": mean_slice(pokes, slice(-200, None)),
        "first_200_write_rate": mean_slice(writes, slice(0, 200)),
        "last_200_write_rate": mean_slice(writes, slice(-200, None)),
        "total_writes": int(np.sum(writes)),
        "assurance_reopens": int(np.sum(assurance_reopens)),
        "alarm_before_erase": pre_erase_alarm.tolist(),
    }


def _summary(runs: list[dict]) -> dict:
    keys = [
        "final_raw_greedy_after_controller_erased",
        "first_200_reward",
        "last_200_reward",
        "first_200_poke_rate",
        "last_200_poke_rate",
        "first_200_write_rate",
        "last_200_write_rate",
        "total_writes",
        "assurance_reopens",
    ]
    out = {}
    for key in keys:
        out[key + "_mean"] = float(np.mean([r[key] for r in runs]))
    out["final_raw_all"] = [r["final_raw_greedy_after_controller_erased"] for r in runs]
    return out


def gate_s12(n_seeds: int = 12) -> dict:
    ungated = [
        train(seed, gated_plasticity=False, active_assurance=False)
        for seed in range(n_seeds)
    ]
    gated = [
        train(seed, gated_plasticity=True, active_assurance=False)
        for seed in range(n_seeds)
    ]
    assured = [
        train(seed, gated_plasticity=True, active_assurance=True)
        for seed in range(n_seeds)
    ]

    a = _summary(ungated)
    b = _summary(gated)
    c = _summary(assured)

    gate = {
        "gate": "S12_HABITUATED_PLASTICITY_AND_ASSURANCE",
        "question": (
            "Does stopping plasticity for familiar successful output prevent an already-good route from monopolizing the shared sheet, "
            "and does re-probing previously learned contexts help preserve both mappings?"
        ),
        "n_seeds": int(n_seeds),
        "reward_every_success_baseline": a,
        "failure_gated_plasticity": b,
        "failure_gated_plus_assurance": c,
        "interpretation": (
            "The controller has no correct-output table. Its only persistent side state is per-context 'needs attention'. "
            "A positive result would mean novelty/relevance can gate *plasticity itself*, not only whether to emit a POKE. "
            "The final score is measured after the controller is erased."
        ),
    }

    # We do not assume in advance that the gated or assurance variants must
    # solve the task; this gate's threshold records the hoped-for handoff.
    gate["pass"] = bool(
        c["final_raw_greedy_after_controller_erased_mean"] >= 0.95
        and c["last_200_reward_mean"] >= 0.90
        and c["last_200_poke_rate_mean"] <= 0.15
        and c["last_200_write_rate_mean"] <= 0.15
        and a["final_raw_greedy_after_controller_erased_mean"] <= 0.60
    )
    return gate


def run_all_gates(n_seeds: int = 12) -> dict:
    gate = gate_s12(n_seeds)
    return {
        "schema": "jellobrain/assurance-plasticity-gates-v1",
        "claim_boundary": (
            "S12 is a toy plasticity-scheduling/assurance audit. It does not claim that chandelier cells implement these variables "
            "or that the AIS performs explicit self-tests."
        ),
        "gates": [gate],
        "all_pass": bool(gate["pass"]),
    }


def write_receipt(path: str | Path = "results/assurance_gates.json", n_seeds: int = 12) -> dict:
    receipt = run_all_gates(n_seeds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    print(json.dumps(write_receipt(), indent=2))
