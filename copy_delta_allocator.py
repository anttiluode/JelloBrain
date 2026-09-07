#!/usr/bin/env python3
"""S19: copy -> perturb -> delta -> choose the least damaging write address.

This is the direct computational version of the old video intuition:

    if a POKE changes the thing being inspected,
    preserve a reference,
    perturb a copy,
    measure the delta,
    then decide where the real change is allowed to live.

Important corrections relative to earlier exploratory gates:

1. The effect vector used for conflict detection is in a FIXED PHYSICAL launcher
   coordinate (C-D for each cue).  It does not change sign when the external
   channel meaning changes and therefore does not know the reward target.
2. A successful exploratory POKE discovers only that a launcher action worked.
   It does not supply a cue->band address.
3. To choose where to store that successful action, every existing anonymous
   band is counterfactually copied and given one candidate ordinary JelloWorld
   write.  We then ask how much that candidate would disturb the OTHER cue's
   currently embodied launcher choice.  The real write goes to the candidate
   with least collateral change.

Thus the allocator knows:
- current cue;
- launcher actually tried;
- binary consequence;
- current material responses;
- reversible internal counterfactual copies.

It does NOT know:
- desired launcher identity before the external POKE;
- a semantic cue->band map;
- the correct output of the other cue.

When opposing physical write deltas are detected, a PREWRITE reference copy is
preserved as a new anonymous band (max four).  That is the address-birth rule.

Attacker: RANDOM WRITE ADDRESS uses the same POKEs, same split rule, same number
of available bands, but sends each successful write to a random band instead of
running the copy-delta allocator.

This is an expensive upper-bound causal allocator, not a biological mechanism.
It tests whether the information in "before / intervention / after" is actually
sufficient to solve the current interference wall before searching for a local
biological approximation.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np

from emergent_address_probe import EmergentAddressConfig, noisy_accuracy, responses_for_cue
from spatial_launcher import (
    NORMAL_CHANNEL,
    SWAPPED_CHANNEL,
    SpatialLaunchConfig,
    launcher_response,
    make_inherited_world,
    reinforce_route,
)


@dataclass(frozen=True)
class AllocatorConfig:
    episodes_per_phase: int = 1800
    cosine_threshold: float = -0.90
    max_bands: int = 4


def raw_effect_vector(world, scfg: SpatialLaunchConfig) -> np.ndarray:
    """Fixed physical read coordinates; independent of channel meaning."""
    out = []
    for cue in (0, 1):
        r = launcher_response(world, cue, config=scfg)
        out.append(float(r[0] - r[1]))  # C - D, always the same coordinate
    return np.asarray(out, dtype=np.float64)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-30 or nb <= 1e-30:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


def system_signature(bands, cue: int, scfg: SpatialLaunchConfig) -> dict:
    responses = responses_for_cue(bands, cue, scfg)
    launchers = np.argmax(responses, axis=1)
    margins = np.abs(responses[:, 1] - responses[:, 0])
    band = int(np.argmax(margins))
    return {
        "band": band,
        "launcher": int(launchers[band]),
        "margin": float(margins[band]),
        "responses": responses,
    }


def evaluate(bands, channel_map, scfg: SpatialLaunchConfig) -> dict:
    correct = 0
    rows = []
    for cue in (0, 1):
        sig = system_signature(bands, cue, scfg)
        delivered = int(channel_map[sig["launcher"]])
        ok = int(delivered == cue)
        correct += ok
        rows.append(
            {
                "cue": cue,
                "band": sig["band"],
                "launcher": sig["launcher"],
                "margin": sig["margin"],
                "correct": bool(ok),
            }
        )
    return {"accuracy": correct / 2.0, "rows": rows}


def margin_toward(response: np.ndarray, launcher: int) -> float:
    return float(response[int(launcher)] - response[1 - int(launcher)])


def candidate_write_metrics(
    bands,
    candidate_band: int,
    cue: int,
    successful_launcher: int,
    scfg: SpatialLaunchConfig,
) -> dict:
    """Counterfactually write one band and quantify self gain + collateral.

    Collateral is defined without reward labels: does the OTHER cue's currently
    committed launcher change, and how much does its winning confidence move?
    """
    other_cue = 1 - int(cue)
    before_other = system_signature(bands, other_cue, scfg)

    trial = [copy.deepcopy(b) for b in bands]
    before_current_response = launcher_response(
        trial[candidate_band], cue, config=scfg
    )
    reinforce_route(
        trial[candidate_band], cue, successful_launcher, config=scfg
    )
    after_current_response = launcher_response(
        trial[candidate_band], cue, config=scfg
    )
    self_gain = (
        margin_toward(after_current_response, successful_launcher)
        - margin_toward(before_current_response, successful_launcher)
    )

    after_other = system_signature(trial, other_cue, scfg)
    other_flip = int(after_other["launcher"] != before_other["launcher"])
    denom = max(abs(before_other["margin"]), 1e-9)
    other_margin_change = abs(after_other["margin"] - before_other["margin"]) / denom

    return {
        "candidate_band": int(candidate_band),
        "self_gain": float(self_gain),
        "other_launcher_flip": int(other_flip),
        "other_margin_change_relative": float(other_margin_change),
    }


def choose_write_band_copy_delta(
    bands,
    cue: int,
    successful_launcher: int,
    scfg: SpatialLaunchConfig,
) -> tuple[int, list[dict]]:
    metrics = [
        candidate_write_metrics(
            bands, b, cue, successful_launcher, scfg
        )
        for b in range(len(bands))
    ]
    # Lexicographic causal safety rule, deliberately no fitted weighting:
    # 1. avoid changing the other cue's current launcher;
    # 2. minimize relative disturbance to its confidence;
    # 3. among equally safe addresses, maximize strengthening of the action
    #    that just succeeded.
    best = min(
        metrics,
        key=lambda m: (
            m["other_launcher_flip"],
            m["other_margin_change_relative"],
            -m["self_gain"],
            m["candidate_band"],
        ),
    )
    return int(best["candidate_band"]), metrics


def maybe_split_after_write(
    bands,
    effect_memory,
    written_band: int,
    reference,
    before_effect: np.ndarray,
    scfg: SpatialLaunchConfig,
    cfg: AllocatorConfig,
    split_events: list,
    phase_name: str,
    episode: int,
    cue: int,
) -> None:
    after_effect = raw_effect_vector(bands[written_band], scfg)
    delta = after_effect - before_effect
    old = effect_memory[written_band]
    conflict = cosine(old, delta) if old is not None else None

    if (
        old is not None
        and conflict is not None
        and conflict <= cfg.cosine_threshold
        and len(bands) < cfg.max_bands
    ):
        # Preserve what existed immediately before the conflicting write.
        bands.append(reference)
        effect_memory.append(copy.deepcopy(old))
        effect_memory[written_band] = delta
        split_events.append(
            {
                "phase": phase_name,
                "episode": int(episode),
                "source_band": int(written_band),
                "new_band": int(len(bands) - 1),
                "cosine": float(conflict),
                "cue": int(cue),
            }
        )
    else:
        effect_memory[written_band] = delta


def run_cycle(seed: int, *, allocator: str) -> dict:
    if allocator not in {"random_write_address", "copy_delta"}:
        raise ValueError(allocator)

    cfg = AllocatorConfig()
    ecfg = EmergentAddressConfig(episodes_per_phase=cfg.episodes_per_phase)
    scfg = SpatialLaunchConfig()
    bands = [make_inherited_world(seed, scfg)]
    effect_memory: list[np.ndarray | None] = [None]
    alarm = np.zeros(2, dtype=np.float64)
    rng = np.random.default_rng(10_100_073 + int(seed))
    split_events = []
    allocation_counts = []
    phases = []

    for phase_name, channel_map in (
        ("learn_swapped", SWAPPED_CHANNEL),
        ("reverse_normal", NORMAL_CHANNEL),
        ("reverse_swapped_again", SWAPPED_CHANNEL),
    ):
        immediate = evaluate(bands, channel_map, scfg)
        rewards = np.zeros(cfg.episodes_per_phase, dtype=np.float64)
        pokes = np.zeros(cfg.episodes_per_phase, dtype=np.float64)
        writes = np.zeros(cfg.episodes_per_phase, dtype=np.float64)
        phase_alloc = np.zeros(cfg.max_bands, dtype=np.int64)

        for ep in range(cfg.episodes_per_phase):
            cue = int(rng.integers(2))
            committed = system_signature(bands, cue, scfg)
            poke = bool(
                rng.random()
                < ecfg.explore_at_full_alarm * float(alarm[cue])
            )
            if poke:
                # External counterfactual action. It carries no storage address.
                chosen_launcher = int(rng.integers(2))
            else:
                chosen_launcher = int(committed["launcher"])

            delivered = int(channel_map[chosen_launcher])
            reward = int(delivered == cue)
            rewards[ep] = reward
            pokes[ep] = float(poke)

            if reward:
                if alarm[cue] >= ecfg.plasticity_alarm_threshold:
                    if allocator == "copy_delta" and len(bands) > 1:
                        written_band, _ = choose_write_band_copy_delta(
                            bands, cue, chosen_launcher, scfg
                        )
                    else:
                        written_band = int(rng.integers(len(bands)))

                    reference = copy.deepcopy(bands[written_band])
                    before_effect = raw_effect_vector(reference, scfg)
                    reinforce_route(
                        bands[written_band], cue, chosen_launcher, config=scfg
                    )
                    writes[ep] = 1.0
                    phase_alloc[written_band] += 1
                    maybe_split_after_write(
                        bands,
                        effect_memory,
                        written_band,
                        reference,
                        before_effect,
                        scfg,
                        cfg,
                        split_events,
                        phase_name,
                        ep,
                        cue,
                    )

                alarm[cue] *= ecfg.success_decay
            else:
                alarm[cue] = 1.0

        state = evaluate(bands, channel_map, scfg)
        eval_rng = np.random.default_rng(
            10_200_079 + 101 * int(seed) + len(phases)
        )
        phases.append(
            {
                "phase": phase_name,
                "immediate_accuracy": immediate["accuracy"],
                "raw_accuracy": state["accuracy"],
                "noisy_accuracy": noisy_accuracy(
                    bands, channel_map, scfg, ecfg, eval_rng
                ),
                "n_bands": len(bands),
                "cue_bands": [row["band"] for row in state["rows"]],
                "last_200_reward": float(np.mean(rewards[-200:])),
                "last_200_poke": float(np.mean(pokes[-200:])),
                "last_200_write": float(np.mean(writes[-200:])),
                "allocation_counts": phase_alloc[: len(bands)].tolist(),
            }
        )
        allocation_counts.append(phase_alloc[: len(bands)].tolist())

    # Only material remains for final decision.
    alarm[:] = 0.0
    effect_memory = [None for _ in bands]
    final = evaluate(bands, SWAPPED_CHANNEL, scfg)
    final_rng = np.random.default_rng(10_300_083 + int(seed))
    return {
        "seed": int(seed),
        "allocator": allocator,
        "phases": phases,
        "split_events": split_events,
        "final_n_bands": len(bands),
        "final_accuracy": final["accuracy"],
        "final_noisy_accuracy": noisy_accuracy(
            bands, SWAPPED_CHANNEL, scfg, ecfg, final_rng
        ),
        "final_cue_bands": [row["band"] for row in final["rows"]],
    }


def summarize(allocator: str, n_seeds: int) -> dict:
    runs = [run_cycle(seed, allocator=allocator) for seed in range(n_seeds)]

    def pm(i: int, key: str) -> float:
        return float(np.mean([r["phases"][i][key] for r in runs]))

    return {
        "condition": allocator,
        "n_seeds": int(n_seeds),
        "mean_final_bands": float(np.mean([r["final_n_bands"] for r in runs])),
        "raw_accuracy": {
            "learn_swapped": pm(0, "raw_accuracy"),
            "reverse_normal": pm(1, "raw_accuracy"),
            "reverse_swapped_again": pm(2, "raw_accuracy"),
        },
        "noisy_accuracy": {
            "learn_swapped": pm(0, "noisy_accuracy"),
            "reverse_normal": pm(1, "noisy_accuracy"),
            "reverse_swapped_again": pm(2, "noisy_accuracy"),
        },
        "last_200_reward": {
            "learn_swapped": pm(0, "last_200_reward"),
            "reverse_normal": pm(1, "last_200_reward"),
            "reverse_swapped_again": pm(2, "last_200_reward"),
        },
        "final_accuracy_all": [r["final_accuracy"] for r in runs],
        "final_noisy_all": [r["final_noisy_accuracy"] for r in runs],
        "final_cue_bands": [r["final_cue_bands"] for r in runs],
        "split_count_all": [len(r["split_events"]) for r in runs],
    }


def main(n_seeds: int = 4) -> dict:
    return {
        "schema": "jellobrain/copy-delta-write-allocation-v1",
        "question": (
            "After a successful POKE, is preserving a pre-write reference and counterfactually measuring collateral deltas sufficient to choose a safe anonymous write address under repeated meaning reversal?"
        ),
        "conditions": [
            summarize("random_write_address", n_seeds),
            summarize("copy_delta", n_seeds),
        ],
        "claim_boundary": (
            "The copy-delta arm is an expensive upper-bound planner. A pass would establish sufficiency of before/intervention/after information for this toy interference problem, not biological plausibility."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
