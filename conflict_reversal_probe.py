#!/usr/bin/env python3
"""S17R: can conflict-born anonymous addresses survive meaning reversal?

S17A showed that opposite local write effects are easy to detect and can trigger
a second anonymous slow band, but the one-phase task did not *need* the split.
This is the actual hard test: SWAPPED -> NORMAL -> SWAPPED.

The system starts with one slow sheet.  A successful write is bracketed by a
reference read, producing a finite intervention delta.  Each anonymous band
remembers only its most recent write-effect vector (no cue label, target label,
or desired output).  If a later successful write on that same band points more
than 90 degrees against the remembered effect, the band may split.

PREWRITE mode preserves the immediately pre-intervention material as the new
branch.  POSTWRITE mode clones the modified material instead.  The effect
fingerprints and alarm state are transient and erased before final evaluation;
only the resulting material bands remain.

If this passes where one shared sheet fails, structural address creation—not a
controller-side cue->band table—has become sufficient for reversal.  If it
fails, then address *birth* and address *current relevance* are separate
problems.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np

from emergent_address_probe import (
    EmergentAddressConfig,
    committed_choice,
    deterministic_state,
    noisy_accuracy,
    responses_for_cue,
)
from spatial_launcher import (
    NORMAL_CHANNEL,
    SWAPPED_CHANNEL,
    SpatialLaunchConfig,
    launcher_response,
    make_inherited_world,
    reinforce_route,
)


@dataclass(frozen=True)
class ReversalSplitConfig:
    cosine_threshold: float = -0.90
    max_bands: int = 4
    episodes_per_phase: int = 1800


def desired_launcher(cue: int, channel_map) -> int:
    return int(tuple(channel_map).index(int(cue)))


def margin_vector(world, channel_map, scfg: SpatialLaunchConfig) -> np.ndarray:
    values = []
    for cue in (0, 1):
        r = launcher_response(world, cue, config=scfg)
        c = desired_launcher(cue, channel_map)
        values.append(float(r[c] - r[1 - c]))
    return np.asarray(values, dtype=np.float64)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-30 or nb <= 1e-30:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


def run_cycle(seed: int, *, mode: str) -> dict:
    if mode not in {"no_split", "prewrite_split", "postwrite_split"}:
        raise ValueError(mode)

    rcfg = ReversalSplitConfig()
    ecfg = EmergentAddressConfig(episodes_per_phase=rcfg.episodes_per_phase)
    scfg = SpatialLaunchConfig()
    bands = [make_inherited_world(seed, scfg)]
    effect_memory: list[np.ndarray | None] = [None]
    alarm = np.zeros(2, dtype=np.float64)
    rng = np.random.default_rng(8_100_037 + int(seed))
    split_events = []
    phases = []

    for phase_name, channel_map in (
        ("learn_swapped", SWAPPED_CHANNEL),
        ("reverse_normal", NORMAL_CHANNEL),
        ("reverse_swapped_again", SWAPPED_CHANNEL),
    ):
        immediate = deterministic_state(bands, channel_map, scfg)
        rewards = np.zeros(rcfg.episodes_per_phase, dtype=np.float64)
        pokes = np.zeros(rcfg.episodes_per_phase, dtype=np.float64)
        writes = np.zeros(rcfg.episodes_per_phase, dtype=np.float64)
        phase_conflicts = []

        for ep in range(rcfg.episodes_per_phase):
            cue = int(rng.integers(2))
            responses = responses_for_cue(bands, cue, scfg)
            committed_band, committed_launcher = committed_choice(responses, rng)

            poke = bool(
                rng.random()
                < ecfg.explore_at_full_alarm * float(alarm[cue])
            )
            if poke:
                chosen_band = int(rng.integers(len(bands)))
                chosen_launcher = int(rng.integers(2))
            else:
                chosen_band = committed_band
                chosen_launcher = committed_launcher

            delivered = int(channel_map[chosen_launcher])
            reward = int(delivered == cue)
            rewards[ep] = reward
            pokes[ep] = float(poke)

            if reward:
                if alarm[cue] >= ecfg.plasticity_alarm_threshold:
                    target = bands[chosen_band]
                    reference = copy.deepcopy(target)
                    before = margin_vector(reference, channel_map, scfg)
                    reinforce_route(target, cue, chosen_launcher, config=scfg)
                    after = margin_vector(target, channel_map, scfg)
                    delta = after - before
                    writes[ep] = 1.0

                    old = effect_memory[chosen_band]
                    conflict = None
                    if old is not None:
                        conflict = cosine(old, delta)
                        phase_conflicts.append(conflict)

                    can_split = (
                        mode != "no_split"
                        and old is not None
                        and conflict is not None
                        and conflict <= rcfg.cosine_threshold
                        and len(bands) < rcfg.max_bands
                    )

                    if can_split:
                        if mode == "prewrite_split":
                            bands.append(reference)
                        else:
                            bands.append(copy.deepcopy(target))
                        # The new branch inherits the old effect history; the
                        # modified branch adopts the new conflicting effect.
                        effect_memory.append(copy.deepcopy(old))
                        effect_memory[chosen_band] = delta
                        split_events.append(
                            {
                                "phase": phase_name,
                                "episode": int(ep),
                                "source_band": int(chosen_band),
                                "new_band": int(len(bands) - 1),
                                "cosine": float(conflict),
                                "cue": int(cue),
                                "mode": mode,
                            }
                        )
                    else:
                        effect_memory[chosen_band] = delta

                alarm[cue] *= ecfg.success_decay
            else:
                alarm[cue] = 1.0

        state = deterministic_state(bands, channel_map, scfg)
        eval_rng = np.random.default_rng(
            8_200_041 + 101 * int(seed) + len(phases)
        )
        phases.append(
            {
                "phase": phase_name,
                "immediate_raw_accuracy": immediate["raw_accuracy"],
                "raw_accuracy": state["raw_accuracy"],
                "noisy_accuracy": noisy_accuracy(
                    bands, channel_map, scfg, ecfg, eval_rng
                ),
                "cue_bands": state["cue_bands"],
                "cue_bands_separated": state["cue_bands_separated"],
                "n_bands": len(bands),
                "last_200_reward": float(np.mean(rewards[-200:])),
                "last_200_poke": float(np.mean(pokes[-200:])),
                "last_200_write": float(np.mean(writes[-200:])),
                "min_phase_conflict_cosine": (
                    float(np.min(phase_conflicts))
                    if phase_conflicts
                    else None
                ),
            }
        )

    alarm[:] = 0.0
    effect_memory = [None for _ in bands]
    final = deterministic_state(bands, SWAPPED_CHANNEL, scfg)
    final_rng = np.random.default_rng(8_300_047 + int(seed))
    return {
        "seed": int(seed),
        "mode": mode,
        "phases": phases,
        "split_events": split_events,
        "final_n_bands": len(bands),
        "final_swapped_after_transients_erased": final,
        "final_swapped_noisy_accuracy": noisy_accuracy(
            bands, SWAPPED_CHANNEL, scfg, ecfg, final_rng
        ),
    }


def summarize(mode: str, n_seeds: int) -> dict:
    runs = [run_cycle(seed, mode=mode) for seed in range(n_seeds)]

    def pm(i: int, key: str) -> float:
        return float(np.mean([r["phases"][i][key] for r in runs]))

    return {
        "condition": mode,
        "n_seeds": int(n_seeds),
        "mean_final_bands": float(np.mean([r["final_n_bands"] for r in runs])),
        "split_count_all": [len(r["split_events"]) for r in runs],
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
        "last_200_poke": {
            "learn_swapped": pm(0, "last_200_poke"),
            "reverse_normal": pm(1, "last_200_poke"),
            "reverse_swapped_again": pm(2, "last_200_poke"),
        },
        "final_swapped_raw_all": [
            r["final_swapped_after_transients_erased"]["raw_accuracy"]
            for r in runs
        ],
        "final_swapped_noisy_all": [
            r["final_swapped_noisy_accuracy"] for r in runs
        ],
        "split_events": [r["split_events"] for r in runs],
    }


def main(n_seeds: int = 4) -> dict:
    return {
        "schema": "jellobrain/conflict-driven-address-reversal-v1",
        "question": (
            "Does conflict-triggered anonymous structural splitting make SWAPPED->NORMAL->SWAPPED reversal possible without a cue->band table?"
        ),
        "conditions": [
            summarize("no_split", n_seeds),
            summarize("prewrite_split", n_seeds),
            summarize("postwrite_split", n_seeds),
        ],
        "claim_boundary": (
            "The temporary alarm and write-effect fingerprints are erased before the final read. A pass would still be a toy structural-memory result, not a biological mechanism. Failure would separate address birth from relevance/reassignment."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
