#!/usr/bin/env python3
"""S18: separate stable address from current relevance at the launch boundary.

S17R can create anonymous slow bands when write-effect vectors conflict, but the
bands become archives: after outward meaning reverses, generic confidence still
prefers an old, strong route.  Address birth and *which address matters now* are
therefore separate problems.

S18 adds one scalar `relevance` per anonymous band.  It is deliberately NOT a
cue->band table.  The scalar is shared across all cues and stores only recent
consequence of that band's committed outputs:

- committed success: recover relevance;
- committed failure: reduce relevance strongly;
- successful exploratory POKE: recover relevance and write the tried route;
- failed exploratory POKE: do not punish the band (a random probe is not an
  established prediction).

Band selection is still based on its local launcher confidence, multiplied by
this current relevance.  Conflict-triggered PREWRITE splitting supplies new
anonymous bands exactly as in S17R.

The key attacker erases relevance back to uniform after learning while leaving
all material untouched.  If behavior collapses, the result is not "material
alone learned reversal"; it means stable route memory and current relevance are
distinct state variables.

This is a computational toy inspired by the earlier salience / AIS discussion.
It is not a model of chandelier cells, basal ganglia, attention, or biological
synaptic tagging.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np

from conflict_reversal_probe import cosine, margin_vector
from emergent_address_probe import EmergentAddressConfig, noisy_accuracy, responses_for_cue
from spatial_launcher import (
    NORMAL_CHANNEL,
    SWAPPED_CHANNEL,
    SpatialLaunchConfig,
    make_inherited_world,
    reinforce_route,
)


@dataclass(frozen=True)
class RelevanceConfig:
    episodes_per_phase: int = 1800
    cosine_threshold: float = -0.90
    max_bands: int = 4
    relevance_floor: float = 0.02
    committed_failure_decay: float = 0.10
    committed_success_recovery: float = 0.15
    poke_success_recovery: float = 0.35


def choose_band_and_launcher(
    responses: np.ndarray,
    relevance: np.ndarray,
    rng: np.random.Generator,
    cfg: RelevanceConfig,
) -> tuple[int, int]:
    launchers = np.argmax(responses, axis=1)
    margins = np.abs(responses[:, 1] - responses[:, 0])
    effective = margins * (cfg.relevance_floor + (1.0 - cfg.relevance_floor) * relevance)
    best = np.flatnonzero(np.isclose(effective, effective.max(), rtol=0.0, atol=1e-14))
    band = int(rng.choice(best))
    return band, int(launchers[band])


def policy_state(bands, relevance, channel_map, scfg, rcfg) -> dict:
    per_cue = []
    correct = 0
    # deterministic tie break for frozen evaluation
    for cue in (0, 1):
        responses = responses_for_cue(bands, cue, scfg)
        launchers = np.argmax(responses, axis=1)
        margins = np.abs(responses[:, 1] - responses[:, 0])
        effective = margins * (
            rcfg.relevance_floor
            + (1.0 - rcfg.relevance_floor) * np.asarray(relevance)
        )
        band = int(np.argmax(effective))
        launcher = int(launchers[band])
        delivered = int(channel_map[launcher])
        ok = int(delivered == cue)
        correct += ok
        per_cue.append(
            {
                "cue": cue,
                "band": band,
                "launcher": launcher,
                "correct": bool(ok),
                "confidence": margins.tolist(),
                "effective_confidence": effective.tolist(),
            }
        )
    return {
        "accuracy": correct / 2.0,
        "cue_bands": [p["band"] for p in per_cue],
        "per_cue": per_cue,
    }


def noisy_policy_accuracy(
    bands,
    relevance,
    channel_map,
    scfg,
    rcfg,
    ecfg,
    rng,
) -> float:
    total = 0
    correct = 0
    rel = np.asarray(relevance, dtype=np.float64)
    for cue in (0, 1):
        base = responses_for_cue(bands, cue, scfg)
        for _ in range(ecfg.noise_samples_per_cue):
            noisy = base + rng.normal(0.0, ecfg.readout_noise_sigma, size=base.shape)
            launchers = np.argmax(noisy, axis=1)
            margins = np.abs(noisy[:, 1] - noisy[:, 0])
            effective = margins * (
                rcfg.relevance_floor + (1.0 - rcfg.relevance_floor) * rel
            )
            band = int(np.argmax(effective))
            launcher = int(launchers[band])
            delivered = int(channel_map[launcher])
            correct += int(delivered == cue)
            total += 1
    return float(correct / total)


def recover(value: float, rate: float) -> float:
    return float(value + rate * (1.0 - value))


def run_cycle(seed: int, *, use_relevance: bool) -> dict:
    rcfg = RelevanceConfig()
    ecfg = EmergentAddressConfig(episodes_per_phase=rcfg.episodes_per_phase)
    scfg = SpatialLaunchConfig()
    bands = [make_inherited_world(seed, scfg)]
    effect_memory: list[np.ndarray | None] = [None]
    relevance = [1.0]
    alarm = np.zeros(2, dtype=np.float64)
    rng = np.random.default_rng(9_100_061 + int(seed))
    split_events = []
    phases = []

    for phase_name, channel_map in (
        ("learn_swapped", SWAPPED_CHANNEL),
        ("reverse_normal", NORMAL_CHANNEL),
        ("reverse_swapped_again", SWAPPED_CHANNEL),
    ):
        immediate = policy_state(bands, relevance, channel_map, scfg, rcfg)
        rewards = np.zeros(rcfg.episodes_per_phase, dtype=np.float64)
        pokes = np.zeros(rcfg.episodes_per_phase, dtype=np.float64)
        writes = np.zeros(rcfg.episodes_per_phase, dtype=np.float64)

        for ep in range(rcfg.episodes_per_phase):
            cue = int(rng.integers(2))
            responses = responses_for_cue(bands, cue, scfg)
            if use_relevance:
                committed_band, committed_launcher = choose_band_and_launcher(
                    responses,
                    np.asarray(relevance),
                    rng,
                    rcfg,
                )
            else:
                # Same structural learner, but relevance is functionally frozen.
                committed_band, committed_launcher = choose_band_and_launcher(
                    responses,
                    np.ones(len(bands), dtype=np.float64),
                    rng,
                    rcfg,
                )

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
                if use_relevance:
                    rate = (
                        rcfg.poke_success_recovery
                        if poke
                        else rcfg.committed_success_recovery
                    )
                    relevance[chosen_band] = recover(relevance[chosen_band], rate)

                if alarm[cue] >= ecfg.plasticity_alarm_threshold:
                    target = bands[chosen_band]
                    reference = copy.deepcopy(target)
                    before = margin_vector(reference, channel_map, scfg)
                    reinforce_route(target, cue, chosen_launcher, config=scfg)
                    after = margin_vector(target, channel_map, scfg)
                    delta = after - before
                    writes[ep] = 1.0

                    old = effect_memory[chosen_band]
                    conflict = cosine(old, delta) if old is not None else None
                    if (
                        old is not None
                        and conflict is not None
                        and conflict <= rcfg.cosine_threshold
                        and len(bands) < rcfg.max_bands
                    ):
                        # PREWRITE split: one child preserves the reference that
                        # existed before the conflicting intervention.
                        bands.append(reference)
                        effect_memory.append(copy.deepcopy(old))
                        relevance.append(float(relevance[chosen_band]))
                        effect_memory[chosen_band] = delta
                        split_events.append(
                            {
                                "phase": phase_name,
                                "episode": int(ep),
                                "source_band": int(chosen_band),
                                "new_band": int(len(bands) - 1),
                                "cosine": float(conflict),
                                "cue": int(cue),
                            }
                        )
                    else:
                        effect_memory[chosen_band] = delta

                alarm[cue] *= ecfg.success_decay
            else:
                if use_relevance and not poke:
                    # Only a failed committed prediction loses trust. A random
                    # POKE is an experiment, not a promise.
                    relevance[chosen_band] *= rcfg.committed_failure_decay
                alarm[cue] = 1.0

        active_rel = relevance if use_relevance else [1.0] * len(bands)
        policy = policy_state(bands, active_rel, channel_map, scfg, rcfg)
        material_only = policy_state(
            bands, [1.0] * len(bands), channel_map, scfg, rcfg
        )
        eval_rng = np.random.default_rng(
            9_200_063 + 101 * int(seed) + len(phases)
        )
        phases.append(
            {
                "phase": phase_name,
                "immediate_policy_accuracy": immediate["accuracy"],
                "policy_accuracy": policy["accuracy"],
                "material_only_accuracy": material_only["accuracy"],
                "noisy_policy_accuracy": noisy_policy_accuracy(
                    bands,
                    active_rel,
                    channel_map,
                    scfg,
                    rcfg,
                    ecfg,
                    eval_rng,
                ),
                "cue_bands": policy["cue_bands"],
                "n_bands": len(bands),
                "relevance": [float(v) for v in active_rel],
                "last_200_reward": float(np.mean(rewards[-200:])),
                "last_200_poke": float(np.mean(pokes[-200:])),
                "last_200_write": float(np.mean(writes[-200:])),
            }
        )

    # Erase genuinely transient state. Relevance is deliberately retained: it
    # is the slow "what matters now" variable under test.
    alarm[:] = 0.0
    effect_memory = [None for _ in bands]
    active_rel = relevance if use_relevance else [1.0] * len(bands)
    final_policy = policy_state(bands, active_rel, SWAPPED_CHANNEL, scfg, rcfg)
    final_reset = policy_state(
        bands, [1.0] * len(bands), SWAPPED_CHANNEL, scfg, rcfg
    )
    final_rng = np.random.default_rng(9_300_067 + int(seed))
    return {
        "seed": int(seed),
        "use_relevance": bool(use_relevance),
        "phases": phases,
        "split_events": split_events,
        "final_n_bands": len(bands),
        "final_relevance": [float(v) for v in active_rel],
        "final_policy_accuracy": final_policy["accuracy"],
        "final_after_relevance_reset": final_reset["accuracy"],
        "final_noisy_policy_accuracy": noisy_policy_accuracy(
            bands,
            active_rel,
            SWAPPED_CHANNEL,
            scfg,
            rcfg,
            ecfg,
            final_rng,
        ),
    }


def summarize(use_relevance: bool, n_seeds: int) -> dict:
    runs = [run_cycle(seed, use_relevance=use_relevance) for seed in range(n_seeds)]

    def pm(i: int, key: str) -> float:
        return float(np.mean([r["phases"][i][key] for r in runs]))

    return {
        "condition": (
            "conflict_split_plus_relevance"
            if use_relevance
            else "conflict_split_relevance_frozen"
        ),
        "n_seeds": int(n_seeds),
        "mean_final_bands": float(np.mean([r["final_n_bands"] for r in runs])),
        "policy_accuracy": {
            "learn_swapped": pm(0, "policy_accuracy"),
            "reverse_normal": pm(1, "policy_accuracy"),
            "reverse_swapped_again": pm(2, "policy_accuracy"),
        },
        "material_only_accuracy": {
            "learn_swapped": pm(0, "material_only_accuracy"),
            "reverse_normal": pm(1, "material_only_accuracy"),
            "reverse_swapped_again": pm(2, "material_only_accuracy"),
        },
        "noisy_policy_accuracy": {
            "learn_swapped": pm(0, "noisy_policy_accuracy"),
            "reverse_normal": pm(1, "noisy_policy_accuracy"),
            "reverse_swapped_again": pm(2, "noisy_policy_accuracy"),
        },
        "last_200_reward": {
            "learn_swapped": pm(0, "last_200_reward"),
            "reverse_normal": pm(1, "last_200_reward"),
            "reverse_swapped_again": pm(2, "last_200_reward"),
        },
        "final_policy_all": [r["final_policy_accuracy"] for r in runs],
        "final_relevance_reset_all": [
            r["final_after_relevance_reset"] for r in runs
        ],
        "final_noisy_all": [r["final_noisy_policy_accuracy"] for r in runs],
        "final_relevance_all": [r["final_relevance"] for r in runs],
        "split_count_all": [len(r["split_events"]) for r in runs],
    }


def main(n_seeds: int = 4) -> dict:
    return {
        "schema": "jellobrain/address-plus-current-relevance-v1",
        "question": (
            "After conflict creates anonymous stable addresses, is a separate consequence-tracked relevance variable sufficient to select which old route matters under meaning reversal?"
        ),
        "conditions": [
            summarize(False, n_seeds),
            summarize(True, n_seeds),
        ],
        "claim_boundary": (
            "Relevance is one scalar per anonymous band, not a cue->band table. If it carries the current regime, the result explicitly demonstrates a two-state-factorization (stable route memory plus current relevance), not material-only continual learning or a biological ChC/AIS mechanism."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
