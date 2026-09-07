#!/usr/bin/env python3
"""S16: can useful band assignment emerge without a cue->band table?

S15 gave each cue an explicitly selected slow-material band and immediately
removed the S13 reversal failure.  That is a useful upper-bound control, but it
repeats an old result from the wider repo lineage: stable addressability helps.

S16 removes the semantic selector.

There are two initially identical slow JelloWorld bands.  A cue is presented to
BOTH.  Each band independently produces its C/D launcher response.  The launch
boundary contains only a generic competition rule:

    candidate launcher = argmax(C, D) within a band
    confidence         = |C - D|
    committed band     = band with the largest confidence

The rule never stores which cue should use which band and never sees the target
launcher.  After a failed consequence a transient alarm permits a random POKE
of one band/launcher pair.  A rewarded POKE writes only into the band that was
actually tried.  Success lets alarm decay again.

If interference makes one shared band brittle, repeated consequence should be
able to make different cue histories claim different anonymous material bands.
The assignment is read back from the slow material itself after the transient
alarm is erased.

Attackers:
- ONE BAND: current collapsed representational class.
- SCRAMBLED WRITE ADDRESS: exploration can find a rewarded action, but the
  resulting material write is deliberately sent to a random band, breaking the
  read/action/write address binding.

All arms traverse SWAPPED -> NORMAL -> SWAPPED so a positive result is not just
one-shot memorization.  The two bands are exact deep copies at birth; no band
has a planted cue role.

This is a toy self-assignment / interference diagnostic, not a biological
compartment, frequency-channel, dendritic, AIS, or cortical model.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np

from spatial_launcher import (
    NORMAL_CHANNEL,
    SWAPPED_CHANNEL,
    SpatialLaunchConfig,
    launcher_response,
    make_inherited_world,
    reinforce_route,
)


@dataclass(frozen=True)
class EmergentAddressConfig:
    episodes_per_phase: int = 1600
    explore_at_full_alarm: float = 0.80
    success_decay: float = 0.55
    plasticity_alarm_threshold: float = 0.10
    readout_noise_sigma: float = 0.001
    noise_samples_per_cue: int = 4000


def make_bands(seed: int, n_bands: int, scfg: SpatialLaunchConfig):
    base = make_inherited_world(seed, scfg)
    # Exact copies: role asymmetry is not planted in the material.
    return [copy.deepcopy(base) for _ in range(int(n_bands))]


def responses_for_cue(bands, cue: int, scfg: SpatialLaunchConfig) -> np.ndarray:
    return np.stack(
        [launcher_response(world, cue, config=scfg) for world in bands], axis=0
    )


def committed_choice(responses: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Generic anonymous-band competition; no cue->band state exists."""
    launchers = np.argmax(responses, axis=1)
    margins = np.abs(responses[:, 1] - responses[:, 0])
    best = np.flatnonzero(np.isclose(margins, margins.max(), rtol=0.0, atol=1e-14))
    band = int(rng.choice(best))
    return band, int(launchers[band])


def deterministic_state(bands, channel_map, scfg) -> dict:
    per_cue = []
    correct = 0
    for cue in (0, 1):
        responses = responses_for_cue(bands, cue, scfg)
        launchers = np.argmax(responses, axis=1)
        margins = np.abs(responses[:, 1] - responses[:, 0])
        band = int(np.argmax(margins))
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
                "band_margins": margins.tolist(),
                "band_responses": responses.tolist(),
            }
        )
    return {
        "raw_accuracy": correct / 2.0,
        "per_cue": per_cue,
        "cue_bands": [p["band"] for p in per_cue],
        "cue_bands_separated": bool(
            len(bands) > 1 and per_cue[0]["band"] != per_cue[1]["band"]
        ),
    }


def noisy_accuracy(bands, channel_map, scfg, cfg, rng) -> float:
    """Evaluate fixed material under S11-sized launcher readout noise."""
    total = 0
    correct = 0
    for cue in (0, 1):
        base = responses_for_cue(bands, cue, scfg)
        for _ in range(cfg.noise_samples_per_cue):
            noisy = base + rng.normal(0.0, cfg.readout_noise_sigma, size=base.shape)
            # Evaluation has no transient policy state.  Ties are practically
            # absent under continuous noise; deterministic first-index fallback
            # is harmless.
            launchers = np.argmax(noisy, axis=1)
            margins = np.abs(noisy[:, 1] - noisy[:, 0])
            band = int(np.argmax(margins))
            launcher = int(launchers[band])
            delivered = int(channel_map[launcher])
            correct += int(delivered == cue)
            total += 1
    return float(correct / total)


def run_phase(
    bands,
    alarm: np.ndarray,
    rng: np.random.Generator,
    channel_map,
    *,
    scramble_write_address: bool,
    cfg: EmergentAddressConfig,
    scfg: SpatialLaunchConfig,
) -> dict:
    n = cfg.episodes_per_phase
    rewards = np.zeros(n, dtype=np.float64)
    pokes = np.zeros(n, dtype=np.float64)
    writes = np.zeros(n, dtype=np.float64)
    write_mismatch = np.zeros(n, dtype=np.float64)
    chosen_bands = np.full(n, -1, dtype=np.int64)

    for ep in range(n):
        cue = int(rng.integers(2))
        responses = responses_for_cue(bands, cue, scfg)
        committed_band, committed_launcher = committed_choice(responses, rng)

        poke = bool(rng.random() < cfg.explore_at_full_alarm * float(alarm[cue]))
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
        chosen_bands[ep] = chosen_band

        if reward:
            if alarm[cue] >= cfg.plasticity_alarm_threshold:
                write_band = chosen_band
                if scramble_write_address and len(bands) > 1:
                    write_band = int(rng.integers(len(bands)))
                    write_mismatch[ep] = float(write_band != chosen_band)
                reinforce_route(
                    bands[write_band], cue, chosen_launcher, config=scfg
                )
                writes[ep] = 1.0
            alarm[cue] *= cfg.success_decay
        else:
            alarm[cue] = 1.0

    state = deterministic_state(bands, channel_map, scfg)
    state.update(
        {
            "first_200_reward": float(np.mean(rewards[:200])),
            "last_200_reward": float(np.mean(rewards[-200:])),
            "first_200_poke_rate": float(np.mean(pokes[:200])),
            "last_200_poke_rate": float(np.mean(pokes[-200:])),
            "last_200_write_rate": float(np.mean(writes[-200:])),
            "write_address_mismatch_rate": float(
                np.sum(write_mismatch) / max(1.0, np.sum(writes))
            ),
            "band0_choice_fraction": float(np.mean(chosen_bands == 0)),
            "total_writes": int(np.sum(writes)),
        }
    )
    return state


def run_cycle(seed: int, *, n_bands: int, scramble_write_address: bool) -> dict:
    cfg = EmergentAddressConfig()
    scfg = SpatialLaunchConfig()
    bands = make_bands(seed, n_bands, scfg)
    alarm = np.zeros(2, dtype=np.float64)
    rng = np.random.default_rng(6_000_127 + int(seed))

    phases = []
    for name, channel in (
        ("learn_swapped", SWAPPED_CHANNEL),
        ("reverse_normal", NORMAL_CHANNEL),
        ("reverse_swapped_again", SWAPPED_CHANNEL),
    ):
        immediate = deterministic_state(bands, channel, scfg)
        phase = run_phase(
            bands,
            alarm,
            rng,
            channel,
            scramble_write_address=scramble_write_address,
            cfg=cfg,
            scfg=scfg,
        )
        eval_rng = np.random.default_rng(6_100_003 + 17 * int(seed) + len(phases))
        phase["noisy_accuracy"] = noisy_accuracy(
            bands, channel, scfg, cfg, eval_rng
        )
        phases.append(
            {
                "phase": name,
                "immediate_raw_accuracy": immediate["raw_accuracy"],
                "immediate_cue_bands": immediate["cue_bands"],
                **phase,
            }
        )

    # Erase the only transient contentful state.  Band selection at evaluation
    # is recomputed from material response; no cue->band memory survives here.
    alarm[:] = 0.0
    final = deterministic_state(bands, SWAPPED_CHANNEL, scfg)
    final_rng = np.random.default_rng(6_200_011 + int(seed))
    return {
        "seed": int(seed),
        "n_bands": int(n_bands),
        "scramble_write_address": bool(scramble_write_address),
        "phases": phases,
        "final_swapped_after_alarm_erased": final,
        "final_swapped_noisy_accuracy": noisy_accuracy(
            bands, SWAPPED_CHANNEL, scfg, cfg, final_rng
        ),
    }


def summarize(label: str, *, n_bands: int, scramble_write_address: bool, n_seeds: int = 4):
    runs = [
        run_cycle(
            seed,
            n_bands=n_bands,
            scramble_write_address=scramble_write_address,
        )
        for seed in range(n_seeds)
    ]

    def pm(i: int, key: str) -> float:
        return float(np.mean([r["phases"][i][key] for r in runs]))

    return {
        "condition": label,
        "n_seeds": int(n_seeds),
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
            "learn_swapped": pm(0, "last_200_poke_rate"),
            "reverse_normal": pm(1, "last_200_poke_rate"),
            "reverse_swapped_again": pm(2, "last_200_poke_rate"),
        },
        "separated_assignment_fraction": {
            "learn_swapped": float(np.mean([r["phases"][0]["cue_bands_separated"] for r in runs])),
            "reverse_normal": float(np.mean([r["phases"][1]["cue_bands_separated"] for r in runs])),
            "reverse_swapped_again": float(np.mean([r["phases"][2]["cue_bands_separated"] for r in runs])),
        },
        "final_swapped_raw_all": [
            r["final_swapped_after_alarm_erased"]["raw_accuracy"] for r in runs
        ],
        "final_swapped_noisy_all": [r["final_swapped_noisy_accuracy"] for r in runs],
        "final_cue_bands": [
            r["final_swapped_after_alarm_erased"]["cue_bands"] for r in runs
        ],
    }


def main(n_seeds: int = 4) -> dict:
    return {
        "schema": "jellobrain/emergent-anonymous-address-v1",
        "claim_boundary": (
            "S16 supplies two physically separate anonymous slow bands but does not supply a cue-to-band mapping. It tests whether consequence-driven local writing can make useful band assignment emerge. It does not show that the bands themselves can emerge from one undifferentiated medium."
        ),
        "conditions": [
            summarize(
                "one_band_collapsed",
                n_bands=1,
                scramble_write_address=False,
                n_seeds=n_seeds,
            ),
            summarize(
                "two_anonymous_bands_bound_write",
                n_bands=2,
                scramble_write_address=False,
                n_seeds=n_seeds,
            ),
            summarize(
                "two_anonymous_bands_scrambled_write_address",
                n_bands=2,
                scramble_write_address=True,
                n_seeds=n_seeds,
            ),
        ],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
