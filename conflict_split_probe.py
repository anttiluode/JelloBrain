#!/usr/bin/env python3
"""S17A: can conflicting write deltas cause an address to be born?

This gate deliberately starts with ONE slow JelloWorld sheet.  There is no
preallocated cue->band table and no second band at birth.

For a successful local write we keep a temporary reference copy of the chosen
sheet, apply the ordinary JelloWorld write, and re-read the two cue margins.
That gives a finite intervention vector

    delta = read(after write) - read(before write)

If two successive successful-write deltas are strongly opposed, the system has
evidence that one writable coordinate is serving incompatible consequences.
Only then is a second slow band created.

The key comparison is how that band is born:

- PREWRITE COPY: keep the reference sheet from immediately before the
  conflicting write while the original sheet keeps the post-write state.
- POSTWRITE COPY: duplicate the already modified sheet instead.  This tests
  whether preserving the pre-intervention state is specifically useful, rather
  than mere capacity expansion.
- NO SPLIT: keep one shared sheet.

After a split, both bands are anonymous and both see every cue.  A generic
confidence competition selects the committed band exactly as in S16.  The
transient delta history and alarm are erased before final evaluation.

This is a toy conflict-driven structural-duplication experiment.  It is not a
claim that brains literally clone dendrites, neurons, AISs, or cortical
circuits when write vectors disagree.
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
    SWAPPED_CHANNEL,
    SpatialLaunchConfig,
    launcher_response,
    make_inherited_world,
    reinforce_route,
)


@dataclass(frozen=True)
class SplitConfig:
    cosine_threshold: float = -0.90
    max_bands: int = 2


def signed_margin(world, cue: int, scfg: SpatialLaunchConfig) -> float:
    r = launcher_response(world, cue, config=scfg)
    # Under SWAPPED_CHANNEL, cue0 requires launcher1 and cue1 launcher0.
    correct = int(SWAPPED_CHANNEL.index(int(cue)))
    wrong = 1 - correct
    return float(r[correct] - r[wrong])


def read_vector(world, scfg: SpatialLaunchConfig) -> np.ndarray:
    return np.asarray([signed_margin(world, cue, scfg) for cue in (0, 1)])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-30 or nb <= 1e-30:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


def run_one(seed: int, *, mode: str) -> dict:
    if mode not in {"no_split", "prewrite_copy", "postwrite_copy"}:
        raise ValueError(mode)

    cfg = EmergentAddressConfig(episodes_per_phase=1800)
    scfg = SpatialLaunchConfig()
    split_cfg = SplitConfig()
    bands = [make_inherited_world(seed, scfg)]
    alarm = np.zeros(2, dtype=np.float64)
    rng = np.random.default_rng(7_300_019 + int(seed))

    rewards = np.zeros(cfg.episodes_per_phase, dtype=np.float64)
    pokes = np.zeros(cfg.episodes_per_phase, dtype=np.float64)
    writes = np.zeros(cfg.episodes_per_phase, dtype=np.float64)
    last_delta = None
    split_event = None
    delta_cosines = []

    for ep in range(cfg.episodes_per_phase):
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

        delivered = int(SWAPPED_CHANNEL[chosen_launcher])
        reward = int(delivered == cue)
        rewards[ep] = reward
        pokes[ep] = float(poke)

        if reward:
            if alarm[cue] >= cfg.plasticity_alarm_threshold:
                target = bands[chosen_band]
                reference = copy.deepcopy(target)
                before = read_vector(reference, scfg)

                reinforce_route(target, cue, chosen_launcher, config=scfg)
                after = read_vector(target, scfg)
                delta = after - before
                writes[ep] = 1.0

                if last_delta is not None:
                    c = cosine(last_delta, delta)
                    delta_cosines.append(c)
                    if (
                        mode != "no_split"
                        and len(bands) < split_cfg.max_bands
                        and c <= split_cfg.cosine_threshold
                    ):
                        if mode == "prewrite_copy":
                            bands.append(reference)
                        else:
                            bands.append(copy.deepcopy(target))
                        split_event = {
                            "episode": int(ep),
                            "cosine": float(c),
                            "cue": int(cue),
                            "mode": mode,
                        }
                last_delta = delta

            alarm[cue] *= cfg.success_decay
        else:
            alarm[cue] = 1.0

    # Erase the transient controller/reference state before judging what the
    # material itself learned.
    alarm[:] = 0.0
    last_delta = None

    final = deterministic_state(bands, SWAPPED_CHANNEL, scfg)
    eval_rng = np.random.default_rng(7_400_031 + int(seed))
    noisy = noisy_accuracy(bands, SWAPPED_CHANNEL, scfg, cfg, eval_rng)

    return {
        "seed": int(seed),
        "mode": mode,
        "n_bands_final": len(bands),
        "split_event": split_event,
        "raw_accuracy": final["raw_accuracy"],
        "noisy_accuracy": noisy,
        "cue_bands": final["cue_bands"],
        "cue_bands_separated": final["cue_bands_separated"],
        "first_200_reward": float(np.mean(rewards[:200])),
        "last_200_reward": float(np.mean(rewards[-200:])),
        "last_200_poke": float(np.mean(pokes[-200:])),
        "last_200_write": float(np.mean(writes[-200:])),
        "total_writes": int(np.sum(writes)),
        "min_observed_delta_cosine": (
            float(np.min(delta_cosines)) if delta_cosines else None
        ),
    }


def summarize(mode: str, n_seeds: int) -> dict:
    runs = [run_one(seed, mode=mode) for seed in range(n_seeds)]
    return {
        "condition": mode,
        "n_seeds": int(n_seeds),
        "split_fraction": float(np.mean([r["n_bands_final"] > 1 for r in runs])),
        "mean_split_episode": float(
            np.mean([
                r["split_event"]["episode"]
                for r in runs
                if r["split_event"] is not None
            ])
        ) if any(r["split_event"] is not None for r in runs) else None,
        "mean_raw_accuracy": float(np.mean([r["raw_accuracy"] for r in runs])),
        "mean_noisy_accuracy": float(np.mean([r["noisy_accuracy"] for r in runs])),
        "mean_last_200_reward": float(np.mean([r["last_200_reward"] for r in runs])),
        "mean_last_200_poke": float(np.mean([r["last_200_poke"] for r in runs])),
        "mean_last_200_write": float(np.mean([r["last_200_write"] for r in runs])),
        "separated_assignment_fraction": float(
            np.mean([r["cue_bands_separated"] for r in runs])
        ),
        "cue_bands": [r["cue_bands"] for r in runs],
        "noisy_accuracy_all": [r["noisy_accuracy"] for r in runs],
        "split_events": [r["split_event"] for r in runs],
        "min_delta_cosine_all": [r["min_observed_delta_cosine"] for r in runs],
    }


def main(n_seeds: int = 8) -> dict:
    return {
        "schema": "jellobrain/conflict-driven-address-birth-v1",
        "question": (
            "Can a single writable sheet detect opposing intervention deltas and create a second anonymous slow address only when interference appears?"
        ),
        "conditions": [
            summarize("no_split", n_seeds),
            summarize("prewrite_copy", n_seeds),
            summarize("postwrite_copy", n_seeds),
        ],
        "claim_boundary": (
            "A positive result would show conflict-triggered structural duplication in this toy, not biological neurogenesis/branching or a general solution to continual learning. The post-write-copy attacker tests whether preserving the pre-intervention reference is specifically needed."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
