#!/usr/bin/env python3
"""S20: assurance needs a receipt saying what is currently safe to preserve.

S19 implemented the old copy/delta intuition literally: after a successful POKE,
try the write on internal copies and choose the address that least changes the
other cue's current behavior.  It failed reversal *worse* than random writing.
The reason is precise: during a regime change, the other cue's current behavior
may itself be obsolete, so blind assurance protects the wrong thing.

S20 adds the smallest missing object: a CURRENT VALIDATION RECEIPT.

A receipt is one bit per externally encountered cue:

    "the material's committed action for this cue has recently succeeded"

It contains no desired launcher and no storage-band identity.  A random POKE
cannot mint a receipt; only a non-exploratory committed success can.  A failed
committed prediction can invalidate receipts because the old assurance set may
no longer describe the current world.

When allocating a successful write, copy/delta collateral protection is applied
ONLY to other cues carrying a live receipt.  Unvalidated behavior is allowed to
change.

Conditions:
- BLIND ASSURANCE: protect the other cue whether validated or not (S19 logic).
- RECEIPT + GLOBAL INVALIDATION: any failed committed prediction clears the old
  receipt set; new committed successes rebuild it.
- STALE RECEIPTS: failure invalidates only the failing cue, leaving the other
  old receipt untouched.  This attacks whether "currently validated" matters.

The material still begins as one sheet.  Opposed physical write deltas can
create anonymous PREWRITE branches exactly as in S19.  Receipts and transient
effect memories are erased before final evaluation; only material remains.

This is a toy causal-learning audit, not a biological claim.  The receipt idea
has explicit ancestry in the repo family (especially Child); S20 tests whether
that old object is the missing control on the new copy/delta allocator.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np

from copy_delta_allocator import (
    cosine,
    evaluate,
    margin_toward,
    raw_effect_vector,
    system_signature,
)
from emergent_address_probe import EmergentAddressConfig, noisy_accuracy
from spatial_launcher import (
    NORMAL_CHANNEL,
    SWAPPED_CHANNEL,
    SpatialLaunchConfig,
    launcher_response,
    make_inherited_world,
    reinforce_route,
)


@dataclass(frozen=True)
class ReceiptConfig:
    episodes_per_phase: int = 1800
    cosine_threshold: float = -0.90
    max_bands: int = 4


def counterfactual_metrics(
    bands,
    candidate_band: int,
    cue: int,
    successful_launcher: int,
    protected_cues: list[int],
    scfg: SpatialLaunchConfig,
) -> dict:
    before_protected = {
        c: system_signature(bands, c, scfg) for c in protected_cues
    }
    trial = [copy.deepcopy(b) for b in bands]
    before_current = launcher_response(trial[candidate_band], cue, config=scfg)
    reinforce_route(
        trial[candidate_band], cue, successful_launcher, config=scfg
    )
    after_current = launcher_response(trial[candidate_band], cue, config=scfg)
    self_gain = (
        margin_toward(after_current, successful_launcher)
        - margin_toward(before_current, successful_launcher)
    )

    flips = 0
    collateral = 0.0
    for c in protected_cues:
        before = before_protected[c]
        after = system_signature(trial, c, scfg)
        flips += int(after["launcher"] != before["launcher"])
        denom = max(abs(before["margin"]), 1e-9)
        collateral += abs(after["margin"] - before["margin"]) / denom

    return {
        "candidate_band": int(candidate_band),
        "self_gain": float(self_gain),
        "protected_launcher_flips": int(flips),
        "protected_margin_change_relative": float(collateral),
    }


def choose_write_band(
    bands,
    cue: int,
    successful_launcher: int,
    protected_cues: list[int],
    scfg: SpatialLaunchConfig,
) -> int:
    metrics = [
        counterfactual_metrics(
            bands,
            b,
            cue,
            successful_launcher,
            protected_cues,
            scfg,
        )
        for b in range(len(bands))
    ]
    # If there is no validated thing to preserve, simply choose the address in
    # which the just-successful action gains most.  Otherwise safety is
    # lexicographically prior to self-gain.
    if not protected_cues:
        best = max(metrics, key=lambda m: (m["self_gain"], -m["candidate_band"]))
    else:
        best = min(
            metrics,
            key=lambda m: (
                m["protected_launcher_flips"],
                m["protected_margin_change_relative"],
                -m["self_gain"],
                m["candidate_band"],
            ),
        )
    return int(best["candidate_band"])


def maybe_split(
    bands,
    effect_memory,
    written_band: int,
    reference,
    before_effect,
    scfg,
    cfg,
    split_events,
    phase_name,
    ep,
    cue,
):
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
        bands.append(reference)
        effect_memory.append(copy.deepcopy(old))
        effect_memory[written_band] = delta
        split_events.append(
            {
                "phase": phase_name,
                "episode": int(ep),
                "source_band": int(written_band),
                "new_band": int(len(bands) - 1),
                "cosine": float(conflict),
                "cue": int(cue),
            }
        )
    else:
        effect_memory[written_band] = delta


def run_cycle(seed: int, *, mode: str) -> dict:
    if mode not in {"blind_assurance", "receipt_global_clear", "receipt_local_clear"}:
        raise ValueError(mode)

    cfg = ReceiptConfig()
    ecfg = EmergentAddressConfig(episodes_per_phase=cfg.episodes_per_phase)
    scfg = SpatialLaunchConfig()
    bands = [make_inherited_world(seed, scfg)]
    effect_memory: list[np.ndarray | None] = [None]
    receipt = np.zeros(2, dtype=bool)
    alarm = np.zeros(2, dtype=np.float64)
    rng = np.random.default_rng(11_100_097 + int(seed))
    split_events = []
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
        receipt_clears = 0

        for ep in range(cfg.episodes_per_phase):
            cue = int(rng.integers(2))
            committed = system_signature(bands, cue, scfg)
            poke = bool(
                rng.random()
                < ecfg.explore_at_full_alarm * float(alarm[cue])
            )
            chosen_launcher = (
                int(rng.integers(2)) if poke else int(committed["launcher"])
            )
            delivered = int(channel_map[chosen_launcher])
            reward = int(delivered == cue)
            rewards[ep] = reward
            pokes[ep] = float(poke)

            if reward:
                if alarm[cue] >= ecfg.plasticity_alarm_threshold:
                    if mode == "blind_assurance":
                        protected = [1 - cue]
                    else:
                        protected = [
                            c for c in (0, 1)
                            if c != cue and bool(receipt[c])
                        ]

                    written_band = choose_write_band(
                        bands,
                        cue,
                        chosen_launcher,
                        protected,
                        scfg,
                    )
                    reference = copy.deepcopy(bands[written_band])
                    before_effect = raw_effect_vector(reference, scfg)
                    reinforce_route(
                        bands[written_band], cue, chosen_launcher, config=scfg
                    )
                    writes[ep] = 1.0
                    maybe_split(
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

                # Only an ordinary committed success validates the currently
                # embodied output. A successful random experiment does not.
                if not poke and mode != "blind_assurance":
                    receipt[cue] = True
                alarm[cue] *= ecfg.success_decay
            else:
                if not poke and mode != "blind_assurance":
                    if mode == "receipt_global_clear":
                        if np.any(receipt):
                            receipt_clears += 1
                        receipt[:] = False
                    else:
                        if receipt[cue]:
                            receipt_clears += 1
                        receipt[cue] = False
                alarm[cue] = 1.0

        state = evaluate(bands, channel_map, scfg)
        eval_rng = np.random.default_rng(
            11_200_101 + 101 * int(seed) + len(phases)
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
                "receipt": receipt.astype(int).tolist(),
                "receipt_clears": int(receipt_clears),
                "last_200_reward": float(np.mean(rewards[-200:])),
                "last_200_poke": float(np.mean(pokes[-200:])),
                "last_200_write": float(np.mean(writes[-200:])),
            }
        )

    # Receipts are an online assurance state, not part of final memory claim.
    receipt[:] = False
    alarm[:] = 0.0
    effect_memory = [None for _ in bands]
    final = evaluate(bands, SWAPPED_CHANNEL, scfg)
    final_rng = np.random.default_rng(11_300_103 + int(seed))
    return {
        "seed": int(seed),
        "mode": mode,
        "phases": phases,
        "split_events": split_events,
        "final_n_bands": len(bands),
        "final_accuracy_after_receipts_erased": final["accuracy"],
        "final_noisy_accuracy_after_receipts_erased": noisy_accuracy(
            bands, SWAPPED_CHANNEL, scfg, ecfg, final_rng
        ),
        "final_cue_bands": [row["band"] for row in final["rows"]],
    }


def summarize(mode: str, n_seeds: int) -> dict:
    runs = [run_cycle(seed, mode=mode) for seed in range(n_seeds)]

    def pm(i: int, key: str) -> float:
        return float(np.mean([r["phases"][i][key] for r in runs]))

    return {
        "condition": mode,
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
        "final_raw_all": [
            r["final_accuracy_after_receipts_erased"] for r in runs
        ],
        "final_noisy_all": [
            r["final_noisy_accuracy_after_receipts_erased"] for r in runs
        ],
        "final_cue_bands": [r["final_cue_bands"] for r in runs],
        "split_count_all": [len(r["split_events"]) for r in runs],
    }


def main(n_seeds: int = 4) -> dict:
    return {
        "schema": "jellobrain/receipt-gated-copy-delta-v1",
        "question": (
            "Does copy/delta assurance work under meaning reversal once collateral protection is restricted to behaviors that are currently validated by consequence?"
        ),
        "conditions": [
            summarize("blind_assurance", n_seeds),
            summarize("receipt_local_clear", n_seeds),
            summarize("receipt_global_clear", n_seeds),
        ],
        "claim_boundary": (
            "Receipts contain recent binary validation only; they store neither desired launcher nor cue->band identity and are erased before final evaluation. Global invalidation is a generic change-point assumption and must be attacked later with local/non-global changes if it is useful."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
