#!/usr/bin/env python3
"""S9: mismatch-gated POKE as a transient bootstrap for spatial route repair.

This experiment returns to the full spatial JelloWorld after S5-S8.

The key constraint is that the transient controller is NOT allowed to store the
correct pulse identity.  It stores only a scalar alarm per private cue:

    "my ordinary launch just failed -- sample alternatives"

When alarm is high, the boundary occasionally emits an exploratory POKE instead
of blindly following the strongest inherited route.  If that exploratory launch
succeeds, the ordinary JelloWorld plasticity rule writes the experienced
cue->launcher route.  Alarm then decays on successful behavior.

The final test zeros the alarm completely and reads the body with the ordinary
raw spatial launcher.  Therefore S9 passes only if the policy has moved into the
spatial material itself.

This is a computational toy, not a chandelier-cell, AIS, or biological mismatch
model.
"""

from __future__ import annotations

from dataclasses import dataclass
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
class PokeConfig:
    episodes: int = 500
    alarm_up: float = 1.0
    success_decay: float = 0.82
    explore_at_full_alarm: float = 0.80
    always_explore_probability: float = 0.80


def train_mismatch_poke(
    seed: int,
    *,
    write_material: bool = True,
    update_alarm_from_consequence: bool = True,
    always_explore: bool = False,
    poke_config: PokeConfig | None = None,
    spatial_config: SpatialLaunchConfig | None = None,
) -> dict:
    """Repair the swapped channel using a transient failure-triggered POKE.

    The controller knows only whether the previous chosen action succeeded.  It
    never receives the target launcher identity.  Exploration samples an action
    uniformly; binary task structure is not used to choose "the opposite" port.
    """
    pcfg = poke_config or PokeConfig()
    scfg = spatial_config or SpatialLaunchConfig()
    world = make_inherited_world(seed, scfg)
    rng = np.random.default_rng(900_001 + int(seed))

    alarm = np.zeros(2, dtype=np.float64)
    rewards = np.zeros(pcfg.episodes, dtype=np.float64)
    pokes = np.zeros(pcfg.episodes, dtype=np.float64)
    raw_choices = np.zeros(pcfg.episodes, dtype=np.int64)
    chosen_actions = np.zeros(pcfg.episodes, dtype=np.int64)

    for ep in range(pcfg.episodes):
        cue = int(rng.integers(2))
        observed = launcher_response(world, cue, config=scfg)
        raw_choice = int(np.argmax(observed))
        raw_choices[ep] = raw_choice

        if always_explore:
            explore_probability = pcfg.always_explore_probability
        else:
            explore_probability = pcfg.explore_at_full_alarm * alarm[cue]

        poke = bool(rng.random() < explore_probability)
        if poke:
            # Deliberately sample uniformly. The controller is not told which
            # alternative is correct.
            chosen = int(rng.integers(2))
        else:
            chosen = raw_choice

        delivered = int(SWAPPED_CHANNEL[chosen])
        reward = int(delivered == cue)
        rewards[ep] = reward
        pokes[ep] = float(poke)
        chosen_actions[ep] = chosen

        if reward and write_material:
            reinforce_route(world, cue, chosen, config=scfg)

        if update_alarm_from_consequence and not always_explore:
            if reward:
                alarm[cue] *= pcfg.success_decay
            else:
                alarm[cue] = min(1.0, alarm[cue] + pcfg.alarm_up)

    # Critical handoff test: erase the transient controller entirely.
    alarm[:] = 0.0
    final_raw = raw_greedy_accuracy(world, SWAPPED_CHANNEL, config=scfg)

    def window(x: np.ndarray, start: int, stop: int) -> float:
        return float(np.mean(x[start:stop]))

    return {
        "seed": int(seed),
        "write_material": bool(write_material),
        "update_alarm_from_consequence": bool(update_alarm_from_consequence),
        "always_explore": bool(always_explore),
        "episodes": pcfg.episodes,
        "final_raw_greedy_after_alarm_erased": float(final_raw),
        "first_100_reward": window(rewards, 0, 100),
        "last_100_reward": window(rewards, -100, None),
        "first_100_poke_rate": window(pokes, 0, 100),
        "last_100_poke_rate": window(pokes, -100, None),
        "total_poke_rate": float(np.mean(pokes)),
        "final_alarm": alarm.tolist(),
    }
