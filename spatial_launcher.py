#!/usr/bin/env python3
"""S-series: an AIS-like launch audit on the full spatial JelloWorld.

The audit isolates one boundary before returning to two full spatial bodies.
A private cue enters a spatial body.  Two output ports (C/D) are candidate
axon-like launchers.  The body is born with two strong horizontal lanes, so its
virgin geometry already prefers A->C and F->D.  That inherited bias is the S0
shortcut.

A narrow two-wire channel maps chosen launcher identity to a receiver.  The
receiver is deliberately trivial here: delivered pulse identity is the action.
This keeps the experiment about the sender's spatial launch boundary rather
than hiding another learning problem downstream.

The negative-image mechanism is a learned predictor of the *self-generated
launcher response* given the cue/corollary context.  The residual

    innovation = log(observed launcher response) - predicted log response

is used for exploration/credit.  Rewarded traffic is replayed through the same
JelloWorld with normal local material writing; no alternate route is edited by
hand after initialization.

This is a computational test inspired by adaptive sensory cancellation in
mormyrid electric fish and by the axon initial segment as an action-potential
launch/boundary compartment.  It is not a biological model of either system.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np

from jellobrain import JelloConfig, JelloWorld


CUE_PORTS = ("A", "F")
LAUNCH_PORTS = ("C", "D")
NORMAL_CHANNEL = (0, 1)
SWAPPED_CHANNEL = (1, 0)


@dataclass(frozen=True)
class SpatialLaunchConfig:
    lane_material: float = 0.15
    lane_half_width: int = 1
    deposit_rate: float = 0.02
    probe_steps: int = 16
    reward_replays: int = 1
    train_episodes: int = 250
    start_temperature: float = 0.50
    end_temperature: float = 0.01
    predictor_track_rate: float = 0.0002
    calibration_repeats: int = 8
    calibration_gain_jitter: float = 0.03
    eps: float = 1e-12


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    t = max(float(temperature), 1e-8)
    z = np.asarray(values, dtype=np.float64) / t
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def make_inherited_world(seed: int, config: SpatialLaunchConfig | None = None) -> JelloWorld:
    """Create a JelloWorld with an intentionally inherited two-lane shortcut.

    The only hand-built structure is the *starting* geography.  This is not a
    performance trick: S0 exists specifically to demonstrate how fixed geometry
    can masquerade as learned protocol.  All post-birth route changes use the
    stock JelloWorld local plasticity rule.
    """
    cfg = config or SpatialLaunchConfig()
    base = JelloConfig(deposit_rate=cfg.deposit_rate)
    world = JelloWorld(seed, base)
    n = world.config.n

    # The A/C and F/D primitive ports live near rows 3 and n-4 respectively.
    # Strengthen horizontal conductance in narrow bands connecting each pair.
    for centre in (3, n - 4):
        for row in range(
            max(0, centre - cfg.lane_half_width),
            min(n, centre + cfg.lane_half_width + 1),
        ):
            world.east[row, :] = cfg.lane_material
            world.west[row, :] = cfg.lane_material
    world.wipe_fast()
    return world


def launcher_response(
    world: JelloWorld,
    cue: int,
    *,
    config: SpatialLaunchConfig | None = None,
    gain: float = 1.0,
) -> np.ndarray:
    """Measure C/D launcher excitation after a cue, with material frozen."""
    cfg = config or SpatialLaunchConfig()
    cue = int(cue)
    world.wipe_fast()
    world.inject(CUE_PORTS[cue], gain=float(gain))
    for _ in range(cfg.probe_steps):
        world.step(write=False)
    values = []
    for port in LAUNCH_PORTS:
        mask = world._masks[world.symbol_index(port)]
        values.append(float(np.sum(world.x * mask)))
    return np.asarray(values, dtype=np.float64)


def raw_greedy_accuracy(
    world: JelloWorld,
    channel_map: tuple[int, int] = NORMAL_CHANNEL,
    *,
    config: SpatialLaunchConfig | None = None,
) -> float:
    """Accuracy when the launcher simply picks the strongest raw output."""
    correct = 0
    for cue in (0, 1):
        chosen = int(np.argmax(launcher_response(world, cue, config=config)))
        delivered = int(channel_map[chosen])
        correct += int(delivered == cue)
    return correct / 2.0


class NegativeImage:
    """Tiny adaptive predictor of expected self-generated launcher activity.

    `prediction[cue]` stores expected log C/D response for the cue-associated
    corollary context.  Calibration is experience with the body's own virgin
    response, not access to its material arrays.  During repair it can track
    slowly; a fast-tracking attacker is tested separately.
    """

    def __init__(self):
        self.prediction = np.zeros((2, 2), dtype=np.float64)
        self._seen = np.zeros(2, dtype=np.int64)

    def clone(self) -> "NegativeImage":
        return copy.deepcopy(self)

    def calibrate(
        self,
        world: JelloWorld,
        rng: np.random.Generator,
        *,
        config: SpatialLaunchConfig | None = None,
    ) -> None:
        cfg = config or SpatialLaunchConfig()
        # Alternate cues so both contexts are observed equally often.  Small
        # amplitude jitter prevents the predictor from being an exact copied
        # snapshot of one deterministic probe.
        for rep in range(cfg.calibration_repeats):
            for cue in (rep % 2, 1 - (rep % 2)):
                jitter = cfg.calibration_gain_jitter
                gain = 1.0 + rng.uniform(-jitter, jitter)
                observed = np.log(
                    launcher_response(world, cue, config=cfg, gain=gain) + cfg.eps
                )
                self._seen[cue] += 1
                if self._seen[cue] == 1:
                    self.prediction[cue] = observed
                else:
                    # Online mean: a genuine learned expectation with no target
                    # mapping or reward information.
                    k = float(self._seen[cue])
                    self.prediction[cue] += (observed - self.prediction[cue]) / k

    def innovation(self, cue: int, observed: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        return np.log(np.asarray(observed, dtype=np.float64) + eps) - self.prediction[int(cue)]

    def track(self, cue: int, observed: np.ndarray, rate: float, eps: float = 1e-12) -> None:
        rate = float(rate)
        if rate <= 0:
            return
        target = np.log(np.asarray(observed, dtype=np.float64) + eps)
        self.prediction[int(cue)] += rate * (target - self.prediction[int(cue)])


def reinforce_route(
    world: JelloWorld,
    cue: int,
    chosen_pulse: int,
    *,
    config: SpatialLaunchConfig | None = None,
) -> None:
    """Rewarded traffic writes via the stock local JelloWorld plasticity rule."""
    cfg = config or SpatialLaunchConfig()
    word = CUE_PORTS[int(cue)] + LAUNCH_PORTS[int(chosen_pulse)]
    for _ in range(cfg.reward_replays):
        world.wipe_fast()
        world.present_word(word, write=True)
    world.wipe_fast()


def _temperature(ep: int, total: int, cfg: SpatialLaunchConfig) -> float:
    if total <= 1:
        return cfg.end_temperature
    f = float(ep) / float(total - 1)
    return cfg.start_temperature + f * (cfg.end_temperature - cfg.start_temperature)


def train_after_swap(
    seed: int,
    *,
    residual_gate: bool,
    write_material: bool = True,
    predictor_track_rate: float | None = None,
    scramble_prediction_context: bool = False,
    global_prediction_context: bool = False,
    config: SpatialLaunchConfig | None = None,
) -> dict:
    """Attempt to repair a wire swap.

    The receiver interprets delivered pulse identity literally.  After the
    physical channel swaps, cue 0 now requires launcher 1 and cue 1 requires
    launcher 0.  No desired launcher identity is supplied to the body; it gets
    only binary reward after its chosen pulse crosses the channel.
    """
    cfg = config or SpatialLaunchConfig()
    world = make_inherited_world(seed, cfg)
    rng = np.random.default_rng(100_003 + int(seed))
    predictor = NegativeImage()
    predictor.calibrate(world, rng, config=cfg)

    if scramble_prediction_context:
        predictor.prediction = predictor.prediction[::-1].copy()
    if global_prediction_context:
        mean_prediction = predictor.prediction.mean(axis=0, keepdims=True)
        predictor.prediction = np.repeat(mean_prediction, 2, axis=0)

    rate = cfg.predictor_track_rate if predictor_track_rate is None else float(predictor_track_rate)
    rewards = np.zeros(cfg.train_episodes, dtype=np.float64)
    choices = np.zeros(cfg.train_episodes, dtype=np.int64)

    for ep in range(cfg.train_episodes):
        cue = int(rng.integers(2))
        observed = launcher_response(world, cue, config=cfg)
        if residual_gate:
            decision = predictor.innovation(cue, observed, cfg.eps)
        else:
            decision = np.log(observed + cfg.eps)

        probs = _softmax(decision, _temperature(ep, cfg.train_episodes, cfg))
        chosen = int(rng.choice(2, p=probs))
        delivered = int(SWAPPED_CHANNEL[chosen])
        reward = int(delivered == cue)
        rewards[ep] = reward
        choices[ep] = chosen

        # Update the expected self-generated response from the pre-write
        # observation.  Critically, this predictor never sees the reward target.
        if residual_gate:
            predictor.track(cue, observed, rate, cfg.eps)

        if reward and write_material:
            reinforce_route(world, cue, chosen, config=cfg)

    greedy_correct = 0
    final = []
    for cue in (0, 1):
        observed = launcher_response(world, cue, config=cfg)
        decision = (
            predictor.innovation(cue, observed, cfg.eps)
            if residual_gate
            else np.log(observed + cfg.eps)
        )
        chosen = int(np.argmax(decision))
        delivered = int(SWAPPED_CHANNEL[chosen])
        ok = int(delivered == cue)
        greedy_correct += ok
        final.append(
            {
                "cue": cue,
                "observed": observed.tolist(),
                "decision": decision.tolist(),
                "chosen_pulse": chosen,
                "delivered_pulse": delivered,
                "correct": bool(ok),
            }
        )

    return {
        "seed": int(seed),
        "residual_gate": bool(residual_gate),
        "write_material": bool(write_material),
        "predictor_track_rate": rate,
        "scramble_prediction_context": bool(scramble_prediction_context),
        "global_prediction_context": bool(global_prediction_context),
        "greedy_accuracy": greedy_correct / 2.0,
        "first_50_reward": float(np.mean(rewards[:50])),
        "last_50_reward": float(np.mean(rewards[-50:])),
        "last_100_reward": float(np.mean(rewards[-100:])),
        "reward_curve": rewards.tolist(),
        "final": final,
    }
