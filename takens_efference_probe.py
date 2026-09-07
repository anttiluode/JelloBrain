#!/usr/bin/env python3
"""S21: forced-delay history + efference copy as a self-effect predictor.

This is a diagnostic bridge between two older strands of the repo family:

- GeometricNeuron: delayed/addressed pulse history can make hidden dynamics
  observable.
- JelloBrain S3: a context-linked negative image can subtract a predictable
  self-generated response and expose innovation.

The experiment asks a narrower mathematical question:

    Does a finite delay history of ONE scalar observation, when augmented by a
    copy of the system's own write action, predict the next self-generated
    change better than either history or action alone?

A single JelloWorld sheet is driven by random local writes.  The material arrays
are hidden from the estimator.  Before and after each write it receives only a
fixed generic scalar projection of launcher responses.  The write identity is
available only in the efference-copy conditions.

Four ridge predictors are trained on self-generated trajectories:

    CURRENT          y_t
    EFFERENCE        y_t + current action
    DELAY            [y_t, y_t-1, ...]
    FORCED_DELAY     delayed y + current/past actions + action/state interaction

Then an attacker inserts occasional *unreported* extra writes.  The model is
not retrained.  If the predictable self-effect has been learned, the innovation

    actual delta - predicted self-delta

should be a better detector of those hidden perturbations than raw |delta|.

This is Takens-*style* forced delay reconstruction, not an application claiming
all hypotheses of the Takens theorem hold for this high-dimensional plastic
substrate.  The action-conditioned formulation is the important bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np

from spatial_launcher import (
    SpatialLaunchConfig,
    launcher_response,
    make_inherited_world,
    reinforce_route,
)


@dataclass(frozen=True)
class ProbeConfig:
    delay: int = 8
    train_seeds: int = 4
    test_seeds: int = 4
    train_steps: int = 500
    test_steps: int = 400
    shock_probability: float = 0.10
    ridge: float = 1e-3
    deposit_rate: float = 0.004
    eps: float = 1e-12


def onehot(action: int) -> np.ndarray:
    v = np.zeros(4, dtype=np.float64)
    v[int(action)] = 1.0
    return v


def scalar_observation(world, scfg: SpatialLaunchConfig, eps: float) -> float:
    """One fixed generic scalar function of the hidden material state.

    The four launcher responses are only intermediate construction machinery;
    the estimator receives the final scalar, never the four-vector or material.
    """
    vals = []
    for cue in (0, 1):
        vals.extend(np.log(launcher_response(world, cue, config=scfg) + eps))
    coeff = np.asarray([0.37, -0.61, 0.83, -0.29], dtype=np.float64)
    coeff /= np.linalg.norm(coeff)
    return float(np.dot(coeff, np.asarray(vals, dtype=np.float64)))


def rollout(
    seed: int,
    steps: int,
    cfg: ProbeConfig,
    *,
    shocks: bool,
) -> dict:
    scfg = SpatialLaunchConfig(deposit_rate=cfg.deposit_rate)
    world = make_inherited_world(seed, scfg)
    rng = np.random.default_rng(21_000_017 + 1009 * int(seed) + int(shocks))

    ys = [scalar_observation(world, scfg, cfg.eps)]
    actions: list[int] = []
    shock_flags: list[int] = []

    for _ in range(int(steps)):
        action = int(rng.integers(4))
        cue, launcher = divmod(action, 2)
        reinforce_route(world, cue, launcher, config=scfg)

        shock = bool(shocks and rng.random() < cfg.shock_probability)
        if shock:
            # Hidden exogenous change: same physical write primitive, but no
            # efference copy is supplied to the estimator.
            hidden = int(rng.integers(4))
            if hidden == action:
                hidden = (hidden + 1 + int(rng.integers(3))) % 4
            hcue, hlauncher = divmod(hidden, 2)
            reinforce_route(world, hcue, hlauncher, config=scfg)

        actions.append(action)
        shock_flags.append(int(shock))
        ys.append(scalar_observation(world, scfg, cfg.eps))

    return {
        "y": np.asarray(ys, dtype=np.float64),
        "actions": np.asarray(actions, dtype=np.int64),
        "shocks": np.asarray(shock_flags, dtype=np.int64),
    }


def features(traj: dict, mode: str, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = traj["y"]
    a = traj["actions"]
    shock = traj["shocks"]
    rows = []
    targets = []
    labels = []

    for t in range(k - 1, len(a)):
        ylags = np.asarray([y[t - j] for j in range(k)], dtype=np.float64)
        current_action = onehot(int(a[t]))

        if mode == "current":
            x = np.asarray([y[t]], dtype=np.float64)
        elif mode == "efference":
            # Interaction lets the effect of an action depend on current state.
            x = np.concatenate(([y[t]], current_action, current_action * y[t]))
        elif mode == "delay":
            x = ylags
        elif mode == "forced_delay":
            ahist = np.concatenate([onehot(int(a[t - j])) for j in range(k)])
            # Current-action x delayed-state interaction is a small lifted
            # forward model; it does not see future outcomes or hidden material.
            interaction = np.concatenate([current_action * z for z in ylags])
            x = np.concatenate((ylags, ahist, interaction))
        else:
            raise ValueError(mode)

        rows.append(x)
        targets.append(y[t + 1] - y[t])
        labels.append(shock[t])

    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
    )


class Ridge:
    def __init__(self, lam: float):
        self.lam = float(lam)
        self.mean = None
        self.scale = None
        self.beta = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        Z = (X - self.mean) / self.scale
        Z = np.column_stack((np.ones(len(Z)), Z))
        reg = np.eye(Z.shape[1], dtype=np.float64) * self.lam
        reg[0, 0] = 0.0
        self.beta = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.scale
        Z = np.column_stack((np.ones(len(Z)), Z))
        return Z @ self.beta


def concat_features(trajectories: list[dict], mode: str, k: int):
    parts = [features(t, mode, k) for t in trajectories]
    return (
        np.concatenate([p[0] for p in parts], axis=0),
        np.concatenate([p[1] for p in parts], axis=0),
        np.concatenate([p[2] for p in parts], axis=0),
    )


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = y - pred
    mse = float(np.mean(err * err))
    baseline = float(np.var(y))
    r2 = float(1.0 - mse / max(baseline, 1e-30))
    return {"mse": mse, "r2": r2}


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = np.asarray(scores[labels == 1], dtype=np.float64)
    neg = np.asarray(scores[labels == 0], dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for s in pos:
        wins += float(np.sum(s > neg)) + 0.5 * float(np.sum(s == neg))
    return float(wins / (len(pos) * len(neg)))


def shuffled_action_attacker(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Destroy action/history correspondence while preserving marginals.

    forced-delay layout = k y-lags | 4k action history | 4k interactions.
    We jointly permute the action-derived portion across rows.
    """
    Xs = X.copy()
    cut = k
    perm = rng.permutation(len(Xs))
    Xs[:, cut:] = Xs[perm, cut:]
    return Xs


def main() -> dict:
    cfg = ProbeConfig()

    train = [
        rollout(seed, cfg.train_steps, cfg, shocks=False)
        for seed in range(cfg.train_seeds)
    ]
    test = [
        rollout(100 + seed, cfg.test_steps, cfg, shocks=False)
        for seed in range(cfg.test_seeds)
    ]
    shock_test = [
        rollout(200 + seed, cfg.test_steps, cfg, shocks=True)
        for seed in range(cfg.test_seeds)
    ]

    models = {}
    metrics = {}
    for mode in ("current", "efference", "delay", "forced_delay"):
        Xtr, ytr, _ = concat_features(train, mode, cfg.delay)
        Xte, yte, _ = concat_features(test, mode, cfg.delay)
        model = Ridge(cfg.ridge).fit(Xtr, ytr)
        pred = model.predict(Xte)
        models[mode] = model
        metrics[mode] = regression_metrics(yte, pred)

    Xs, ys, labels = concat_features(shock_test, "forced_delay", cfg.delay)
    forced_pred = models["forced_delay"].predict(Xs)
    innovation = np.abs(ys - forced_pred)
    raw_change = np.abs(ys)

    rng = np.random.default_rng(21_999_983)
    Xscr = shuffled_action_attacker(Xs, cfg.delay, rng)
    scrambled_pred = models["forced_delay"].predict(Xscr)

    Xclean, yclean, _ = concat_features(test, "forced_delay", cfg.delay)
    clean_pred = models["forced_delay"].predict(Xclean)
    Xclean_scr = shuffled_action_attacker(Xclean, cfg.delay, rng)
    clean_scr_pred = models["forced_delay"].predict(Xclean_scr)

    return {
        "schema": "jellobrain/forced-delay-efference-v1",
        "question": (
            "Can delayed scalar history plus a copy of the system's own writes predict self-generated change and expose unreported perturbations as innovation?"
        ),
        "config": {
            "delay": cfg.delay,
            "train_trajectories": cfg.train_seeds,
            "test_trajectories": cfg.test_seeds,
            "train_steps": cfg.train_steps,
            "test_steps": cfg.test_steps,
            "shock_probability": cfg.shock_probability,
        },
        "held_out_self_prediction": metrics,
        "forced_delay_attackers": {
            "clean_mse_correct_efference": regression_metrics(yclean, clean_pred)["mse"],
            "clean_mse_scrambled_efference": regression_metrics(yclean, clean_scr_pred)["mse"],
            "mse_scramble_ratio": float(
                regression_metrics(yclean, clean_scr_pred)["mse"]
                / max(regression_metrics(yclean, clean_pred)["mse"], 1e-30)
            ),
        },
        "hidden_shock_detection": {
            "n_shocks": int(labels.sum()),
            "n_clean": int((labels == 0).sum()),
            "auc_raw_abs_delta": auc(raw_change, labels),
            "auc_abs_innovation": auc(innovation, labels),
            "auc_abs_innovation_scrambled_efference": auc(
                np.abs(ys - scrambled_pred), labels
            ),
        },
        "claim_boundary": (
            "This is a finite, linear lifted predictor on a forced plastic toy. A positive result supports the utility of action-conditioned delay history for self-effect prediction; it is not a proof that Takens' theorem applies to JelloWorld, a dendritic implementation claim, or an AIS-grating mechanism."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
