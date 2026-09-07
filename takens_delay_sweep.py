#!/usr/bin/env python3
"""S21R: does delay depth buy anything beyond the efference copy itself?

S21 found a strong action-copy effect and only a modest gain from an 8-step
forced delay history.  This sweep prevents us from turning that modest gain into
an attractive Takens story without testing the obvious attacker.

For k in {1,2,4,8,16,32}, all models are evaluated on the same k-truncated held
out transitions.  k=1 FORCED_DELAY is effectively the current-state + current
action model.  Therefore any improvement for k>1 is the empirical contribution
of temporal history in this assay.
"""

from __future__ import annotations

import json
import numpy as np

from takens_efference_probe import (
    ProbeConfig,
    Ridge,
    auc,
    concat_features,
    regression_metrics,
    rollout,
    shuffled_action_attacker,
)


def main() -> dict:
    cfg = ProbeConfig()
    delays = [1, 2, 4, 8, 16, 32]

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

    rows = []
    rng = np.random.default_rng(22_000_021)

    for k in delays:
        # Same truncation for all conditions at this k.
        models = {}
        metrics = {}
        for mode in ("current", "efference", "delay", "forced_delay"):
            Xtr, ytr, _ = concat_features(train, mode, k)
            Xte, yte, _ = concat_features(test, mode, k)
            model = Ridge(cfg.ridge).fit(Xtr, ytr)
            models[mode] = model
            metrics[mode] = regression_metrics(yte, model.predict(Xte))

        Xs, ys, labels = concat_features(shock_test, "forced_delay", k)
        pred = models["forced_delay"].predict(Xs)
        Xscr = shuffled_action_attacker(Xs, k, rng)
        pred_scr = models["forced_delay"].predict(Xscr)

        eff_mse = metrics["efference"]["mse"]
        forced_mse = metrics["forced_delay"]["mse"]
        rows.append(
            {
                "delay": int(k),
                "current_r2": metrics["current"]["r2"],
                "efference_r2": metrics["efference"]["r2"],
                "delay_only_r2": metrics["delay"]["r2"],
                "forced_delay_r2": metrics["forced_delay"]["r2"],
                "forced_mse_over_efference_mse": float(
                    forced_mse / max(eff_mse, 1e-30)
                ),
                "shock_auc_raw_delta": auc(np.abs(ys), labels),
                "shock_auc_innovation": auc(np.abs(ys - pred), labels),
                "shock_auc_scrambled_efference": auc(
                    np.abs(ys - pred_scr), labels
                ),
            }
        )

    best = min(rows, key=lambda r: r["forced_mse_over_efference_mse"])
    return {
        "schema": "jellobrain/forced-delay-depth-sweep-v1",
        "question": (
            "How much predictive information does temporal delay history add beyond the current efference copy in S21?"
        ),
        "rows": rows,
        "best_relative_delay": best,
        "claim_boundary": (
            "A monotone or reproducible history gain would support a finite-memory state-reconstruction role in this toy. It would still not establish a Takens-theorem implementation or a biological delay line."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
