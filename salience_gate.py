#!/usr/bin/env python3
"""S5-S8: a minimal salience/ping isolator for the JelloBrain launch boundary.

This file does NOT model chandelier cells biologically.  It isolates one
computational distinction suggested by the recent chandelier-cell literature
and by the failure of S4:

    surprise != relevance

An event can become perfectly expected and still remain behaviorally important.

The toy gate sees two event identities in two contexts.  A fast predictor learns
the ordinary sensory magnitude of each event.  A separate relevance state learns
from a delayed consequence signal.  The launch/"ping" decision may use either:

    surprise only
        |observed - predicted|

or

    surprise + learned relevance(context, event)

The experiment first familiarizes all events until novelty responses vanish.
Then a previously familiar event becomes consequential in only one context.
The gate is evaluated with all learning frozen.

The outcome/relevance signal is deliberately external in this isolator: this
experiment asks what happens *if* a consequence signal exists, not how the
brain routes that signal to chandelier cells or the AIS.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np


@dataclass(frozen=True)
class PingGateConfig:
    prediction_rate: float = 0.25
    relevance_rate: float = 0.08
    relevance_gain: float = 1.0
    threshold: float = 0.20
    observation_noise: float = 0.02
    familiarization_steps: int = 240
    conditioning_steps: int = 480
    eval_repeats: int = 120


def is_relevant(context: int, event: int) -> int:
    """Ground-truth consequence used only to train the relevance trace.

    The formerly familiar event 0 becomes important in context 0 and remains
    irrelevant in context 1.  Event 1 remains irrelevant everywhere.
    """
    return int(int(context) == 0 and int(event) == 0)


class SaliencePingGate:
    """Prediction + relevance state controlling a sparse outward ping."""

    BASE_EVENT_MAGNITUDES = (0.75, 1.00)

    def __init__(
        self,
        seed: int = 0,
        config: PingGateConfig | None = None,
        *,
        context_specific_relevance: bool = True,
    ):
        self.seed = int(seed)
        self.config = config or PingGateConfig()
        self.rng = np.random.default_rng(self.seed)
        self.context_specific_relevance = bool(context_specific_relevance)

        # Fast expectation: what magnitude does this event normally produce?
        self.prediction = np.zeros(2, dtype=np.float64)

        # Separate learned "does this matter here?" state.
        rows = 2 if self.context_specific_relevance else 1
        self.relevance = np.zeros((rows, 2), dtype=np.float64)

    def clone(self) -> "SaliencePingGate":
        return copy.deepcopy(self)

    def observe(self, event: int) -> float:
        event = int(event)
        if event not in (0, 1):
            raise ValueError("event must be 0 or 1")
        return float(
            self.BASE_EVENT_MAGNITUDES[event]
            + self.rng.normal(0.0, self.config.observation_noise)
        )

    def relevance_value(self, context: int, event: int) -> float:
        row = int(context) if self.context_specific_relevance else 0
        return float(self.relevance[row, int(event)])

    def decision(
        self,
        context: int,
        event: int,
        observed: float,
        *,
        mode: str = "dual",
    ) -> tuple[float, float, float]:
        """Return (launch_score, surprise, relevance).

        `mode="surprise"` ignores learned relevance.
        `mode="dual"` uses surprise + relevance.
        """
        if mode not in ("surprise", "dual"):
            raise ValueError("mode must be 'surprise' or 'dual'")

        surprise = abs(float(observed) - self.prediction[int(event)])
        if mode == "surprise":
            return surprise, surprise, 0.0

        relevance = self.relevance_value(context, event)
        score = surprise + self.config.relevance_gain * relevance
        return float(score), float(surprise), float(relevance)

    def step(
        self,
        context: int,
        event: int,
        outcome: int,
        *,
        mode: str = "dual",
        learn: bool = True,
    ) -> dict:
        """Observe one event and optionally update expectation/relevance.

        Importantly, the ping decision is made before the current outcome is
        applied. Therefore current-trial consequence cannot leak directly into
        the launch decision.
        """
        context = int(context)
        event = int(event)
        outcome = int(outcome)
        if context not in (0, 1) or event not in (0, 1) or outcome not in (0, 1):
            raise ValueError("context, event, outcome must be binary")

        observed = self.observe(event)
        score, surprise, relevance = self.decision(
            context, event, observed, mode=mode
        )
        ping = bool(score > self.config.threshold)

        if learn:
            self.prediction[event] += self.config.prediction_rate * (
                observed - self.prediction[event]
            )
            row = context if self.context_specific_relevance else 0
            self.relevance[row, event] += self.config.relevance_rate * (
                float(outcome) - self.relevance[row, event]
            )

        return {
            "context": context,
            "event": event,
            "outcome": outcome,
            "observed": observed,
            "surprise": surprise,
            "relevance": relevance,
            "score": score,
            "ping": ping,
        }

    def eval_ping_rate(
        self,
        context: int,
        event: int,
        *,
        mode: str = "dual",
        repeats: int | None = None,
    ) -> float:
        """Evaluate with all state frozen."""
        n = self.config.eval_repeats if repeats is None else int(repeats)
        pings = []
        for _ in range(n):
            observed = self.observe(event)
            score, _, _ = self.decision(context, event, observed, mode=mode)
            pings.append(score > self.config.threshold)
        return float(np.mean(pings))


def familiarization(
    gate: SaliencePingGate,
    *,
    mode: str = "dual",
) -> dict:
    """All events are initially inconsequential; novelty should habituate."""
    pings: list[float] = []
    surprises: list[float] = []
    for _ in range(gate.config.familiarization_steps):
        context = int(gate.rng.integers(2))
        event = int(gate.rng.integers(2))
        out = gate.step(context, event, 0, mode=mode, learn=True)
        pings.append(float(out["ping"]))
        surprises.append(float(out["surprise"]))
    return {
        "first_12_ping_rate": float(np.mean(pings[:12])),
        "last_80_ping_rate": float(np.mean(pings[-80:])),
        "first_12_surprise": float(np.mean(surprises[:12])),
        "last_80_surprise": float(np.mean(surprises[-80:])),
    }


def conditioning(
    gate: SaliencePingGate,
    *,
    mode: str = "dual",
) -> None:
    """A familiar event becomes relevant in only one context."""
    for _ in range(gate.config.conditioning_steps):
        context = int(gate.rng.integers(2))
        event = int(gate.rng.integers(2))
        gate.step(
            context,
            event,
            is_relevant(context, event),
            mode=mode,
            learn=True,
        )


def evaluate_grid(gate: SaliencePingGate, *, mode: str = "dual") -> dict:
    """Frozen evaluation for every context/event pair."""
    out: dict[str, float] = {}
    for context in (0, 1):
        for event in (0, 1):
            out[f"c{context}_e{event}"] = gate.eval_ping_rate(
                context, event, mode=mode
            )
    return out


def run_protocol(
    seed: int,
    *,
    mode: str = "dual",
    prediction_rate: float | None = None,
    context_specific_relevance: bool = True,
) -> dict:
    cfg = PingGateConfig(
        prediction_rate=(
            PingGateConfig.prediction_rate
            if prediction_rate is None
            else float(prediction_rate)
        )
    )
    gate = SaliencePingGate(
        seed,
        cfg,
        context_specific_relevance=context_specific_relevance,
    )
    familiar = familiarization(gate, mode=mode)
    conditioning(gate, mode=mode)
    grid = evaluate_grid(gate, mode=mode)
    return {
        "seed": int(seed),
        "mode": mode,
        "prediction_rate": cfg.prediction_rate,
        "context_specific_relevance": bool(context_specific_relevance),
        "familiarization": familiar,
        "grid": grid,
        "prediction": gate.prediction.tolist(),
        "relevance": gate.relevance.tolist(),
    }


def relevance_ablation_after_learning(seed: int) -> dict:
    """Train the dual gate, then erase only relevance before frozen testing."""
    cfg = PingGateConfig()
    gate = SaliencePingGate(seed, cfg, context_specific_relevance=True)
    familiarization(gate, mode="dual")
    conditioning(gate, mode="dual")
    before = evaluate_grid(gate, mode="dual")
    gate.relevance.fill(0.0)
    after = evaluate_grid(gate, mode="dual")
    return {
        "seed": int(seed),
        "before": before,
        "after": after,
    }
