#!/usr/bin/env python3
"""JelloBrain: words as interventions on a writable dynamical substrate.

The model is intentionally a toy. Six meaningless primitives (A-F) are short
spatiotemporal excitation patterns. They propagate through a 2-D directed
material field. Fast excitation/refractory state decays; slow directed material
is written by local temporal co-activity and persists after fast state is wiped.

There is no word embedding, tokenizer, word ID, semantic label, or pretrained
model inside the substrate. A pseudoword is only a sequence of primitive pulses.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence
import copy
import numpy as np


@dataclass(frozen=True)
class JelloConfig:
    n: int = 16
    fast_decay: float = 0.42
    refractory_decay: float = 0.88
    refractory_gain: float = 0.18
    propagation: float = 0.45
    nonlinearity: float = 1.0
    deposit_rate: float = 0.008
    material_decay: float = 1.0
    base_edge: float = 0.03
    edge_gain: float = 4.0
    steps_per_symbol: int = 4
    input_gain: float = 0.65
    eligibility_decay: float = 0.75
    initial_material: float = 0.02
    individuality: float = 0.00002


class JelloWorld:
    """2-D excitable medium with a persistent, directed material operator.

    Four arrays store directed nearest-neighbour conductivities:
      east:  (r,c)   -> (r,c+1)
      west:  (r,c+1) -> (r,c)
      south: (r,c)   -> (r+1,c)
      north: (r+1,c) -> (r,c)

    Plasticity is local: a decaying eligibility trace at a source cell and
    present activity at its neighbour deposit material onto that directed edge.
    That makes temporal order capable of leaving a spatially directed trace.
    """

    SYMBOLS = "ABCDEF"

    def __init__(self, seed: int = 0, config: JelloConfig | None = None):
        self.config = config or JelloConfig()
        c = self.config
        if c.n < 10:
            raise ValueError("grid must be at least 10x10")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        n = c.n

        self.x = np.zeros((n, n), dtype=np.float64)
        self.refractory = np.zeros_like(self.x)
        self.eligibility = np.zeros_like(self.x)

        def material(shape: tuple[int, ...]) -> np.ndarray:
            if c.individuality <= 0:
                return np.full(shape, c.initial_material, dtype=np.float64)
            return np.clip(
                self.rng.normal(c.initial_material, c.individuality, shape),
                0.0,
                1.0,
            )

        self.east = material((n, n - 1))
        self.west = material((n, n - 1))
        self.south = material((n - 1, n))
        self.north = material((n - 1, n))

        # Six primitive ports. A symbol is a pulse pattern, not a stored word.
        p = [
            (3, 3),
            (2, n // 2),
            (3, n - 4),
            (n - 4, n - 4),
            (n - 3, n // 2),
            (n - 4, 3),
        ]
        yy, xx = np.mgrid[:n, :n]
        self._masks = [
            np.exp(-((yy - rr) ** 2 + (xx - cc) ** 2) / (2.0 * 1.35**2))
            for rr, cc in p
        ]

    def clone(self) -> "JelloWorld":
        return copy.deepcopy(self)

    def wipe_fast(self) -> None:
        """Erase excitation, refractory state, and eligibility; preserve material."""
        self.x.fill(0.0)
        self.refractory.fill(0.0)
        self.eligibility.fill(0.0)

    def inject(self, symbol: int | str, gain: float = 1.0) -> None:
        k = self.symbol_index(symbol)
        self.x += float(gain) * self.config.input_gain * self._masks[k]

    @classmethod
    def symbol_index(cls, symbol: int | str) -> int:
        if isinstance(symbol, str):
            symbol = symbol.upper()
            if symbol not in cls.SYMBOLS:
                raise ValueError(f"unknown primitive {symbol!r}")
            return cls.SYMBOLS.index(symbol)
        k = int(symbol)
        if not 0 <= k < len(cls.SYMBOLS):
            raise ValueError("primitive index must be in 0..5")
        return k

    @classmethod
    def parse_word(cls, word: str | Sequence[int | str]) -> tuple[int, ...]:
        if isinstance(word, str):
            return tuple(cls.symbol_index(ch) for ch in word if not ch.isspace())
        return tuple(cls.symbol_index(ch) for ch in word)

    def _conductivity(self, material: np.ndarray) -> np.ndarray:
        c = self.config
        return c.base_edge + c.edge_gain * material

    def step(self, write: bool = True) -> None:
        c = self.config
        x = self.x
        incoming = np.zeros_like(x)

        ge = self._conductivity(self.east)
        gw = self._conductivity(self.west)
        gs = self._conductivity(self.south)
        gn = self._conductivity(self.north)

        incoming[:, 1:] += ge * x[:, :-1]
        incoming[:, :-1] += gw * x[:, 1:]
        incoming[1:, :] += gs * x[:-1, :]
        incoming[:-1, :] += gn * x[1:, :]

        drive = c.fast_decay * x + c.propagation * incoming
        drive -= c.refractory_gain * self.refractory
        new_x = np.maximum(0.0, np.tanh(c.nonlinearity * drive))

        self.refractory = (
            c.refractory_decay * self.refractory
            + (1.0 - c.refractory_decay) * new_x
        )

        if write:
            q = self.eligibility
            d = c.deposit_rate
            md = c.material_decay
            self.east = np.clip(md * self.east + d * q[:, :-1] * new_x[:, 1:], 0.0, 1.0)
            self.west = np.clip(md * self.west + d * q[:, 1:] * new_x[:, :-1], 0.0, 1.0)
            self.south = np.clip(md * self.south + d * q[:-1, :] * new_x[1:, :], 0.0, 1.0)
            self.north = np.clip(md * self.north + d * q[1:, :] * new_x[:-1, :], 0.0, 1.0)

        self.eligibility = c.eligibility_decay * self.eligibility + (1.0 - c.eligibility_decay) * x
        self.x = new_x

    def present_symbol(self, symbol: int | str, *, write: bool = True, gap: int = 0) -> None:
        self.inject(symbol)
        for _ in range(self.config.steps_per_symbol):
            self.step(write=write)
        for _ in range(int(gap)):
            self.step(write=write)

    def present_word(
        self,
        word: str | Sequence[int | str],
        *,
        write: bool = True,
        gap: int = 0,
    ) -> None:
        for symbol in self.parse_word(word):
            self.present_symbol(symbol, write=write, gap=gap)

    def material_vector(self) -> np.ndarray:
        return np.concatenate(
            [self.east.ravel(), self.west.ravel(), self.south.ravel(), self.north.ravel()]
        )

    def fast_vector(self) -> np.ndarray:
        return np.concatenate([self.x.ravel(), self.refractory.ravel()])

    def material_stats(self) -> dict[str, float]:
        m = self.material_vector()
        return {
            "mean": float(m.mean()),
            "std": float(m.std()),
            "min": float(m.min()),
            "max": float(m.max()),
        }

    def response_to_probe(self, symbol: int | str, steps: int = 8) -> np.ndarray:
        """Wipe fast state, apply one primitive, return its trajectory without writing."""
        self.wipe_fast()
        self.inject(symbol)
        frames = []
        for _ in range(int(steps)):
            self.step(write=False)
            frames.append(self.x.copy())
        return np.concatenate([frame.ravel() for frame in frames])

    def probe_battery(self, steps: int = 8) -> np.ndarray:
        """Response fingerprint from six neutral primitive probes."""
        return np.concatenate([self.response_to_probe(k, steps) for k in range(6)])

    def response_to_word(self, word: str | Sequence[int | str]) -> np.ndarray:
        """Fast trajectory for a pseudoword with material frozen."""
        self.wipe_fast()
        frames: list[np.ndarray] = []
        for symbol in self.parse_word(word):
            self.inject(symbol)
            for _ in range(self.config.steps_per_symbol):
                self.step(write=False)
                frames.append(self.x.copy())
        return np.concatenate([frame.ravel() for frame in frames])

    def shuffle_material(self, seed: int) -> None:
        """Spatial attacker: preserve all material values, destroy their placement/direction."""
        rng = np.random.default_rng(int(seed))
        values = self.material_vector().copy()
        rng.shuffle(values)
        offset = 0
        for name in ("east", "west", "south", "north"):
            arr = getattr(self, name)
            count = arr.size
            setattr(self, name, values[offset : offset + count].reshape(arr.shape).copy())
            offset += count


def grow_world(
    seed: int,
    corpus: Iterable[str | Sequence[int | str]],
    *,
    repeats: int = 5,
    config: JelloConfig | None = None,
    gap: int = 0,
) -> JelloWorld:
    world = JelloWorld(seed, config)
    corpus = list(corpus)
    for _ in range(int(repeats)):
        for word in corpus:
            world.present_word(word, write=True, gap=gap)
    world.wipe_fast()
    return world


def with_individuality(config: JelloConfig | None, individuality: float) -> JelloConfig:
    return replace(config or JelloConfig(), individuality=float(individuality))
