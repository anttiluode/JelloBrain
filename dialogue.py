#!/usr/bin/env python3
"""Phase II JelloBrain: two private protocol gels linked by a narrow pulse channel.

This is deliberately smaller than the spatial 2-D JelloWorld.  It isolates the
question: can two bounded systems, each with private state, grow a useful pulse
convention from consequences alone?

The only transmitted object is one of two meaningless pulse identities.  The
sender sees cue x, the receiver sees cue y, and the cooperative target is
x XOR y.  Neither agent can inspect the other's slow material.  Slow material
stores transition preferences and is changed by reward-modulated traffic.

There is no hand-written mapping such as pulse 0 == cue 0.  Different random
seeds can settle on opposite conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np


@dataclass(frozen=True)
class DialogueConfig:
    baseline: float = 0.5
    init_sigma: float = 0.03
    deposit_rate: float = 0.08
    erosion_rate: float = 0.03
    relax_rate: float = 0.0005
    start_temperature: float = 0.40
    end_temperature: float = 0.05
    eval_temperature: float = 0.08


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    t = max(float(temperature), 1e-6)
    z = np.asarray(values, dtype=np.float64) / t
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


class DialoguePair:
    """Two private adaptive materials joined only by a two-symbol channel.

    sender_material[x, pulse]
        slow road strength from sender cue x to emitted pulse identity.

    receiver_material[y, pulse, action]
        slow road strength from receiver cue y and received pulse to action.

    The matrices are *private*.  The receiver never reads sender_material and
    the sender never reads receiver_material.  Only an integer pulse crosses
    the channel.
    """

    def __init__(self, seed: int = 0, config: DialogueConfig | None = None):
        self.seed = int(seed)
        self.config = config or DialogueConfig()
        self.rng = np.random.default_rng(self.seed)
        c = self.config
        self.sender_material = np.clip(
            c.baseline + self.rng.normal(0.0, c.init_sigma, size=(2, 2)),
            0.0,
            1.0,
        )
        self.receiver_material = np.clip(
            c.baseline + self.rng.normal(0.0, c.init_sigma, size=(2, 2, 2)),
            0.0,
            1.0,
        )
        self.episodes_seen = 0

    def clone(self) -> "DialoguePair":
        return copy.deepcopy(self)

    def sender_probs(self, x: int, temperature: float | None = None) -> np.ndarray:
        t = self.config.eval_temperature if temperature is None else temperature
        return _softmax(self.sender_material[int(x)], t)

    def receiver_probs(
        self, y: int, delivered_pulse: int, temperature: float | None = None
    ) -> np.ndarray:
        t = self.config.eval_temperature if temperature is None else temperature
        return _softmax(self.receiver_material[int(y), int(delivered_pulse)], t)

    def _temperature(self, episode: int, total_episodes: int) -> float:
        c = self.config
        if total_episodes <= 1:
            return c.end_temperature
        frac = float(episode) / float(total_episodes - 1)
        return c.start_temperature + frac * (c.end_temperature - c.start_temperature)

    def _relax(self) -> None:
        c = self.config
        self.sender_material += c.relax_rate * (c.baseline - self.sender_material)
        self.receiver_material += c.relax_rate * (c.baseline - self.receiver_material)

    def _reinforce(self, x: int, chosen_pulse: int, y: int, delivered_pulse: int, action: int, reward: int) -> None:
        c = self.config
        self._relax()
        if reward:
            s = self.sender_material[x, chosen_pulse]
            r = self.receiver_material[y, delivered_pulse, action]
            self.sender_material[x, chosen_pulse] = s + c.deposit_rate * (1.0 - s)
            self.receiver_material[y, delivered_pulse, action] = r + c.deposit_rate * (1.0 - r)
        else:
            self.sender_material[x, chosen_pulse] *= 1.0 - c.erosion_rate
            self.receiver_material[y, delivered_pulse, action] *= 1.0 - c.erosion_rate
        np.clip(self.sender_material, 0.0, 1.0, out=self.sender_material)
        np.clip(self.receiver_material, 0.0, 1.0, out=self.receiver_material)

    def episode(
        self,
        x: int,
        y: int,
        *,
        episode_index: int,
        total_episodes: int,
        adapt: bool = True,
        channel_map: tuple[int, int] = (0, 1),
    ) -> dict:
        """Run one cooperative episode.

        `channel_map` is a physical remapping attacker.  For example (1, 0)
        swaps the two pulse wires *after* the sender chooses a pulse.
        """
        x = int(x)
        y = int(y)
        if x not in (0, 1) or y not in (0, 1):
            raise ValueError("private cues must be binary")
        if sorted(channel_map) != [0, 1]:
            raise ValueError("channel_map must be a permutation of (0, 1)")

        temperature = self._temperature(episode_index, total_episodes)
        pulse_probs = _softmax(self.sender_material[x], temperature)
        chosen_pulse = int(self.rng.choice(2, p=pulse_probs))
        delivered_pulse = int(channel_map[chosen_pulse])
        action_probs = _softmax(self.receiver_material[y, delivered_pulse], temperature)
        action = int(self.rng.choice(2, p=action_probs))
        target = x ^ y
        reward = int(action == target)

        if adapt:
            self._reinforce(x, chosen_pulse, y, delivered_pulse, action, reward)
        self.episodes_seen += 1
        return {
            "x": x,
            "y": y,
            "chosen_pulse": chosen_pulse,
            "delivered_pulse": delivered_pulse,
            "action": action,
            "target": target,
            "reward": reward,
            "temperature": temperature,
        }

    def train(
        self,
        episodes: int = 2000,
        *,
        channel_map: tuple[int, int] = (0, 1),
    ) -> np.ndarray:
        rewards = np.zeros(int(episodes), dtype=np.float64)
        for ep in range(int(episodes)):
            x = int(self.rng.integers(2))
            y = int(self.rng.integers(2))
            out = self.episode(
                x,
                y,
                episode_index=ep,
                total_episodes=int(episodes),
                adapt=True,
                channel_map=channel_map,
            )
            rewards[ep] = out["reward"]
        return rewards

    def greedy_accuracy(self, channel_map: tuple[int, int] = (0, 1)) -> float:
        correct = 0
        total = 0
        for x in (0, 1):
            chosen_pulse = int(np.argmax(self.sender_material[x]))
            delivered = int(channel_map[chosen_pulse])
            for y in (0, 1):
                action = int(np.argmax(self.receiver_material[y, delivered]))
                correct += int(action == (x ^ y))
                total += 1
        return correct / total

    def sender_coupling(self) -> float:
        """How strongly pulse identity distinguishes the sender's private cue."""
        p0 = self.sender_probs(0)[1]
        p1 = self.sender_probs(1)[1]
        return float(abs(p1 - p0))

    def predictive_correct_probability(self) -> float:
        """Receiver probability of the correct joint action after the pulse arrives."""
        values = []
        for x in (0, 1):
            pulse = int(np.argmax(self.sender_material[x]))
            for y in (0, 1):
                p = self.receiver_probs(y, pulse)
                values.append(float(p[x ^ y]))
        return float(np.mean(values))

    def no_pulse_correct_probability(self) -> float:
        """Best receiver probability when pulse identity is hidden/marginalized.

        This is an attacker that gives B its own cue y but erases the channel.
        For the XOR task it should remain near chance even after a protocol has
        formed.
        """
        values = []
        for x in (0, 1):
            for y in (0, 1):
                p = 0.5 * self.receiver_probs(y, 0) + 0.5 * self.receiver_probs(y, 1)
                values.append(float(p[x ^ y]))
        return float(np.mean(values))

    def orientation(self) -> tuple[int, int]:
        """Raw pulse convention: pulse used for x=0 and pulse used for x=1."""
        return (
            int(np.argmax(self.sender_material[0])),
            int(np.argmax(self.sender_material[1])),
        )


def crossplay_accuracy(sender: DialoguePair, receiver: DialoguePair) -> float:
    """Use sender's private code with a foreign receiver's private code."""
    correct = 0
    total = 0
    for x in (0, 1):
        pulse = int(np.argmax(sender.sender_material[x]))
        for y in (0, 1):
            action = int(np.argmax(receiver.receiver_material[y, pulse]))
            correct += int(action == (x ^ y))
            total += 1
    return correct / total
