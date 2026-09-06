#!/usr/bin/env python3
"""Phase-II D0-D4 communication gates for JelloBrain."""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
import numpy as np

from dialogue import DialoguePair, crossplay_accuracy


def _pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def gate_d0(n_pairs: int = 16, episodes: int = 2000) -> dict:
    pairs = []
    for seed in range(n_pairs):
        pair = DialoguePair(seed)
        pair.train(episodes)
        pairs.append(pair)
    intact = [p.greedy_accuracy() for p in pairs]
    swapped = [p.greedy_accuracy((1, 0)) for p in pairs]
    foreign = [
        crossplay_accuracy(sender, receiver)
        for i, sender in enumerate(pairs)
        for j, receiver in enumerate(pairs)
        if i != j
    ]
    return {
        "gate": "D0_COMMUNICATION_SPECIFIC",
        "question": "Does the narrow pulse channel carry pair-specific task information rather than generic synchrony?",
        "pairs": n_pairs,
        "training_episodes": episodes,
        "intact_mean_reward": float(np.mean(intact)),
        "swapped_channel_mean_reward": float(np.mean(swapped)),
        "foreign_code_crossplay_mean_reward": float(np.mean(foreign)),
        "chance": 0.5,
        "pass": bool(np.mean(intact) >= 0.95 and np.mean(swapped) <= 0.10 and 0.35 <= np.mean(foreign) <= 0.65),
    }


def gate_d1(seed: int = 7, episodes: int = 1200) -> dict:
    """Discrete predictive analogue; not a reproduction of fMRI lead/lag."""
    pair = DialoguePair(seed)
    early_post = pair.predictive_correct_probability()
    early_no_pulse = pair.no_pulse_correct_probability()
    pair.train(episodes)
    late_post = pair.predictive_correct_probability()
    late_no_pulse = pair.no_pulse_correct_probability()
    return {
        "gate": "D1_PREDICTIVE_PULSE_EFFECT",
        "question": "Does an incoming learned pulse become an immediate predictive intervention for the receiver?",
        "note": "discrete causal analogue only; this gate does not test brain-scale anticipatory lead/lag",
        "early_post_pulse_correct_probability": early_post,
        "late_post_pulse_correct_probability": late_post,
        "early_no_pulse_correct_probability": early_no_pulse,
        "late_no_pulse_correct_probability": late_no_pulse,
        "late_pulse_gain": late_post - late_no_pulse,
        "learning_gain": late_post - early_post,
        "pass": bool(late_post >= 0.95 and late_no_pulse <= 0.55 and late_post - late_no_pulse >= 0.40 and late_post - early_post >= 0.25),
    }


def gate_d2() -> dict:
    budgets = [20, 40, 80, 120, 200, 400, 800, 1600]
    rows, couplings, rewards = [], [], []
    for bi, budget in enumerate(budgets):
        for rep in range(6):
            seed = 100 + 10 * bi + rep
            pair = DialoguePair(seed)
            pair.train(budget)
            coupling = pair.sender_coupling()
            reward = pair.greedy_accuracy()
            couplings.append(coupling)
            rewards.append(reward)
            rows.append({"seed": seed, "episodes": budget, "sender_coupling": coupling, "held_out_table_reward": reward})
    corr = _pearson(couplings, rewards)
    return {
        "gate": "D2_COUPLING_TRACKS_SUCCESS",
        "question": "Across independent pairs at different learning stages, does task-relevant pulse coupling track reward?",
        "pearson_r": corr,
        "n_pairs": len(rows),
        "budget_reward_means": {str(b): float(np.mean([r["held_out_table_reward"] for r in rows if r["episodes"] == b])) for b in budgets},
        "pass": bool(corr >= 0.70 and np.mean(rewards[-6:]) >= 0.95),
    }


def gate_d3(n_pairs: int = 12, pretrain: int = 1600, repair_episodes: int = 600) -> dict:
    broken, frozen, repaired, repair_tail = [], [], [], []
    for seed in range(n_pairs):
        pair = DialoguePair(300 + seed)
        pair.train(pretrain)
        broken.append(pair.greedy_accuracy((1, 0)))
        frozen_pair = pair.clone()
        frozen.append(frozen_pair.greedy_accuracy((1, 0)))
        pair.rng = np.random.default_rng(9000 + seed)
        rewards = pair.train(repair_episodes, channel_map=(1, 0))
        repaired.append(pair.greedy_accuracy((1, 0)))
        repair_tail.append(float(np.mean(rewards[-100:])))
    return {
        "gate": "D3_REPAIR",
        "question": "After the two physical pulse wires are swapped, can the pair renegotiate without being told the new code?",
        "pairs": n_pairs,
        "pretrain_episodes": pretrain,
        "repair_episodes": repair_episodes,
        "immediate_post_swap_reward": float(np.mean(broken)),
        "frozen_code_reward": float(np.mean(frozen)),
        "repaired_greedy_reward": float(np.mean(repaired)),
        "repaired_tail_training_reward": float(np.mean(repair_tail)),
        "pass": bool(np.mean(broken) <= 0.10 and np.mean(frozen) <= 0.10 and np.mean(repaired) >= 0.95 and np.mean(repair_tail) >= 0.90),
    }


def gate_d4(n_pairs: int = 20, episodes: int = 2000) -> dict:
    pairs = []
    for seed in range(500, 500 + n_pairs):
        pair = DialoguePair(seed)
        pair.train(episodes)
        pairs.append(pair)
    orientations = [p.orientation() for p in pairs]
    counts = Counter(orientations)
    offdiag = [crossplay_accuracy(a, b) for i, a in enumerate(pairs) for j, b in enumerate(pairs) if i != j]
    canonical_correct = 0
    canonical_total = 0
    for p in pairs:
        orientation = p.orientation()
        if orientation[0] == orientation[1]:
            continue
        role_for_pulse = {orientation[0]: 0, orientation[1]: 1}
        for x in (0, 1):
            pulse = orientation[x]
            role = role_for_pulse[pulse]
            for y in (0, 1):
                canonical_action = role ^ y
                actual_action = int(np.argmax(p.receiver_material[y, pulse]))
                canonical_correct += int(actual_action == canonical_action)
                canonical_total += 1
    canonical_accuracy = canonical_correct / max(canonical_total, 1)
    both_conventions = len([k for k, v in counts.items() if v > 0]) >= 2
    return {
        "gate": "D4_PRIVATE_CODE_SHARED_RELATION",
        "question": "Do independent pairs use different raw pulse identities while converging on the same relational task structure?",
        "pairs": n_pairs,
        "orientation_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "both_binary_conventions_present": both_conventions,
        "foreign_raw_crossplay_mean_reward": float(np.mean(offdiag)),
        "canonical_relational_accuracy": float(canonical_accuracy),
        "pass": bool(both_conventions and 0.35 <= np.mean(offdiag) <= 0.65 and canonical_accuracy >= 0.95),
    }


def run_all_dialogue_gates() -> dict:
    gates = [gate_d0(), gate_d1(), gate_d2(), gate_d3(), gate_d4()]
    return {
        "schema": "jellobrain/dialogue-gates-v1",
        "task": "A sees x, B sees y, one meaningless binary pulse crosses A->B, B must output x XOR y",
        "claim_boundary": (
            "Passing D0-D4 shows an emergent, pair-specific, repairable signaling convention in a minimal "
            "reward-modulated material toy. It does not establish language understanding, neural coupling, "
            "qualia transfer, consciousness, or a biological mechanism. D1 is only a discrete predictive analogue."
        ),
        "gates": gates,
        "all_pass": bool(all(g["pass"] for g in gates)),
    }


def write_receipt(path: str | Path = "results/dialogue_gates.json") -> dict:
    result = run_all_dialogue_gates()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(write_receipt(), indent=2))
