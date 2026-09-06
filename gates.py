#!/usr/bin/env python3
"""Reproducible JelloBrain W0-W5 falsification gates."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from jellobrain import JelloConfig, JelloWorld, grow_world, with_individuality


def _split_indices(labels: np.ndarray, seeds: np.ndarray, modulus: int = 4):
    test = seeds % modulus == 0
    train = ~test
    return train, test


def _standardize(train_x: np.ndarray, x: np.ndarray):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-7
    return (x - mean) / std, mean, std


def _centroid_accuracy(x: np.ndarray, y: np.ndarray, seeds: np.ndarray) -> float:
    train, test = _split_indices(y, seeds)
    z, mean, std = _standardize(x[train], x)
    labels = np.unique(y)
    centroids = np.stack([z[train][y[train] == k].mean(axis=0) for k in labels])
    distances = ((z[test, None, :] - centroids[None, :, :]) ** 2).mean(axis=-1)
    pred = labels[distances.argmin(axis=1)]
    return float(np.mean(pred == y[test]))


def _intact_and_shuffled_accuracy(
    intact: np.ndarray,
    shuffled: np.ndarray,
    y: np.ndarray,
    seeds: np.ndarray,
):
    train, test = _split_indices(y, seeds)
    mean = intact[train].mean(axis=0)
    std = intact[train].std(axis=0) + 1e-7
    zi = (intact - mean) / std
    zs = (shuffled - mean) / std
    labels = np.unique(y)
    centroids = np.stack([zi[train][y[train] == k].mean(axis=0) for k in labels])

    def score(z):
        d = ((z[test, None, :] - centroids[None, :, :]) ** 2).mean(axis=-1)
        pred = labels[d.argmin(axis=1)]
        return float(np.mean(pred == y[test]))

    return score(zi), score(zs)


def _collect_corpus_fingerprints(corpora, n_seeds=24, repeats=5, config=None):
    features = []
    labels = []
    seeds = []
    for label, corpus in enumerate(corpora):
        for seed in range(n_seeds):
            world = grow_world(seed, corpus, repeats=repeats, config=config)
            features.append(world.probe_battery(steps=7))
            labels.append(label)
            seeds.append(seed)
    return np.asarray(features), np.asarray(labels), np.asarray(seeds)


def gate_w0() -> dict:
    """Persistent history after all fast variables are erased."""
    corpus_0 = ["ABC", "ABC", "DEF"]
    corpus_1 = ["DEF", "DEF", "ABC"]
    x, y, seeds = _collect_corpus_fingerprints([corpus_0, corpus_1])
    accuracy = _centroid_accuracy(x, y, seeds)
    return {
        "gate": "W0_PERSISTENT_WORD_TRACE",
        "question": "After fast state is wiped, can neutral probes identify which corpus shaped the gel?",
        "decoder": "nearest centroid; held-out world seeds",
        "accuracy": accuracy,
        "chance": 0.5,
        "pass": accuracy >= 0.75,
    }


def gate_w1() -> dict:
    """Same primitive counts, different ordering."""
    corpus_0 = ["ABCD", "DCBA", "ABCD", "DCBA"]
    corpus_1 = ["ACBD", "DBCA", "ACBD", "DBCA"]

    def counts(corpus):
        out = {k: 0 for k in JelloWorld.SYMBOLS}
        for word in corpus:
            for ch in word:
                out[ch] += 1
        return out

    c0, c1 = counts(corpus_0), counts(corpus_1)
    x, y, seeds = _collect_corpus_fingerprints([corpus_0, corpus_1], repeats=6)
    accuracy = _centroid_accuracy(x, y, seeds)
    return {
        "gate": "W1_ORDER_NOT_FREQUENCY",
        "question": "With exactly equal primitive counts, does order leave a decodable slow trace?",
        "corpus_0_counts": c0,
        "corpus_1_counts": c1,
        "counts_identical": c0 == c1,
        "accuracy": accuracy,
        "chance": 0.5,
        "pass": bool(c0 == c1 and accuracy >= 0.75),
    }


def gate_w2() -> dict:
    """Preserve material values but destroy geometry/direction."""
    corpus_0 = ["ABCD", "DCBA", "ABCD", "DCBA"]
    corpus_1 = ["ACBD", "DBCA", "ACBD", "DBCA"]
    intact, shuffled, labels, seeds = [], [], [], []
    for label, corpus in enumerate([corpus_0, corpus_1]):
        for seed in range(24):
            world = grow_world(seed, corpus, repeats=6)
            intact.append(world.probe_battery(steps=7))
            attacked = world.clone()
            attacked.shuffle_material(10000 + seed + 100 * label)
            shuffled.append(attacked.probe_battery(steps=7))
            labels.append(label)
            seeds.append(seed)
    intact = np.asarray(intact)
    shuffled = np.asarray(shuffled)
    labels = np.asarray(labels)
    seeds = np.asarray(seeds)
    intact_acc, shuffled_acc = _intact_and_shuffled_accuracy(intact, shuffled, labels, seeds)
    return {
        "gate": "W2_GEOMETRY_MATTERS",
        "question": "If all material values are preserved but spatially/directionally shuffled, does decoding collapse?",
        "intact_accuracy": intact_acc,
        "shuffled_accuracy": shuffled_acc,
        "delta": intact_acc - shuffled_acc,
        "chance": 0.5,
        "pass": bool(intact_acc >= 0.75 and shuffled_acc <= 0.65 and intact_acc - shuffled_acc >= 0.20),
    }


def gate_w3() -> dict:
    """Same target pulse, different immediately preceding context."""
    corpus = ["ABCD", "DCBA", "EFEF"]
    features, labels, seeds = [], [], []
    contexts = [("AB", 0), ("CD", 1)]
    for seed in range(24):
        world = grow_world(seed, corpus, repeats=5)
        for context, label in contexts:
            world.wipe_fast()
            for ch in context:
                world.present_symbol(ch, write=False)
            world.inject("E")
            frames = []
            for _ in range(8):
                world.step(write=False)
                frames.append(world.x.copy())
            features.append(np.concatenate([f.ravel() for f in frames]))
            labels.append(label)
            seeds.append(seed)
    x = np.asarray(features)
    y = np.asarray(labels)
    seeds = np.asarray(seeds)
    accuracy = _centroid_accuracy(x, y, seeds)
    return {
        "gate": "W3_CONTEXT",
        "question": "Does the identical target primitive E evoke distinguishable responses after AB versus CD context?",
        "target": "E",
        "contexts": ["AB", "CD"],
        "accuracy": accuracy,
        "chance": 0.5,
        "pass": accuracy >= 0.75,
    }


def _ridge_dual_fit_predict(x_train, y_train, x_test, lam=1e-2):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    a = (x_train - mean) / std
    b = (x_test - mean) / std
    a = np.column_stack([np.ones(len(a)), a])
    b = np.column_stack([np.ones(len(b)), b])
    gram = a @ a.T + lam * np.eye(a.shape[0])
    alpha = np.linalg.solve(gram, y_train)
    return b @ (a.T @ alpha)


def _best_linear_xor_accuracy() -> float:
    """Attacker-friendly 0/1 error search for a line on the four XOR corners."""
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    best = 0.0
    angles = np.linspace(0.0, np.pi, 721)
    biases = np.linspace(-1.5, 1.5, 601)
    for angle in angles:
        w = np.array([np.cos(angle), np.sin(angle)])
        base = x @ w
        pred = base[None, :] + biases[:, None] >= 0.0
        score = (pred == y[None, :]).mean(axis=1).max()
        best = max(best, float(score))
    return best


def gate_w4() -> dict:
    """Reservoir readout solves temporal XOR better than a raw linear attacker."""
    corpus = ["ABCD", "DCBA", "EFEF"]
    x, y, seeds = [], [], []
    for seed in range(16):
        world = grow_world(seed, corpus, repeats=5)
        for gap in (0, 1, 2):
            for b1 in (0, 1):
                for b2 in (0, 1):
                    world.wipe_fast()
                    world.present_symbol("A" if b1 == 0 else "B", write=False)
                    for _ in range(gap):
                        world.step(write=False)
                    world.present_symbol("C" if b2 == 0 else "D", write=False)
                    x.append(world.fast_vector())
                    y.append(b1 ^ b2)
                    seeds.append(seed)
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    seeds = np.asarray(seeds)
    test = seeds % 4 == 0
    train = ~test
    scores = _ridge_dual_fit_predict(x[train], y[train], x[test])
    pred = scores >= 0.5
    reservoir_acc = float(np.mean(pred == y[test]))
    raw_attacker = _best_linear_xor_accuracy()
    return {
        "gate": "W4_COMPUTATION",
        "question": "Can a linear readout of gel state solve temporal XOR better than any linear boundary on the two raw bits?",
        "reservoir_accuracy": reservoir_acc,
        "raw_linear_attacker_upper_accuracy": raw_attacker,
        "delta": reservoir_acc - raw_attacker,
        "pass": bool(reservoir_acc >= 0.90 and reservoir_acc - raw_attacker >= 0.15),
    }


def gate_w5() -> dict:
    """Cross-world response geometry under strong individualized material seeds."""
    words = ["ABC", "ACB", "BAC", "BCA"]
    corpus = words * 2
    config = with_individuality(JelloConfig(), 0.012)
    features, labels, seeds = [], [], []
    for seed in range(20):
        world = grow_world(seed, corpus, repeats=4, config=config)
        for label, word in enumerate(words):
            features.append(world.response_to_word(word))
            labels.append(label)
            seeds.append(seed)
    x = np.asarray(features)
    y = np.asarray(labels)
    seeds = np.asarray(seeds)
    train = seeds < 10
    test = seeds >= 10
    mean = x[train].mean(axis=0)
    std = x[train].std(axis=0) + 1e-6
    z = (x - mean) / std
    centroids = np.stack([z[train][y[train] == k].mean(axis=0) for k in range(len(words))])
    distances = ((z[test, None, :] - centroids[None, :, :]) ** 2).mean(axis=-1)
    pred = distances.argmin(axis=1)
    accuracy = float(np.mean(pred == y[test]))
    return {
        "gate": "W5_SHARED_VS_INDIVIDUAL_CODE",
        "question": "Can a decoder trained on 10 individualized gels identify order-only pseudowords in 10 unseen gels?",
        "individuality_sigma": config.individuality,
        "train_worlds": 10,
        "test_worlds": 10,
        "classes": words,
        "accuracy": accuracy,
        "chance": 0.25,
        "pass": accuracy >= 0.65,
    }


def run_all_gates() -> dict:
    gates = [gate_w0(), gate_w1(), gate_w2(), gate_w3(), gate_w4(), gate_w5()]
    return {
        "schema": "jellobrain/gates-v1",
        "claim_boundary": (
            "Passing gates supports persistent history, order sensitivity, geometry-dependent "
            "responses, context dependence, reservoir computation, and cross-world decodability "
            "in this toy. It does not establish thought, understanding, consciousness, or a brain model."
        ),
        "gates": gates,
        "all_pass": bool(all(g["pass"] for g in gates)),
    }


def write_receipt(path: str | Path = "results/gates.json") -> dict:
    receipt = run_all_gates()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    print(json.dumps(write_receipt(), indent=2))
