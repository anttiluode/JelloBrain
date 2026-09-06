import numpy as np

from dialogue import DialoguePair, crossplay_accuracy


def test_pair_learns_and_channel_swap_is_causal():
    pair = DialoguePair(500)
    pair.train(800)
    assert pair.greedy_accuracy() == 1.0
    assert pair.greedy_accuracy((1, 0)) == 0.0


def test_private_conventions_can_be_incompatible():
    a = DialoguePair(500)
    b = DialoguePair(501)
    a.train(800)
    b.train(800)
    assert a.orientation() != b.orientation()
    assert crossplay_accuracy(a, b) == 0.0


def test_repair_after_wire_swap():
    pair = DialoguePair(503)
    pair.train(800)
    assert pair.greedy_accuracy((1, 0)) == 0.0
    pair.rng = np.random.default_rng(12345)
    rewards = pair.train(500, channel_map=(1, 0))
    assert pair.greedy_accuracy((1, 0)) == 1.0
    assert rewards[-100:].mean() > 0.9
