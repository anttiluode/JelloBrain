import numpy as np

from jellobrain import JelloWorld, grow_world


def test_wipe_fast_preserves_slow_material():
    world = grow_world(3, ["ABC", "ACB"], repeats=2)
    before = world.material_vector().copy()
    world.present_word("DEF", write=False)
    assert world.x.max() > 0
    world.wipe_fast()
    assert np.allclose(world.x, 0)
    assert np.allclose(world.refractory, 0)
    assert np.allclose(world.eligibility, 0)
    assert np.array_equal(before, world.material_vector())


def test_words_are_only_primitive_sequences():
    assert JelloWorld.parse_word("ABC") == (0, 1, 2)
    assert JelloWorld.parse_word(["A", "C", "B"]) == (0, 2, 1)


def test_material_shuffle_preserves_values_but_changes_geometry():
    world = grow_world(1, ["ABCD", "DCBA"], repeats=3)
    before = world.material_vector().copy()
    world.shuffle_material(99)
    after = world.material_vector()
    assert np.allclose(np.sort(before), np.sort(after))
    assert not np.array_equal(before, after)


def test_writing_changes_future_probe_response():
    fresh = JelloWorld(0)
    grown = grow_world(0, ["ABC", "ABC", "ABC"], repeats=4)
    a = fresh.probe_battery(steps=6)
    b = grown.probe_battery(steps=6)
    assert np.linalg.norm(a - b) > 1e-6
