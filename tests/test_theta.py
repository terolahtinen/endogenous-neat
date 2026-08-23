from random import Random

import pytest

from endogenous_neat.theta import THETA_FIELD_NAMES, Theta


def test_default_theta_within_bounds():
    theta = Theta.default()
    for name in THETA_FIELD_NAMES:
        lo, hi = Theta.BOUNDS[name]
        value = getattr(theta, name)
        assert lo <= value <= hi


def test_mutation_preserves_bounds():
    rng = Random(0)
    theta = Theta.default()
    for _ in range(200):
        theta = theta.mutated(rng, rate=1.0, power=0.2)
        for name in THETA_FIELD_NAMES:
            lo, hi = Theta.BOUNDS[name]
            value = getattr(theta, name)
            assert lo <= value <= hi


def test_unit_interval_vector_maps_bounds():
    lo = Theta.from_dict({name: Theta.BOUNDS[name][0] for name in THETA_FIELD_NAMES})
    hi = Theta.from_dict({name: Theta.BOUNDS[name][1] for name in THETA_FIELD_NAMES})
    mid = Theta.default()
    assert lo.to_unit_interval_vector() == pytest.approx([0.0] * 8)
    assert hi.to_unit_interval_vector() == pytest.approx([1.0] * 8)
    units = mid.to_unit_interval_vector()
    assert all(0.0 <= u <= 1.0 for u in units)


def test_meta_output_transformation_within_bounds():
    # Meta-heads are already sigmoid-activated; map [0, 1] linearly into bounds.
    raw_outputs = [0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9, 0.5]
    theta = Theta.from_meta_outputs(raw_outputs)
    for name, raw in zip(THETA_FIELD_NAMES, raw_outputs, strict=True):
        lo, hi = Theta.BOUNDS[name]
        value = getattr(theta, name)
        assert lo <= value <= hi
        assert value == pytest.approx(lo + raw * (hi - lo))


def test_meta_output_clamps_out_of_range():
    raw_outputs = [-1.0, 2.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5]
    theta = Theta.from_meta_outputs(raw_outputs)
    assert theta.node_add_prob == pytest.approx(Theta.BOUNDS["node_add_prob"][0])
    assert theta.node_delete_prob == pytest.approx(Theta.BOUNDS["node_delete_prob"][1])


def test_crossover_within_bounds():
    rng = Random(1)
    t1 = Theta.from_dict({name: Theta.BOUNDS[name][0] for name in THETA_FIELD_NAMES})
    t2 = Theta.from_dict({name: Theta.BOUNDS[name][1] for name in THETA_FIELD_NAMES})
    for _ in range(50):
        child = Theta.crossover(t1, t2, rng)
        for name in THETA_FIELD_NAMES:
            lo, hi = Theta.BOUNDS[name]
            value = getattr(child, name)
            assert lo <= value <= hi


def test_vector_round_trip():
    theta = Theta.default()
    restored = Theta.from_vector(theta.to_vector())
    assert restored.to_vector() == theta.to_vector()
