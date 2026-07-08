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


def test_meta_output_transformation_within_bounds():
    raw_outputs = [-10.0, -1.0, 0.0, 1.0, 5.0, 10.0, -5.0, 3.0]
    theta = Theta.from_meta_outputs(raw_outputs)
    for name in THETA_FIELD_NAMES:
        lo, hi = Theta.BOUNDS[name]
        value = getattr(theta, name)
        assert lo <= value <= hi


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
