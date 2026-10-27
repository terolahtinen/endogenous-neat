"""Tests for random-λ start (shared θ_rand for standard vs meta-theta)."""

from random import Random

import pytest

from endogenous_neat.experiment import format_theta_meta_lines, sample_theta_for_run
from endogenous_neat.genome import EndogenousGenome, apply_theta_to_config
from endogenous_neat.tasks.base import theta_for_genome
from endogenous_neat.theta import THETA_FIELD_NAMES, Theta


def test_uniform_random_within_bounds():
    rng = Random(0)
    for _ in range(100):
        theta = Theta.uniform_random(rng)
        for name in THETA_FIELD_NAMES:
            lo, hi = Theta.BOUNDS[name]
            assert lo <= getattr(theta, name) <= hi


def test_uniform_random_differs_from_defaults():
    theta = Theta.uniform_random(Random(42))
    assert theta.l1_distance_from_defaults() > 0.0


def test_sample_theta_for_run_paired_by_run_index():
    """Same seed + run_index → identical θ for standard/meta pair."""
    a = sample_theta_for_run(run_index=3, seed=7)
    b = sample_theta_for_run(run_index=3, seed=7)
    assert a.to_vector() == b.to_vector()
    other = sample_theta_for_run(run_index=4, seed=7)
    assert a.to_vector() != other.to_vector()


def test_apply_theta_freezes_standard_config(genome_config):
    theta = Theta.uniform_random(Random(1))
    local = apply_theta_to_config(genome_config, theta)
    for name in THETA_FIELD_NAMES:
        config_key = Theta.NEAT_CONFIG_KEYS[name]
        assert getattr(local, config_key) == pytest.approx(getattr(theta, name))


def test_configure_new_uses_theta_initial(genome_config):
    initial = Theta.uniform_random(Random(9))
    genome_config.theta_initial = initial
    genome = EndogenousGenome(0)
    genome.configure_new(genome_config)
    assert genome.theta.to_vector() == initial.to_vector()


def test_theta_for_genome_fixed_theta_for_standard():
    class PlainGenome:
        pass

    fixed = Theta.uniform_random(Random(2))
    assert theta_for_genome(PlainGenome(), fixed_theta=fixed).to_vector() == (
        fixed.to_vector()
    )
    assert theta_for_genome(PlainGenome()).to_vector() == Theta.default().to_vector()


def test_l1_distance_between_thetas():
    a = Theta.default()
    b = Theta.from_dict({**Theta.DEFAULTS, "weight_mutate_power": 2.5})
    assert a.l1_distance(b) == pytest.approx(2.0)


def test_format_theta_meta_lines_uses_given_prefix():
    lines = format_theta_meta_lines(Theta.default(), "theta_final")
    keys = [line.split("=", 1)[0] for line in lines]
    assert "theta_final_l1_from_defaults" in keys
    assert "theta_final_weight_mutate_power" in keys
    assert all(not key.startswith("theta_initial_") for key in keys)
