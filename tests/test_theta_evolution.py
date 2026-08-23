from random import Random

import pytest

from endogenous_neat.genome import EndogenousGenome
from endogenous_neat.theta import THETA_FIELD_NAMES, Theta
from endogenous_neat.theta_config import ThetaEvolutionConfig


def test_randomized_theta_within_bounds():
    rng = Random(0)
    for _ in range(50):
        theta = Theta.randomized(rng, spread=0.1)
        for name in THETA_FIELD_NAMES:
            lo, hi = Theta.BOUNDS[name]
            assert lo <= getattr(theta, name) <= hi


def test_log_normal_mutation_stays_bounded():
    rng = Random(1)
    theta = Theta.default()
    config = ThetaEvolutionConfig(mutate_rate=1.0, mutate_power=0.05, mutate_mode="log_normal")
    for _ in range(100):
        theta = theta.mutated_with_config(rng, config)
        for name in THETA_FIELD_NAMES:
            lo, hi = Theta.BOUNDS[name]
            assert lo <= getattr(theta, name) <= hi


def test_weight_mutate_power_cap_applied():
    rng = Random(2)
    theta = Theta.from_dict({**Theta.DEFAULTS, "weight_mutate_power": 4.0})
    mutated = theta.mutated(
        rng,
        rate=0.0,
        power=0.0,
        mode="gaussian",
        weight_mutate_power_cap=1.5,
    )
    assert mutated.weight_mutate_power == pytest.approx(1.5)


def test_parse_config_attaches_theta_settings(minimal_config_path):
    import neat

    from endogenous_neat.reproduction import EndogenousReproduction

    config = neat.Config(
        EndogenousGenome,
        EndogenousReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(minimal_config_path),
    )
    genome_config = config.genome_config
    assert genome_config.theta_mutate_mode == "log_normal"
    assert genome_config.theta_distance_coefficient == 0.0
    assert genome_config.weight_mutate_power_cap == pytest.approx(1.5)
