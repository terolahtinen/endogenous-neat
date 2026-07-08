from random import Random

import pytest

from endogenous_neat.genome import EndogenousGenome
from endogenous_neat.theta import THETA_FIELD_NAMES, Theta


def test_children_receive_theta(neat_config):
    g1 = EndogenousGenome(1)
    g2 = EndogenousGenome(2)
    g1.configure_new(neat_config.genome_config)
    g2.configure_new(neat_config.genome_config)
    g1.theta = Theta.from_dict({**Theta.DEFAULTS, "node_add_prob": 0.05})
    g2.theta = Theta.from_dict({**Theta.DEFAULTS, "node_add_prob": 0.45})
    g1.fitness = 1.0
    g2.fitness = 0.5

    child = EndogenousGenome(3)
    child.configure_crossover(g1, g2, neat_config.genome_config)
    assert child.theta.node_add_prob in (0.05, 0.45)


def test_child_theta_stays_bounded_after_mutation(neat_config):
    genome = EndogenousGenome(4)
    genome.configure_new(neat_config.genome_config)
    rng = Random(0)
    for _ in range(100):
        genome.theta = genome.theta.mutated(rng, rate=1.0, power=0.3)
        genome.mutate(neat_config.genome_config)
        for name in THETA_FIELD_NAMES:
            lo, hi = Theta.BOUNDS[name]
            assert lo <= getattr(genome.theta, name) <= hi


def test_high_conn_add_prob_adds_more_connections(neat_config):
    """Parents with higher conn_add_prob should tend to add more connections."""

    def count_connections(theta: Theta) -> int:
        totals = []
        for _ in range(30):
            genome = EndogenousGenome(10)
            genome.configure_new(neat_config.genome_config)
            genome.theta = theta
            start = len(genome.connections)
            for _ in range(50):
                genome.mutate(neat_config.genome_config)
            totals.append(len(genome.connections) - start)
        return sum(totals) / len(totals)

    low = Theta.from_dict({**Theta.DEFAULTS, "conn_add_prob": 0.01, "conn_delete_prob": 0.0})
    high = Theta.from_dict({**Theta.DEFAULTS, "conn_add_prob": 0.75, "conn_delete_prob": 0.0})

    assert count_connections(high) > count_connections(low)
