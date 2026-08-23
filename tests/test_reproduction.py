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
    """Higher conn_add_prob should tend to add more connections under mutation."""

    def count_net_growth(theta: Theta) -> float:
        totals = []
        for seed in range(40):
            genome = EndogenousGenome(1000 + seed)
            genome.configure_new(neat_config.genome_config)
            genome.theta = theta
            start = len(genome.connections)
            for _ in range(80):
                genome.mutate(neat_config.genome_config)
            totals.append(len(genome.connections) - start)
        return sum(totals) / len(totals)

    # Freeze structural deletes/adds of nodes so conn_add_prob dominates.
    shared = {
        **Theta.DEFAULTS,
        "node_add_prob": 0.0,
        "node_delete_prob": 0.0,
        "conn_delete_prob": 0.0,
    }
    low = Theta.from_dict({**shared, "conn_add_prob": 0.01})
    high = Theta.from_dict({**shared, "conn_add_prob": 0.9})

    assert count_net_growth(high) > count_net_growth(low)
