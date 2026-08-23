import copy
import pickle
from itertools import count

from endogenous_neat.genome import (
    EndogenousGenome,
    apply_theta_to_config,
    sync_node_indexer,
)
from endogenous_neat.theta import Theta


def test_genome_has_theta_after_construction():
    genome = EndogenousGenome(0)
    assert isinstance(genome.theta, Theta)


def test_configure_new_sets_theta(genome_config):
    genome = EndogenousGenome(1)
    genome.configure_new(genome_config)
    assert isinstance(genome.theta, Theta)
    assert genome.theta.l1_distance_from_defaults() >= 0.0


def test_mutation_preserves_theta(genome_config):
    genome = EndogenousGenome(2)
    genome.configure_new(genome_config)
    genome.mutate(genome_config)
    assert isinstance(genome.theta, Theta)


def test_crossover_creates_theta(genome_config):
    g1 = EndogenousGenome(3)
    g2 = EndogenousGenome(4)
    g1.configure_new(genome_config)
    g2.configure_new(genome_config)
    g1.theta = Theta.from_dict({**Theta.DEFAULTS, "conn_add_prob": 0.1})
    g2.theta = Theta.from_dict({**Theta.DEFAULTS, "conn_add_prob": 0.7})
    g1.fitness = 1.0
    g2.fitness = 0.5

    child = EndogenousGenome(5)
    child.configure_crossover(g1, g2, genome_config)
    assert isinstance(child.theta, Theta)
    assert child.theta.conn_add_prob in (0.1, 0.7)


def test_distance_includes_theta_component(genome_config):
    g1 = EndogenousGenome(6)
    g2 = EndogenousGenome(7)
    g1.configure_new(genome_config)
    g2.configure_new(genome_config)
    g1.theta = Theta.default()
    g2.theta = Theta.from_dict({**Theta.DEFAULTS, "conn_add_prob": 0.8})
    genome_config.theta_distance_coefficient = 0.1

    d_same = g1.distance(g1, genome_config)
    d_diff = g1.distance(g2, genome_config)
    assert d_diff > d_same
    assert d_diff - d_same >= 0.1 * 0.6


def test_distance_ignores_theta_when_coeff_zero(genome_config):
    g1 = EndogenousGenome(8)
    g2 = EndogenousGenome(9)
    g1.configure_new(genome_config)
    g1.fitness = 1.0
    g2.configure_crossover(g1, g1, genome_config)
    g1.theta = Theta.default()
    g2.theta = Theta.from_dict({**Theta.DEFAULTS, "conn_add_prob": 0.8})
    genome_config.theta_distance_coefficient = 0.0
    assert g1.distance(g2, genome_config) == g1.distance(g1, genome_config)


def test_apply_theta_does_not_mutate_global_config(genome_config):
    original = genome_config.conn_add_prob
    theta = Theta.from_dict({**Theta.DEFAULTS, "conn_add_prob": 0.05})
    local = apply_theta_to_config(genome_config, theta)
    assert local.conn_add_prob == 0.05
    assert genome_config.conn_add_prob == original


def test_many_mutations_do_not_alter_global_config(genome_config):
    original = copy.copy(
        {name: getattr(genome_config, Theta.NEAT_CONFIG_KEYS[name]) for name in Theta.BOUNDS}
    )
    for i in range(20):
        genome = EndogenousGenome(i)
        genome.configure_new(genome_config)
        genome.mutate(genome_config)
    for name, value in original.items():
        assert getattr(genome_config, Theta.NEAT_CONFIG_KEYS[name]) == value


def test_pickle_round_trip(genome_config):
    genome = EndogenousGenome(99)
    genome.configure_new(genome_config)
    data = pickle.dumps(genome)
    restored = pickle.loads(data)
    assert restored.theta.to_vector() == genome.theta.to_vector()


def test_sync_node_indexer_skips_existing_keys(genome_config):
    genome_config.node_indexer = count(0)
    nodes = {0: None, 1: None, 5: None}
    sync_node_indexer(genome_config, nodes)
    assert next(genome_config.node_indexer) == 6


def test_mutate_with_stale_indexer_does_not_assert(genome_config):
    """Reproduce AssertionError when indexer lags behind genome node keys."""
    genome = EndogenousGenome(100)
    genome.configure_new(genome_config)
    # Simulate crossover inheriting a high node id while indexer is behind.
    high_key = max(genome.nodes) + 50
    genome.nodes[high_key] = copy.deepcopy(next(iter(genome.nodes.values())))
    genome_config.node_indexer = count(0)
    genome.theta = Theta.from_dict(
        {**Theta.DEFAULTS, "node_add_prob": 1.0, "conn_add_prob": 0.0,
         "node_delete_prob": 0.0, "conn_delete_prob": 0.0}
    )
    # Should not raise AssertionError from get_new_node_key.
    genome.mutate(genome_config)
    assert isinstance(genome.theta, Theta)
