"""Tests for starting a population from saved genomes."""

from __future__ import annotations

import pickle

import neat
import pytest

from endogenous_neat.genome import EndogenousGenome, sync_node_indexer
from endogenous_neat.seed import apply_seeds, clone_counts, genomes_for_type, load_genome
from endogenous_neat.theta import Theta


def _signature(genome):
    weights = tuple(sorted(round(conn.weight, 5) for conn in genome.connections.values()))
    theta = round(genome.theta.conn_add_prob, 5)
    return (tuple(sorted(genome.nodes)), weights, theta)


def _make_seed(config, weight: float, conn_add: float):
    genome = EndogenousGenome(1)
    genome.configure_new(config.genome_config)
    sync_node_indexer(config.genome_config, genome.nodes)
    node_key = config.genome_config.get_new_node_key(genome.nodes)
    genome.nodes[node_key] = genome.create_node(config.genome_config, node_key)
    for connection in genome.connections.values():
        connection.weight = weight
    connection = next(iter(genome.connections.values()))
    connection.innovation = 9999
    genome.fitness = 3.0
    genome.theta = Theta.from_dict({**Theta.DEFAULTS, "conn_add_prob": conn_add})
    return genome


def test_clone_counts_gives_remainder_to_earlier_seeds():
    assert clone_counts(2, 4) == [2, 2]
    assert clone_counts(2, 5) == [3, 2]


def test_elite_fill_keeps_one_copy_and_random_rest(neat_config, tmp_path):
    seed = _make_seed(neat_config, weight=7.5, conn_add=0.11)
    path = tmp_path / "best_genome.pkl"
    with path.open("wb") as handle:
        pickle.dump(seed, handle)
    loaded = load_genome(path)

    population = neat.Population(neat_config)
    apply_seeds(population, neat_config, [loaded], "elite")

    assert len(population.population) == neat_config.pop_size
    matches = [
        genome
        for genome in population.population.values()
        if _signature(genome) == _signature(loaded)
    ]
    assert len(matches) == 1
    assert matches[0].fitness is None
    assert loaded.fitness == 3.0
    assert neat_config.genome_config.innovation_tracker.global_counter >= 9999
    assert neat_config.genome_config.get_new_node_key({}) > max(loaded.nodes)
    assert population.species.species
    assert population._skip_first_evaluation is False


def test_clone_fill_spreads_several_seeds(neat_config):
    first = _make_seed(neat_config, weight=7.5, conn_add=0.11)
    second = _make_seed(neat_config, weight=8.5, conn_add=0.22)
    population = neat.Population(neat_config)
    apply_seeds(population, neat_config, [first, second], "clones")

    counts = {}
    for genome in population.population.values():
        assert genome.fitness is None
        counts[_signature(genome)] = counts.get(_signature(genome), 0) + 1
    assert counts[_signature(first)] == 2
    assert counts[_signature(second)] == 2
    assert sum(counts.values()) == neat_config.pop_size


def test_wrong_genome_class_is_rejected(neat_config, tmp_path):
    plain = neat.DefaultGenome(1)
    path = tmp_path / "plain.pkl"
    with path.open("wb") as handle:
        pickle.dump(plain, handle)
    loaded, paths = genomes_for_type([path], EndogenousGenome)
    assert loaded == []
    assert paths == []

    population = neat.Population(neat_config)
    with pytest.raises(ValueError, match="DefaultGenome"):
        apply_seeds(population, neat_config, [plain], "elite")
