"""Endogenous genome extending neat-python DefaultGenome."""

from __future__ import annotations

import copy
from itertools import count
from random import Random

import neat

from endogenous_neat.theta import THETA_FIELD_NAMES, Theta
from endogenous_neat.theta_config import ThetaEvolutionConfig

# Legacy module constant; distance() reads config.theta_distance_coefficient.
THETA_DISTANCE_COEFF = 0.0


def apply_theta_to_config(config, theta: Theta):
    """Return a shallow copy of genome config with theta overlaid on mutation rates."""
    local = copy.copy(config)
    for field_name in THETA_FIELD_NAMES:
        config_key = Theta.NEAT_CONFIG_KEYS[field_name]
        setattr(local, config_key, getattr(theta, field_name))
    return local


def sync_node_indexer(config, node_dict) -> None:
    """Advance config.node_indexer past keys already present in node_dict.

    Genomes can acquire high node IDs via crossover/innovation while the shared
    indexer lags behind. neat-python's get_new_node_key then asserts
    ``new_id not in node_dict``. Advance (and rebind) the indexer so the next
    allocated key is unused in this genome.
    """
    max_id = max(node_dict) if node_dict else -1
    start_at = max(max_id + 1, getattr(config, "num_outputs", 0))
    if getattr(config, "node_indexer", None) is None:
        config.node_indexer = count(start_at)
        return

    while True:
        candidate = next(config.node_indexer)
        if candidate > max_id:
            # Rebind so the chosen candidate is the next yielded value.
            config.node_indexer = count(candidate)
            return


class EndogenousGenome(neat.DefaultGenome):
    """NEAT genome with per-individual endogenous mutation parameters."""

    @classmethod
    def parse_config(cls, param_dict):
        genome_params, theta_config = ThetaEvolutionConfig.extract_from_param_dict(
            param_dict
        )
        genome_params["node_gene_type"] = neat.DefaultNodeGene
        genome_params["connection_gene_type"] = neat.DefaultConnectionGene
        config = neat.genome.DefaultGenomeConfig(genome_params, cls.__name__)
        theta_config.attach_to(config)
        return config

    def __init__(self, key):
        super().__init__(key)
        self.theta = Theta.default()

    def configure_new(self, config):
        super().configure_new(config)
        initial = getattr(config, "theta_initial", None)
        if initial is not None:
            self.theta = initial
            return
        spread = getattr(config, "theta_init_spread", 0.1)
        self.theta = Theta.randomized(Random(), spread=spread)

    def configure_crossover(self, genome1, genome2, config, fitness_criterion=None):
        super().configure_crossover(genome1, genome2, config, fitness_criterion)
        if hasattr(genome1, "theta") and hasattr(genome2, "theta"):
            self.theta = Theta.crossover(genome1.theta, genome2.theta, Random())
        else:
            self.theta = Theta.default()

    def mutate(self, config):
        # Use this genome's theta for structural and attribute mutation rates.
        local_config = apply_theta_to_config(config, self.theta)
        sync_node_indexer(local_config, self.nodes)
        # Propagate indexer rebinding to the shared population config.
        config.node_indexer = local_config.node_indexer
        super().mutate(local_config)
        theta_config = ThetaEvolutionConfig.from_genome_config(config)
        self.theta = self.theta.mutated_with_config(Random(), theta_config)

    def distance(self, other, config):
        base = super().distance(other, config)
        if not hasattr(other, "theta"):
            return base
        coeff = getattr(config, "theta_distance_coefficient", THETA_DISTANCE_COEFF)
        if coeff <= 0:
            return base
        theta_dist = sum(
            abs(a - b)
            for a, b in zip(self.theta.to_vector(), other.theta.to_vector(), strict=True)
        )
        return base + coeff * theta_dist
