"""Genomes that can add and delete letter-output nodes.

Meta outputs stay the fixed ``config.output_keys``. Letter outputs are extra
nodes (keys allocated by the usual node indexer) listed in
``letter_output_keys``. Their count is the spelled word's length.
"""

from __future__ import annotations

from random import Random, random, randrange

import neat

from endogenous_neat.genome import (
    EndogenousGenome,
    apply_theta_to_config,
    pin_frozen_theta,
    sync_node_indexer,
)
from endogenous_neat.theta_config import ThetaEvolutionConfig

WORD_CONFIG_DEFAULTS: dict[str, float | int] = {
    "output_add_prob": 0.2,
    "output_delete_prob": 0.2,
    "initial_word_length": 1,
    "max_word_length": 256,
}


def extract_word_config(params: dict) -> dict[str, float | int]:
    """Remove word-length keys before building a DefaultGenomeConfig."""
    extracted: dict[str, float | int] = {}
    for key, default in WORD_CONFIG_DEFAULTS.items():
        raw = params.pop(key, default)
        if key.endswith("_prob"):
            value = float(raw)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be in [0, 1], got {value}")
            extracted[key] = value
        else:
            value = int(raw)
            if value < 0:
                raise ValueError(f"{key} must be >= 0, got {value}")
            extracted[key] = value
    if extracted["max_word_length"] < extracted["initial_word_length"]:
        raise ValueError(
            "max_word_length must be >= initial_word_length "
            f"({extracted['max_word_length']} < {extracted['initial_word_length']})"
        )
    return extracted


def attach_word_config(genome_config, extracted: dict[str, float | int]) -> None:
    for key, value in extracted.items():
        setattr(genome_config, key, value)


def merge_letter_keys(fitter, other, child_nodes: dict) -> list[int]:
    """Letter keys the child actually inherited, fitter parent first."""
    keys: list[int] = []
    seen: set[int] = set()
    for parent in (fitter, other):
        for key in getattr(parent, "letter_output_keys", []):
            if key in child_nodes and key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _fitter_parent(genome1, genome2, fitness_criterion: str | None):
    criterion = fitness_criterion or "max"
    if genome1.fitness is None or genome2.fitness is None:
        return genome1, genome2
    if criterion == "min":
        genome1_better = genome1.fitness < genome2.fitness
    else:
        genome1_better = genome1.fitness > genome2.fitness
    if genome1_better:
        return genome1, genome2
    return genome2, genome1


def _mutate_structure_protecting_letters(genome, config, mutate) -> None:
    """Run structural mutation without letting it delete letter outputs."""
    saved = list(config.output_keys)
    extra = [key for key in genome.letter_output_keys if key in genome.nodes]
    config.output_keys = saved + extra
    try:
        mutate(genome, config)
    finally:
        config.output_keys = saved
    genome.letter_output_keys = [
        key for key in genome.letter_output_keys if key in genome.nodes
    ]


class LetterOutputGenome:
    """Mixin: ordered letter-output nodes on top of a NEAT genome."""

    def __init__(self, key):
        super().__init__(key)
        self.letter_output_keys: list[int] = []

    def configure_new(self, config):
        super().configure_new(config)
        self.letter_output_keys = []
        for _ in range(int(config.initial_word_length)):
            self._add_letter_output(config)

    def configure_crossover(self, genome1, genome2, config, fitness_criterion=None):
        super().configure_crossover(genome1, genome2, config, fitness_criterion)
        fitter, other = _fitter_parent(genome1, genome2, fitness_criterion)
        self.letter_output_keys = merge_letter_keys(fitter, other, self.nodes)

    def _add_letter_output(self, config) -> None:
        if len(self.letter_output_keys) >= int(config.max_word_length):
            return
        sync_node_indexer(config, self.nodes)
        node_key = config.get_new_node_key(self.nodes)
        self.nodes[node_key] = self.create_node(config, node_key)
        index = randrange(len(self.letter_output_keys) + 1)
        self.letter_output_keys.insert(index, node_key)

    def _delete_letter_at(self, config, index: int) -> None:
        key = self.letter_output_keys.pop(index)
        for conn_key in [conn for conn in self.connections if key in conn]:
            del self.connections[conn_key]
        if key in self.nodes:
            del self.nodes[key]
        saved = list(config.output_keys)
        remaining = [k for k in self.letter_output_keys if k in self.nodes]
        config.output_keys = saved + remaining
        try:
            self._prune_dangling_nodes(config)
        finally:
            config.output_keys = saved
        self.letter_output_keys = [
            k for k in self.letter_output_keys if k in self.nodes
        ]

    def _mutate_word_length(self, config) -> None:
        if random() < float(config.output_add_prob):
            self._add_letter_output(config)
        if self.letter_output_keys and random() < float(config.output_delete_prob):
            self._delete_letter_at(config, randrange(len(self.letter_output_keys)))


def _parse_standard_config(cls, param_dict):
    params = dict(param_dict)
    word_config = extract_word_config(params)
    params["node_gene_type"] = neat.DefaultNodeGene
    params["connection_gene_type"] = neat.DefaultConnectionGene
    config = neat.genome.DefaultGenomeConfig(params, cls.__name__)
    attach_word_config(config, word_config)
    return config


def _parse_endogenous_config(cls, param_dict):
    genome_params, theta_config = ThetaEvolutionConfig.extract_from_param_dict(
        param_dict
    )
    word_config = extract_word_config(genome_params)
    genome_params["node_gene_type"] = neat.DefaultNodeGene
    genome_params["connection_gene_type"] = neat.DefaultConnectionGene
    config = neat.genome.DefaultGenomeConfig(genome_params, cls.__name__)
    theta_config.attach_to(config)
    attach_word_config(config, word_config)
    return config


class WordGenome(LetterOutputGenome, neat.DefaultGenome):
    """Standard NEAT genome that can change the number of letter outputs."""

    @classmethod
    def parse_config(cls, param_dict):
        return _parse_standard_config(cls, param_dict)

    def mutate(self, config):
        _mutate_structure_protecting_letters(self, config, neat.DefaultGenome.mutate)
        self._mutate_word_length(config)


class WordEndogenousGenome(LetterOutputGenome, EndogenousGenome):
    """Endogenous genome that can change the number of letter outputs."""

    @classmethod
    def parse_config(cls, param_dict):
        return _parse_endogenous_config(cls, param_dict)

    def mutate(self, config):
        # Frozen ablation fields are defaults before they set structural rates.
        self.theta = pin_frozen_theta(self.theta, config)
        local_config = apply_theta_to_config(config, self.theta)
        sync_node_indexer(local_config, self.nodes)
        config.node_indexer = local_config.node_indexer
        _mutate_structure_protecting_letters(
            self, local_config, neat.DefaultGenome.mutate
        )
        theta_config = ThetaEvolutionConfig.from_genome_config(config)
        self.theta = self.theta.mutated_with_config(Random(), theta_config)
        self.theta = pin_frozen_theta(self.theta, config)
        self._mutate_word_length(local_config)
