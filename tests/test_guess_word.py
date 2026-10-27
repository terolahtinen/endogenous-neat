"""Tests for the guess-a-word task and variable letter outputs."""

from __future__ import annotations

from pathlib import Path

import neat
import pytest
from neat.innovation import InnovationTracker

from endogenous_neat.reproduction import EndogenousReproduction
from endogenous_neat.tasks import get_task
from endogenous_neat.tasks.base import NUM_META_OUTPUTS
from endogenous_neat.tasks.guess_word import (
    ALPHABET,
    FITNESS_THRESHOLD,
    TARGET_WORD,
    activation_to_char,
    edit_distance,
    evaluate_guess_word,
    fitness_threshold,
    make_eval_genomes,
    network_for_genome,
    normalize_answer,
)
from endogenous_neat.word_genome import WordEndogenousGenome, WordGenome

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeNet:
    def __init__(self, outputs: list[float]):
        self.outputs = outputs
        self.last_inputs = None

    def activate(self, inputs):
        self.last_inputs = list(inputs)
        return list(self.outputs)


def _config(genome_type):
    reproduction = (
        EndogenousReproduction
        if genome_type is WordEndogenousGenome
        else neat.DefaultReproduction
    )
    config = neat.Config(
        genome_type,
        reproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(FIXTURES / "guess_word_minimal.cfg"),
    )
    config.genome_config.innovation_tracker = InnovationTracker()
    return config


def test_edit_distance_cases():
    assert edit_distance("same", "same") == 0
    assert edit_distance("", "a") == 1
    assert edit_distance("a", "") == 1
    assert edit_distance("a", "b") == 1
    assert edit_distance("ac", "abc") == 1
    assert edit_distance("abc", "ac") == 1
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("a", "neat") == 3


def test_activation_maps_onto_alphabet_including_space():
    assert activation_to_char(0.0) == "a"
    assert activation_to_char(-1.0) == "a"
    assert activation_to_char(1.5 / len(ALPHABET)) == "b"
    assert activation_to_char(1.0) == " "
    assert len(ALPHABET) == 27


def test_normalize_answer_lowercases_and_allows_spaces():
    assert normalize_answer(None) == TARGET_WORD == "neat"
    assert normalize_answer("NEAT") == "neat"
    assert normalize_answer("hello world") == "hello world"
    with pytest.raises(ValueError):
        normalize_answer("")
    with pytest.raises(ValueError):
        normalize_answer("hello!")


def test_new_genome_starts_at_initial_word_length():
    config = _config(WordGenome)
    genome = WordGenome(1)
    genome.configure_new(config.genome_config)
    assert len(genome.letter_output_keys) == 1
    assert genome.letter_output_keys[0] not in config.genome_config.output_keys
    assert set(config.genome_config.output_keys) <= set(genome.nodes)


def test_add_and_delete_change_letter_count():
    config = _config(WordGenome)
    genome_config = config.genome_config
    genome_config.initial_word_length = 1
    genome_config.max_word_length = 3
    genome = WordGenome(1)
    genome.configure_new(genome_config)
    genome._add_letter_output(genome_config)
    assert len(genome.letter_output_keys) == 2
    genome._add_letter_output(genome_config)
    genome._add_letter_output(genome_config)
    assert len(genome.letter_output_keys) == 3
    genome._delete_letter_at(genome_config, 0)
    assert len(genome.letter_output_keys) == 2
    assert all(key in genome.nodes for key in genome.letter_output_keys)


def test_mutate_add_and_delete_respect_probabilities():
    config = _config(WordGenome)
    genome_config = config.genome_config
    genome_config.node_add_prob = 0.0
    genome_config.node_delete_prob = 0.0
    genome_config.conn_add_prob = 0.0
    genome_config.conn_delete_prob = 0.0
    genome_config.initial_word_length = 2
    genome_config.output_add_prob = 1.0
    genome_config.output_delete_prob = 0.0
    genome = WordGenome(1)
    genome.configure_new(genome_config)
    genome.mutate(genome_config)
    assert len(genome.letter_output_keys) == 3

    genome_config.output_add_prob = 0.0
    genome_config.output_delete_prob = 1.0
    genome.mutate(genome_config)
    assert len(genome.letter_output_keys) == 2


def test_structural_node_delete_does_not_drop_letters():
    config = _config(WordGenome)
    genome_config = config.genome_config
    genome_config.node_add_prob = 0.0
    genome_config.node_delete_prob = 1.0
    genome_config.conn_add_prob = 0.0
    genome_config.conn_delete_prob = 0.0
    genome_config.output_add_prob = 0.0
    genome_config.output_delete_prob = 0.0
    genome = WordGenome(1)
    genome.configure_new(genome_config)
    before = list(genome.letter_output_keys)
    for _ in range(10):
        genome.mutate(genome_config)
    assert genome.letter_output_keys == before


def test_crossover_keeps_fitter_parent_letter_order():
    config = _config(WordGenome)
    genome_config = config.genome_config
    genome_config.initial_word_length = 0
    fitter = WordGenome(1)
    fitter.configure_new(genome_config)
    fitter._add_letter_output(genome_config)
    fitter._add_letter_output(genome_config)
    fitter.fitness = 1.0

    other = WordGenome(2)
    other.configure_new(genome_config)
    other.fitness = 0.0
    shared = fitter.letter_output_keys[0]
    other.nodes[shared] = fitter.nodes[shared].copy()
    other._add_letter_output(genome_config)
    other.letter_output_keys = [shared, other.letter_output_keys[-1]]

    child = WordGenome(3)
    child.configure_crossover(other, fitter, genome_config, fitness_criterion="max")
    assert child.letter_output_keys == list(fitter.letter_output_keys)
    assert all(key in child.nodes for key in child.letter_output_keys)


def test_evaluate_fitness_is_negative_distance():
    outputs = [0.0, *([0.5] * NUM_META_OUTPUTS)]
    net = _FakeNet(outputs)
    fitness, meta, word = evaluate_guess_word(
        net, update_meta=True, n_letters=1, answer="a"
    )
    assert fitness == 0.0
    assert word == "a"
    assert meta is not None and len(meta) == NUM_META_OUTPUTS
    fitness, _, word = evaluate_guess_word(
        net, update_meta=False, n_letters=1, answer="neat"
    )
    assert word == "a"
    assert fitness == -float(edit_distance("a", "neat"))
    assert len(net.last_inputs) == 8


def test_real_network_fitness_matches_decoded_word():
    config = _config(WordGenome)
    genome = WordGenome(1)
    genome.configure_new(config.genome_config)
    for key in genome.letter_output_keys:
        genome.nodes[key].bias = -20.0
    net, n_letters = network_for_genome(genome, config)
    fitness, _, word = evaluate_guess_word(
        net, update_meta=False, n_letters=n_letters, answer="a"
    )
    assert n_letters == 1
    assert fitness == 0.0
    assert word == "a"
    fitness, _, word = evaluate_guess_word(
        net, update_meta=False, n_letters=n_letters, answer="ab"
    )
    assert word == "a"
    assert fitness == -1.0


def test_registry_and_threshold():
    task = get_task("guess-word")
    assert fitness_threshold() == FITNESS_THRESHOLD == 0.0
    assert task.default_config_path().name == "guess-word-endogenous.cfg"
    assert task.genome_type_for_mode("standard") is WordGenome
    assert task.genome_type_for_mode("meta-theta") is WordEndogenousGenome


def test_meta_eval_sets_fitness_and_theta():
    config = _config(WordEndogenousGenome)
    genome = WordEndogenousGenome(1)
    genome.configure_new(config.genome_config)
    make_eval_genomes("meta-theta", answer="neat")([(1, genome)], config)
    assert genome.fitness <= 0.0
    assert len(genome.output_word) == len(genome.letter_output_keys)
    assert len(genome.theta.to_vector()) == 8


def test_answer_longer_than_max_raises():
    config = _config(WordGenome)
    genome = WordGenome(1)
    genome.configure_new(config.genome_config)
    eval_fn = make_eval_genomes("standard", answer="a" * 13)
    with pytest.raises(ValueError, match="max_word_length"):
        eval_fn([(1, genome)], config)


def test_snapshot_labels_letters_in_word_order():
    from endogenous_neat.structure import snapshot

    config = _config(WordGenome)
    genome = WordGenome(1)
    genome.configure_new(config.genome_config)
    genome._add_letter_output(config.genome_config)
    genome.output_word = "qz"
    _row, graph = snapshot(genome, config, None, 3)
    letters = sorted(
        (node for node in graph["nodes"] if node["role"] == "letter"),
        key=lambda node: node["order"],
    )
    assert [node["label"] for node in letters] == ["q", "z"]
    assert [node["key"] for node in letters] == list(genome.letter_output_keys)
    assert graph["word"] == "qz"


def test_one_generation_runs():
    config = _config(WordGenome)
    population = neat.Population(config)
    population.run(make_eval_genomes("standard", answer="neat"), 1)
