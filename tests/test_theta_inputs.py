"""Tests for theta-as-input helpers and XOR evaluation."""

from __future__ import annotations

from pathlib import Path

import neat
from neat.innovation import InnovationTracker

from endogenous_neat.genome import EndogenousGenome
from endogenous_neat.reproduction import EndogenousReproduction
from endogenous_neat.tasks.base import (
    NUM_THETA_INPUTS,
    augment_inputs_with_theta,
    theta_for_genome,
)
from endogenous_neat.tasks.xor import evaluate_xor, make_eval_genomes
from endogenous_neat.theta import Theta


FIXTURES = Path(__file__).parent / "fixtures"


class _FakeNet:
    def __init__(self, outputs: list[float]):
        self.outputs = outputs
        self.last_inputs = None
        self.input_history: list[list[float]] = []

    def activate(self, inputs):
        self.last_inputs = list(inputs)
        self.input_history.append(list(inputs))
        return list(self.outputs)


def test_augment_inputs_appends_normalized_theta():
    theta = Theta.from_dict({**Theta.DEFAULTS, "enable_disable_rate": 0.5})
    out = augment_inputs_with_theta((0.0, 1.0), theta)
    assert len(out) == 2 + NUM_THETA_INPUTS
    assert out[:2] == [0.0, 1.0]
    assert out[-1] == 1.0  # enable_disable at upper bound


def test_theta_for_genome_uses_live_theta():
    genome = EndogenousGenome(1)
    genome.theta = Theta.from_dict({**Theta.DEFAULTS, "conn_add_prob": 0.8})
    assert theta_for_genome(genome).conn_add_prob == 0.8
    assert theta_for_genome(object()).to_vector() == Theta.default().to_vector()


def test_evaluate_xor_feeds_theta_inputs():
    net = _FakeNet([0.5] + [0.0] * 8)
    theta = Theta.from_dict({**Theta.DEFAULTS, "weight_mutate_power": 0.001})
    fitness, _ = evaluate_xor(net, update_meta=False, theta=theta)
    assert fitness is not None
    assert len(net.input_history) == 4
    assert all(len(inp) == 10 for inp in net.input_history)
    # weight_mutate_power at lower bound -> unit 0
    assert net.last_inputs[2 + 5] == 0.0


def test_meta_theta_rewrite_updates_genome_for_next_inputs():
    config = neat.Config(
        EndogenousGenome,
        EndogenousReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(FIXTURES / "minimal.cfg"),
    )
    config.genome_config.innovation_tracker = InnovationTracker()

    genome = EndogenousGenome(1)
    genome.configure_new(config.genome_config)
    before = genome.theta.to_unit_interval_vector()

    eval_fn = make_eval_genomes("meta-theta")
    eval_fn([(1, genome)], config)

    after = genome.theta.to_unit_interval_vector()
    # Inputs for a follow-up eval must reflect the rewritten theta.
    net = _FakeNet([0.5] + [0.0] * 8)
    evaluate_xor(net, update_meta=False, theta=theta_for_genome(genome))
    assert net.last_inputs[2:] == after
    assert after != before or genome.fitness is not None
