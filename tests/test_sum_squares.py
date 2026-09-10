"""Tests for sum-of-squares function-fitting task."""

from __future__ import annotations

from pathlib import Path

import neat
import pytest
from neat.innovation import InnovationTracker

from endogenous_neat.genome import EndogenousGenome
from endogenous_neat.reproduction import EndogenousReproduction
from endogenous_neat.tasks import get_task
from endogenous_neat.tasks.base import NUM_META_OUTPUTS
from endogenous_neat.tasks.sum_squares import (
    FITNESS_THRESHOLD,
    NUM_CASES,
    SUM_SQUARES_CASES,
    evaluate_sum_squares,
    fitness_threshold,
    make_eval_genomes,
)
from endogenous_neat.theta import THETA_FIELD_NAMES, Theta

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeNet:
    """Network that returns a fixed output vector, or per-case targets."""

    def __init__(self, outputs: list[float] | None = None, *, perfect: bool = False):
        self.outputs = outputs
        self.perfect = perfect
        self.calls = 0
        self.last_inputs = None
        self._case_index = 0

    def activate(self, inputs):
        self.calls += 1
        self.last_inputs = list(inputs)
        if self.perfect:
            expected = SUM_SQUARES_CASES[self._case_index][1]
            self._case_index += 1
            return [expected] + [0.5] * NUM_META_OUTPUTS
        assert self.outputs is not None
        return list(self.outputs)


def test_grid_has_25_cases_with_scaled_targets():
    assert NUM_CASES == 25
    assert len(SUM_SQUARES_CASES) == 25
    for (x1, x2), t in SUM_SQUARES_CASES:
        assert -1.0 <= x1 <= 1.0
        assert -1.0 <= x2 <= 1.0
        assert t == pytest.approx((x1 * x1 + x2 * x2) / 2.0)
        assert 0.0 <= t <= 1.0


def test_perfect_net_scores_25():
    net = _FakeNet(perfect=True)
    fitness, mean_meta = evaluate_sum_squares(net, update_meta=False)
    assert fitness == pytest.approx(25.0)
    assert mean_meta is None
    assert net.calls == 25


def test_evaluate_meta_average_length_and_theta_inputs():
    meta = [0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9, 0.5]
    net = _FakeNet([0.5, *meta])
    fitness, mean_meta = evaluate_sum_squares(net, update_meta=True)
    assert mean_meta is not None
    assert len(mean_meta) == NUM_META_OUTPUTS
    assert mean_meta == pytest.approx(meta)
    assert len(net.last_inputs) == 10  # 2 task + 8 theta
    assert fitness < 25.0
    theta = Theta.from_meta_outputs(mean_meta)
    for name in THETA_FIELD_NAMES:
        lo, hi = Theta.BOUNDS[name]
        assert lo <= getattr(theta, name) <= hi


def test_evaluate_feeds_custom_theta_as_inputs():
    net = _FakeNet([0.5] + [0.0] * 8)
    custom = Theta.from_dict({**Theta.DEFAULTS, "weight_mutate_power": 5.0})
    evaluate_sum_squares(net, update_meta=False, theta=custom)
    assert len(net.last_inputs) == 10
    # weight_mutate_power is field index 5 among the 8 theta inputs
    assert net.last_inputs[2 + 5] == pytest.approx(1.0)


def test_fitness_threshold_constant():
    assert fitness_threshold() == FITNESS_THRESHOLD == 24.5


def test_registry_accepts_sum_squares():
    task = get_task("sum-squares")
    assert task.fitness_threshold() == 24.5
    assert task.default_config_path().name == "sum-squares-endogenous.cfg"


def test_meta_theta_eval_sets_genome_theta():
    config = neat.Config(
        EndogenousGenome,
        EndogenousReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(FIXTURES / "sum_squares_minimal.cfg"),
    )
    config.genome_config.innovation_tracker = InnovationTracker()

    genome = EndogenousGenome(1)
    genome.configure_new(config.genome_config)

    eval_fn = make_eval_genomes("meta-theta")
    eval_fn([(1, genome)], config)

    assert genome.fitness is not None
    assert 0.0 <= genome.fitness <= 25.0
    assert isinstance(genome.theta, Theta)
    assert len(genome.theta.to_vector()) == 8


def test_standard_eval_sets_fitness():
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(FIXTURES / "sum_squares_minimal.cfg"),
    )
    config.genome_config.innovation_tracker = InnovationTracker()

    genome = neat.DefaultGenome(1)
    genome.configure_new(config.genome_config)

    eval_fn = make_eval_genomes("standard")
    eval_fn([(1, genome)], config)
    assert genome.fitness is not None
    assert 0.0 <= genome.fitness <= 25.0
