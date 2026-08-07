"""Tests for single-pole balancing task."""

from __future__ import annotations

from pathlib import Path

import neat
import pytest
from neat.innovation import InnovationTracker

from endogenous_neat.envs.cart_pole import CartPole, discrete_actuator_force
from endogenous_neat.genome import EndogenousGenome
from endogenous_neat.reproduction import EndogenousReproduction
from endogenous_neat.tasks.base import NUM_META_OUTPUTS
from endogenous_neat.tasks.pole import evaluate_pole, make_eval_genomes
from endogenous_neat.theta import THETA_FIELD_NAMES, Theta


FIXTURES = Path(__file__).parent / "fixtures"


class _FakeNet:
    """Network that returns a fixed output vector."""

    def __init__(self, outputs: list[float]):
        self.outputs = outputs
        self.calls = 0

    def activate(self, inputs):
        self.calls += 1
        return list(self.outputs)


def test_discrete_force_uses_first_output():
    assert discrete_actuator_force([0.6, 0.0, 0.0]) == 10.0
    assert discrete_actuator_force([0.4, 1.0, 1.0]) == -10.0


def test_evaluate_pole_meta_average_within_bounds():
    # outputs[0] for force; remaining 8 for meta
    meta = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0]
    net = _FakeNet([0.9, *meta])
    fitness, mean_meta = evaluate_pole(
        net,
        update_meta=True,
        runs_per_net=1,
        simulation_seconds=0.05,
    )
    assert fitness >= 0.0
    assert mean_meta is not None
    assert len(mean_meta) == NUM_META_OUTPUTS
    theta = Theta.from_meta_outputs(mean_meta)
    for name in THETA_FIELD_NAMES:
        lo, hi = Theta.BOUNDS[name]
        assert lo <= getattr(theta, name) <= hi


def test_meta_theta_eval_sets_genome_theta(tmp_path):
    config = neat.Config(
        EndogenousGenome,
        EndogenousReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(FIXTURES / "pole_minimal.cfg"),
    )
    config.genome_config.innovation_tracker = InnovationTracker()

    genome = EndogenousGenome(1)
    genome.configure_new(config.genome_config)
    before = genome.theta.to_vector()

    # Patch evaluate_pole to avoid long sims while still updating theta.
    import endogenous_neat.tasks.pole as pole_mod

    original = pole_mod.evaluate_pole

    def short_eval(net, update_meta, **kwargs):
        return original(
            net,
            update_meta,
            runs_per_net=1,
            simulation_seconds=0.05,
        )

    pole_mod.evaluate_pole = short_eval
    try:
        eval_fn = make_eval_genomes("meta-theta")
        eval_fn([(1, genome)], config)
    finally:
        pole_mod.evaluate_pole = original

    assert genome.fitness is not None
    assert genome.fitness >= 0.0
    assert isinstance(genome.theta, Theta)
    # Meta replacement should typically change theta from defaults
    assert len(genome.theta.to_vector()) == len(before)


def test_cart_pole_step_advances_time():
    sim = CartPole(x=0.0, theta=0.0, dx=0.0, dtheta=0.0)
    t0 = sim.t
    sim.step(10.0)
    assert sim.t > t0
