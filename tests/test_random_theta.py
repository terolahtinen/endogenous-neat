"""Random between-generation theta updates, the control for meta-theta."""

from __future__ import annotations

from pathlib import Path
from random import Random

import neat
import pytest
from neat.innovation import InnovationTracker

from endogenous_neat.genome import EndogenousGenome
from endogenous_neat.reproduction import EndogenousReproduction
from endogenous_neat.tasks.base import apply_random_theta, genome_type_for_mode
from endogenous_neat.tasks.dynamic_nk import NKSettings, make_eval_genomes as make_nk_eval
from endogenous_neat.tasks.guess_word import genome_type_for_mode as guess_word_genome_type
from endogenous_neat.tasks.xor import evaluate_xor, make_eval_genomes
from endogenous_neat.theta import THETA_FIELD_NAMES, Theta
from endogenous_neat.word_genome import WordEndogenousGenome

FIXTURES = Path(__file__).parent / "fixtures"


def _within_bounds(theta: Theta) -> None:
    for name in THETA_FIELD_NAMES:
        lo, hi = Theta.BOUNDS[name]
        value = getattr(theta, name)
        assert lo <= value <= hi


def test_apply_random_theta_draws_inside_bounds_and_changes():
    genome = EndogenousGenome(1)
    genome.theta = Theta.default()
    before = genome.theta.to_vector()
    apply_random_theta(genome, Random(0))
    _within_bounds(genome.theta)
    assert genome.theta.to_vector() != before
    again = EndogenousGenome(2)
    again.theta = Theta.default()
    apply_random_theta(again, Random(0))
    assert again.theta.to_vector() == genome.theta.to_vector()


def test_random_theta_uses_the_endogenous_genome():
    assert genome_type_for_mode("random-theta") is EndogenousGenome
    assert guess_word_genome_type("random-theta") is WordEndogenousGenome
    with pytest.raises(ValueError, match="random-theta"):
        genome_type_for_mode("nope")


def test_random_theta_eval_ignores_meta_and_replaces_theta(monkeypatch):
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
    genome.theta = Theta.default()
    replacement = Theta.uniform_random(Random(1))

    seen = {}
    real = evaluate_xor

    def spy(net, update_meta, theta):
        seen["update_meta"] = update_meta
        seen["theta"] = theta.to_vector()
        return real(net, update_meta, theta)

    monkeypatch.setattr("endogenous_neat.tasks.xor.evaluate_xor", spy)
    monkeypatch.setattr(
        Theta,
        "uniform_random",
        classmethod(lambda cls, rng: replacement),
    )

    make_eval_genomes("random-theta")([(1, genome)], config)

    assert seen["update_meta"] is False
    assert seen["theta"] == Theta.default().to_vector()
    assert genome.theta.to_vector() == replacement.to_vector()
    assert genome.fitness is not None


def test_dynamic_nk_accepts_random_theta():
    evaluator = make_nk_eval("random-theta", settings=NKSettings(n=4))
    assert evaluator.mode == "random-theta"
