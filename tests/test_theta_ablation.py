"""Single-component meta-theta ablation: one live field, seven pinned."""

from __future__ import annotations

from pathlib import Path

import neat
import pytest
from neat.innovation import InnovationTracker

from endogenous_neat.experiment import ablation_arms
from endogenous_neat.genome import EndogenousGenome, apply_theta_to_config
from endogenous_neat.reproduction import EndogenousReproduction
from endogenous_neat.tasks.base import apply_meta_theta
from endogenous_neat.theta import THETA_FIELD_NAMES, Theta
from endogenous_neat.word_genome import WordEndogenousGenome

FIXTURES = Path(__file__).parent / "fixtures"
FREE = "bias_mutate_rate"


def _upper() -> Theta:
    return Theta.from_dict({name: Theta.BOUNDS[name][1] for name in THETA_FIELD_NAMES})


def _assert_only_free(theta: Theta, free: str, expected_free: float) -> None:
    assert getattr(theta, free) == pytest.approx(expected_free)
    for name in THETA_FIELD_NAMES:
        if name == free:
            continue
        assert getattr(theta, name) == pytest.approx(Theta.DEFAULTS[name])


def test_with_only_pins_other_fields_at_defaults():
    pinned = _upper().with_only((FREE,))
    _assert_only_free(pinned, FREE, Theta.BOUNDS[FREE][1])


def test_with_only_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown theta fields"):
        Theta.default().with_only(("not_a_field",))


def test_ablation_arms_cover_the_control_and_each_field():
    arms = ablation_arms()
    assert arms[0] == ("meta-theta", THETA_FIELD_NAMES)
    assert [suffix for suffix, _fields in arms[1:]] == [
        f"meta-theta-{name}" for name in THETA_FIELD_NAMES
    ]
    assert [fields for _suffix, fields in arms[1:]] == [
        (name,) for name in THETA_FIELD_NAMES
    ]


def test_apply_meta_theta_without_mask_replaces_all_eight():
    genome = EndogenousGenome(1)
    genome.theta = Theta.default()
    apply_meta_theta(genome, [1.0] * 8)
    for name in THETA_FIELD_NAMES:
        assert getattr(genome.theta, name) == pytest.approx(Theta.BOUNDS[name][1])


def test_apply_meta_theta_unset_attribute_replaces_all_eight(neat_config):
    genome = EndogenousGenome(2)
    genome.theta = Theta.default()
    neat_config.genome_config.theta_free_fields = None
    apply_meta_theta(genome, [1.0] * 8, neat_config)
    for name in THETA_FIELD_NAMES:
        assert getattr(genome.theta, name) == pytest.approx(Theta.BOUNDS[name][1])


def test_apply_meta_theta_mask_keeps_one_field(neat_config):
    genome = EndogenousGenome(3)
    genome.theta = Theta.default()
    neat_config.genome_config.theta_free_fields = (FREE,)
    apply_meta_theta(genome, [1.0] * 8, neat_config)
    _assert_only_free(genome.theta, FREE, Theta.BOUNDS[FREE][1])


def test_full_free_set_keeps_every_meta_field(neat_config):
    genome = EndogenousGenome(4)
    genome.theta = Theta.default()
    neat_config.genome_config.theta_free_fields = THETA_FIELD_NAMES
    apply_meta_theta(genome, [1.0] * 8, neat_config)
    for name in THETA_FIELD_NAMES:
        assert getattr(genome.theta, name) == pytest.approx(Theta.BOUNDS[name][1])


def test_configure_new_pins_frozen_fields(genome_config):
    genome_config.theta_free_fields = (FREE,)
    genome_config.theta_init_spread = 1.0
    genome = EndogenousGenome(5)
    genome.configure_new(genome_config)
    for name in THETA_FIELD_NAMES:
        if name == FREE:
            continue
        assert getattr(genome.theta, name) == pytest.approx(Theta.DEFAULTS[name])


def test_crossover_pins_frozen_fields(genome_config):
    genome_config.theta_free_fields = ("conn_add_prob",)
    g1 = EndogenousGenome(6)
    g2 = EndogenousGenome(7)
    g1.configure_new(genome_config)
    g2.configure_new(genome_config)
    g1.theta = _upper()
    g2.theta = Theta.from_dict({**Theta.DEFAULTS, "conn_add_prob": 0.05})
    g1.fitness = 1.0
    g2.fitness = 0.0
    child = EndogenousGenome(8)
    child.configure_crossover(g1, g2, genome_config)
    assert child.theta.conn_add_prob in (
        pytest.approx(Theta.BOUNDS["conn_add_prob"][1]),
        pytest.approx(0.05),
    )
    for name in THETA_FIELD_NAMES:
        if name == "conn_add_prob":
            continue
        assert getattr(child.theta, name) == pytest.approx(Theta.DEFAULTS[name])


def test_mutate_pins_frozen_fields_and_uses_them_as_rates(genome_config, monkeypatch):
    genome_config.theta_free_fields = (FREE,)
    genome_config.theta_mutate_rate = 0.0
    genome = EndogenousGenome(9)
    genome.configure_new(genome_config)
    genome.theta = _upper()
    seen = {}
    real = apply_theta_to_config

    def spy(config, theta):
        seen["theta"] = theta
        return real(config, theta)

    monkeypatch.setattr("endogenous_neat.genome.apply_theta_to_config", spy)
    genome.mutate(genome_config)
    _assert_only_free(seen["theta"], FREE, Theta.BOUNDS[FREE][1])
    _assert_only_free(genome.theta, FREE, Theta.BOUNDS[FREE][1])


def test_word_mutate_pins_frozen_fields():
    config = neat.Config(
        WordEndogenousGenome,
        EndogenousReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(FIXTURES / "guess_word_minimal.cfg"),
    )
    genome_config = config.genome_config
    genome_config.innovation_tracker = InnovationTracker()
    genome_config.theta_free_fields = (FREE,)
    genome_config.theta_mutate_rate = 0.0
    genome = WordEndogenousGenome(10)
    genome.configure_new(genome_config)
    genome.theta = _upper()
    genome.mutate(genome_config)
    _assert_only_free(genome.theta, FREE, Theta.BOUNDS[FREE][1])
