"""Tests for experiment logging of theta values."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import neat
from neat.innovation import InnovationTracker

from endogenous_neat.genome import (
    EndogenousGenome,
    apply_theta_to_config,
    theta_from_genome_config,
)
from endogenous_neat.logging_utils import ThetaReporter
from endogenous_neat.structure import draw_network, load_structure_jsonl
from endogenous_neat.reproduction import EndogenousReproduction
from endogenous_neat.theta import THETA_FIELD_NAMES, Theta

FIXTURES = Path(__file__).parent / "fixtures"


def _minimal_config():
    config = neat.Config(
        EndogenousGenome,
        EndogenousReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(FIXTURES / "minimal.cfg"),
    )
    config.genome_config.innovation_tracker = InnovationTracker()
    return config


def test_theta_from_genome_config_round_trip():
    config = _minimal_config().genome_config
    theta = Theta.from_dict({**Theta.DEFAULTS, "weight_mutate_power": 1.25})
    local = apply_theta_to_config(config, theta)
    restored = theta_from_genome_config(local)
    assert restored.to_vector() == theta.to_vector()


def test_reporter_logs_endogenous_theta(tmp_path):
    config = _minimal_config()
    genome = EndogenousGenome(1)
    genome.configure_new(config.genome_config)
    genome.fitness = 1.0
    genome.theta = Theta.from_dict({**Theta.DEFAULTS, "conn_add_prob": 0.33})

    reporter = ThetaReporter()
    reporter.post_evaluate(config, {1: genome}, {}, genome)

    reporter.write_csv(tmp_path)
    reporter.write_summary(
        tmp_path, mode="meta-theta", generations=1, fitness_threshold=3.9
    )

    rows = list(csv.DictReader((tmp_path / "theta_best.csv").open()))
    assert len(rows) == 1
    assert float(rows[0]["conn_add_prob"]) == 0.33

    summary = (tmp_path / "summary.txt").read_text()
    assert "Final best genome theta:" in summary
    assert "conn_add_prob: 0.330000" in summary


def test_reporter_logs_fixed_theta_for_standard(tmp_path):
    fixed = Theta.from_dict({**Theta.DEFAULTS, "weight_mutate_power": 2.0})
    reporter = ThetaReporter(fixed_theta=fixed)

    genome = neat.DefaultGenome(1)
    genome.fitness = 1.0
    reporter.post_evaluate(None, {1: genome}, {}, genome)
    reporter.post_evaluate(None, {1: genome}, {}, genome)

    reporter.write_csv(tmp_path)
    reporter.write_summary(
        tmp_path, mode="standard", generations=2, fitness_threshold=24.5, task="sum-squares"
    )

    best_rows = list(csv.DictReader((tmp_path / "theta_best.csv").open()))
    assert len(best_rows) == 2
    assert float(best_rows[0]["weight_mutate_power"]) == 2.0
    assert float(best_rows[1]["weight_mutate_power"]) == 2.0

    pop_rows = list(csv.DictReader((tmp_path / "theta_population.csv").open()))
    assert len(pop_rows) == 2
    assert float(pop_rows[0]["weight_mutate_power_mean"]) == 2.0
    assert float(pop_rows[0]["weight_mutate_power_std"]) == 0.0

    summary = (tmp_path / "summary.txt").read_text()
    assert "Fixed global lambda (standard mode):" in summary
    assert "weight_mutate_power: 2.000000" in summary

    assert reporter.effective_theta().weight_mutate_power == 2.0


def test_reporter_writes_structure_snapshot(tmp_path):
    config = _minimal_config()
    genome = EndogenousGenome(1)
    genome.configure_new(config.genome_config)
    genome.fitness = 1.5
    genome.output_word = "ab"

    class _Species:
        species = {1: None, 2: None}

    reporter = ThetaReporter()
    reporter.post_evaluate(config, {1: genome}, _Species(), genome)
    reporter.write_csv(tmp_path)

    rows = list(csv.DictReader((tmp_path / "structure.csv").open()))
    assert len(rows) == 1
    assert int(rows[0]["n_nodes"]) == len(genome.nodes)
    assert int(rows[0]["n_enabled"]) == sum(
        1 for conn in genome.connections.values() if conn.enabled
    )
    assert int(rows[0]["n_species"]) == 2
    assert int(rows[0]["n_letters"]) == 0

    graphs = load_structure_jsonl(tmp_path / "structure_best.jsonl")
    assert len(graphs) == 1
    gene_nodes = [node for node in graphs[0]["nodes"] if node["role"] != "input"]
    assert len(gene_nodes) == len(genome.nodes)
    assert len(graphs[0]["connections"]) == len(genome.connections)
    assert json.loads((tmp_path / "structure_best.jsonl").read_text().splitlines()[0])["generation"] == 0

    assert graphs[0]["word"] == "ab"
    word_rows = list(csv.DictReader((tmp_path / "words.csv").open()))
    assert word_rows == [{"generation": "0", "word": "ab", "score": "1.5"}]

    figure = draw_network(graphs[0])
    assert figure.axes
    import matplotlib.pyplot as plt

    plt.close(figure)
