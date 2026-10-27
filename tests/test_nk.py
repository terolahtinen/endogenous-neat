"""Tests for NK landscapes, the dynamic schedule, and the dynamic-nk task."""

from __future__ import annotations

import csv
import itertools
import json
import random
from pathlib import Path
from random import Random

import neat
import numpy as np
import pytest
from neat.innovation import InnovationTracker

from endogenous_neat.compare import compare_runs, discover_runs
from endogenous_neat.genome import EndogenousGenome
from endogenous_neat.nk import DynamicNKTask, NKLandscape, parse_schedule, table_size
from endogenous_neat.nk_report import event_markers, recovery_for_change, recovery_rows
from endogenous_neat.reproduction import EndogenousReproduction
from endogenous_neat.tasks import get_task
from endogenous_neat.tasks.base import NUM_META_OUTPUTS
from endogenous_neat.tasks.dynamic_nk import (
    NKSettings,
    decode_bits,
    load_settings,
    make_eval_genomes,
    prepare_config,
)

FIXTURES = Path(__file__).parent / "fixtures"
NK_CFG = FIXTURES / "nk_minimal.cfg"


def _random_bits(n: int, seed: int) -> list[int]:
    rng = Random(seed)
    return [rng.randint(0, 1) for _ in range(n)]


def _config(genome_type=EndogenousGenome, reproduction=EndogenousReproduction):
    config = neat.Config(
        genome_type,
        reproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(NK_CFG),
    )
    config.genome_config.innovation_tracker = InnovationTracker()
    return config


# --- NKLandscape -----------------------------------------------------------


def test_same_seed_same_fitness():
    bits = _random_bits(12, seed=3)
    first = NKLandscape(12, 3, seed=7)
    second = NKLandscape(12, 3, seed=7)
    assert first.evaluate(bits) == second.evaluate(bits)
    assert np.array_equal(first.tables, second.tables)
    assert np.array_equal(first.dependencies, second.dependencies)


def test_different_seeds_differ():
    first = NKLandscape(12, 3, seed=7)
    second = NKLandscape(12, 3, seed=8)
    assert not np.array_equal(first.tables, second.tables)


def test_k_zero_depends_only_on_self():
    landscape = NKLandscape(10, 0, seed=1)
    assert landscape.dependencies.shape == (10, 0)
    assert landscape.tables.shape == (10, 2)
    assert table_size(0) == 2
    # Flipping one bit changes only that locus's contribution.
    bits = [0] * 10
    base = landscape.evaluate(bits)
    flipped = list(bits)
    flipped[4] = 1
    expected = base + (landscape.tables[4][1] - landscape.tables[4][0]) / 10
    assert landscape.evaluate(flipped) == pytest.approx(expected)


def test_k_three_has_sixteen_entries():
    landscape = NKLandscape(10, 3, seed=1)
    assert landscape.tables.shape == (10, 16)
    assert table_size(3) == 16


def test_dependencies_are_valid():
    n, k = 10, 4
    landscape = NKLandscape(n, k, seed=5)
    for i in range(n):
        deps = list(landscape.dependencies[i])
        assert len(deps) == k
        assert i not in deps
        assert len(set(deps)) == k
        assert all(0 <= d < n for d in deps)


def test_k_must_be_below_n():
    with pytest.raises(ValueError):
        NKLandscape(4, 4, seed=1)
    with pytest.raises(ValueError):
        DynamicNKTask(n=4, initial_k=0, k_schedule=((5, 4),))


def test_fitness_bounds():
    landscape = NKLandscape(20, 5, seed=11)
    for seed in range(50):
        value = landscape.evaluate(_random_bits(20, seed))
        assert 0.0 <= value <= 1.0


def test_evaluate_matches_definition():
    n, k = 6, 2
    landscape = NKLandscape(n, k, seed=2)
    bits = [1, 0, 1, 1, 0, 0]
    total = 0.0
    for i in range(n):
        index = bits[i]
        for d in landscape.dependencies[i]:
            index = index * 2 + bits[d]
        total += landscape.tables[i][index]
    assert landscape.evaluate(bits) == pytest.approx(total / n)


def test_exact_optimum_matches_brute_force():
    landscape = NKLandscape(6, 2, seed=9)
    best = max(
        (landscape.evaluate(bits), bits) for bits in itertools.product((0, 1), repeat=6)
    )
    optimum = landscape.exact_optimum(max_n=20)
    assert optimum is not None
    assert optimum[0] == pytest.approx(best[0])
    assert landscape.evaluate(optimum[1]) == pytest.approx(best[0])
    assert NKLandscape(30, 2, seed=1).exact_optimum(max_n=20) is None


def test_perturbation_changes_only_requested_loci():
    n = 100
    landscape = NKLandscape(n, 3, seed=4)
    perturbed, loci = landscape.perturbed(0.10, seed="p")
    assert len(loci) == round(n * 0.10) == 10
    assert perturbed.k == landscape.k
    assert np.array_equal(perturbed.dependencies, landscape.dependencies)
    for i in range(n):
        if i in loci:
            assert not np.array_equal(perturbed.tables[i], landscape.tables[i])
        else:
            assert np.array_equal(perturbed.tables[i], landscape.tables[i])
    again, loci_again = landscape.perturbed(0.10, seed="p")
    assert loci_again == loci
    assert np.array_equal(again.tables, perturbed.tables)


# --- schedule --------------------------------------------------------------


def test_parse_schedule():
    assert parse_schedule("0:0, 100:2 200:4") == ((0, 0), (100, 2), (200, 4))
    assert parse_schedule("150:0.10, 250:0.2", float) == ((150, 0.1), (250, 0.2))
    assert parse_schedule(None) == ()
    with pytest.raises(ValueError):
        parse_schedule("100")


V1_SCHEDULE = ((0, 0), (100, 2), (200, 4), (300, 8), (400, 12))


def test_k_at_boundaries():
    task = DynamicNKTask(n=64, initial_k=0, k_schedule=V1_SCHEDULE)
    assert task.k_at(0) == 0
    assert task.k_at(99) == 0
    assert task.k_at(100) == 2
    assert task.k_at(199) == 2
    assert task.k_at(200) == 4
    assert task.k_at(299) == 4
    assert task.k_at(300) == 8
    assert task.k_at(400) == 12
    assert task.k_at(10_000) == 12


def test_advance_changes_landscape_exactly_at_boundaries():
    task = DynamicNKTask(n=16, initial_k=0, k_schedule=((0, 0), (3, 2), (6, 4)))
    seen = {}
    for generation in range(8):
        events = task.advance(generation)
        seen[generation] = (task.landscape.k, task.version, [e["event"] for e in events])
    assert seen[0] == (0, 0, ["initial_landscape"])
    assert seen[2] == (0, 0, [])
    assert seen[3] == (2, 1, ["k_change"])
    assert seen[5] == (2, 1, [])
    assert seen[6] == (4, 2, ["k_change"])
    assert seen[7] == (4, 2, [])
    changes = [e for e in task.events if e["event"] == "k_change"]
    assert [(e["old_k"], e["new_k"]) for e in changes] == [(0, 2), (2, 4)]


def test_static_task_never_changes():
    task = DynamicNKTask(n=16, initial_k=3)
    assert task.is_dynamic is False
    task.advance(0)
    first = task.landscape
    for generation in range(1, 10):
        assert task.advance(generation) == []
    assert task.landscape is first


def test_perturbation_event_and_k_unchanged():
    task = DynamicNKTask(n=20, initial_k=2, perturbations=((2, 0.25),))
    task.advance(0)
    before = task.landscape
    task.advance(1)
    events = task.advance(2)
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "landscape_perturbation"
    assert event["changed_loci"] == 5
    assert event["landscape_version"] == 1
    assert task.landscape.k == 2
    assert np.array_equal(task.landscape.dependencies, before.dependencies)


def test_advance_requires_consecutive_generations():
    task = DynamicNKTask(n=8)
    task.advance(0)
    with pytest.raises(ValueError):
        task.advance(2)


def test_landscape_rng_is_isolated_from_evolution():
    """Consuming evolutionary randomness must not change the landscapes."""
    schedule = ((0, 0), (2, 2), (4, 3))
    perturbs = ((3, 0.5),)
    quiet = DynamicNKTask(n=12, k_schedule=schedule, perturbations=perturbs, landscape_seed="s")
    noisy = DynamicNKTask(n=12, k_schedule=schedule, perturbations=perturbs, landscape_seed="s")
    random.seed(1)
    evolution_rng = Random(99)
    for generation in range(6):
        quiet.advance(generation)
        for _ in range(generation * 17 + 3):
            random.random()
            evolution_rng.gauss(0, 1)
        noisy.advance(generation)
        assert np.array_equal(quiet.landscape.tables, noisy.landscape.tables)
        assert np.array_equal(quiet.landscape.dependencies, noisy.landscape.dependencies)
    assert quiet.version == noisy.version == 3


# --- task wiring -----------------------------------------------------------


def test_load_settings_and_overrides():
    settings = load_settings(NK_CFG)
    assert settings.n == 8
    assert settings.k_schedule == ((0, 0), (2, 2), (4, 3))
    assert settings.perturbations == ((3, 0.25),)
    static = load_settings(NK_CFG, {"k": 5})
    assert static.initial_k == 5
    assert static.k_schedule == ()
    bigger = load_settings(NK_CFG, {"n": 32, "k_schedule": "0:1, 10:4", "landscape_seed": 3})
    assert bigger.n == 32
    assert bigger.num_outputs == 40
    assert bigger.k_schedule == ((0, 1), (10, 4))
    assert bigger.landscape_seed == 3
    with pytest.raises(ValueError):
        load_settings(NK_CFG, {"k": 8})


def test_prepare_config_sets_output_keys():
    config = _config()
    prepare_config(config, NK_CFG, {"n": 12})
    assert config.genome_config.num_outputs == 12 + NUM_META_OUTPUTS
    assert config.genome_config.output_keys == list(range(20))


def test_decode_bits_threshold():
    assert decode_bits([0.0, 0.49, 0.5, 0.51, 1.0, 0.9], 5) == [0, 0, 1, 1, 1]


def test_registry_and_threshold():
    task = get_task("dynamic-nk")
    assert task.fitness_threshold() == 1.0
    assert task.default_config_path().name == "nk-dynamic-v1.cfg"
    assert task.genome_type_for_mode("standard") is neat.DefaultGenome
    assert task.genome_type_for_mode("meta-theta") is EndogenousGenome


def test_evaluator_rejects_wrong_output_count():
    config = _config()
    evaluator = make_eval_genomes("standard", settings=NKSettings(n=9))
    genome = neat.DefaultGenome(1)
    genome.configure_new(config.genome_config)
    with pytest.raises(ValueError, match="num_outputs"):
        evaluator([(1, genome)], config)


def test_evaluator_logs_and_same_population_response(tmp_path):
    config = _config(neat.DefaultGenome, neat.DefaultReproduction)
    settings = load_settings(NK_CFG)
    evaluator = make_eval_genomes("standard", settings=settings, run_index=0, experiment_seed=1)
    genomes = []
    for key in range(4):
        genome = neat.DefaultGenome(key)
        genome.configure_new(config.genome_config)
        genomes.append((key, genome))

    fitness_by_generation = []
    for _generation in range(5):
        evaluator(genomes, config)
        fitness_by_generation.append([genome.fitness for _k, genome in genomes])
        for _k, genome in genomes:
            assert 0.0 <= genome.fitness <= 1.0

    # Same genomes, landscape unchanged between generations 0 and 1.
    assert fitness_by_generation[0] == fitness_by_generation[1]
    # K changed at generation 2: fitness of the same population changes immediately.
    assert fitness_by_generation[1] != fitness_by_generation[2]

    evaluator.write_logs(tmp_path)
    rows = list(csv.DictReader((tmp_path / "nk.csv").open()))
    assert [int(row["k"]) for row in rows] == [0, 0, 2, 2, 3]
    assert [int(row["landscape_version"]) for row in rows] == [0, 0, 1, 2, 3]
    assert all(row["global_optimum_fitness"] for row in rows)  # n=8 <= 20
    for row in rows:
        assert 0.0 <= float(row["best_fitness"]) <= 1.0
        assert 0.0 <= float(row["mean_fitness"]) <= 1.0
        assert 0.0 <= float(row["median_fitness"]) <= 1.0

    events = [json.loads(line) for line in (tmp_path / "nk_events.jsonl").read_text().splitlines()]
    kinds = [event["event"] for event in events]
    assert kinds == ["initial_landscape", "k_change", "landscape_perturbation", "k_change"]
    change = events[1]
    assert change["generation"] == 2
    assert change["same_population_pre_best"] == pytest.approx(max(fitness_by_generation[1]))
    assert change["same_population_post_best"] == pytest.approx(max(fitness_by_generation[2]))
    assert (tmp_path / "nk_settings.json").exists()


def test_modes_with_same_run_index_share_landscapes():
    settings = load_settings(NK_CFG)
    standard = make_eval_genomes("standard", settings=settings, run_index=2, experiment_seed=5)
    meta = make_eval_genomes("meta-theta", settings=settings, run_index=2, experiment_seed=5)
    other = make_eval_genomes("standard", settings=settings, run_index=3, experiment_seed=5)
    for generation in range(5):
        standard.task.advance(generation)
        meta.task.advance(generation)
        other.task.advance(generation)
        assert np.array_equal(standard.task.landscape.tables, meta.task.landscape.tables)
    assert not np.array_equal(standard.task.landscape.tables, other.task.landscape.tables)


def test_population_survives_landscape_change():
    """The population at the change generation descends from the one before it."""
    config = _config()
    population = neat.Population(config)
    settings = load_settings(NK_CFG)
    evaluator = make_eval_genomes("meta-theta", settings=settings)
    keys_by_generation: list[set[int]] = []

    def eval_and_record(genomes, cfg):
        keys_by_generation.append({key for key, _genome in genomes})
        evaluator(genomes, cfg)

    population.run(eval_and_record, 4)
    # Generation 2 is the first K change in the fixture schedule.
    before, after = keys_by_generation[1], keys_by_generation[2]
    ancestors = population.reproduction.ancestors
    for key in after:
        parents = ancestors.get(key, ())
        assert key in before or all(parent in before for parent in parents)
    assert evaluator.task.landscape.k == 2 or evaluator.task.landscape.k == 3
    for genome in population.population.values():
        assert len(genome.theta.to_vector()) == 8


# --- recovery and comparison report -------------------------------------------


def test_recovery_metric_and_censoring():
    curve = [0.5, 0.6, 0.7, 0.4, 0.5, 0.66, 0.7]
    event = {"generation": 3, "event": "k_change", "old_k": 0, "new_k": 2}
    result = recovery_for_change(curve, event)
    assert result["pre_change_best"] == 0.7
    assert result["post_change_best"] == 0.4
    assert result["fitness_drop"] == pytest.approx(0.3)
    # target 0.665: 0.66 at index 5 is below it, 0.7 at index 6 recovers -> 3 generations
    assert result["recovery_generations_95"] == 3
    censored = recovery_for_change([0.5, 0.7, 0.3, 0.3], event | {"generation": 2})
    assert censored["recovery_generations_95"] is None
    assert censored["recovered"] is False


def _write_run(run_dir: Path, mode: str, best: list[float]) -> None:
    run_dir.mkdir(parents=True)
    with (run_dir / "fitness.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["generation", "best_fitness", "mean_fitness"])
        writer.writeheader()
        for gen, value in enumerate(best):
            writer.writerow({"generation": gen, "best_fitness": value, "mean_fitness": value - 0.1})
    events = [
        {"generation": 0, "event": "initial_landscape", "k": 0, "landscape_version": 0},
        {
            "generation": 2,
            "event": "k_change",
            "old_k": 0,
            "new_k": 2,
            "landscape_version": 1,
            "same_population_best_drop": 0.2,
        },
    ]
    (run_dir / "nk_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    with (run_dir / "nk.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["generation", "k", "landscape_version"])
        writer.writeheader()
        for gen in range(len(best)):
            writer.writerow({"generation": gen, "k": 0 if gen < 2 else 2, "landscape_version": 0 if gen < 2 else 1})


def test_compare_runs_adds_recovery_report(tmp_path):
    _write_run(tmp_path / "20260101_120000_run0_standard", "standard", [0.5, 0.6, 0.4, 0.5, 0.6])
    _write_run(tmp_path / "20260101_120000_run0_meta-theta", "meta-theta", [0.5, 0.6, 0.45, 0.6, 0.6])
    by_mode = discover_runs(tmp_path)
    assert event_markers(by_mode) == [(2, "K 0->2")]
    rows = recovery_rows(by_mode)
    assert {row["mode"] for row in rows} == {"standard", "meta-theta"}

    out = compare_runs(tmp_path, output_dir=tmp_path / "comparison", fitness_threshold=1.0, task_name="dynamic-nk")
    assert (out / "compare_recovery.csv").exists()
    assert (out / "compare_recovery_summary.csv").exists()
    assert (out / "compare_mean_fitness.png").exists()
    assert (out / "compare_nk_schedule.png").exists()
    summary = (out / "compare_summary.txt").read_text()
    assert "Dynamic NK recovery" in summary
    assert "K 0->2" in summary
