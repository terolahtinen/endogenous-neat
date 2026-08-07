"""Tests for mode comparison helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from endogenous_neat.compare import (
    COMPARE_MODES,
    compare_runs,
    discover_runs,
    mode_stats,
    write_comparison_csv,
)


def _write_fitness(path: Path, best_values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["generation", "best_fitness", "mean_fitness"]
        )
        writer.writeheader()
        for gen, best in enumerate(best_values):
            writer.writerow(
                {
                    "generation": gen,
                    "best_fitness": best,
                    "mean_fitness": best - 0.2,
                }
            )


def _write_theta_best(path: Path, generations: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "generation",
        "node_add_prob",
        "node_delete_prob",
        "conn_add_prob",
        "conn_delete_prob",
        "weight_mutate_rate",
        "weight_mutate_power",
        "bias_mutate_rate",
        "enable_disable_rate",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for gen in range(generations):
            writer.writerow(
                {
                    "generation": gen,
                    "node_add_prob": 0.2 + 0.01 * gen,
                    "node_delete_prob": 0.2,
                    "conn_add_prob": 0.5,
                    "conn_delete_prob": 0.5,
                    "weight_mutate_rate": 0.8,
                    "weight_mutate_power": 0.5,
                    "bias_mutate_rate": 0.7,
                    "enable_disable_rate": 0.01,
                }
            )


def test_discover_runs_groups_by_mode(tmp_path: Path):
    for mode in COMPARE_MODES:
        run_dir = tmp_path / f"20260101_120000_run0_{mode}"
        _write_fitness(run_dir / "fitness.csv", [2.0, 2.5, 3.0])

    by_mode = discover_runs(tmp_path)
    assert set(by_mode) == set(COMPARE_MODES)
    for mode in COMPARE_MODES:
        assert len(by_mode[mode]) == 1


def test_mode_stats_solve_rate(tmp_path: Path):
    solved = tmp_path / "solved"
    unsolved = tmp_path / "unsolved"
    _write_fitness(solved / "fitness.csv", [2.0, 3.0, 3.95])
    _write_fitness(unsolved / "fitness.csv", [2.0, 2.5, 3.0])

    stats = mode_stats([solved, unsolved], fitness_threshold=3.9)
    assert stats["n_runs"] == 2
    assert stats["solved"] == 1
    assert stats["solve_rate"] == 0.5
    assert stats["median_first_solve_gen"] == 2


def test_compare_runs_writes_artifacts(tmp_path: Path):
    for mode in COMPARE_MODES:
        run_dir = tmp_path / f"20260101_120000_run0_{mode}"
        _write_fitness(run_dir / "fitness.csv", [2.0, 2.8, 3.2, 3.5])
        if mode != "standard":
            _write_theta_best(run_dir / "theta_best.csv", generations=4)

    out = compare_runs(tmp_path, output_dir=tmp_path / "comparison")
    assert (out / "compare_fitness.png").exists()
    assert (out / "compare_theta.png").exists()
    assert (out / "compare_summary.txt").exists()
    assert (out / "compare_stats.csv").exists()

    summary = (out / "compare_summary.txt").read_text()
    assert "mode: standard" in summary
    assert "mode: meta-theta" in summary


def test_write_comparison_csv_rows(tmp_path: Path):
    run_dir = tmp_path / "20260101_120000_run0_standard"
    _write_fitness(run_dir / "fitness.csv", [2.0, 3.0])
    by_mode = discover_runs(tmp_path)
    csv_path = write_comparison_csv(by_mode, tmp_path / "out")
    rows = list(csv.DictReader(csv_path.open()))
    assert {row["mode"] for row in rows} == set(COMPARE_MODES)
