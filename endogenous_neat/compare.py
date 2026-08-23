"""Compare standard NEAT vs weak endogenous NEAT modes."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from endogenous_neat.theta import THETA_FIELD_NAMES, Theta

COMPARE_MODES = ("standard", "inherited-theta", "meta-theta")
FITNESS_THRESHOLD = 3.9


def load_fitness_csv(path: Path) -> dict[str, list[float]]:
    """Load generation, best_fitness, mean_fitness columns from a run CSV."""
    generations: list[float] = []
    best: list[float] = []
    mean: list[float] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            generations.append(float(row["generation"]))
            best.append(float(row["best_fitness"]))
            mean.append(float(row["mean_fitness"]))
    return {"generation": generations, "best_fitness": best, "mean_fitness": mean}


def load_theta_best_csv(path: Path) -> dict[str, list[float]]:
    """Load generation and theta columns from theta_best.csv."""
    columns: dict[str, list[float]] = defaultdict(list)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            columns["generation"].append(float(row["generation"]))
            for name in THETA_FIELD_NAMES:
                if name in row:
                    columns[name].append(float(row[name]))
    return dict(columns)


def discover_runs(runs_root: Path) -> dict[str, list[Path]]:
    """Group run directories under runs_root by mode suffix."""
    by_mode: dict[str, list[Path]] = {mode: [] for mode in COMPARE_MODES}
    if not runs_root.exists():
        return by_mode

    for path in sorted(runs_root.iterdir()):
        if not path.is_dir():
            continue
        for mode in COMPARE_MODES:
            if path.name.endswith(f"_{mode}"):
                if (path / "fitness.csv").exists():
                    by_mode[mode].append(path)
                break
    return by_mode


def aggregate_fitness(
    run_dirs: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate best fitness across runs: gens, median, q25, q75."""
    series = []
    for run_dir in run_dirs:
        data = load_fitness_csv(run_dir / "fitness.csv")
        series.append(data["best_fitness"])

    if not series:
        empty = np.array([])
        return empty, empty, empty, empty

    min_len = min(len(s) for s in series)
    arr = np.array([s[:min_len] for s in series], dtype=float)
    generations = np.arange(min_len, dtype=float)
    median = np.median(arr, axis=0)
    q25 = np.percentile(arr, 25, axis=0)
    q75 = np.percentile(arr, 75, axis=0)
    return generations, median, q25, q75


def final_theta_drift_l1(run_dir: Path) -> float | None:
    """L1 distance of final best-genome theta from defaults."""
    theta_path = run_dir / "theta_best.csv"
    if not theta_path.exists():
        return None
    rows = list(csv.DictReader(theta_path.open()))
    if not rows:
        return None
    final = Theta.from_dict(
        {name: float(rows[-1][name]) for name in THETA_FIELD_NAMES}
    )
    return final.l1_distance_from_defaults()


def mode_stats(run_dirs: list[Path], fitness_threshold: float = FITNESS_THRESHOLD) -> dict:
    """Compute summary statistics for one mode across runs."""
    final_best: list[float] = []
    solved = 0
    first_solve_gens: list[int] = []

    for run_dir in run_dirs:
        data = load_fitness_csv(run_dir / "fitness.csv")
        if not data["best_fitness"]:
            continue
        best_curve = data["best_fitness"]
        final = max(best_curve)
        final_best.append(final)
        if final >= fitness_threshold:
            solved += 1
            for gen, value in enumerate(best_curve):
                if value >= fitness_threshold:
                    first_solve_gens.append(gen)
                    break

    n = len(final_best)
    drifts = [d for d in (final_theta_drift_l1(run_dir) for run_dir in run_dirs) if d is not None]
    material_drift_runs = sum(1 for d in drifts if d > 0.05)
    return {
        "n_runs": n,
        "solved": solved,
        "solve_rate": (solved / n) if n else 0.0,
        "final_best_mean": float(np.mean(final_best)) if final_best else float("nan"),
        "final_best_std": float(np.std(final_best)) if final_best else float("nan"),
        "final_best_median": float(np.median(final_best)) if final_best else float("nan"),
        "median_first_solve_gen": (
            float(np.median(first_solve_gens)) if first_solve_gens else float("nan")
        ),
        "theta_drift_l1_mean": float(np.mean(drifts)) if drifts else float("nan"),
        "theta_drift_l1_median": float(np.median(drifts)) if drifts else float("nan"),
        "theta_material_drift_runs": material_drift_runs,
    }


def write_comparison_plots(
    by_mode: dict[str, list[Path]],
    output_dir: Path,
    fitness_threshold: float = FITNESS_THRESHOLD,
) -> None:
    """Write overlay fitness and theta comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, run_dirs in by_mode.items():
        if not run_dirs:
            continue
        generations, median, q25, q75 = aggregate_fitness(run_dirs)
        if len(generations) == 0:
            continue
        ax.plot(generations, median, label=f"{mode} (median, n={len(run_dirs)})")
        ax.fill_between(generations, q25, q75, alpha=0.2)

    ax.axhline(
        fitness_threshold, color="gray", linestyle="--", linewidth=1, label="threshold"
    )
    ax.set_xlabel("generation")
    ax.set_ylabel("best fitness")
    ax.set_title("Mode comparison: best fitness (median ± IQR)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "compare_fitness.png", dpi=120)
    plt.close(fig)

    endogenous = {
        mode: dirs
        for mode, dirs in by_mode.items()
        if mode != "standard" and dirs
    }
    if not endogenous:
        return

    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    axes_flat = axes.flatten()
    for i, name in enumerate(THETA_FIELD_NAMES):
        ax = axes_flat[i]
        for mode, run_dirs in endogenous.items():
            series = []
            for run_dir in run_dirs:
                theta_path = run_dir / "theta_best.csv"
                if not theta_path.exists():
                    continue
                data = load_theta_best_csv(theta_path)
                if name not in data:
                    continue
                series.append(data[name])
            if not series:
                continue
            min_len = min(len(s) for s in series)
            arr = np.array([s[:min_len] for s in series], dtype=float)
            gens = np.arange(min_len, dtype=float)
            ax.plot(gens, np.median(arr, axis=0), label=mode)
            ax.fill_between(
                gens,
                np.percentile(arr, 25, axis=0),
                np.percentile(arr, 75, axis=0),
                alpha=0.2,
            )
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Best-genome theta by mode (median ± IQR)")
    fig.tight_layout()
    fig.savefig(output_dir / "compare_theta.png", dpi=120)
    plt.close(fig)


def write_comparison_summary(
    by_mode: dict[str, list[Path]],
    output_dir: Path,
    fitness_threshold: float = FITNESS_THRESHOLD,
) -> Path:
    """Write comparison summary.txt and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Endogenous NEAT mode comparison",
        "===============================",
        f"fitness threshold: {fitness_threshold}",
        "",
    ]

    for mode in COMPARE_MODES:
        run_dirs = by_mode.get(mode, [])
        stats = mode_stats(run_dirs, fitness_threshold=fitness_threshold)
        lines.extend(
            [
                f"mode: {mode}",
                f"  runs: {stats['n_runs']}",
                f"  solved: {stats['solved']} ({100 * stats['solve_rate']:.1f}%)",
                f"  final best fitness mean±std: "
                f"{stats['final_best_mean']:.6f} ± {stats['final_best_std']:.6f}",
                f"  final best fitness median: {stats['final_best_median']:.6f}",
                f"  median first-solve generation: {stats['median_first_solve_gen']}",
            ]
        )
        if mode != "standard" and stats["n_runs"]:
            lines.extend(
                [
                    f"  theta drift L1 (mean / median): "
                    f"{stats['theta_drift_l1_mean']:.4f} / {stats['theta_drift_l1_median']:.4f}",
                    f"  runs with material theta drift (>0.05 L1): "
                    f"{stats['theta_material_drift_runs']}/{stats['n_runs']}",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "Interpretation",
            "--------------",
            "standard vs inherited-theta: effect of per-genome theta vs fixed global λ",
            "inherited-theta vs meta-theta: effect of network meta-output theta updates",
            "standard vs meta-theta: overall weak endogenous NEAT vs classical NEAT",
            "",
            "All modes use the same matched-capacity config "
            "(task outputs + 8 meta outputs).",
        ]
    )

    summary_path = output_dir / "compare_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def write_comparison_csv(
    by_mode: dict[str, list[Path]],
    output_dir: Path,
    fitness_threshold: float = FITNESS_THRESHOLD,
) -> Path:
    """Write per-mode aggregate stats to compare_stats.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "compare_stats.csv"
    fieldnames = [
        "mode",
        "n_runs",
        "solved",
        "solve_rate",
        "final_best_mean",
        "final_best_std",
        "final_best_median",
        "median_first_solve_gen",
        "theta_drift_l1_mean",
        "theta_drift_l1_median",
        "theta_material_drift_runs",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mode in COMPARE_MODES:
            stats = mode_stats(by_mode.get(mode, []), fitness_threshold=fitness_threshold)
            writer.writerow({"mode": mode, **stats})
    return path


def compare_runs(
    runs_root: Path,
    output_dir: Path | None = None,
    fitness_threshold: float = FITNESS_THRESHOLD,
) -> Path:
    """Discover runs under runs_root, write comparison artifacts, return output dir."""
    by_mode = discover_runs(runs_root)
    if output_dir is None:
        output_dir = runs_root / "comparison"

    write_comparison_plots(
        by_mode, output_dir, fitness_threshold=fitness_threshold
    )
    write_comparison_summary(by_mode, output_dir, fitness_threshold=fitness_threshold)
    write_comparison_csv(by_mode, output_dir, fitness_threshold=fitness_threshold)
    return output_dir
