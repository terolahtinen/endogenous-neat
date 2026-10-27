"""Compare standard NEAT vs weak endogenous NEAT modes."""

from __future__ import annotations

import csv
from collections import defaultdict
from math import isnan
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from endogenous_neat import nk_report
from endogenous_neat.theta import THETA_FIELD_NAMES, Theta

COMPARE_MODES = ("standard", "inherited-theta", "meta-theta")
# Comparison arms. Classical NEAT is run twice: known-good config λ, and a
# frozen uniform-random λ (directory suffix ``standard-random``).
# ``random-theta`` keeps the meta-theta network and replaces θ with a uniform
# draw after each evaluation.
COMPARE_ARMS = (
    "standard",
    "standard-random",
    "inherited-theta",
    "meta-theta",
    "random-theta",
)
FITNESS_THRESHOLD = 3.9


def is_fixed_lambda(mode: str) -> bool:
    """True for classical NEAT arms, whose λ does not evolve."""
    return mode == "standard" or mode.startswith("standard-")


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
    """Group run directories under runs_root by comparison-arm suffix.

    ``standard-random`` is matched before ``standard`` so a random-λ run is
    not filed with the known-good arm. ``random-theta`` is its own suffix and
    is not filed with ``meta-theta``.
    """
    by_mode: dict[str, list[Path]] = {mode: [] for mode in COMPARE_ARMS}
    if not runs_root.exists():
        return by_mode

    # Longest suffix first so ``_standard-random`` is not claimed by ``_standard``.
    suffixes = sorted(COMPARE_ARMS, key=len, reverse=True)
    for path in sorted(runs_root.iterdir()):
        if not path.is_dir() or not (path / "fitness.csv").exists():
            continue
        for mode in suffixes:
            if path.name.endswith(f"_{mode}"):
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


def load_run_meta(run_dir: Path) -> dict[str, str]:
    """Parse key=value lines from run_meta.txt if present."""
    path = run_dir / "run_meta.txt"
    if not path.exists():
        return {}
    meta: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip()] = value.strip()
    return meta


def initial_theta_from_meta(meta: dict[str, str]) -> Theta | None:
    """Rebuild θ_rand from run_meta theta_initial_* fields when present."""
    try:
        values = {
            name: float(meta[f"theta_initial_{name}"]) for name in THETA_FIELD_NAMES
        }
    except (KeyError, ValueError):
        return None
    return Theta.from_dict(values)


def final_theta_from_run(run_dir: Path) -> Theta | None:
    theta_path = run_dir / "theta_best.csv"
    if not theta_path.exists():
        return None
    rows = list(csv.DictReader(theta_path.open()))
    if not rows:
        return None
    return Theta.from_dict(
        {name: float(rows[-1][name]) for name in THETA_FIELD_NAMES}
    )


def mode_stats(run_dirs: list[Path], fitness_threshold: float = FITNESS_THRESHOLD) -> dict:
    """Compute summary statistics for one mode across runs."""
    final_best: list[float] = []
    solved = 0
    first_solve_gens: list[int] = []
    initial_l1s: list[float] = []
    leave_initial_l1s: list[float] = []

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

        meta = load_run_meta(run_dir)
        if "theta_initial_l1_from_defaults" in meta:
            try:
                initial_l1s.append(float(meta["theta_initial_l1_from_defaults"]))
            except ValueError:
                pass
        initial = initial_theta_from_meta(meta)
        final_theta = final_theta_from_run(run_dir)
        if initial is not None and final_theta is not None:
            leave_initial_l1s.append(final_theta.l1_distance(initial))

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
        "theta_initial_l1_mean": (
            float(np.mean(initial_l1s)) if initial_l1s else float("nan")
        ),
        "theta_leave_initial_l1_mean": (
            float(np.mean(leave_initial_l1s)) if leave_initial_l1s else float("nan")
        ),
        "theta_leave_initial_l1_median": (
            float(np.median(leave_initial_l1s)) if leave_initial_l1s else float("nan")
        ),
    }


def write_comparison_plots(
    by_mode: dict[str, list[Path]],
    output_dir: Path,
    fitness_threshold: float = FITNESS_THRESHOLD,
    task_name: str | None = None,
    markers: list[tuple[int, str]] | None = None,
) -> None:
    """Write overlay fitness and theta comparison plots.

    ``markers`` are ``(generation, label)`` pairs drawn as vertical lines, used
    for dynamic-nk landscape changes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    task_prefix = f"{task_name}: " if task_name else ""
    markers = markers or []

    fig, ax = plt.subplots(figsize=(9, 5))
    for mode in COMPARE_ARMS:
        run_dirs = by_mode.get(mode, [])
        if not run_dirs:
            continue
        generations, median, q25, q75 = aggregate_fitness(run_dirs)
        if len(generations) == 0:
            continue
        ax.plot(generations, median, label=f"{mode} (median, n={len(run_dirs)})")
        ax.fill_between(generations, q25, q75, alpha=0.2)

    if not markers:
        ax.axhline(
            fitness_threshold, color="gray", linestyle="--", linewidth=1, label="threshold"
        )
    nk_report.draw_markers(ax, markers)
    ax.set_xlabel("generation")
    ax.set_ylabel("best fitness")
    ax.set_title(f"{task_prefix}best fitness (median ± IQR)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "compare_fitness.png", dpi=120)
    plt.close(fig)

    endogenous = {
        mode: dirs
        for mode, dirs in by_mode.items()
        if dirs and not is_fixed_lambda(mode)
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
        nk_report.draw_markers(ax, markers, label_axis=False)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0 and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)

    fig.suptitle(f"{task_prefix}best-genome theta by mode (median ± IQR)")
    fig.tight_layout()
    fig.savefig(output_dir / "compare_theta.png", dpi=120)
    plt.close(fig)


def write_comparison_summary(
    by_mode: dict[str, list[Path]],
    output_dir: Path,
    fitness_threshold: float = FITNESS_THRESHOLD,
    task_name: str | None = None,
    extra_lines: list[str] | None = None,
) -> Path:
    """Write comparison summary.txt and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Endogenous NEAT mode comparison",
        "===============================",
        f"task: {task_name or 'unspecified'}",
        f"fitness threshold: {fitness_threshold}",
        "",
    ]
    if extra_lines:
        lines.extend(extra_lines)
        lines.append("")

    for mode in COMPARE_ARMS:
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
        if not is_fixed_lambda(mode) and stats["n_runs"]:
            lines.extend(
                [
                    f"  theta drift L1 (mean / median): "
                    f"{stats['theta_drift_l1_mean']:.4f} / {stats['theta_drift_l1_median']:.4f}",
                    f"  runs with material theta drift (>0.05 L1): "
                    f"{stats['theta_material_drift_runs']}/{stats['n_runs']}",
                ]
            )
        if stats["n_runs"] and not isnan(stats["theta_initial_l1_mean"]):
            lines.append(
                f"  initial θ L1 from defaults (mean): "
                f"{stats['theta_initial_l1_mean']:.4f}"
            )
        if (
            not is_fixed_lambda(mode)
            and stats["n_runs"]
            and not isnan(stats["theta_leave_initial_l1_mean"])
        ):
            lines.extend(
                [
                    f"  final θ L1 from initial θ_rand (mean / median): "
                    f"{stats['theta_leave_initial_l1_mean']:.4f} / "
                    f"{stats['theta_leave_initial_l1_median']:.4f}",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "Interpretation",
            "--------------",
            "standard: classical NEAT frozen at the known-good config λ",
            "standard-random: classical NEAT frozen at a uniform-random λ "
            "(one draw per run)",
            "random-theta: meta-theta network, with θ replaced by a uniform "
            "draw after each evaluation",
            "standard vs standard-random: effect of the hand-tuned λ",
            "standard vs inherited-theta: effect of per-genome theta vs fixed global λ",
            "inherited-theta vs meta-theta: effect of network meta-output theta updates",
            "meta-theta vs random-theta: network-chosen θ updates versus a "
            "uniform redraw of θ after each evaluation",
            "standard vs meta-theta: overall weak endogenous NEAT vs classical NEAT",
            "compare-random-lambda: shared uniform θ_rand; standard freezes λ, "
            "meta may rewrite — does meta recover from a bad random config?",
            "",
            "All modes use the same matched-capacity config for this task.",
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
        "theta_initial_l1_mean",
        "theta_leave_initial_l1_mean",
        "theta_leave_initial_l1_median",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mode in COMPARE_ARMS:
            stats = mode_stats(by_mode.get(mode, []), fitness_threshold=fitness_threshold)
            writer.writerow({"mode": mode, **stats})
    return path


def compare_runs(
    runs_root: Path,
    output_dir: Path | None = None,
    fitness_threshold: float = FITNESS_THRESHOLD,
    task_name: str | None = None,
) -> Path:
    """Discover runs under runs_root, write comparison artifacts, return output dir.

    ``fitness_threshold`` and ``task_name`` should come from the task being
    compared (XOR 3.9, pole 60, sum-squares 24.5, guess-word 0).
    """
    by_mode = discover_runs(runs_root)
    if output_dir is None:
        output_dir = runs_root / "comparison"

    # Dynamic NK runs carry landscape-change events; mark them on every plot
    # and add the recovery table to the summary.
    markers = nk_report.event_markers(by_mode)
    extra_lines = nk_report.write_nk_report(by_mode, output_dir) if markers else None

    write_comparison_plots(
        by_mode,
        output_dir,
        fitness_threshold=fitness_threshold,
        task_name=task_name,
        markers=markers,
    )
    write_comparison_summary(
        by_mode,
        output_dir,
        fitness_threshold=fitness_threshold,
        task_name=task_name,
        extra_lines=extra_lines,
    )
    write_comparison_csv(by_mode, output_dir, fitness_threshold=fitness_threshold)
    return output_dir
