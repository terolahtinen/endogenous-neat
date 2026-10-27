"""Dynamic NK additions to the mode comparison: change markers and recovery."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RECOVERY_FRACTION = 0.95
CHANGE_EVENTS = ("k_change", "landscape_perturbation")


def load_events(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "nk_events.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def load_nk_rows(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "nk.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def change_events(events: list[dict]) -> list[dict]:
    return [event for event in events if event.get("event") in CHANGE_EVENTS]


def change_label(event: dict) -> str:
    if event["event"] == "k_change":
        return f"K {event['old_k']}->{event['new_k']}"
    return f"perturb {round(100 * float(event['fraction']))}%"


def event_markers(by_mode: dict[str, list[Path]]) -> list[tuple[int, str]]:
    """``(generation, label)`` for each change, from the first run that has events.

    All arms of one comparison share the same schedule, so one run suffices.
    """
    for run_dirs in by_mode.values():
        for run_dir in run_dirs:
            events = change_events(load_events(run_dir))
            if events:
                return [(int(event["generation"]), change_label(event)) for event in events]
    return []


def draw_markers(ax, markers: list[tuple[int, str]], *, label_axis: bool = True) -> None:
    for generation, label in markers:
        ax.axvline(generation, color="gray", linestyle=":", linewidth=1)
        if label_axis:
            ax.text(
                generation,
                ax.get_ylim()[1],
                label,
                rotation=90,
                va="top",
                ha="right",
                fontsize=6,
                color="gray",
            )


def recovery_for_change(
    best_curve: list[float], event: dict, fraction: float = RECOVERY_FRACTION
) -> dict | None:
    """Drop and recovery for one change on one run's best-fitness curve.

    ``pre_change_best`` is the best fitness in the generation before the
    change and ``post_change_best`` the best in the change generation.
    ``recovery_generations_95`` counts generations from the change until best
    fitness is at least ``fraction`` of ``pre_change_best``; it is ``None``
    (censored) when that never happens within the run.
    """
    generation = int(event["generation"])
    if generation <= 0 or generation >= len(best_curve):
        return None
    pre = float(best_curve[generation - 1])
    post = float(best_curve[generation])
    target = fraction * pre
    recovery = None
    for index in range(generation, len(best_curve)):
        if best_curve[index] >= target:
            recovery = index - generation
            break
    return {
        "generation": generation,
        "change": change_label(event),
        "pre_change_best": pre,
        "post_change_best": post,
        "fitness_drop": pre - post,
        "recovery_generations_95": recovery,
        "recovered": recovery is not None,
        "generations_observed": len(best_curve) - generation,
        "same_population_best_drop": event.get("same_population_best_drop"),
    }


def _best_curve(run_dir: Path) -> list[float]:
    path = Path(run_dir) / "fitness.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [float(row["best_fitness"]) for row in csv.DictReader(handle)]


def recovery_rows(by_mode: dict[str, list[Path]]) -> list[dict]:
    rows = []
    for mode, run_dirs in by_mode.items():
        for run_dir in run_dirs:
            events = change_events(load_events(run_dir))
            if not events:
                continue
            curve = _best_curve(run_dir)
            for event in events:
                result = recovery_for_change(curve, event)
                if result is None:
                    continue
                rows.append({"mode": mode, "run": Path(run_dir).name, **result})
    return rows


def recovery_summary(rows: list[dict]) -> list[dict]:
    """Aggregate per (mode, change): mean drop, median recovery, censored count."""
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["mode"], row["generation"], row["change"]), []).append(row)
    summary = []
    for (mode, generation, change), group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        drops = [row["fitness_drop"] for row in group]
        recoveries = [
            row["recovery_generations_95"]
            for row in group
            if row["recovery_generations_95"] is not None
        ]
        same_pop = [
            row["same_population_best_drop"]
            for row in group
            if row["same_population_best_drop"] is not None
        ]
        summary.append(
            {
                "mode": mode,
                "generation": generation,
                "change": change,
                "n_runs": len(group),
                "fitness_drop_mean": float(np.mean(drops)),
                "fitness_drop_std": float(np.std(drops)),
                "same_population_drop_mean": (
                    float(np.mean(same_pop)) if same_pop else float("nan")
                ),
                "recovered": len(recoveries),
                "censored": len(group) - len(recoveries),
                "recovery_generations_95_median": (
                    float(np.median(recoveries)) if recoveries else float("nan")
                ),
                "recovery_generations_95_mean": (
                    float(np.mean(recoveries)) if recoveries else float("nan")
                ),
            }
        )
    return summary


def write_mean_fitness_plot(
    by_mode: dict[str, list[Path]], output_dir: Path, markers: list[tuple[int, str]]
) -> Path | None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for mode, run_dirs in by_mode.items():
        series = []
        for run_dir in run_dirs:
            path = Path(run_dir) / "fitness.csv"
            if not path.exists():
                continue
            with path.open(newline="") as handle:
                series.append([float(row["mean_fitness"]) for row in csv.DictReader(handle)])
        if not series:
            continue
        length = min(len(values) for values in series)
        arr = np.array([values[:length] for values in series], dtype=float)
        gens = np.arange(length)
        ax.plot(gens, np.median(arr, axis=0), label=f"{mode} (median, n={len(series)})")
        ax.fill_between(
            gens, np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0), alpha=0.2
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    draw_markers(ax, markers)
    ax.set_xlabel("generation")
    ax.set_ylabel("mean population fitness")
    ax.set_title("Dynamic NK: mean population fitness (median ± IQR)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "compare_mean_fitness.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_k_plot(by_mode: dict[str, list[Path]], output_dir: Path) -> Path | None:
    """K in force per generation, from the first run with nk.csv."""
    for run_dirs in by_mode.values():
        for run_dir in run_dirs:
            rows = load_nk_rows(run_dir)
            if rows:
                fig, ax = plt.subplots(figsize=(9, 2.5))
                gens = [int(row["generation"]) for row in rows]
                ax.step(gens, [int(row["k"]) for row in rows], where="post", label="K")
                ax.step(
                    gens,
                    [int(row["landscape_version"]) for row in rows],
                    where="post",
                    label="landscape version",
                )
                ax.set_xlabel("generation")
                ax.legend(fontsize=8)
                fig.tight_layout()
                path = output_dir / "compare_nk_schedule.png"
                fig.savefig(path, dpi=120)
                plt.close(fig)
                return path
    return None


def write_nk_report(by_mode: dict[str, list[Path]], output_dir: Path) -> list[str]:
    """Write recovery tables and NK plots. Returns lines for compare_summary.txt."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markers = event_markers(by_mode)
    rows = recovery_rows(by_mode)
    summary = recovery_summary(rows)

    row_fields = [
        "mode",
        "run",
        "generation",
        "change",
        "pre_change_best",
        "post_change_best",
        "fitness_drop",
        "same_population_best_drop",
        "recovery_generations_95",
        "recovered",
        "generations_observed",
    ]
    with (output_dir / "compare_recovery.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in row_fields})

    summary_fields = [
        "mode",
        "generation",
        "change",
        "n_runs",
        "fitness_drop_mean",
        "fitness_drop_std",
        "same_population_drop_mean",
        "recovered",
        "censored",
        "recovery_generations_95_median",
        "recovery_generations_95_mean",
    ]
    with (output_dir / "compare_recovery_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    write_mean_fitness_plot(by_mode, output_dir, markers)
    write_k_plot(by_mode, output_dir)

    lines = ["Dynamic NK recovery (best fitness back to 95% of pre-change best)", ""]
    if not summary:
        lines.append("  no landscape changes recorded")
        return lines
    lines.append(
        f"  {'mode':<18}{'change':<14}{'gen':>5}{'drop':>9}{'same-pop':>10}"
        f"{'recovered':>11}{'median gens':>13}"
    )
    for entry in summary:
        median = entry["recovery_generations_95_median"]
        median_text = "censored" if np.isnan(median) else f"{median:.0f}"
        same_pop = entry["same_population_drop_mean"]
        same_pop_text = "" if np.isnan(same_pop) else f"{same_pop:.4f}"
        lines.append(
            f"  {entry['mode']:<18}{entry['change']:<14}{entry['generation']:>5}"
            f"{entry['fitness_drop_mean']:>9.4f}{same_pop_text:>10}"
            f"{entry['recovered']:>5}/{entry['n_runs']:<5}{median_text:>13}"
        )
    lines.append("")
    lines.append(
        "  drop = best(gen-1) - best(gen) on the evolving population; same-pop = "
        "previous generation's genomes re-scored on the new landscape."
    )
    lines.append("  censored = never reached 95% of the pre-change best within the run.")
    return lines
