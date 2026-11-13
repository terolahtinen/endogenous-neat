"""Publication layout of the three recorded performance curves and the two evolved circuits."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from architecture_figures import circuit_panel


ROOT = Path(__file__).resolve().parent
ARMS = (
    ("standard", "Standard", "#22558b", "-", "o"),
    ("standard-random", "Standard-random", "#bb5a06", "--", "s"),
    ("random-theta", r"Random-$\theta$", "#247961", "-.", "^"),
    ("inherited-theta", r"Inherited-$\theta$", "#813f87", (0, (1.1, 1.2)), "D"),
    ("meta-theta", r"Meta-$\theta$", "#bb3d52", "-", "X"),
)


def main():
    with (ROOT / "results/fitness_by_generation.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    networks = json.loads((ROOT / "data/sensor_networks.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 12.2, "axes.labelcolor": "#19252b", "text.color": "#19252b",
                         "axes.spines.right": False, "axes.spines.top": False, "pdf.fonttype": 42,
                         "axes.edgecolor": "#76838a", "xtick.color": "#38454a", "ytick.color": "#38454a"})
    fig = plt.figure(figsize=(8.65, 6.55))
    chart_axes = [fig.add_axes([.068, bottom, .415, .173]) for bottom in (.752, .467, .181)]
    names = ["(a)  Median best-so-far fitness", "(b)  Runs above 0.80", "(c)  Median population mean fitness"]
    for i, (ax, title) in enumerate(zip(chart_axes, names, strict=True)):
        ax.set_title(title, loc="left", fontweight="semibold", fontsize=14.7, pad=3)
        ax.set_xlim(1, 1000)
        ax.set_xticks([1, 250, 500, 750, 1000], ["1", "250", "500", "750", "1,000"])
        ax.tick_params(labelsize=11.5, length=2.8, width=.65, pad=2)
        ax.grid(axis="y", color="#e0e3e5", lw=.66)
        ax.axvline(100, color="#708088", lw=.85, ls=(0, (1.2, 2.1)), zorder=1)
        if i != 1:
            ax.axhline(.75, color="#74838b", lw=.85, ls=(0, (1.2, 2.1)), zorder=1)
        if i < 2:
            ax.tick_params(labelbottom=False)
    chart_axes[0].set(ylim=(.709,1.012), yticks=(.75,.85,.95,1.0), yticklabels=(".75",".85",".95","1"))
    chart_axes[1].set(ylim=(-.30,10.35), yticks=(0,2,4,6,8,10))
    chart_axes[2].set(ylim=(.49,.835), yticks=(.50,.60,.70,.80), yticklabels=(".50",".60",".70",".80"))
    chart_axes[2].set_xlabel("Generation", labelpad=1, fontsize=12.6)
    generations = np.arange(1, 1001)
    for i, (arm, label, color, style, marker) in enumerate(ARMS):
        best = np.full((10, 1000), np.nan); population = np.full_like(best, np.nan)
        for row in rows:
            if row["arm"] == arm:
                key = int(row["seed"]) - 1, int(row["generation"]) - 1
                best[key] = float(row["best_so_far"])
                population[key] = float(row["population_mean"])
        assert (np.isfinite(best) & np.isfinite(population)).all()
        series = (np.median(best, axis=0), np.sum(best > .80, axis=0), np.median(population, axis=0))
        for j, (ax, y) in enumerate(zip(chart_axes, series, strict=True)):
            ax.plot(generations, y, color=color, ls=style, lw=1.45 if i == 4 else 1.16,
                    marker=marker, ms=3.5, mew=.55, markevery=(23 + 35*i, 239), label=label,
                    drawstyle="steps-post" if j == 1 else "default")
    for y, key, title, meta in [
        (.572, "standard_04_1000", "(d)  Standard, seed 4", False),
        (.181, "meta-theta_02_1000", r"(e)  Meta-$\theta$, seed 2", True),
    ]:
        ax = fig.add_axes([.538, y, .455, .32])
        circuit_panel(ax, networks[key], meta)
        fig.text(.541, y+.326, title, va="bottom", fontsize=14.7, fontweight="semibold")
    fig.add_artist(Line2D([.515,.515],[.161,.939], lw=.58, color="#d1d8db", transform=fig.transFigure))
    fig.legend(*chart_axes[0].get_legend_handles_labels(), ncol=5, loc="lower center",
               bbox_to_anchor=(.510,.011), frameon=False, handlelength=1.7, fontsize=12.0,
               columnspacing=1.0, handletextpad=.34)
    fig.savefig(ROOT / "figures/sensor_case_study.pdf",
                metadata={"Title": "Sensor calibration: search and evolved prediction circuits"})
    plt.close(fig)


if __name__ == "__main__":
    main()
