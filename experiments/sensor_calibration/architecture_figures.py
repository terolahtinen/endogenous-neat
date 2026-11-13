"""Draw the task and the recovered prediction circuits from the archived exports."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as DrawPath
import numpy as np


INK = "#192f3b"
MUTED = "#4b626c"
GRID = "#d5dfe1"
POS = "#08665c"
NEG = "#a34834"
DARK = "#294e66"
LIGHT = "#e9b853"
HILITE = "#963851"
ROOT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def records(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(fig, stem, dest, png=False):
    fig.savefig(dest / (stem + ".pdf"), facecolor="white",
                metadata={"Creator": "Sensor calibration study"})
    if png:
        fig.savefig(dest / (stem + ".png"), dpi=175, facecolor="white")
    plt.close(fig)


def training_data(cases, dest):
    fig, ax = plt.subplots(figsize=(9.25, 2.92))
    fig.subplots_adjust(left=.183, right=.982, top=.74, bottom=.245)
    ax.set_facecolor("white")
    for b in (0, 1):
        ax.axhline(b, color=GRID, lw=1.15, zorder=0)
    for x in (-9 / 16, 9 / 16):
        ax.add_patch(FancyBboxPatch((x - .045, -.22), .09, 1.44,
                     boxstyle="round,pad=0,rounding_size=.027", fc="#f8eaf0",
                     ec="#bd8595", lw=.9, zorder=1))
        ax.text(x, 1.34, ("−" if x < 0 else "+") + "9/16", color=HILITE,
                fontsize=11.3, ha="center", va="bottom", weight="semibold")
    ax.axvline(0, color="#708188", lw=1, ls=(0, (3, 3)), zorder=1)
    for y, marker, color, edge in [(0, "s", DARK, DARK), (1, "o", LIGHT, "#8f6624")]:
        points = [(float(r["x"]), int(r["reference"])) for r in cases if int(r["target"]) == y]
        assert len(points) == 16
        ax.scatter(*zip(*points), s=111 if y else 91, marker=marker,
                   c=color, edgecolors=edge, linewidths=1.0, zorder=3)
    ax.set_ylim(-.40, 1.57)
    ax.set_xlim(-1.02, 1.02)
    ax.set_yticks([0, 1], ["Dark reference\n$b=0$", "Light reference\n$b=1$"])
    ax.set_xticks([-1, -.5, 0, .5, 1], ["−1", "−1/2", "0", "+1/2", "+1"])
    ax.set_xlabel("Query reading  $x$", fontsize=12.5)
    ax.tick_params(axis="y", length=0, pad=10, colors=INK, labelsize=12)
    ax.tick_params(axis="x", colors=MUTED, labelsize=11, length=3)
    for side in ("right", "top", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#96a6aa")
    legend = [
        Line2D([], [], ls="", marker="o", markersize=9, mfc=LIGHT, mec="#8f6624", label="Light target (1)"),
        Line2D([], [], ls="", marker="s", markersize=8, mfc=DARK, mec=DARK, label="Dark target (0)"),
        Rectangle((0, 0), 1, 1, fc="#f8eaf0", ec="#bd8595", label="Worked cases"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, bbox_to_anchor=(.55, .995),
               columnspacing=1.6, handlelength=1.25, handletextpad=.5,
               frameon=False, fontsize=11.8, labelcolor=INK)
    save_figure(fig, "training_cases", dest, png=True)


def signed(weight):
    return f"{weight:+.2f}".replace("-", "−")


def arrow(ax, start, end, control, label_xy, weight):
    """Quadratic routes with fixed ports keep the direct input-output links visible."""
    path = DrawPath([start, control, end],
                    [DrawPath.MOVETO, DrawPath.CURVE3, DrawPath.CURVE3])
    color = POS if weight > 0 else NEG
    arrow = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=12.7,
                            lw=1.7, color=color, zorder=2, capstyle="round")
    ax.add_patch(arrow)
    ax.text(*label_xy, signed(weight), color=color, fontsize=11.5, fontweight="medium",
            ha="center", va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=.10", ec="none", fc="white", alpha=.96))


def circuit_node(ax, point, label, kind):
    x, y = point
    if kind == "input":
        ax.add_patch(FancyBboxPatch((x - .245, y - .205), .49, .41, zorder=4,
                     boxstyle="round,pad=0,rounding_size=.08", facecolor="#e8eef0",
                     edgecolor="#94a6ab", lw=1))
    else:
        ax.add_patch(Circle(point, .245, facecolor=INK if kind == "output" else "#e1f0eb",
                            edgecolor=INK if kind == "output" else POS, linewidth=1.35, zorder=4))
    ax.text(x, y, label, color="white" if kind == "output" else INK, zorder=6,
            fontsize=15.5, va="center", ha="center")


def circuit_panel(ax, graph, meta):
    ax.set_xlim(-.10, 5.44)
    ax.set_ylim(-.36, 3.40)
    ax.set_aspect("equal")
    ax.axis("off")
    hidden = [18945, 7733] if meta else [43]
    x, b, p = (.40, 2.30), (.40, .80), (4.96, 1.55)
    points = {-1: x, -2: b, 0: p}
    labels = {-1: "$x$", -2: "$b$", 0: "$p$"}
    if meta:
        points.update({7733: (2.72, 2.36), 18945: (2.72, .79)})
        labels.update({7733: "A", 18945: "B"})
        routes = {
            (-1, 7733): ((.65, 2.31), (2.47, 2.36), (1.61, 2.34), (1.55, 2.55)),
            (-2, 7733): ((.64, .94), (2.51, 2.20), (1.53, 1.48), (1.00, 1.43)),
            (-1, 18945): ((.64, 2.16), (2.51, .93), (1.61, 1.60), (2.05, 1.42)),
            (7733, 0): ((2.96, 2.29), (4.71, 1.63), (3.81, 1.98), (3.82, 2.22)),
            (18945, 0): ((2.95, .86), (4.71, 1.46), (3.79, 1.14), (3.83, .96)),
            (-2, 0): ((.64, .69), (4.87, 1.31), (3.20, -.61), (2.70, .18)),
        }
    else:
        points[43] = (2.72, 1.55)
        labels[43] = "$h$"
        routes = {
            (-1, 43): ((.64, 2.24), (2.47, 1.63), (1.50, 1.98), (1.49, 2.19)),
            (-2, 43): ((.64, .86), (2.47, 1.47), (1.50, 1.12), (1.50, .95)),
            (43, 0): ((2.97, 1.55), (4.70, 1.55), (3.82, 1.55), (3.84, 1.80)),
            (-1, 0): ((.64, 2.42), (4.86, 1.80), (3.25, 3.72), (2.71, 2.97)),
            (-2, 0): ((.64, .68), (4.86, 1.30), (3.25, -.62), (2.71, .16)),
        }
    folded = {r["id"]: r for r in graph["folded"]}
    assert set(folded) == {0, *hidden}
    weights = {(source, key): w for key, r in folded.items() for source, w in r["inputs"]}
    assert weights.keys() == routes.keys()
    for link, route in routes.items():
        arrow(ax, *route, weights[link])
    for key, point in points.items():
        circuit_node(ax, point, labels[key], "input" if key < 0 else "output" if key == 0 else "hidden")
    order = [7733, 18945, 0] if meta else [43, 0]
    bias = "      ".join(labels[i] + "  " + signed(folded[i]["bias"]) for i in order)
    ax.text(.50, -.018, "Biases     " + bias, transform=ax.transAxes, ha="center",
            fontsize=10.7, color=MUTED, va="top")


def architectures(graphs, dest):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.03))
    fig.subplots_adjust(left=.015, right=.985, top=.79, bottom=.115, wspace=.085)
    for ax, meta, title, subtitle, name in (
            (axes[0], False, "Standard NEAT  ·  seed 4", "One hidden unit", "standard_04_1000"),
            (axes[1], True, r"Meta-$\theta$  ·  seed 2", "Two hidden units", "meta-theta_02_1000")):
        circuit_panel(ax, graphs[name], meta)
        bb = ax.get_position()
        fig.text((bb.x0 + bb.x1) / 2, .95, title, ha="center", va="top",
                 fontsize=15.5, fontweight="semibold", color=INK)
        fig.text((bb.x0 + bb.x1) / 2, .85, subtitle, ha="center", va="top",
                 fontsize=11.8, color=MUTED)
    fig.add_artist(Line2D([.5, .5], [.105, .94], transform=fig.transFigure, color=GRID, lw=.75))
    save_figure(fig, "champion_circuits", dest, png=True)


def entire_network(graph, dest):
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.68))
    fig.subplots_adjust(left=.055, right=.986, top=.78, bottom=.20, wspace=.14)
    out = graph["output_ids"]
    hidden = [7733, 18945] + sorted(n["id"] for n in graph["nodes"]
                                    if n["id"] not in {*out, 7733, 18945})
    tags = dict(zip(hidden, "ABCDEFGHIJ", strict=True))
    tags.update({-1: "x", -2: "b", 0: "p", **{-2-i: f"u{i}" for i in range(1, 9)},
                 **{i: f"m{i}" for i in range(1, 9)}})
    target = [*hidden, *out]
    edges = {(k, n["id"]): w for n in graph["nodes"] for k, w in n["inputs"]}
    ancestors = set(graph["task_ancestors"])
    assert len(edges) == 37 and not (set(s for s, t in edges) & set(out))
    for ax, sources, title in [(axes[0], graph["input_ids"], "Connections from the inputs"),
                               (axes[1], hidden, "Connections from the hidden units")]:
        ax.set_xlim(-.5, len(target) - .5)
        ax.set_ylim(len(sources) - .5, -.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(len(target)), [tags[t] for t in target], fontsize=9)
        ax.set_yticks(range(len(sources)), [tags[s] for s in sources], fontsize=9.7)
        ax.tick_params(axis="both", length=0, pad=2.5, colors=INK)
        ax.xaxis.tick_top()
        for x in np.arange(-.5, len(target), 1):
            ax.axvline(x, c=GRID, lw=.44, zorder=0)
        for y in np.arange(-.5, len(sources), 1):
            ax.axhline(y, c=GRID, lw=.44, zorder=0)
        ax.axvline(9.5, c="#647983", lw=1.05)
        ax.axvline(10.5, c="#647983", lw=1.05)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, s in enumerate(sources):
            for j, t in enumerate(target):
                if (s, t) not in edges:
                    continue
                positive = edges[s, t] > 0
                ax.scatter(j, i, marker="o" if positive else "D", color=POS if positive else NEG,
                           s=32 if positive else 26, lw=0, zorder=2)
                if t in ancestors:
                    ax.add_patch(Rectangle((j - .42, i - .42), .84, .84,
                                           fc="none", ec=HILITE, lw=1.15, zorder=3))
        ax.set_title(title, fontsize=13.3, fontweight="semibold", color=INK, pad=16)
    handles = [
        Line2D([], [], ls="", marker="o", color=POS, markersize=7, label="Positive weight"),
        Line2D([], [], ls="", marker="D", color=NEG, markersize=6, label="Negative weight"),
        Rectangle((0, 0), 1, 1, fc="white", ec=HILITE, lw=1.2, label="Reaches prediction $p$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(.51, .012),
               frameon=False, fontsize=10.9, handletextpad=.5, columnspacing=1.8)
    save_figure(fig, "full_champion_network", dest)


def main():
    data, dest = ROOT / "data", ROOT / "figures"
    dest.mkdir(exist_ok=True)
    graphs = json.loads((data / "sensor_networks.json").read_text(encoding="utf-8"))
    training_data(records(data / "training_cases.csv"), dest)
    architectures(graphs, dest)
    entire_network(graphs["meta-theta_02_1000"], dest)


if __name__ == "__main__":
    main()
