"""Per-generation network structure snapshots and a simple drawing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STRUCTURE_FIELDS = (
    "generation",
    "n_nodes",
    "n_hidden",
    "n_enabled",
    "n_disabled",
    "n_species",
    "n_letters",
)

_ROLE_X = {"input": 0.0, "hidden": 1.0, "output": 2.0, "letter": 3.0}
_ROLE_COLOR = {
    "input": "#7f8c8d",
    "hidden": "#2980b9",
    "output": "#27ae60",
    "letter": "#e67e22",
}


def species_count(species) -> int:
    inner = getattr(species, "species", None)
    if isinstance(inner, dict):
        return len(inner)
    return 0


def letter_keys(genome) -> list[int]:
    nodes = getattr(genome, "nodes", {})
    return [key for key in getattr(genome, "letter_output_keys", []) if key in nodes]


def snapshot(genome, config, species, generation: int) -> tuple[dict, dict]:
    """Return a structure.csv row and a JSON-ready graph for ``genome``."""
    genome_config = getattr(config, "genome_config", None) if config is not None else None
    letters = letter_keys(genome)
    letter_set = set(letters)
    output_keys = set(getattr(genome_config, "output_keys", []) or [])
    nodes = genome.nodes
    hidden = [
        key
        for key in nodes
        if key not in output_keys and key not in letter_set
    ]
    enabled = 0
    disabled = 0
    connections = []
    for (src, dst), conn in genome.connections.items():
        is_enabled = bool(conn.enabled)
        if is_enabled:
            enabled += 1
        else:
            disabled += 1
        connections.append(
            {
                "in": int(src),
                "out": int(dst),
                "weight": float(conn.weight),
                "enabled": is_enabled,
            }
        )

    row = {
        "generation": generation,
        "n_nodes": len(nodes),
        "n_hidden": len(hidden),
        "n_enabled": enabled,
        "n_disabled": disabled,
        "n_species": species_count(species),
        "n_letters": len(letters),
    }

    word = getattr(genome, "output_word", None) or ""
    label_by_key = {key: word[index] for index, key in enumerate(letters) if index < len(word)}
    order_by_key = {key: index for index, key in enumerate(letters)}

    graph_nodes = []
    if genome_config is not None:
        for key in genome_config.input_keys:
            graph_nodes.append({"key": int(key), "role": "input", "bias": None})
    for key, node in nodes.items():
        if key in letter_set:
            role = "letter"
        elif key in output_keys:
            role = "output"
        else:
            role = "hidden"
        bias = getattr(node, "bias", None)
        entry = {
            "key": int(key),
            "role": role,
            "bias": None if bias is None else float(bias),
        }
        if key in label_by_key:
            entry["label"] = label_by_key[key]
            entry["order"] = order_by_key[key]
        graph_nodes.append(entry)

    graph = {
        "generation": generation,
        "word": word,
        "nodes": graph_nodes,
        "connections": connections,
    }
    return row, graph


def write_structure(output_dir: Path, rows: list[dict], graphs: list[dict]) -> None:
    """Write structure.csv and structure_best.jsonl."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "structure.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(STRUCTURE_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in STRUCTURE_FIELDS})
    with (output_dir / "structure_best.jsonl").open("w") as handle:
        for graph in graphs:
            handle.write(json.dumps(graph) + "\n")


def load_structure_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_structure_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    graphs = []
    for line in path.read_text().splitlines():
        if line.strip():
            graphs.append(json.loads(line))
    return graphs


def plot_structure_counts(by_mode: dict[str, list[Path]]):
    """Median node and enabled-connection counts across runs of each arm.

    Returns None when no run has a structure.csv.
    """
    plotted = False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for mode, run_dirs in by_mode.items():
        node_series = []
        enabled_series = []
        for run_dir in run_dirs:
            rows = load_structure_csv(Path(run_dir) / "structure.csv")
            if not rows:
                continue
            node_series.append([float(row["n_nodes"]) for row in rows])
            enabled_series.append([float(row["n_enabled"]) for row in rows])
        if not node_series:
            continue
        plotted = True
        for ax, series in ((axes[0], node_series), (axes[1], enabled_series)):
            length = min(len(values) for values in series)
            arr = np.array([values[:length] for values in series], dtype=float)
            gens = list(range(length))
            ax.plot(gens, np.median(arr, axis=0), label=f"{mode} (n={len(series)})")
    if not plotted:
        plt.close(fig)
        return None
    axes[0].set_title("nodes")
    axes[1].set_title("enabled connections")
    for ax in axes:
        ax.set_xlabel("generation")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def _visible_word(word: str) -> str:
    """Show spaces so a trailing blank is not invisible."""
    return word.replace(" ", "␣")


def _display_label(node: dict) -> str:
    """Letter nodes show the character they spell. Other nodes show their id."""
    label = node.get("label")
    if isinstance(label, str) and label:
        return _visible_word(label)
    return str(node["key"])


def draw_network(graph: dict):
    """Draw one snapshot. Inputs sit left, then hidden, outputs, and letters."""
    fig, ax = plt.subplots(figsize=(8, 5))
    nodes = graph.get("nodes") or []
    if not nodes:
        ax.set_axis_off()
        ax.set_title("No nodes")
        return fig

    by_role: dict[str, list[dict]] = {role: [] for role in _ROLE_X}
    for node in nodes:
        by_role.setdefault(node["role"], []).append(node)

    positions: dict[int, tuple[float, float]] = {}
    for role, group in by_role.items():
        ordered = sorted(group, key=lambda item: (item.get("order", item["key"]), item["key"]))
        count = len(ordered)
        for index, node in enumerate(ordered):
            y = 0.0 if count == 1 else 1.0 - (index / (count - 1))
            positions[node["key"]] = (_ROLE_X.get(role, 1.0), y)

    for conn in graph.get("connections") or []:
        start = positions.get(conn["in"])
        end = positions.get(conn["out"])
        if start is None or end is None:
            continue
        weight = float(conn["weight"])
        if conn.get("enabled", True):
            color = "#c0392b" if weight < 0 else "#2c3e50"
            style = "-"
            alpha = min(1.0, 0.35 + abs(weight) / 4.0)
        else:
            color = "#bdc3c7"
            style = "--"
            alpha = 0.7
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linestyle": style,
                "alpha": alpha,
                "lw": 1.0,
            },
        )

    for node in nodes:
        x, y = positions[node["key"]]
        ax.scatter(
            [x],
            [y],
            s=180,
            color=_ROLE_COLOR.get(node["role"], "#2980b9"),
            zorder=3,
        )
        ax.text(x, y - 0.08, _display_label(node), ha="center", va="top", fontsize=11)

    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(-0.35, 1.25)
    ax.set_xticks(list(_ROLE_X.values()))
    ax.set_xticklabels(list(_ROLE_X.keys()))
    ax.set_yticks([])
    generation = graph.get("generation")
    title = "Best network"
    if generation is not None:
        title = f"Best network at generation {generation}"
    word = graph.get("word") or ""
    if word:
        title = f"{title}: {_visible_word(word)}"
    ax.set_title(title)
    fig.tight_layout()
    return fig
