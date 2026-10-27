"""Local browser for launching runs and inspecting theta and network structure.

Start it with ``python -m endogenous_neat.ui`` (requires the ``ui`` extra).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from endogenous_neat.compare import COMPARE_MODES, discover_runs
from endogenous_neat.seed import (
    compatible_modes,
    describe_saved_genome,
    find_saved_genomes,
    load_genome,
)
from endogenous_neat.structure import (
    draw_network,
    load_structure_csv,
    load_structure_jsonl,
    plot_structure_counts,
    snapshot,
)
from endogenous_neat import nk_report
from endogenous_neat.tasks import TASKS, get_task
from endogenous_neat.theta import THETA_FIELD_NAMES

MODES = [
    *COMPARE_MODES,
    "random-theta",
    "compare",
    "compare-random-lambda",
    "meta-theta-ablation",
]


def build_experiment_command(
    *,
    task: str,
    mode: str,
    generations: int,
    runs: int,
    output_dir: str,
    seed: int | None = None,
    answer: str | None = None,
    seed_fill: str = "elite",
    continue_after_solution: bool = False,
    seed_genomes: list[str] | None = None,
    config: str | None = None,
) -> list[str]:
    """Arguments for ``python -m endogenous_neat.experiment``."""
    command = [
        sys.executable,
        "-m",
        "endogenous_neat.experiment",
        "--task",
        task,
        "--mode",
        mode,
        "--generations",
        str(generations),
        "--runs",
        str(runs),
        "--output-dir",
        output_dir,
        "--seed-fill",
        seed_fill,
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if answer:
        command.extend(["--answer", answer])
    if config:
        command.extend(["--config", config])
    if continue_after_solution:
        command.append("--continue-after-solution")
    if seed_genomes:
        command.append("--seed-genomes")
        command.extend(seed_genomes)
    return command


def _newest_first(paths: list[Path]) -> list[Path]:
    """Order run artifacts by file time, newest first."""
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


def _genome_rows(root: Path, task_name: str) -> list[dict]:
    task = get_task(task_name)
    rows = []
    for path in _newest_first(find_saved_genomes(root)):
        info = describe_saved_genome(path)
        genome = info.pop("genome")
        compatible = compatible_modes(genome, task)
        info["can_seed"] = ", ".join(compatible) if compatible else "no"
        rows.append(info)
    return rows


def _fitness_figure(rows: list[dict]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    gens = [int(row["generation"]) for row in rows]
    ax.plot(gens, [float(row["best_fitness"]) for row in rows], label="best")
    ax.plot(gens, [float(row["mean_fitness"]) for row in rows], label="mean")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.legend()
    fig.tight_layout()
    return fig


def _theta_figure(rows: list[dict]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    gens = [int(row["generation"]) for row in rows]
    for name in THETA_FIELD_NAMES:
        if name in rows[0]:
            ax.plot(gens, [float(row[name]) for row in rows], label=name)
    ax.set_xlabel("generation")
    ax.set_ylabel("theta")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def _structure_figure(rows: list[dict]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    gens = [int(row["generation"]) for row in rows]
    for name in ("n_nodes", "n_hidden", "n_enabled", "n_disabled", "n_letters"):
        ax.plot(gens, [float(row[name]) for row in rows], label=name)
    ax.set_xlabel("generation")
    ax.set_ylabel("count")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def _show(fig) -> None:
    import matplotlib.pyplot as plt
    import streamlit as st

    st.pyplot(fig)
    plt.close(fig)


def _launch_view() -> None:
    import streamlit as st

    st.header("Launch")
    task = st.selectbox("Task", sorted(TASKS), index=sorted(TASKS).index("xor"))
    mode = st.selectbox("Mode", MODES, index=MODES.index("compare"))
    generations = st.number_input("Generations", min_value=1, value=100, step=1)
    runs = st.number_input("Runs", min_value=1, value=1, step=1)
    seed_text = st.text_input("Random seed (empty for none)", value="0")
    output_dir = st.text_input("Output directory", value="runs/")
    library = st.text_input(
        "Directory to search for saved genomes",
        value=output_dir,
    )
    seed_fill = st.selectbox(
        "How to place saved genomes",
        ["elite", "clones"],
        help="elite inserts each genome once among random genomes. "
        "clones fills the population with copies.",
    )
    continue_after = st.checkbox(
        "Keep evolving after a solution",
        value=False,
        help="Turn this on when a saved genome already meets the fitness threshold.",
    )
    answer = None
    if task == "guess-word":
        answer = st.text_input("Target word", value="neat")
    config = None
    if task == "dynamic-nk":
        presets = sorted(str(path) for path in Path("configs").glob("nk-*.cfg"))
        if presets:
            default = str(get_task(task).default_config_path())
            config = st.selectbox(
                "NK preset",
                presets,
                index=presets.index(default) if default in presets else 0,
                help="nk-smoke: N=16, 60 generations. The others expect --generations 500.",
            )

    seed_value = int(seed_text) if seed_text.strip() else None
    rows = _genome_rows(Path(library), task) if library.strip() else []
    st.subheader("Saved genomes")
    if rows:
        st.dataframe(
            [{key: value for key, value in row.items()} for row in rows],
            hide_index=True,
        )
        labels = {
            row["path"]: (
                f"{Path(row['path']).parent.name}  "
                f"({row['genome_type']}, fitness={row['fitness']}, "
                f"can seed: {row['can_seed']})"
            )
            for row in rows
        }
        chosen = st.multiselect(
            "Start from these genomes",
            options=list(labels),
            format_func=lambda path: labels[path],
        )
    else:
        st.caption("No best_genome.pkl files in that directory.")
        chosen = []

    command = build_experiment_command(
        task=task,
        mode=mode,
        generations=int(generations),
        runs=int(runs),
        output_dir=output_dir,
        seed=seed_value,
        answer=answer,
        seed_fill=seed_fill,
        continue_after_solution=continue_after,
        seed_genomes=chosen or None,
        config=config,
    )
    st.code(" ".join(command))
    if st.button("Run"):
        log = st.empty()
        collected: list[str] = []
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert process.stdout is not None
        for line in process.stdout:
            collected.append(line)
            log.code("".join(collected[-30:]))
        process.wait()
        if process.returncode == 0:
            st.success(f"Finished. Output is in {output_dir}")
        else:
            st.error(f"The run exited with status {process.returncode}")


def _run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    found = [
        path
        for path in root.rglob("fitness.csv")
        if path.parent.name != "comparison"
    ]
    return [path.parent for path in _newest_first(found)]


def _run_view() -> None:
    import streamlit as st

    st.header("Run")
    root = Path(st.text_input("Runs directory", value="runs/", key="run_root"))
    runs = _run_dirs(root)
    if not runs:
        st.caption("No finished runs there yet.")
        return
    labels = {str(path): path for path in runs}
    selected = Path(st.selectbox("Run", list(labels)))
    fitness_path = selected / "fitness.csv"
    if fitness_path.exists():
        import csv

        with fitness_path.open(newline="") as handle:
            fitness_rows = list(csv.DictReader(handle))
        if fitness_rows:
            st.subheader("Fitness")
            _show(_fitness_figure(fitness_rows))

    theta_path = selected / "theta_best.csv"
    if theta_path.exists():
        import csv

        with theta_path.open(newline="") as handle:
            theta_rows = list(csv.DictReader(handle))
        if theta_rows:
            st.subheader("Theta")
            _show(_theta_figure(theta_rows))
    else:
        st.caption("This run has no theta_best.csv.")

    nk_events = nk_report.change_events(nk_report.load_events(selected))
    if nk_events:
        st.subheader("NK landscape changes")
        st.dataframe(
            [
                {
                    "generation": event["generation"],
                    "change": nk_report.change_label(event),
                    "landscape_version": event.get("landscape_version"),
                    "same-population best before": event.get("same_population_pre_best"),
                    "same-population best after": event.get("same_population_post_best"),
                }
                for event in nk_events
            ],
            hide_index=True,
        )

    structure_rows = load_structure_csv(selected / "structure.csv")
    if structure_rows:
        st.subheader("Structure counts")
        _show(_structure_figure(structure_rows))
    else:
        st.caption("This run has no structure.csv (it predates structure logging).")

    graphs = load_structure_jsonl(selected / "structure_best.jsonl")
    words_path = selected / "words.csv"
    word_rows = []
    if words_path.exists():
        import csv

        with words_path.open(newline="") as handle:
            word_rows = list(csv.DictReader(handle))
    st.subheader("Network")
    if graphs:
        index = st.slider("Generation", 0, len(graphs) - 1, len(graphs) - 1)
        if word_rows and index < len(word_rows):
            spelled = word_rows[index]["word"].replace(" ", "␣")
            st.markdown(
                f"Generation {word_rows[index]['generation']}: `{spelled}`"
                + (
                    f"  (score {word_rows[index]['score']})"
                    if word_rows[index].get("score")
                    else ""
                )
            )
        _show(draw_network(graphs[index]))
    if word_rows:
        st.subheader("Spelled word")
        st.dataframe(word_rows, hide_index=True)
    else:
        pickle_path = selected / "best_genome.pkl"
        if pickle_path.exists():
            _row, graph = snapshot(load_genome(pickle_path), None, None, 0)
            st.caption("Final saved genome. Per-generation drawings are not in this run.")
            _show(draw_network(graph))
        else:
            st.caption("No network snapshot in this run.")


def _comparison_dirs(root: Path) -> tuple[Path, Path] | None:
    if (root / "compare_summary.txt").exists():
        return root.parent, root
    if (root / "comparison" / "compare_summary.txt").exists():
        return root, root / "comparison"
    return None


def _compare_view() -> None:
    import streamlit as st

    st.header("Compare")
    root = Path(st.text_input("Comparison or runs directory", value="runs/", key="cmp_root"))
    resolved = _comparison_dirs(root)
    if resolved is None:
        st.caption("No compare_summary.txt there. Run with --mode compare first.")
        return
    runs_root, comparison = resolved
    summary = (comparison / "compare_summary.txt").read_text()
    st.text(summary)
    for name in ("compare_fitness.png", "compare_theta.png"):
        image = comparison / name
        if image.exists():
            st.image(str(image))
    by_mode = discover_runs(runs_root)
    figure = plot_structure_counts(by_mode)
    if figure is None:
        st.caption("No structure.csv files to overlay. Fitness and theta above still apply.")
    else:
        st.subheader("Structure")
        _show(figure)


def render() -> None:
    import streamlit as st

    st.set_page_config(page_title="Endogenous NEAT", layout="wide")
    st.title("Endogenous NEAT")
    view = st.sidebar.radio("View", ["Launch", "Run", "Compare"])
    if view == "Launch":
        _launch_view()
    elif view == "Run":
        _run_view()
    else:
        _compare_view()


def _under_streamlit() -> bool:
    if os.environ.get("ENDOGENOUS_NEAT_UI") == "1":
        return True
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def main() -> None:
    if _under_streamlit():
        render()
        return
    os.environ["ENDOGENOUS_NEAT_UI"] = "1"
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
