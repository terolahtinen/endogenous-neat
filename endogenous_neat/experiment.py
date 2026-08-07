"""Task-aware experiment runner for endogenous-parameter NEAT."""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import neat

from endogenous_neat.compare import COMPARE_MODES, compare_runs
from endogenous_neat.logging_utils import ThetaReporter
from endogenous_neat.reproduction import EndogenousReproduction
from endogenous_neat.tasks import TASKS, get_task
from endogenous_neat.tasks.base import genome_type_for_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run endogenous-parameter NEAT experiments."
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="xor",
        help="Task to run (default: xor).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to NEAT config file (default: task-specific config).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of generations per run.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent runs per mode.",
    )
    parser.add_argument(
        "--mode",
        choices=[*COMPARE_MODES, "compare"],
        default="meta-theta",
        help=(
            "Experiment mode. Use 'compare' to run all three modes "
            "(standard, inherited-theta, meta-theta) and write overlay plots."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="runs/",
        help="Directory for run outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed (requires seed in config or neat support).",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help=(
            "Skip evolution and only build comparison plots/summary from "
            "existing run directories under --output-dir."
        ),
    )
    return parser.parse_args()


def run_single(
    config_path: Path,
    generations: int,
    mode: str,
    output_dir: Path,
    run_index: int,
    task_name: str,
    eval_genomes,
    fitness_threshold: float,
) -> neat.DefaultGenome:
    genome_type = genome_type_for_mode(mode)
    reproduction_type = (
        neat.DefaultReproduction if mode == "standard" else EndogenousReproduction
    )

    config = neat.Config(
        genome_type,
        reproduction_type,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    population = neat.Population(config)
    theta_reporter = ThetaReporter()
    population.add_reporter(neat.StdOutReporter(False))
    population.add_reporter(theta_reporter)
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes, generations)

    output_dir.mkdir(parents=True, exist_ok=True)
    theta_reporter.write_csv(output_dir)
    theta_reporter.write_plots(output_dir)
    theta_reporter.write_summary(
        output_dir,
        mode=mode,
        generations=generations,
        fitness_threshold=fitness_threshold,
        task=task_name,
    )

    best = theta_reporter.best_genome or winner
    with (output_dir / "best_genome.pkl").open("wb") as f:
        pickle.dump(best, f)

    run_meta = output_dir / "run_meta.txt"
    run_meta.write_text(
        f"run_index={run_index}\ntask={task_name}\nmode={mode}\n"
        f"generations={generations}\nconfig={config_path}\n"
    )

    return best


def run_modes(
    config_path: Path,
    generations: int,
    modes: tuple[str, ...],
    runs: int,
    base_output: Path,
    task_name: str,
    task,
) -> list[Path]:
    """Run each mode for the requested number of independent runs."""
    run_dirs: list[Path] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold = task.fitness_threshold()

    for mode in modes:
        eval_genomes = task.make_eval_genomes(mode)
        for run_index in range(runs):
            run_dir = base_output / f"{timestamp}_run{run_index}_{mode}"
            run_single(
                config_path=config_path,
                generations=generations,
                mode=mode,
                output_dir=run_dir,
                run_index=run_index,
                task_name=task_name,
                eval_genomes=eval_genomes,
                fitness_threshold=threshold,
            )
            run_dirs.append(run_dir)
            print(f"[{task_name}/{mode}] Run {run_index + 1}/{runs} complete: {run_dir}")

    return run_dirs


def main() -> None:
    args = parse_args()
    task = get_task(args.task)
    config_path = Path(args.config) if args.config else task.default_config_path()
    base_output = Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    threshold = task.fitness_threshold()

    if args.compare_only:
        comparison_dir = compare_runs(
            base_output, fitness_threshold=threshold
        )
        print(f"Comparison written to: {comparison_dir}")
        return

    if args.mode == "compare":
        run_modes(
            config_path=config_path,
            generations=args.generations,
            modes=COMPARE_MODES,
            runs=args.runs,
            base_output=base_output,
            task_name=args.task,
            task=task,
        )
        comparison_dir = compare_runs(
            base_output, fitness_threshold=threshold
        )
        print(f"Comparison written to: {comparison_dir}")
        return

    run_modes(
        config_path=config_path,
        generations=args.generations,
        modes=(args.mode,),
        runs=args.runs,
        base_output=base_output,
        task_name=args.task,
        task=task,
    )


if __name__ == "__main__":
    main()
