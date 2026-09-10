"""Task-aware experiment runner for endogenous-parameter NEAT."""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path
from random import Random

import neat

from endogenous_neat.compare import COMPARE_MODES, compare_runs
from endogenous_neat.genome import apply_theta_to_config
from endogenous_neat.logging_utils import ThetaReporter
from endogenous_neat.reproduction import EndogenousReproduction
from endogenous_neat.tasks import TASKS, get_task
from endogenous_neat.tasks.base import genome_type_for_mode
from endogenous_neat.theta import THETA_FIELD_NAMES, Theta

RANDOM_LAMBDA_MODES = ("standard", "meta-theta")


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
        choices=[*COMPARE_MODES, "compare", "compare-random-lambda"],
        default="meta-theta",
        help=(
            "Experiment mode. Use 'compare' for all three modes; "
            "'compare-random-lambda' for standard vs meta-theta from a shared "
            "uniform-in-bounds random θ per run."
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
        help=(
            "Optional seed. For compare-random-lambda, seeds the θ sampler as "
            "seed + run_index so standard/meta pairs share the same θ_rand."
        ),
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help=(
            "Skip evolution and only build comparison plots/summary from "
            "existing run directories under --output-dir."
        ),
    )
    parser.add_argument(
        "--theta-distance-coeff",
        type=float,
        default=None,
        help="Override theta_distance_coefficient on the genome config.",
    )
    parser.add_argument(
        "--theta-mutate-mode",
        choices=("gaussian", "log_normal"),
        default=None,
        help="Override theta_mutate_mode on the genome config.",
    )
    return parser.parse_args()


def apply_theta_cli_overrides(config: neat.Config, args: argparse.Namespace) -> None:
    genome_config = config.genome_config
    if args.theta_distance_coeff is not None:
        genome_config.theta_distance_coefficient = args.theta_distance_coeff
    if args.theta_mutate_mode is not None:
        genome_config.theta_mutate_mode = args.theta_mutate_mode


def sample_theta_for_run(run_index: int, seed: int | None) -> Theta:
    """Sample uniform-in-bounds θ; deterministic when seed is set."""
    rng_seed = (0 if seed is None else seed) + run_index
    return Theta.uniform_random(Random(rng_seed))


def format_theta_meta_lines(theta: Theta) -> list[str]:
    lines = [
        f"theta_initial_l1_from_defaults={theta.l1_distance_from_defaults():.6f}",
    ]
    for name in THETA_FIELD_NAMES:
        lines.append(f"theta_initial_{name}={getattr(theta, name):.6f}")
    return lines


def run_single(
    config_path: Path,
    generations: int,
    mode: str,
    output_dir: Path,
    run_index: int,
    task_name: str,
    eval_genomes,
    fitness_threshold: float,
    theta_overrides: argparse.Namespace | None = None,
    initial_theta: Theta | None = None,
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
    if theta_overrides is not None:
        apply_theta_cli_overrides(config, theta_overrides)

    if initial_theta is not None:
        if mode == "standard":
            # Freeze random λ into the shared genome config for the whole run.
            config.genome_config = apply_theta_to_config(
                config.genome_config, initial_theta
            )
        else:
            config.genome_config.theta_initial = initial_theta

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

    meta_lines = [
        f"run_index={run_index}",
        f"task={task_name}",
        f"mode={mode}",
        f"generations={generations}",
        f"config={config_path}",
    ]
    if initial_theta is not None:
        meta_lines.extend(format_theta_meta_lines(initial_theta))
    (output_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n")

    return best


def run_modes(
    config_path: Path,
    generations: int,
    modes: tuple[str, ...],
    runs: int,
    base_output: Path,
    task_name: str,
    task,
    theta_overrides: argparse.Namespace | None = None,
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
                theta_overrides=theta_overrides,
            )
            run_dirs.append(run_dir)
            print(f"[{task_name}/{mode}] Run {run_index + 1}/{runs} complete: {run_dir}")

    return run_dirs


def run_random_lambda_compare(
    config_path: Path,
    generations: int,
    runs: int,
    base_output: Path,
    task_name: str,
    task,
    seed: int | None,
    theta_overrides: argparse.Namespace | None = None,
) -> list[Path]:
    """Paired standard vs meta-theta from a shared uniform random θ per run_index."""
    run_dirs: list[Path] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold = task.fitness_threshold()

    for run_index in range(runs):
        theta_rand = sample_theta_for_run(run_index, seed)
        for mode in RANDOM_LAMBDA_MODES:
            fixed_theta = theta_rand if mode == "standard" else None
            eval_genomes = task.make_eval_genomes(mode, fixed_theta=fixed_theta)
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
                theta_overrides=theta_overrides,
                initial_theta=theta_rand,
            )
            run_dirs.append(run_dir)
            print(
                f"[{task_name}/{mode}] Run {run_index + 1}/{runs} "
                f"(random-λ L1={theta_rand.l1_distance_from_defaults():.3f}) "
                f"complete: {run_dir}"
            )

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
            theta_overrides=args,
        )
        comparison_dir = compare_runs(
            base_output, fitness_threshold=threshold
        )
        print(f"Comparison written to: {comparison_dir}")
        return

    if args.mode == "compare-random-lambda":
        run_random_lambda_compare(
            config_path=config_path,
            generations=args.generations,
            runs=args.runs,
            base_output=base_output,
            task_name=args.task,
            task=task,
            seed=args.seed,
            theta_overrides=args,
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
        theta_overrides=args,
    )


if __name__ == "__main__":
    main()
