"""XOR experiment for endogenous-parameter NEAT."""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import neat

from endogenous_neat.evaluation import genome_type_for_mode, make_eval_genomes
from endogenous_neat.logging_utils import ThetaReporter
from endogenous_neat.reproduction import EndogenousReproduction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run XOR experiment with endogenous-parameter NEAT."
    )
    parser.add_argument(
        "--config",
        default="configs/xor-endogenous.cfg",
        help="Path to NEAT config file.",
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
        help="Number of independent runs.",
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "inherited-theta", "meta-theta"],
        default="meta-theta",
        help="Experiment mode.",
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
    return parser.parse_args()


def run_single(
    config_path: Path,
    generations: int,
    mode: str,
    output_dir: Path,
    run_index: int,
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

    eval_genomes = make_eval_genomes(mode)
    winner = population.run(eval_genomes, generations)

    output_dir.mkdir(parents=True, exist_ok=True)
    theta_reporter.write_csv(output_dir)
    theta_reporter.write_plots(output_dir)
    theta_reporter.write_summary(
        output_dir,
        mode=mode,
        generations=generations,
        fitness_threshold=config.fitness_threshold,
    )

    best = theta_reporter.best_genome or winner
    with (output_dir / "best_genome.pkl").open("wb") as f:
        pickle.dump(best, f)

    run_meta = output_dir / "run_meta.txt"
    run_meta.write_text(
        f"run_index={run_index}\nmode={mode}\ngenerations={generations}\n"
        f"config={config_path}\n"
    )

    return best


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    base_output = Path(args.output_dir)

    for run_index in range(args.runs):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_output / f"{timestamp}_run{run_index}_{args.mode}"
        run_single(
            config_path=config_path,
            generations=args.generations,
            mode=args.mode,
            output_dir=run_dir,
            run_index=run_index,
        )
        print(f"Run {run_index + 1}/{args.runs} complete: {run_dir}")


if __name__ == "__main__":
    main()
