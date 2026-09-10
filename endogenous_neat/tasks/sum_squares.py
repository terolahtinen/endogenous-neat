"""Sum-of-squares function-fitting task for endogenous-parameter NEAT."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import neat

from endogenous_neat.tasks.base import (
    NUM_META_OUTPUTS,
    TASK_OUTPUT_INDEX,
    apply_meta_theta,
    augment_inputs_with_theta,
    genome_type_for_mode,
    theta_for_genome,
)
from endogenous_neat.theta import Theta

# 5x5 grid over {-1, -0.5, 0, 0.5, 1}^2; target t = (x1^2 + x2^2) / 2 in [0, 1].
_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
SUM_SQUARES_CASES: list[tuple[tuple[float, float], float]] = [
    ((x1, x2), (x1 * x1 + x2 * x2) / 2.0) for x1 in _GRID for x2 in _GRID
]
NUM_CASES = len(SUM_SQUARES_CASES)  # 25
FITNESS_THRESHOLD = 24.5


def default_config_path() -> Path:
    return Path("configs/sum-squares-endogenous.cfg")


def fitness_threshold() -> float:
    return FITNESS_THRESHOLD


def evaluate_sum_squares(
    net,
    update_meta: bool,
    theta: Theta | None = None,
) -> tuple[float, list[float] | None]:
    """Evaluate sum-of-squares fitting fitness; optionally collect mean meta-outputs.

    ``theta`` is concatenated (normalized) onto each grid case as network inputs.
    Fitness is ``NUM_CASES - SSE`` so perfect fit scores 25.
    """
    if theta is None:
        theta = Theta.default()

    fitness = float(NUM_CASES)
    meta_sums = [0.0] * NUM_META_OUTPUTS

    for case_inputs, expected in SUM_SQUARES_CASES:
        inputs = augment_inputs_with_theta(case_inputs, theta)
        outputs = net.activate(inputs)
        fitness -= (outputs[TASK_OUTPUT_INDEX] - expected) ** 2
        if update_meta:
            for i, value in enumerate(outputs[TASK_OUTPUT_INDEX + 1 :]):
                meta_sums[i] += value

    if update_meta:
        mean_meta = [value / NUM_CASES for value in meta_sums]
        return fitness, mean_meta
    return fitness, None


def make_eval_genomes(
    mode: str, fixed_theta: Theta | None = None
) -> Callable:
    """Build an eval_genomes callback for the given experiment mode.

    ``fixed_theta`` is used as network θ inputs in standard mode (e.g. random-λ
    experiments). Endogenous modes ignore it and read ``genome.theta``.
    """
    if mode == "standard":

        def eval_genomes(genomes, config):
            for _genome_id, genome in genomes:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                fitness, _ = evaluate_sum_squares(
                    net,
                    update_meta=False,
                    theta=theta_for_genome(genome, fixed_theta=fixed_theta),
                )
                genome.fitness = fitness

        return eval_genomes

    if mode == "inherited-theta":

        def eval_genomes(genomes, config):
            for _genome_id, genome in genomes:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                fitness, _ = evaluate_sum_squares(
                    net, update_meta=False, theta=theta_for_genome(genome)
                )
                genome.fitness = fitness

        return eval_genomes

    if mode == "meta-theta":

        def eval_genomes(genomes, config):
            for _genome_id, genome in genomes:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                fitness, mean_meta = evaluate_sum_squares(
                    net, update_meta=True, theta=theta_for_genome(genome)
                )
                genome.fitness = fitness
                apply_meta_theta(genome, mean_meta)

        return eval_genomes

    raise ValueError(
        f"Unknown mode {mode!r}; expected 'standard', 'inherited-theta', or 'meta-theta'"
    )


__all__ = [
    "FITNESS_THRESHOLD",
    "NUM_CASES",
    "SUM_SQUARES_CASES",
    "default_config_path",
    "evaluate_sum_squares",
    "fitness_threshold",
    "genome_type_for_mode",
    "make_eval_genomes",
]
