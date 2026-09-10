"""XOR task for endogenous-parameter NEAT."""

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

XOR_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
XOR_OUTPUTS = [0.0, 1.0, 1.0, 0.0]
FITNESS_THRESHOLD = 3.9


def default_config_path() -> Path:
    return Path("configs/xor-endogenous.cfg")


def fitness_threshold() -> float:
    return FITNESS_THRESHOLD


def evaluate_xor(
    net,
    update_meta: bool,
    theta: Theta | None = None,
) -> tuple[float, list[float] | None]:
    """Evaluate XOR fitness and optionally collect mean meta-outputs.

    ``theta`` is concatenated (normalized) onto each XOR case as network inputs.
    """
    if theta is None:
        theta = Theta.default()

    fitness = 4.0
    meta_sums = [0.0] * NUM_META_OUTPUTS

    for case_inputs, expected in zip(XOR_INPUTS, XOR_OUTPUTS, strict=True):
        inputs = augment_inputs_with_theta(case_inputs, theta)
        outputs = net.activate(inputs)
        fitness -= (outputs[TASK_OUTPUT_INDEX] - expected) ** 2
        if update_meta:
            for i, value in enumerate(outputs[TASK_OUTPUT_INDEX + 1 :]):
                meta_sums[i] += value

    if update_meta:
        mean_meta = [value / len(XOR_INPUTS) for value in meta_sums]
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
                fitness, _ = evaluate_xor(
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
                fitness, _ = evaluate_xor(
                    net, update_meta=False, theta=theta_for_genome(genome)
                )
                genome.fitness = fitness

        return eval_genomes

    if mode == "meta-theta":

        def eval_genomes(genomes, config):
            for _genome_id, genome in genomes:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                # Inputs use pre-eval theta; rewrite updates genome.theta so the
                # next generation (and any later activate) sees the new values.
                fitness, mean_meta = evaluate_xor(
                    net, update_meta=True, theta=theta_for_genome(genome)
                )
                genome.fitness = fitness
                apply_meta_theta(genome, mean_meta)

        return eval_genomes

    raise ValueError(
        f"Unknown mode {mode!r}; expected 'standard', 'inherited-theta', or 'meta-theta'"
    )


__all__ = [
    "XOR_INPUTS",
    "XOR_OUTPUTS",
    "default_config_path",
    "evaluate_xor",
    "fitness_threshold",
    "genome_type_for_mode",
    "make_eval_genomes",
]
