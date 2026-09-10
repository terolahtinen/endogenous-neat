"""Single-pole balancing task for endogenous-parameter NEAT."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import neat

from endogenous_neat.envs.cart_pole import CartPole, discrete_actuator_force
from endogenous_neat.tasks.base import (
    NUM_META_OUTPUTS,
    TASK_OUTPUT_INDEX,
    apply_meta_theta,
    augment_inputs_with_theta,
    genome_type_for_mode,
    theta_for_genome,
)
from endogenous_neat.theta import Theta

FITNESS_THRESHOLD = 60.0
RUNS_PER_NET = 5
SIMULATION_SECONDS = 60.0


def default_config_path() -> Path:
    return Path("configs/pole-endogenous.cfg")


def fitness_threshold() -> float:
    return FITNESS_THRESHOLD


def evaluate_pole(
    net,
    update_meta: bool,
    *,
    theta: Theta | None = None,
    runs_per_net: int = RUNS_PER_NET,
    simulation_seconds: float = SIMULATION_SECONDS,
) -> tuple[float, list[float] | None]:
    """Evaluate cart-pole balance time; optionally average meta-outputs.

    ``theta`` is concatenated (normalized) onto the scaled cart-pole state at
    every timestep so the controller can condition on current operator params.
    """
    if theta is None:
        theta = Theta.default()

    fitnesses: list[float] = []
    meta_sums = [0.0] * NUM_META_OUTPUTS
    meta_count = 0

    for _ in range(runs_per_net):
        sim = CartPole()
        while sim.t < simulation_seconds:
            inputs = augment_inputs_with_theta(sim.get_scaled_state(), theta)
            outputs = net.activate(inputs)
            force = discrete_actuator_force(outputs)
            sim.step(force)

            if update_meta:
                for i, value in enumerate(outputs[TASK_OUTPUT_INDEX + 1 :]):
                    meta_sums[i] += value
                meta_count += 1

            if (
                abs(sim.x) >= sim.position_limit
                or abs(sim.theta) >= sim.angle_limit_radians
            ):
                break

        fitnesses.append(sim.t)

    fitness = min(fitnesses) if fitnesses else 0.0
    if update_meta and meta_count > 0:
        mean_meta = [value / meta_count for value in meta_sums]
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
                fitness, _ = evaluate_pole(
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
                fitness, _ = evaluate_pole(
                    net, update_meta=False, theta=theta_for_genome(genome)
                )
                genome.fitness = fitness

        return eval_genomes

    if mode == "meta-theta":

        def eval_genomes(genomes, config):
            for _genome_id, genome in genomes:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                fitness, mean_meta = evaluate_pole(
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
    "RUNS_PER_NET",
    "SIMULATION_SECONDS",
    "default_config_path",
    "evaluate_pole",
    "fitness_threshold",
    "genome_type_for_mode",
    "make_eval_genomes",
]
