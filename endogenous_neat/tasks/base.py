"""Shared helpers for endogenous NEAT tasks."""

from __future__ import annotations

from endogenous_neat.genome import EndogenousGenome
from endogenous_neat.theta import Theta

TASK_OUTPUT_INDEX = 0
NUM_META_OUTPUTS = 8
NUM_THETA_INPUTS = 8


def theta_for_genome(genome) -> Theta:
    """Return the theta vector that should be fed as network inputs.

    Endogenous genomes use their live ``genome.theta`` (updated by inheritance,
    mutation, and meta rewrite). Standard genomes have no theta attribute, so
    they receive the fixed config defaults for matched input capacity.
    """
    if isinstance(genome, EndogenousGenome):
        return genome.theta
    return Theta.default()


def augment_inputs_with_theta(task_inputs, theta: Theta) -> list[float]:
    """Concatenate task inputs with normalized current theta."""
    return [*task_inputs, *theta.to_unit_interval_vector()]


def apply_meta_theta(genome, mean_meta: list[float]) -> None:
    """Replace genome theta from averaged meta-outputs (meta-theta mode).

    Subsequent evaluations read the new ``genome.theta`` when building network
    inputs, so the input channel stays in sync with the operator-control state.
    """
    if mean_meta is not None and isinstance(genome, EndogenousGenome):
        genome.theta = Theta.from_meta_outputs(mean_meta)


def genome_type_for_mode(mode: str):
    import neat

    if mode == "standard":
        return neat.DefaultGenome
    if mode in ("inherited-theta", "meta-theta"):
        return EndogenousGenome
    raise ValueError(
        f"Unknown mode {mode!r}; expected 'standard', 'inherited-theta', or 'meta-theta'"
    )
