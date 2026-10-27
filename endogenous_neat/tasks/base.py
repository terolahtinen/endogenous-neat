"""Shared helpers for endogenous NEAT tasks."""

from __future__ import annotations

from random import Random

from endogenous_neat.genome import EndogenousGenome, pin_frozen_theta
from endogenous_neat.theta import Theta

TASK_OUTPUT_INDEX = 0
NUM_META_OUTPUTS = 8
NUM_THETA_INPUTS = 8
ENDOGENOUS_MODES = ("inherited-theta", "meta-theta", "random-theta")


def theta_for_genome(genome, fixed_theta: Theta | None = None) -> Theta:
    """Return the theta vector that should be fed as network inputs.

    Endogenous genomes use their live ``genome.theta`` (updated by inheritance,
    mutation, and meta rewrite). Standard genomes have no theta attribute, so
    they receive ``fixed_theta`` when provided (random-λ experiments), otherwise
    the fixed config defaults for matched input capacity.
    """
    if isinstance(genome, EndogenousGenome):
        return genome.theta
    if fixed_theta is not None:
        return fixed_theta
    return Theta.default()


def augment_inputs_with_theta(task_inputs, theta: Theta) -> list[float]:
    """Concatenate task inputs with normalized current theta."""
    return [*task_inputs, *theta.to_unit_interval_vector()]


def apply_meta_theta(genome, mean_meta: list[float], config=None) -> None:
    """Replace genome theta from averaged meta-outputs (meta-theta mode).

    Subsequent evaluations read the new ``genome.theta`` when building network
    inputs, so the input channel stays in sync with the operator-control state.
    When ``config.genome_config.theta_free_fields`` names a subset, only those
    fields are taken from the meta outputs; the rest stay at the hand-tuned
    defaults. ``None`` replaces every field.
    """
    if mean_meta is not None and isinstance(genome, EndogenousGenome):
        theta = Theta.from_meta_outputs(mean_meta)
        if config is not None:
            theta = pin_frozen_theta(theta, config)
        genome.theta = theta


def apply_random_theta(genome, rng: Random | None = None) -> None:
    """Replace genome theta with a uniform draw inside the bounds.

    This is the random-change control for meta-theta. The network is unchanged
    and its meta outputs are ignored. After evaluation, every θ field is
    replaced by an independent uniform sample, in the same slot where
    meta-theta would write the network's choice. The next generation inherits
    that draw.
    """
    if isinstance(genome, EndogenousGenome):
        genome.theta = Theta.uniform_random(rng or Random())


def genome_type_for_mode(mode: str):
    import neat

    if mode == "standard":
        return neat.DefaultGenome
    if mode in ENDOGENOUS_MODES:
        return EndogenousGenome
    raise ValueError(
        f"Unknown mode {mode!r}; expected 'standard', 'inherited-theta', "
        "'meta-theta', or 'random-theta'"
    )
