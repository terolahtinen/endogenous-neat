"""Shared helpers for endogenous NEAT tasks."""

from __future__ import annotations

from endogenous_neat.genome import EndogenousGenome
from endogenous_neat.theta import Theta

TASK_OUTPUT_INDEX = 0
NUM_META_OUTPUTS = 8


def apply_meta_theta(genome, mean_meta: list[float]) -> None:
    """Replace genome theta from averaged meta-outputs (meta-theta mode)."""
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
