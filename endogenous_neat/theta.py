"""Per-genome endogenous mutation-control parameters (theta)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import ClassVar

THETA_FIELD_NAMES: tuple[str, ...] = (
    "node_add_prob",
    "node_delete_prob",
    "conn_add_prob",
    "conn_delete_prob",
    "weight_mutate_rate",
    "weight_mutate_power",
    "bias_mutate_rate",
    "enable_disable_rate",
)


def sigmoid(x: float) -> float:
    """Map a raw network output to (0, 1)."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class Theta:
    """Endogenous operator-control parameters encoded in each genome."""

    node_add_prob: float
    node_delete_prob: float
    conn_add_prob: float
    conn_delete_prob: float
    weight_mutate_rate: float
    weight_mutate_power: float
    bias_mutate_rate: float
    enable_disable_rate: float

    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "node_add_prob": (0.0, 0.5),
        "node_delete_prob": (0.0, 0.5),
        "conn_add_prob": (0.0, 0.8),
        "conn_delete_prob": (0.0, 0.8),
        "weight_mutate_rate": (0.0, 1.0),
        "weight_mutate_power": (0.001, 5.0),
        "bias_mutate_rate": (0.0, 1.0),
        "enable_disable_rate": (0.0, 0.5),
    }

    DEFAULTS: ClassVar[dict[str, float]] = {
        "node_add_prob": 0.2,
        "node_delete_prob": 0.2,
        "conn_add_prob": 0.5,
        "conn_delete_prob": 0.5,
        "weight_mutate_rate": 0.8,
        "weight_mutate_power": 0.5,
        "bias_mutate_rate": 0.7,
        "enable_disable_rate": 0.01,
    }

    # Maps theta field names to neat-python DefaultGenomeConfig attribute names.
    NEAT_CONFIG_KEYS: ClassVar[dict[str, str]] = {
        "node_add_prob": "node_add_prob",
        "node_delete_prob": "node_delete_prob",
        "conn_add_prob": "conn_add_prob",
        "conn_delete_prob": "conn_delete_prob",
        "weight_mutate_rate": "weight_mutate_rate",
        "weight_mutate_power": "weight_mutate_power",
        "bias_mutate_rate": "bias_mutate_rate",
        "enable_disable_rate": "enabled_mutate_rate",
    }

    @classmethod
    def default(cls) -> Theta:
        """Return theta initialized from global config defaults."""
        return cls.from_dict(cls.DEFAULTS)

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> Theta:
        kwargs = {name: values[name] for name in THETA_FIELD_NAMES}
        return cls(**kwargs).clip()

    def clip(self) -> Theta:
        """Return a copy with all fields clipped to bounds."""
        clipped = {}
        for name in THETA_FIELD_NAMES:
            lo, hi = self.BOUNDS[name]
            clipped[name] = min(hi, max(lo, getattr(self, name)))
        return Theta(**clipped)

    def mutated(
        self,
        rng: Random,
        rate: float = 0.1,
        power: float = 0.05,
    ) -> Theta:
        """Apply small Gaussian perturbations, then clip."""
        values = self.to_dict()
        for name in THETA_FIELD_NAMES:
            if rng.random() < rate:
                values[name] += rng.gauss(0.0, power)
        return Theta.from_dict(values)

    @classmethod
    def crossover(cls, theta1: Theta, theta2: Theta, rng: Random) -> Theta:
        """Per-field uniform inheritance from either parent."""
        values = {}
        for name in THETA_FIELD_NAMES:
            parent = theta1 if rng.random() < 0.5 else theta2
            values[name] = getattr(parent, name)
        return cls.from_dict(values)

    def to_vector(self) -> list[float]:
        return [getattr(self, name) for name in THETA_FIELD_NAMES]

    @classmethod
    def from_vector(cls, vector: list[float]) -> Theta:
        if len(vector) != len(THETA_FIELD_NAMES):
            raise ValueError(
                f"Expected {len(THETA_FIELD_NAMES)} values, got {len(vector)}"
            )
        return cls.from_dict(dict(zip(THETA_FIELD_NAMES, vector, strict=True)))

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in THETA_FIELD_NAMES}

    @classmethod
    def from_meta_outputs(cls, raw_outputs: list[float]) -> Theta:
        """Map averaged meta-outputs to bounded theta via sigmoid."""
        if len(raw_outputs) != len(THETA_FIELD_NAMES):
            raise ValueError(
                f"Expected {len(THETA_FIELD_NAMES)} meta-outputs, got {len(raw_outputs)}"
            )
        values = {}
        for name, raw in zip(THETA_FIELD_NAMES, raw_outputs, strict=True):
            lo, hi = cls.BOUNDS[name]
            values[name] = lo + sigmoid(raw) * (hi - lo)
        return cls.from_dict(values)

    def field_names(self) -> tuple[str, ...]:
        return THETA_FIELD_NAMES
