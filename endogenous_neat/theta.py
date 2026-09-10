"""Per-genome endogenous mutation-control parameters (theta)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import ClassVar

from endogenous_neat.theta_config import ThetaEvolutionConfig

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


def _unit_interval(x: float) -> float:
    """Clamp a value into [0, 1].

    Meta-heads already use neat-python's sigmoid activation, so outputs are in
    (0, 1). A second sigmoid would squash them into ~[0.5, 0.73] and push θ
    toward the upper half of each bound (overly aggressive mutation).
    """
    return min(1.0, max(0.0, x))


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

    @classmethod
    def randomized(cls, rng: Random, spread: float = 0.1) -> Theta:
        """Return theta with small uniform perturbations around defaults."""
        values = {}
        for name in THETA_FIELD_NAMES:
            default = cls.DEFAULTS[name]
            lo, hi = cls.BOUNDS[name]
            span = hi - lo
            values[name] = default + rng.uniform(-spread, spread) * span
        return cls.from_dict(values)

    @classmethod
    def uniform_random(cls, rng: Random) -> Theta:
        """Return theta with each field sampled uniformly within its bounds."""
        values = {}
        for name in THETA_FIELD_NAMES:
            lo, hi = cls.BOUNDS[name]
            values[name] = rng.uniform(lo, hi)
        return cls.from_dict(values)

    def l1_distance_from_defaults(self) -> float:
        return sum(
            abs(getattr(self, name) - self.DEFAULTS[name]) for name in THETA_FIELD_NAMES
        )

    def l1_distance(self, other: Theta) -> float:
        return sum(
            abs(getattr(self, name) - getattr(other, name))
            for name in THETA_FIELD_NAMES
        )

    def mutated(
        self,
        rng: Random,
        rate: float = 0.1,
        power: float = 0.05,
        mode: str = "gaussian",
        weight_mutate_power_cap: float | None = None,
    ) -> Theta:
        """Apply small perturbations, then clip (and optional power cap)."""
        if mode == "log_normal":
            return self._mutated_log_normal(
                rng, rate=rate, power=power, weight_mutate_power_cap=weight_mutate_power_cap
            )
        return self._mutated_gaussian(
            rng, rate=rate, power=power, weight_mutate_power_cap=weight_mutate_power_cap
        )

    def mutated_with_config(
        self, rng: Random, config: ThetaEvolutionConfig
    ) -> Theta:
        return self.mutated(
            rng,
            rate=config.mutate_rate,
            power=config.mutate_power,
            mode=config.mutate_mode,
            weight_mutate_power_cap=config.weight_mutate_power_cap,
        )

    def _mutated_gaussian(
        self,
        rng: Random,
        *,
        rate: float,
        power: float,
        weight_mutate_power_cap: float | None,
    ) -> Theta:
        values = self.to_dict()
        for name in THETA_FIELD_NAMES:
            if rng.random() < rate:
                values[name] += rng.gauss(0.0, power)
        return self._finalize_mutation(values, weight_mutate_power_cap)

    def _mutated_log_normal(
        self,
        rng: Random,
        *,
        rate: float,
        power: float,
        weight_mutate_power_cap: float | None,
    ) -> Theta:
        values = self.to_dict()
        for name in THETA_FIELD_NAMES:
            if rng.random() >= rate:
                continue
            lo, hi = self.BOUNDS[name]
            if name == "weight_mutate_power":
                values[name] += rng.gauss(0.0, power)
            else:
                # Multiplicative step on positive rates/probabilities.
                scale = math.exp(rng.gauss(0.0, power))
                base = max(values[name], lo + 1e-9)
                values[name] = base * scale
        return self._finalize_mutation(values, weight_mutate_power_cap)

    def _finalize_mutation(
        self, values: dict[str, float], weight_mutate_power_cap: float | None
    ) -> Theta:
        if weight_mutate_power_cap is not None:
            lo, _ = self.BOUNDS["weight_mutate_power"]
            values["weight_mutate_power"] = min(
                values["weight_mutate_power"],
                max(lo, weight_mutate_power_cap),
            )
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

    def to_unit_interval_vector(self) -> list[float]:
        """Normalize each field into [0, 1] for use as network inputs."""
        values = []
        for name in THETA_FIELD_NAMES:
            lo, hi = self.BOUNDS[name]
            span = hi - lo
            if span <= 0:
                values.append(0.0)
            else:
                values.append(_unit_interval((getattr(self, name) - lo) / span))
        return values

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
        """Map averaged meta-outputs (already in [0, 1]) linearly into bounds."""
        if len(raw_outputs) != len(THETA_FIELD_NAMES):
            raise ValueError(
                f"Expected {len(THETA_FIELD_NAMES)} meta-outputs, got {len(raw_outputs)}"
            )
        values = {}
        for name, raw in zip(THETA_FIELD_NAMES, raw_outputs, strict=True):
            lo, hi = cls.BOUNDS[name]
            values[name] = lo + _unit_interval(raw) * (hi - lo)
        return cls.from_dict(values)

    def field_names(self) -> tuple[str, ...]:
        return THETA_FIELD_NAMES
