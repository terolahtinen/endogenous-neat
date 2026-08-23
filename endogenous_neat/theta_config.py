"""Configuration for inherited theta evolution."""

from __future__ import annotations

from dataclasses import dataclass

THETA_CONFIG_KEYS: tuple[str, ...] = (
    "theta_mutate_rate",
    "theta_mutate_power",
    "theta_mutate_mode",
    "theta_init_spread",
    "theta_distance_coefficient",
    "weight_mutate_power_cap",
)


@dataclass(frozen=True)
class ThetaEvolutionConfig:
    """Controls how per-genome theta is initialized and mutated."""

    mutate_rate: float = 0.05
    mutate_power: float = 0.02
    mutate_mode: str = "log_normal"
    init_spread: float = 0.1
    distance_coefficient: float = 0.0
    weight_mutate_power_cap: float = 1.5

    @classmethod
    def defaults(cls) -> ThetaEvolutionConfig:
        return cls()

    @classmethod
    def extract_from_param_dict(
        cls, param_dict: dict
    ) -> tuple[dict, ThetaEvolutionConfig]:
        """Remove theta evolution keys before building a DefaultGenomeConfig."""
        params = dict(param_dict)
        defaults = cls.defaults()
        config = cls(
            mutate_rate=float(params.pop("theta_mutate_rate", defaults.mutate_rate)),
            mutate_power=float(params.pop("theta_mutate_power", defaults.mutate_power)),
            mutate_mode=str(params.pop("theta_mutate_mode", defaults.mutate_mode)),
            init_spread=float(params.pop("theta_init_spread", defaults.init_spread)),
            distance_coefficient=float(
                params.pop("theta_distance_coefficient", defaults.distance_coefficient)
            ),
            weight_mutate_power_cap=float(
                params.pop("weight_mutate_power_cap", defaults.weight_mutate_power_cap)
            ),
        )
        return params, config

    def attach_to(self, genome_config) -> None:
        """Store settings on a neat genome config object."""
        genome_config.theta_mutate_rate = self.mutate_rate
        genome_config.theta_mutate_power = self.mutate_power
        genome_config.theta_mutate_mode = self.mutate_mode
        genome_config.theta_init_spread = self.init_spread
        genome_config.theta_distance_coefficient = self.distance_coefficient
        genome_config.weight_mutate_power_cap = self.weight_mutate_power_cap

    @classmethod
    def from_genome_config(cls, genome_config) -> ThetaEvolutionConfig:
        defaults = cls.defaults()
        return cls(
            mutate_rate=getattr(
                genome_config, "theta_mutate_rate", defaults.mutate_rate
            ),
            mutate_power=getattr(
                genome_config, "theta_mutate_power", defaults.mutate_power
            ),
            mutate_mode=getattr(
                genome_config, "theta_mutate_mode", defaults.mutate_mode
            ),
            init_spread=getattr(
                genome_config, "theta_init_spread", defaults.init_spread
            ),
            distance_coefficient=getattr(
                genome_config,
                "theta_distance_coefficient",
                defaults.distance_coefficient,
            ),
            weight_mutate_power_cap=getattr(
                genome_config,
                "weight_mutate_power_cap",
                defaults.weight_mutate_power_cap,
            ),
        )
