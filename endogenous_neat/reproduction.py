"""Reproduction for endogenous NEAT.

Endogenous control enters via EndogenousGenome lifecycle hooks:
  reproduce -> configure_crossover (inherits theta) -> mutate (uses theta via local config)

Elites are copied unchanged by DefaultReproduction and keep their theta verbatim.
"""

import neat


class EndogenousReproduction(neat.DefaultReproduction):
    """Default NEAT reproduction with endogenous genomes."""

    @classmethod
    def parse_config(cls, param_dict):
        return neat.DefaultReproduction.parse_config(param_dict)
