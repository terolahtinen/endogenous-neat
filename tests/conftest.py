from pathlib import Path

import neat
import pytest
from neat.innovation import InnovationTracker

from endogenous_neat.genome import EndogenousGenome
from endogenous_neat.reproduction import EndogenousReproduction

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def minimal_config_path() -> Path:
    return FIXTURES / "minimal.cfg"


@pytest.fixture
def genome_config(minimal_config_path):
    config = neat.Config(
        EndogenousGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(minimal_config_path),
    )
    config.genome_config.innovation_tracker = InnovationTracker()
    return config.genome_config


@pytest.fixture
def neat_config(minimal_config_path):
    config = neat.Config(
        EndogenousGenome,
        EndogenousReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(minimal_config_path),
    )
    config.genome_config.innovation_tracker = InnovationTracker()
    return config
