"""Start a population from one or more saved genomes."""

from __future__ import annotations

import pickle
from pathlib import Path

import neat

from endogenous_neat.genome import sync_node_indexer


def load_genome(path: Path):
    """Unpickle a genome saved by the experiment runner.

    Genome classes are imported first so pickle can resolve them.
    """
    import endogenous_neat.genome  # noqa: F401
    import endogenous_neat.word_genome  # noqa: F401
    import neat.genome  # noqa: F401

    with Path(path).open("rb") as handle:
        genome = pickle.load(handle)
    if not hasattr(genome, "nodes") or not hasattr(genome, "connections"):
        raise ValueError(f"{path} is not a NEAT genome")
    return genome


def clone_genome(genome, new_key):
    """Deep-copy ``genome`` under ``new_key`` with fitness cleared."""
    import copy

    cloned = copy.deepcopy(genome)
    cloned.key = new_key
    cloned.fitness = None
    return cloned


def clone_counts(n_seeds: int, pop_size: int) -> list[int]:
    """How many copies of each seed fill a population.

    The remainder of ``pop_size / n_seeds`` is given to the earliest seeds.
    """
    if n_seeds < 1:
        raise ValueError("at least one seed genome is required")
    if n_seeds > pop_size:
        raise ValueError(
            f"{n_seeds} seed genomes do not fit in a population of {pop_size}"
        )
    base, remainder = divmod(pop_size, n_seeds)
    return [base + (1 if index < remainder else 0) for index in range(n_seeds)]


def genomes_for_type(paths: list[Path], genome_type) -> tuple[list, list[Path]]:
    """Return genomes whose class is exactly ``genome_type``, and their paths."""
    genomes = []
    matched = []
    for path in paths:
        genome = load_genome(path)
        if type(genome) is genome_type:
            genomes.append(genome)
            matched.append(Path(path))
    return genomes, matched


def _max_innovation(genomes) -> int:
    highest = 0
    for genome in genomes:
        for connection in genome.connections.values():
            innovation = getattr(connection, "innovation", None)
            if isinstance(innovation, int) and innovation > highest:
                highest = innovation
    return highest


def _check_seeds(seeds, genome_type, pop_size: int) -> None:
    if len(seeds) > pop_size:
        raise ValueError(
            f"{len(seeds)} seed genomes do not fit in a population of {pop_size}"
        )
    for genome in seeds:
        if type(genome) is not genome_type:
            raise ValueError(
                f"Saved genome is {type(genome).__name__}, "
                f"but this run requires {genome_type.__name__}."
            )


def apply_seeds(population: neat.Population, config: neat.Config, seeds, fill: str) -> None:
    """Replace members of an existing population with copies of ``seeds``.

    ``elite`` keeps one copy of each seed and leaves the other members as the
    random genomes ``Population`` just created. ``clones`` replaces every
    member with copies of the seeds. Fitness is cleared and the first
    evaluation is not skipped. The node indexer and innovation counter are
    moved past any id already used by the seeds, then the population is
    speciated again.
    """
    if fill not in ("elite", "clones"):
        raise ValueError(f"Unknown seed fill {fill!r}; expected 'elite' or 'clones'")
    if not seeds:
        return

    members = population.population
    keys = list(members.keys())
    _check_seeds(seeds, config.genome_type, len(keys))

    if fill == "elite":
        replacements = list(zip(keys, seeds))
    else:
        assigned: list = []
        for genome, count in zip(seeds, clone_counts(len(seeds), len(keys))):
            assigned.extend([genome] * count)
        replacements = list(zip(keys, assigned))

    for key, genome in replacements:
        members[key] = clone_genome(genome, key)

    all_nodes = {}
    for genome in members.values():
        all_nodes.update(genome.nodes)
    sync_node_indexer(config.genome_config, all_nodes)

    tracker = getattr(config.genome_config, "innovation_tracker", None)
    if tracker is not None:
        tracker.global_counter = max(tracker.global_counter, _max_innovation(members.values()))

    population.species.species.clear()
    population.species.genome_to_species.clear()
    population.species.speciate(config, members, population.generation)


def parse_run_meta(path: Path) -> dict[str, str]:
    """Parse ``key=value`` lines from run_meta.txt."""
    if not path.exists():
        return {}
    meta: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key] = value
    return meta


def best_fitness_from_summary(path: Path) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("best fitness:"):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def find_saved_genomes(root: Path) -> list[Path]:
    """Paths of ``best_genome.pkl`` files under ``root``."""
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("best_genome.pkl") if path.is_file())


def compatible_modes(genome, task) -> list[str]:
    """Modes of ``task`` whose genome class is exactly this genome's class.

    ``standard-random`` uses the same class as ``standard``.
    """
    from endogenous_neat.tasks.base import genome_type_for_mode

    getter = getattr(task, "genome_type_for_mode", genome_type_for_mode)
    modes = [
        mode
        for mode in ("standard", "inherited-theta", "meta-theta", "random-theta")
        if type(genome) is getter(mode)
    ]
    if "standard" in modes:
        modes.append("standard-random")
    return modes


def describe_saved_genome(path: Path) -> dict:
    """Summary of one saved genome for the run browser."""
    genome = load_genome(path)
    run_dir = path.parent
    meta = parse_run_meta(run_dir / "run_meta.txt")
    enabled = sum(1 for conn in genome.connections.values() if conn.enabled)
    return {
        "path": str(path),
        "task": meta.get("task", ""),
        "mode": meta.get("mode", ""),
        "fitness": best_fitness_from_summary(run_dir / "summary.txt"),
        "n_nodes": len(genome.nodes),
        "n_connections": len(genome.connections),
        "n_enabled": enabled,
        "genome_type": type(genome).__name__,
        "genome": genome,
    }
