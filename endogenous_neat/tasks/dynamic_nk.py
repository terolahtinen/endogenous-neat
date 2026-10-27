"""Dynamic NK landscape task for endogenous-parameter NEAT.

The network has 8 inputs (normalized θ only, like guess-word) and ``n + 8``
outputs: ``n`` bit outputs followed by the 8 fixed meta outputs. Each bit
output is a sigmoid activation in ``(0, 1)``; it is decoded as
``1 if activation >= 0.5 else 0``. The resulting vector is scored on the NK
landscape in force for the current generation.

Task settings live in an ``[NK]`` section of the config file (ignored by
neat-python) and can be overridden from the CLI with ``--nk-*`` flags. The
landscape sequence depends only on ``landscape_seed``, the experiment
``--seed``, and the run index, so every mode of the same run index sees the
same landscapes.
"""

from __future__ import annotations

import csv
import json
import statistics
from configparser import ConfigParser
from dataclasses import dataclass, replace
from pathlib import Path

import neat

from endogenous_neat.nk import DynamicNKTask, events_to_jsonl, parse_schedule
from endogenous_neat.tasks.base import (
    NUM_META_OUTPUTS,
    NUM_THETA_INPUTS,
    apply_meta_theta,
    apply_random_theta,
    augment_inputs_with_theta,
    genome_type_for_mode,
    theta_for_genome,
)
from endogenous_neat.theta import Theta

# NK fitness is normalized to [0, 1] and the attainable optimum is usually
# below 1, so the task is never "solved"; configs set no_fitness_termination.
FITNESS_THRESHOLD = 1.0
BIT_THRESHOLD = 0.5
CONFIG_SECTION = "NK"

NK_FIELDS = (
    "generation",
    "n",
    "k",
    "landscape_version",
    "stage_seed",
    "global_optimum_fitness",
    "best_fitness",
    "mean_fitness",
    "median_fitness",
)


def default_config_path() -> Path:
    return Path("configs/nk-dynamic-v1.cfg")


def fitness_threshold() -> float:
    return FITNESS_THRESHOLD


@dataclass(frozen=True)
class NKSettings:
    n: int = 64
    initial_k: int = 0
    k_schedule: tuple[tuple[int, int], ...] = ()
    perturbations: tuple[tuple[int, float], ...] = ()
    landscape_seed: int = 12345
    exact_optimum_max_n: int = 20

    @property
    def num_outputs(self) -> int:
        return self.n + NUM_META_OUTPUTS


def load_settings(config_path: Path | str, overrides: dict | None = None) -> NKSettings:
    """Read the ``[NK]`` section and apply CLI overrides.

    Overrides: ``n``, ``k`` (static: sets initial_k and clears the schedule),
    ``k_schedule`` (text), ``perturbations`` (text), ``landscape_seed``.
    """
    parser = ConfigParser()
    with Path(config_path).open() as handle:
        parser.read_file(handle)
    settings = NKSettings()
    if parser.has_section(CONFIG_SECTION):
        section = parser[CONFIG_SECTION]
        settings = NKSettings(
            n=section.getint("n", settings.n),
            initial_k=section.getint("initial_k", settings.initial_k),
            k_schedule=parse_schedule(section.get("k_schedule", None), int),
            perturbations=parse_schedule(section.get("perturbations", None), float),
            landscape_seed=section.getint("landscape_seed", settings.landscape_seed),
            exact_optimum_max_n=section.getint(
                "exact_optimum_max_n", settings.exact_optimum_max_n
            ),
        )
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    if "n" in overrides:
        settings = replace(settings, n=int(overrides["n"]))
    if "k" in overrides:
        settings = replace(settings, initial_k=int(overrides["k"]), k_schedule=())
    if "k_schedule" in overrides:
        settings = replace(settings, k_schedule=parse_schedule(overrides["k_schedule"], int))
    if "perturbations" in overrides:
        settings = replace(
            settings, perturbations=parse_schedule(overrides["perturbations"], float)
        )
    if "landscape_seed" in overrides:
        settings = replace(settings, landscape_seed=int(overrides["landscape_seed"]))
    for _generation, k in ((0, settings.initial_k), *settings.k_schedule):
        if not 0 <= k < settings.n:
            raise ValueError(f"NK requires 0 <= k < n; got k={k}, n={settings.n}")
    return settings


def options_from_args(args) -> dict | None:
    """Collect ``--nk-*`` CLI values; ``None`` when none were given."""
    options = {
        "n": getattr(args, "nk_n", None),
        "k": getattr(args, "nk_k", None),
        "k_schedule": getattr(args, "nk_k_schedule", None),
        "perturbations": getattr(args, "nk_perturbations", None),
        "landscape_seed": getattr(args, "nk_landscape_seed", None),
    }
    options = {key: value for key, value in options.items() if value is not None}
    return options or None


def prepare_config(config: neat.Config, config_path: Path | str, options: dict | None) -> None:
    """Make the genome config's output count match ``n + 8``.

    ``--nk-n`` can change ``n`` after the config file was written, so the
    output keys are rebuilt here before the population is created.
    """
    settings = load_settings(config_path, options)
    genome_config = config.genome_config
    if genome_config.num_inputs != NUM_THETA_INPUTS:
        raise ValueError(
            f"dynamic-nk needs num_inputs = {NUM_THETA_INPUTS} (theta only), "
            f"got {genome_config.num_inputs}"
        )
    genome_config.num_outputs = settings.num_outputs
    genome_config.output_keys = list(range(settings.num_outputs))


def run_landscape_seed(settings: NKSettings, run_index: int, experiment_seed: int | None) -> str:
    """Seed string shared by every mode of one run index."""
    return f"{settings.landscape_seed}:{experiment_seed}:{run_index}"


def decode_bits(outputs, n: int) -> list[int]:
    return [1 if outputs[i] >= BIT_THRESHOLD else 0 for i in range(n)]


class NKEvaluator:
    """``eval_genomes`` callable that also tracks the landscape schedule.

    One instance per run. Each call is one generation: the schedule is
    advanced first, then the population is scored on the landscape now in
    force. When a change happens, the previous generation's decoded bit
    vectors are re-scored on the new landscape so the event records the
    immediate response of the same population.
    """

    def __init__(
        self,
        mode: str,
        settings: NKSettings,
        *,
        fixed_theta: Theta | None = None,
        run_index: int = 0,
        experiment_seed: int | None = None,
    ) -> None:
        if mode not in ("standard", "inherited-theta", "meta-theta", "random-theta"):
            raise ValueError(
                f"Unknown mode {mode!r}; expected 'standard', 'inherited-theta', "
                "'meta-theta', or 'random-theta'"
            )
        self.mode = mode
        self.settings = settings
        self.fixed_theta = fixed_theta
        self.run_index = run_index
        self.experiment_seed = experiment_seed
        self.task = DynamicNKTask(
            n=settings.n,
            initial_k=settings.initial_k,
            k_schedule=settings.k_schedule,
            perturbations=settings.perturbations,
            landscape_seed=run_landscape_seed(settings, run_index, experiment_seed),
            exact_optimum_max_n=settings.exact_optimum_max_n,
        )
        self.generation = 0
        self.rows: list[dict] = []
        self.events: list[dict] = []
        self._last_bits: list[list[int]] = []
        self._last_fitness: list[float] = []

    def _check_config(self, config) -> None:
        genome_config = config.genome_config
        expected = self.settings.num_outputs
        if genome_config.num_outputs != expected or len(genome_config.output_keys) != expected:
            raise ValueError(
                f"dynamic-nk with n={self.settings.n} needs num_outputs = {expected}; "
                f"config has {genome_config.num_outputs}"
            )

    def _theta_for(self, genome) -> Theta:
        if self.mode == "standard":
            return theta_for_genome(genome, fixed_theta=self.fixed_theta)
        return theta_for_genome(genome)

    def _record_same_population_response(self, events: list[dict]) -> None:
        changes = [event for event in events if event["event"] != "initial_landscape"]
        if not changes or not self._last_bits:
            return
        landscape = self.task.landscape
        post = [landscape.evaluate(bits) for bits in self._last_bits]
        pre_best = max(self._last_fitness)
        post_best = max(post)
        for event in changes:
            event["same_population_pre_best"] = pre_best
            event["same_population_post_best"] = post_best
            event["same_population_pre_mean"] = statistics.fmean(self._last_fitness)
            event["same_population_post_mean"] = statistics.fmean(post)
            event["same_population_best_drop"] = pre_best - post_best

    def __call__(self, genomes, config) -> None:
        self._check_config(config)
        events = self.task.advance(self.generation)
        self._record_same_population_response(events)
        self.events.extend(events)
        landscape = self.task.landscape
        n = self.settings.n

        bits_seen: list[list[int]] = []
        fitnesses: list[float] = []
        for _genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            outputs = net.activate(augment_inputs_with_theta((), self._theta_for(genome)))
            if len(outputs) < n + NUM_META_OUTPUTS:
                raise RuntimeError(
                    f"expected at least {n + NUM_META_OUTPUTS} outputs, got {len(outputs)}"
                )
            bits = decode_bits(outputs, n)
            fitness = landscape.evaluate(bits)
            genome.fitness = fitness
            if self.mode == "meta-theta":
                apply_meta_theta(
                    genome, list(outputs[n : n + NUM_META_OUTPUTS]), config
                )
            elif self.mode == "random-theta":
                apply_random_theta(genome)
            bits_seen.append(bits)
            fitnesses.append(fitness)

        optimum = self.task.optimum()
        self.rows.append(
            {
                "generation": self.generation,
                "n": n,
                "k": landscape.k,
                "landscape_version": self.task.version,
                "stage_seed": self.task.current_stage_seed(),
                "global_optimum_fitness": "" if optimum is None else optimum[0],
                "best_fitness": max(fitnesses) if fitnesses else float("nan"),
                "mean_fitness": statistics.fmean(fitnesses) if fitnesses else float("nan"),
                "median_fitness": statistics.median(fitnesses) if fitnesses else float("nan"),
            }
        )
        self._last_bits = bits_seen
        self._last_fitness = fitnesses
        self.generation += 1

    def write_logs(self, output_dir: Path) -> None:
        """Write nk.csv (per generation) and nk_events.jsonl (changes)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "nk.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(NK_FIELDS))
            writer.writeheader()
            writer.writerows(self.rows)
        (output_dir / "nk_events.jsonl").write_text(events_to_jsonl(self.events))
        (output_dir / "nk_settings.json").write_text(
            json.dumps(
                {
                    **self.task.describe(),
                    "run_index": self.run_index,
                    "experiment_seed": self.experiment_seed,
                    "bit_threshold": BIT_THRESHOLD,
                },
                indent=2,
            )
            + "\n"
        )

    def meta_lines(self) -> list[str]:
        """Extra ``run_meta.txt`` lines."""
        return [
            f"nk_n={self.settings.n}",
            f"nk_initial_k={self.settings.initial_k}",
            f"nk_landscape_seed={self.task.landscape_seed}",
            f"nk_dynamic={'yes' if self.task.is_dynamic else 'no'}",
        ]


def make_eval_genomes(
    mode: str,
    fixed_theta: Theta | None = None,
    *,
    config_path: Path | str | None = None,
    options: dict | None = None,
    run_index: int = 0,
    experiment_seed: int | None = None,
    settings: NKSettings | None = None,
) -> NKEvaluator:
    """Build the per-run evaluator. Pass ``settings`` directly or a ``config_path``."""
    if settings is None:
        if config_path is None:
            raise ValueError("make_eval_genomes needs settings or a config_path")
        settings = load_settings(config_path, options)
    return NKEvaluator(
        mode,
        settings,
        fixed_theta=fixed_theta,
        run_index=run_index,
        experiment_seed=experiment_seed,
    )


__all__ = [
    "BIT_THRESHOLD",
    "FITNESS_THRESHOLD",
    "NKEvaluator",
    "NKSettings",
    "decode_bits",
    "default_config_path",
    "fitness_threshold",
    "genome_type_for_mode",
    "load_settings",
    "make_eval_genomes",
    "options_from_args",
    "prepare_config",
    "run_landscape_seed",
]
