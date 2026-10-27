"""Task-aware experiment runner for endogenous-parameter NEAT."""

from __future__ import annotations

import argparse
import inspect
import pickle
from datetime import datetime
from pathlib import Path
from random import Random

import neat

from endogenous_neat.compare import COMPARE_MODES, compare_runs
from endogenous_neat.genome import (
    apply_theta_to_config,
    pin_frozen_theta,
    theta_from_genome_config,
)
from endogenous_neat.logging_utils import ThetaReporter
from endogenous_neat.reproduction import EndogenousReproduction
from endogenous_neat.seed import apply_seeds, genomes_for_type
from endogenous_neat.tasks import TASKS, get_task
from endogenous_neat.tasks.base import genome_type_for_mode
from endogenous_neat.theta import THETA_FIELD_NAMES, Theta

RANDOM_LAMBDA_MODES = ("standard", "meta-theta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run endogenous-parameter NEAT experiments."
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="xor",
        help="Task to run (default: xor).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to NEAT config file (default: task-specific config).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of generations per run.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent runs per mode.",
    )
    parser.add_argument(
        "--mode",
        choices=[
            *COMPARE_MODES,
            "random-theta",
            "compare",
            "compare-random-lambda",
            "meta-theta-ablation",
        ],
        default="meta-theta",
        help=(
            "Experiment mode. Use 'compare' for this task's full comparison: "
            "classical NEAT at the known-good λ, classical NEAT at a random λ, "
            "inherited-theta, meta-theta, and random-theta. "
            "'random-theta' keeps the meta-theta network but, after each "
            "evaluation, replaces θ with a uniform draw instead of the "
            "network's meta outputs. "
            "'compare-random-lambda' pairs standard and meta-theta from one "
            "shared random θ per run. "
            "'meta-theta-ablation' runs full meta-theta plus one arm per θ "
            "field, with the other seven fields pinned at the hand-tuned λ."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="runs/",
        help="Directory for run outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional experiment seed. Seeds the random-λ draw for 'compare' "
            "(standard-random) and 'compare-random-lambda' as seed + run_index, "
            "and is folded into the dynamic-nk landscape seed so every mode of "
            "one run index sees the same landscapes."
        ),
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help=(
            "Skip evolution and only build comparison plots/summary from "
            "existing run directories under --output-dir."
        ),
    )
    parser.add_argument(
        "--theta-distance-coeff",
        type=float,
        default=None,
        help="Override theta_distance_coefficient on the genome config.",
    )
    parser.add_argument(
        "--theta-mutate-mode",
        choices=("gaussian", "log_normal"),
        default=None,
        help="Override theta_mutate_mode on the genome config.",
    )
    parser.add_argument(
        "--answer",
        default=None,
        help=(
            "Target word for --task guess-word (letters a-z and spaces). "
            "Default: neat."
        ),
    )
    nk_group = parser.add_argument_group(
        "dynamic-nk", "Overrides for the [NK] section of the config (--task dynamic-nk only)."
    )
    nk_group.add_argument("--nk-n", type=int, default=None, help="Number of loci N.")
    nk_group.add_argument(
        "--nk-k",
        type=int,
        default=None,
        help="Static K for the whole run; clears any K schedule from the config.",
    )
    nk_group.add_argument(
        "--nk-k-schedule",
        default=None,
        help='K schedule as "gen:k, gen:k", e.g. "0:0, 100:2, 200:4".',
    )
    nk_group.add_argument(
        "--nk-perturbations",
        default=None,
        help='Perturbations as "gen:fraction", e.g. "150:0.10, 250:0.10".',
    )
    nk_group.add_argument(
        "--nk-landscape-seed",
        type=int,
        default=None,
        help="Base seed for the landscape sequence (combined with --seed and run index).",
    )
    parser.add_argument(
        "--seed-genomes",
        nargs="+",
        default=None,
        help=(
            "Pickle files (best_genome.pkl) to start from. A genome is used "
            "only when its class matches the arm. A single-mode run rejects "
            "a file that does not match."
        ),
    )
    parser.add_argument(
        "--seed-fill",
        choices=("elite", "clones"),
        default="elite",
        help=(
            "elite: one copy of each saved genome, the rest random. "
            "clones: the whole population is copies of the saved genomes."
        ),
    )
    parser.add_argument(
        "--continue-after-solution",
        action="store_true",
        help=(
            "Keep evolving after fitness reaches the task threshold. "
            "Needed when a saved genome is already a solution."
        ),
    )
    return parser.parse_args()


def apply_theta_cli_overrides(config: neat.Config, args: argparse.Namespace) -> None:
    genome_config = config.genome_config
    if args.theta_distance_coeff is not None:
        genome_config.theta_distance_coefficient = args.theta_distance_coeff
    if args.theta_mutate_mode is not None:
        genome_config.theta_mutate_mode = args.theta_mutate_mode


def _make_eval_genomes(
    task,
    mode: str,
    fixed_theta: Theta | None,
    answer: str | None,
    *,
    config_path: Path | None = None,
    task_options: dict | None = None,
    run_index: int = 0,
    experiment_seed: int | None = None,
):
    """Build a task's eval callable, passing only the keywords it accepts.

    Tasks whose evaluation has per-run state (dynamic-nk) take the config
    path, CLI options, run index, and experiment seed; the older tasks only
    take ``fixed_theta`` and ``answer``.
    """
    kwargs = {}
    if fixed_theta is not None:
        kwargs["fixed_theta"] = fixed_theta
    if answer is not None:
        kwargs["answer"] = answer
    accepted = inspect.signature(task.make_eval_genomes).parameters
    context = {
        "config_path": config_path,
        "options": task_options,
        "run_index": run_index,
        "experiment_seed": experiment_seed,
    }
    for name, value in context.items():
        if name in accepted:
            kwargs[name] = value
    return task.make_eval_genomes(mode, **kwargs)


def _task_options(task, args: argparse.Namespace) -> dict | None:
    """Task-specific CLI options, or an error if they were given to another task."""
    nk_values = {
        name: getattr(args, name, None)
        for name in ("nk_n", "nk_k", "nk_k_schedule", "nk_perturbations", "nk_landscape_seed")
    }
    if hasattr(task, "options_from_args"):
        return task.options_from_args(args)
    given = [name for name, value in nk_values.items() if value is not None]
    if given:
        flags = ", ".join("--" + name.replace("_", "-") for name in given)
        raise ValueError(f"{flags} only apply to --task dynamic-nk")
    return None


def _genome_type(task, mode: str):
    if hasattr(task, "genome_type_for_mode"):
        return task.genome_type_for_mode(mode)
    return genome_type_for_mode(mode)


def _resolve_seeds(task, mode: str, seed_paths: list[Path] | None, *, require_match: bool):
    """Load seeds whose class matches this mode.

    A single-mode run (``require_match``) fails if none match. A comparison
    arm starts from scratch instead, and the caller logs the paths that applied.
    """
    if not seed_paths:
        return [], []
    genome_type = _genome_type(task, mode)
    genomes, paths = genomes_for_type(seed_paths, genome_type)
    if require_match and not genomes:
        names = ", ".join(path.name for path in seed_paths)
        raise ValueError(
            f"None of the saved genomes ({names}) are {genome_type.__name__}, "
            f"which --mode {mode} requires."
        )
    return genomes, paths


def sample_theta_for_run(run_index: int, seed: int | None) -> Theta:
    """Sample uniform-in-bounds θ; deterministic when seed is set."""
    rng_seed = (0 if seed is None else seed) + run_index
    return Theta.uniform_random(Random(rng_seed))


def _answer_for_task(args: argparse.Namespace) -> str | None:
    if args.task == "guess-word":
        from endogenous_neat.tasks.guess_word import normalize_answer

        return normalize_answer(args.answer)
    if args.answer is not None:
        raise ValueError("--answer is only valid with --task guess-word")
    return None


def format_theta_meta_lines(theta: Theta, prefix: str) -> list[str]:
    """Format one θ snapshot as ``{prefix}_*`` lines for run_meta.txt.

    ``prefix`` is ``theta_initial`` for the θ a run starts from (fixed λ, or a
    shared random-λ draw) and ``theta_final`` for the θ in force at the end.
    """
    lines = [
        f"{prefix}_l1_from_defaults={theta.l1_distance_from_defaults():.6f}",
    ]
    for name in THETA_FIELD_NAMES:
        lines.append(f"{prefix}_{name}={getattr(theta, name):.6f}")
    return lines


def run_single(
    config_path: Path,
    generations: int,
    mode: str,
    output_dir: Path,
    run_index: int,
    task_name: str,
    fitness_threshold: float,
    task,
    theta_overrides: argparse.Namespace | None = None,
    initial_theta: Theta | None = None,
    answer: str | None = None,
    lambda_start: str | None = None,
    seed_genomes: list | None = None,
    seed_paths: list[Path] | None = None,
    seed_fill: str = "elite",
    continue_after_solution: bool = False,
    task_options: dict | None = None,
    experiment_seed: int | None = None,
    theta_free_fields: tuple[str, ...] | None = None,
) -> neat.DefaultGenome:
    genome_type = _genome_type(task, mode)
    reproduction_type = (
        neat.DefaultReproduction if mode == "standard" else EndogenousReproduction
    )

    config = neat.Config(
        genome_type,
        reproduction_type,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    if hasattr(task, "prepare_config"):
        task.prepare_config(config, config_path, task_options)
    if theta_overrides is not None:
        apply_theta_cli_overrides(config, theta_overrides)
    if theta_free_fields is not None:
        unknown = [name for name in theta_free_fields if name not in THETA_FIELD_NAMES]
        if unknown:
            known = ", ".join(THETA_FIELD_NAMES)
            raise ValueError(f"Unknown theta fields {unknown}; expected {known}")
        # None means every field is live. A tuple is the ablation mask: only
        # these fields are inherited, mutated, and taken from the meta outputs.
        config.genome_config.theta_free_fields = tuple(theta_free_fields)

    # Standard mode feeds the frozen λ (random or default) as θ inputs; the
    # endogenous modes read genome.theta, so they get no fixed_theta.
    eval_genomes = _make_eval_genomes(
        task,
        mode,
        fixed_theta=initial_theta if mode == "standard" else None,
        answer=answer,
        config_path=config_path,
        task_options=task_options,
        run_index=run_index,
        experiment_seed=experiment_seed,
    )

    if initial_theta is not None:
        if mode == "standard":
            # Freeze random λ into the shared genome config for the whole run.
            config.genome_config = apply_theta_to_config(
                config.genome_config, initial_theta
            )
        else:
            config.genome_config.theta_initial = initial_theta

    if continue_after_solution:
        config.no_fitness_termination = True

    fixed_theta = (
        theta_from_genome_config(config.genome_config) if mode == "standard" else None
    )

    population = neat.Population(config)
    if seed_genomes:
        apply_seeds(population, config, seed_genomes, seed_fill)
    if theta_free_fields is not None:
        # Seeds skip configure_new. Pin them before the first evaluation.
        for genome in population.population.values():
            if hasattr(genome, "theta"):
                genome.theta = pin_frozen_theta(genome.theta, config.genome_config)
    theta_reporter = ThetaReporter(fixed_theta=fixed_theta)
    population.add_reporter(neat.StdOutReporter(False))
    population.add_reporter(theta_reporter)
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes, generations)

    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(eval_genomes, "write_logs"):
        eval_genomes.write_logs(output_dir)
    theta_reporter.write_csv(output_dir)
    theta_reporter.write_plots(output_dir, task=task_name)
    theta_reporter.write_summary(
        output_dir,
        mode=mode,
        generations=generations,
        fitness_threshold=fitness_threshold,
        task=task_name,
    )

    best = theta_reporter.best_genome or winner
    with (output_dir / "best_genome.pkl").open("wb") as f:
        pickle.dump(best, f)

    meta_lines = [
        f"run_index={run_index}",
        f"task={task_name}",
        f"mode={mode}",
        f"generations={generations}",
        f"config={config_path}",
    ]
    if theta_free_fields is not None:
        meta_lines.append("theta_free_fields=" + ",".join(theta_free_fields))
    if answer is not None:
        meta_lines.append(f"answer={answer}")
    if lambda_start is not None:
        meta_lines.append(f"lambda_start={lambda_start}")
    if seed_paths is not None:
        meta_lines.append(
            "seed_genomes=" + ",".join(str(path) for path in seed_paths)
        )
        meta_lines.append(f"seed_fill={seed_fill}")
    if continue_after_solution:
        meta_lines.append("continue_after_solution=yes")
    if experiment_seed is not None:
        meta_lines.append(f"experiment_seed={experiment_seed}")
    if hasattr(eval_genomes, "meta_lines"):
        meta_lines.extend(eval_genomes.meta_lines())
    # theta_initial is the θ the run actually started from. Endogenous runs
    # without a shared draw have no single initial θ (each genome is spread
    # around the defaults), so only the final snapshot is logged for them.
    if initial_theta is not None:
        meta_lines.extend(format_theta_meta_lines(initial_theta, "theta_initial"))
    elif fixed_theta is not None:
        meta_lines.extend(format_theta_meta_lines(fixed_theta, "theta_initial"))
    logged_theta = theta_reporter.effective_theta()
    if logged_theta is not None:
        meta_lines.extend(format_theta_meta_lines(logged_theta, "theta_final"))
    (output_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n")

    return best


def run_modes(
    config_path: Path,
    generations: int,
    modes: tuple[str, ...],
    runs: int,
    base_output: Path,
    task_name: str,
    task,
    theta_overrides: argparse.Namespace | None = None,
    answer: str | None = None,
    seed_paths: list[Path] | None = None,
    seed_fill: str = "elite",
    continue_after_solution: bool = False,
    task_options: dict | None = None,
    experiment_seed: int | None = None,
) -> list[Path]:
    """Run each mode for the requested number of independent runs."""
    run_dirs: list[Path] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold = task.fitness_threshold()

    for mode in modes:
        seed_genomes, used_paths = _resolve_seeds(
            task, mode, seed_paths, require_match=True
        )
        for run_index in range(runs):
            run_dir = base_output / f"{timestamp}_run{run_index}_{mode}"
            run_single(
                config_path=config_path,
                generations=generations,
                mode=mode,
                output_dir=run_dir,
                run_index=run_index,
                task_name=task_name,
                fitness_threshold=threshold,
                task=task,
                theta_overrides=theta_overrides,
                answer=answer,
                seed_genomes=seed_genomes or None,
                seed_paths=used_paths if seed_paths else None,
                seed_fill=seed_fill,
                continue_after_solution=continue_after_solution,
                task_options=task_options,
                experiment_seed=experiment_seed,
            )
            run_dirs.append(run_dir)
            print(f"[{task_name}/{mode}] Run {run_index + 1}/{runs} complete: {run_dir}")

    return run_dirs


def run_task_compare(
    config_path: Path,
    generations: int,
    runs: int,
    base_output: Path,
    task_name: str,
    task,
    seed: int | None,
    theta_overrides: argparse.Namespace | None = None,
    answer: str | None = None,
    seed_paths: list[Path] | None = None,
    seed_fill: str = "elite",
    continue_after_solution: bool = False,
    task_options: dict | None = None,
) -> list[Path]:
    """One comparison for this task.

    Classical NEAT is included twice: frozen at the known-good config λ, and
    frozen at a fresh uniform-random λ per run. Inherited-θ and meta-θ use
    their usual initialization. Random-θ uses the meta-θ network and replaces
    θ with a uniform draw after each evaluation. ``seed`` is also the
    experiment seed handed to tasks with their own randomness (dynamic-nk), so
    arms with the same run index share the same task instance.
    """
    run_dirs: list[Path] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold = task.fitness_threshold()
    fixed_arms = (
        ("standard", "standard", "default"),
        ("inherited-theta", "inherited-theta", None),
        ("meta-theta", "meta-theta", None),
        ("random-theta", "random-theta", None),
    )

    for genome_mode, suffix, lambda_start in fixed_arms:
        seed_genomes, used_paths = _resolve_seeds(
            task, genome_mode, seed_paths, require_match=False
        )
        if seed_paths and not seed_genomes:
            print(
                f"[{task_name}/{suffix}] no saved genome matches "
                f"{_genome_type(task, genome_mode).__name__}; starting from scratch"
            )
        for run_index in range(runs):
            run_dir = base_output / f"{timestamp}_run{run_index}_{suffix}"
            run_single(
                config_path=config_path,
                generations=generations,
                mode=genome_mode,
                output_dir=run_dir,
                run_index=run_index,
                task_name=task_name,
                fitness_threshold=threshold,
                task=task,
                theta_overrides=theta_overrides,
                answer=answer,
                lambda_start=lambda_start,
                seed_genomes=seed_genomes or None,
                seed_paths=used_paths if seed_paths else None,
                seed_fill=seed_fill,
                continue_after_solution=continue_after_solution,
                task_options=task_options,
                experiment_seed=seed,
            )
            run_dirs.append(run_dir)
            print(f"[{task_name}/{suffix}] Run {run_index + 1}/{runs} complete: {run_dir}")

    random_seeds, random_paths = _resolve_seeds(
        task, "standard", seed_paths, require_match=False
    )
    if seed_paths and not random_seeds:
        print(
            f"[{task_name}/standard-random] no saved genome matches "
            f"{_genome_type(task, 'standard').__name__}; starting from scratch"
        )
    for run_index in range(runs):
        theta_rand = sample_theta_for_run(run_index, seed)
        run_dir = base_output / f"{timestamp}_run{run_index}_standard-random"
        run_single(
            config_path=config_path,
            generations=generations,
            mode="standard",
            output_dir=run_dir,
            run_index=run_index,
            task_name=task_name,
            fitness_threshold=threshold,
            task=task,
            theta_overrides=theta_overrides,
            initial_theta=theta_rand,
            answer=answer,
            lambda_start="random",
            seed_genomes=random_seeds or None,
            seed_paths=random_paths if seed_paths else None,
            seed_fill=seed_fill,
            continue_after_solution=continue_after_solution,
            task_options=task_options,
            experiment_seed=seed,
        )
        run_dirs.append(run_dir)
        print(
            f"[{task_name}/standard-random] Run {run_index + 1}/{runs} "
            f"(random-λ L1={theta_rand.l1_distance_from_defaults():.3f}) "
            f"complete: {run_dir}"
        )

    return run_dirs


def run_random_lambda_compare(
    config_path: Path,
    generations: int,
    runs: int,
    base_output: Path,
    task_name: str,
    task,
    seed: int | None,
    theta_overrides: argparse.Namespace | None = None,
    answer: str | None = None,
    seed_paths: list[Path] | None = None,
    seed_fill: str = "elite",
    continue_after_solution: bool = False,
    task_options: dict | None = None,
) -> list[Path]:
    """Paired standard vs meta-theta from a shared uniform random θ per run_index."""
    run_dirs: list[Path] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold = task.fitness_threshold()
    seeds_by_mode = {
        mode: _resolve_seeds(task, mode, seed_paths, require_match=False)
        for mode in RANDOM_LAMBDA_MODES
    }
    for mode, (genomes, _paths) in seeds_by_mode.items():
        if seed_paths and not genomes:
            print(
                f"[{task_name}/{mode}] no saved genome matches "
                f"{_genome_type(task, mode).__name__}; starting from scratch"
            )

    for run_index in range(runs):
        theta_rand = sample_theta_for_run(run_index, seed)
        for mode in RANDOM_LAMBDA_MODES:
            seed_genomes, used_paths = seeds_by_mode[mode]
            run_dir = base_output / f"{timestamp}_run{run_index}_{mode}"
            run_single(
                config_path=config_path,
                generations=generations,
                mode=mode,
                output_dir=run_dir,
                run_index=run_index,
                task_name=task_name,
                fitness_threshold=threshold,
                task=task,
                theta_overrides=theta_overrides,
                initial_theta=theta_rand,
                answer=answer,
                seed_genomes=seed_genomes or None,
                seed_paths=used_paths if seed_paths else None,
                seed_fill=seed_fill,
                continue_after_solution=continue_after_solution,
                task_options=task_options,
                experiment_seed=seed,
            )
            run_dirs.append(run_dir)
            print(
                f"[{task_name}/{mode}] Run {run_index + 1}/{runs} "
                f"(random-λ L1={theta_rand.l1_distance_from_defaults():.3f}) "
                f"complete: {run_dir}"
            )

    return run_dirs


def ablation_arms() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Directory suffix and the θ fields left live for each ablation arm.

    The first arm is full meta-theta. Each later arm keeps one field and pins
    the other seven at the hand-tuned defaults.
    """
    arms = [("meta-theta", THETA_FIELD_NAMES)]
    for name in THETA_FIELD_NAMES:
        arms.append((f"meta-theta-{name}", (name,)))
    return tuple(arms)


def run_meta_theta_ablation(
    config_path: Path,
    generations: int,
    runs: int,
    base_output: Path,
    task_name: str,
    task,
    theta_overrides: argparse.Namespace | None = None,
    answer: str | None = None,
    seed_paths: list[Path] | None = None,
    seed_fill: str = "elite",
    continue_after_solution: bool = False,
    task_options: dict | None = None,
    experiment_seed: int | None = None,
) -> list[Path]:
    """Full meta-theta plus one single-field arm for each θ component."""
    run_dirs: list[Path] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold = task.fitness_threshold()
    seed_genomes, used_paths = _resolve_seeds(
        task, "meta-theta", seed_paths, require_match=False
    )
    if seed_paths and not seed_genomes:
        print(
            f"[{task_name}/meta-theta-ablation] no saved genome matches "
            f"{_genome_type(task, 'meta-theta').__name__}; starting from scratch"
        )

    for suffix, free_fields in ablation_arms():
        for run_index in range(runs):
            run_dir = base_output / f"{timestamp}_run{run_index}_{suffix}"
            run_single(
                config_path=config_path,
                generations=generations,
                mode="meta-theta",
                output_dir=run_dir,
                run_index=run_index,
                task_name=task_name,
                fitness_threshold=threshold,
                task=task,
                theta_overrides=theta_overrides,
                answer=answer,
                seed_genomes=seed_genomes or None,
                seed_paths=used_paths if seed_paths else None,
                seed_fill=seed_fill,
                continue_after_solution=continue_after_solution,
                task_options=task_options,
                experiment_seed=experiment_seed,
                theta_free_fields=free_fields,
            )
            run_dirs.append(run_dir)
            print(
                f"[{task_name}/{suffix}] Run {run_index + 1}/{runs} complete: {run_dir}"
            )

    return run_dirs


def main() -> None:
    args = parse_args()
    task = get_task(args.task)
    answer = _answer_for_task(args)
    config_path = Path(args.config) if args.config else task.default_config_path()
    base_output = Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    threshold = task.fitness_threshold()

    if args.compare_only:
        comparison_dir = compare_runs(
            base_output, fitness_threshold=threshold, task_name=args.task
        )
        print(f"Comparison written to: {comparison_dir}")
        return

    seed_paths = [Path(path) for path in args.seed_genomes] if args.seed_genomes else None
    task_options = _task_options(task, args)

    if args.mode == "compare":
        run_task_compare(
            config_path=config_path,
            generations=args.generations,
            runs=args.runs,
            base_output=base_output,
            task_name=args.task,
            task=task,
            seed=args.seed,
            theta_overrides=args,
            answer=answer,
            seed_paths=seed_paths,
            seed_fill=args.seed_fill,
            continue_after_solution=args.continue_after_solution,
            task_options=task_options,
        )
        comparison_dir = compare_runs(
            base_output, fitness_threshold=threshold, task_name=args.task
        )
        print(f"Comparison written to: {comparison_dir}")
        return

    if args.mode == "compare-random-lambda":
        run_random_lambda_compare(
            config_path=config_path,
            generations=args.generations,
            runs=args.runs,
            base_output=base_output,
            task_name=args.task,
            task=task,
            seed=args.seed,
            theta_overrides=args,
            answer=answer,
            seed_paths=seed_paths,
            seed_fill=args.seed_fill,
            continue_after_solution=args.continue_after_solution,
            task_options=task_options,
        )
        comparison_dir = compare_runs(
            base_output, fitness_threshold=threshold, task_name=args.task
        )
        print(f"Comparison written to: {comparison_dir}")
        return

    if args.mode == "meta-theta-ablation":
        run_meta_theta_ablation(
            config_path=config_path,
            generations=args.generations,
            runs=args.runs,
            base_output=base_output,
            task_name=args.task,
            task=task,
            theta_overrides=args,
            answer=answer,
            seed_paths=seed_paths,
            seed_fill=args.seed_fill,
            continue_after_solution=args.continue_after_solution,
            task_options=task_options,
            experiment_seed=args.seed,
        )
        return

    run_modes(
        config_path=config_path,
        generations=args.generations,
        modes=(args.mode,),
        runs=args.runs,
        base_output=base_output,
        task_name=args.task,
        task=task,
        theta_overrides=args,
        answer=answer,
        seed_paths=seed_paths,
        seed_fill=args.seed_fill,
        continue_after_solution=args.continue_after_solution,
        task_options=task_options,
        experiment_seed=args.seed,
    )


if __name__ == "__main__":
    main()
