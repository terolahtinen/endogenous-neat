"""Logging and reporting for endogenous NEAT experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import neat
import numpy as np

from endogenous_neat.genome import EndogenousGenome
from endogenous_neat.structure import snapshot, write_structure
from endogenous_neat.theta import THETA_FIELD_NAMES, Theta


class ThetaReporter(neat.reporting.BaseReporter):
    """Track fitness and theta statistics across generations."""

    def __init__(self, fixed_theta: Theta | None = None):
        self.fixed_theta = fixed_theta
        self.fitness_rows: list[dict] = []
        self.theta_best_rows: list[dict] = []
        self.theta_population_rows: list[dict] = []
        self.structure_rows: list[dict] = []
        self.structure_graphs: list[dict] = []
        self.word_rows: list[dict] = []
        self.best_genome = None
        self.best_fitness = float("-inf")

    def _append_theta_rows(self, generation: int, theta: Theta, population: dict) -> None:
        self.theta_best_rows.append({"generation": generation, **theta.to_dict()})

        vectors = [
            g.theta.to_vector()
            for g in population.values()
            if isinstance(g, EndogenousGenome)
        ]
        if not vectors and self.fixed_theta is not None:
            vectors = [self.fixed_theta.to_vector()]

        if vectors:
            arr = np.array(vectors)
            means = arr.mean(axis=0)
            stds = arr.std(axis=0)
            pop_row = {"generation": generation}
            for i, name in enumerate(THETA_FIELD_NAMES):
                pop_row[f"{name}_mean"] = float(means[i])
                pop_row[f"{name}_std"] = float(stds[i])
            self.theta_population_rows.append(pop_row)

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        best_fitness = max(fitnesses) if fitnesses else float("-inf")
        mean_fitness = float(np.mean(fitnesses)) if fitnesses else float("nan")

        generation = len(self.fitness_rows)
        self.fitness_rows.append(
            {
                "generation": generation,
                "best_fitness": best_fitness,
                "mean_fitness": mean_fitness,
            }
        )

        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_genome = best_genome

        if isinstance(best_genome, EndogenousGenome):
            self._append_theta_rows(generation, best_genome.theta, population)
        elif self.fixed_theta is not None:
            self._append_theta_rows(generation, self.fixed_theta, population)

        if best_genome is not None and hasattr(best_genome, "nodes"):
            row, graph = snapshot(best_genome, config, species, generation)
            self.structure_rows.append(row)
            self.structure_graphs.append(graph)
            word = getattr(best_genome, "output_word", None)
            if word is not None:
                self.word_rows.append(
                    {
                        "generation": generation,
                        "word": word,
                        "score": best_genome.fitness,
                    }
                )

    def write_csv(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        with (output_dir / "fitness.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["generation", "best_fitness", "mean_fitness"]
            )
            writer.writeheader()
            writer.writerows(self.fitness_rows)

        if self.theta_best_rows:
            with (output_dir / "theta_best.csv").open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["generation", *THETA_FIELD_NAMES]
                )
                writer.writeheader()
                writer.writerows(self.theta_best_rows)

        if self.theta_population_rows:
            pop_fields = ["generation"]
            for name in THETA_FIELD_NAMES:
                pop_fields.extend([f"{name}_mean", f"{name}_std"])
            with (output_dir / "theta_population.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=pop_fields)
                writer.writeheader()
                writer.writerows(self.theta_population_rows)

        if self.structure_rows:
            write_structure(output_dir, self.structure_rows, self.structure_graphs)

        if self.word_rows:
            with (output_dir / "words.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["generation", "word", "score"])
                writer.writeheader()
                writer.writerows(self.word_rows)

    def write_plots(self, output_dir: Path, task: str = "xor") -> None:
        if not self.fitness_rows:
            return

        generations = [row["generation"] for row in self.fitness_rows]
        best = [row["best_fitness"] for row in self.fitness_rows]
        mean = [row["mean_fitness"] for row in self.fitness_rows]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(generations, best, label="best")
        ax.plot(generations, mean, label="mean")
        ax.set_xlabel("generation")
        ax.set_ylabel("fitness")
        ax.set_title(f"{task} fitness")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "fitness.png", dpi=120)
        plt.close(fig)

        if self.theta_best_rows:
            fig, ax = plt.subplots(figsize=(10, 6))
            for name in THETA_FIELD_NAMES:
                values = [row[name] for row in self.theta_best_rows]
                ax.plot(generations[: len(values)], values, label=name)
            ax.set_xlabel("generation")
            ax.set_ylabel("theta value")
            title = "Best genome theta"
            if self.fixed_theta is not None:
                title = "Fixed lambda (standard mode)"
            ax.set_title(title)
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(output_dir / "theta_best.png", dpi=120)
            plt.close(fig)

    def write_summary(
        self,
        output_dir: Path,
        mode: str,
        generations: int,
        fitness_threshold: float,
        task: str = "xor",
    ) -> None:
        solved = self.best_fitness >= fitness_threshold
        lines = [
            "Endogenous NEAT experiment summary",
            "==================================",
            f"task: {task}",
            f"mode: {mode}",
            f"generations: {generations}",
            f"fitness threshold: {fitness_threshold}",
            f"best fitness: {self.best_fitness:.6f}",
            f"solved: {'yes' if solved else 'no'}",
            "",
        ]

        if isinstance(self.best_genome, EndogenousGenome):
            lines.append("Final best genome theta:")
            for name, value in self.best_genome.theta.to_dict().items():
                lines.append(f"  {name}: {value:.6f}")
            lines.append("")
            lines.append(
                "Theta evolved via inheritance, mutation, selection, "
                "and (in meta-theta mode) network meta-outputs between generations."
            )
        elif self.fixed_theta is not None:
            lines.append("Fixed global lambda (standard mode):")
            for name, value in self.fixed_theta.to_dict().items():
                lines.append(f"  {name}: {value:.6f}")
            lines.append("")
            lines.append(
                "Standard NEAT mode: mutation rates fixed from config for the whole run; "
                "9-output topology with fitness from output[0] only."
            )
        else:
            lines.append(
                "Standard NEAT mode: fixed global mutation parameters; "
                "9-output topology with fitness from output[0] only."
            )

        lines.append("")
        lines.append(
            "This is weak endogeneity / self-adaptive NEAT — reproduction "
            "machinery is still external."
        )

        (output_dir / "summary.txt").write_text("\n".join(lines) + "\n")

    def effective_theta(self) -> Theta | None:
        """Return the logged theta for the run, if any."""
        if isinstance(self.best_genome, EndogenousGenome):
            return self.best_genome.theta
        return self.fixed_theta
