"""Task 3: learn a light sensor's polarity from one calibration label.

Import score(net.activate) to evaluate any feedforward predictor. Input order is
(reading, calibration_label, *extra_inputs); return (task_probability, *controls).
For a standalone fixed-rate pilot: pip install neat-python==2.0.0
then: python light_sensor.py --seeds 1 2 3 4 5 --generations 100 --log pilot.csv
The scorer itself uses only the Python standard library.
"""
import argparse
import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory


def grid(denominator):
    return tuple((s * n / denominator, b, b if s == 1 else 1 - b)
                 for b in (0, 1) for s in (-1, 1)
                 for n in range(1, denominator, 2))


TRAIN = grid(16)   # 32 cue/reading pairs
TEST = grid(32)    # 64 new pairs, with smaller spacing near the boundary


@dataclass(frozen=True)
class Score:
    fitness: float
    accuracy: float
    mean_controls: tuple


def score(activate, cases=TRAIN, extra_inputs=(), uninformative_cue=False):
    """All calls must be stateless. Control outputs are raw means, never rewards.

    Set uninformative_cue=True only for diagnostics: average over both possible
    displayed labels for each true wiring, making the cue independent of the rule.
    The held-out and diagnostic scores must not update theta or affect selection.
    """
    loss = correct = count = 0
    totals = None
    for reading, b, target in cases:
        for shown in ((0, 1) if uninformative_cue else (b,)):
            p, *controls = activate((reading, shown, *extra_inputs))
            if not (0.0 <= p <= 1.0) or not all(map(isfinite, controls)):
                raise ValueError("Expected a probability and finite control outputs")
            if totals is None:
                totals = [0.0] * len(controls)
            if len(totals) != len(controls):
                raise ValueError("Inconsistent number of control outputs")
            for i, v in enumerate(controls):
                totals[i] += v
            loss += (p - target) ** 2
            correct += int((p >= 0.5) == target)
            count += 1
    if not count:
        raise ValueError("No cases")
    return Score(1.0 - loss / count, correct / count, tuple(v / count for v in totals))


# Fixed-rate, task-output-only feasibility pilot. The five-way comparison uses
# the study's own matched networks and reproduction code, not this pilot config.
CONFIG = """
[NEAT]
fitness_criterion = max
fitness_threshold = 1
pop_size = 120
reset_on_extinction = False
no_fitness_termination = True

[DefaultGenome]
num_inputs = 2
num_outputs = 1
num_hidden = 0
feed_forward = True
initial_connection = full_direct
node_add_prob = 0.2
node_delete_prob = 0.2
conn_add_prob = 0.5
conn_delete_prob = 0.5
weight_mutate_rate = 0.8
weight_mutate_power = 0.5
bias_mutate_rate = 0.7
enabled_mutate_rate = 0.01
enabled_default = True
weight_init_mean = 0
weight_init_stdev = 1
weight_max_value = 30
weight_min_value = -30
weight_replace_rate = 0.1
bias_init_mean = 0
bias_init_stdev = 1
bias_max_value = 30
bias_min_value = -30
bias_replace_rate = 0.1
bias_mutate_power = 0.5
activation_default = sigmoid
activation_options = sigmoid
activation_mutate_rate = 0
aggregation_default = sum
aggregation_options = sum
aggregation_mutate_rate = 0
response_init_mean = 1
response_init_stdev = 0
response_max_value = 30
response_min_value = -30
response_mutate_power = 0
response_mutate_rate = 0
response_replace_rate = 0
compatibility_disjoint_coefficient = 1
compatibility_weight_coefficient = 0.5

[DefaultSpeciesSet]
compatibility_threshold = 3
[DefaultStagnation]
species_fitness_func = max
max_stagnation = 20
species_elitism = 2
[DefaultReproduction]
elitism = 2
survival_threshold = 0.2
min_species_size = 2
"""


def pilot(seed, generations, config_file):
    import neat
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_file))
    population = neat.Population(config, seed=seed)
    first = None
    evaluations = 0

    def evaluate(genomes, config):
        nonlocal first, evaluations
        for _, genome in genomes:
            result = score(neat.nn.FeedForwardNetwork.create(genome, config).activate)
            genome.fitness = result.fitness
            evaluations += 1
        if first is None:
            first = max(g.fitness for _, g in genomes)

    winner = population.run(evaluate, generations)
    predict = neat.nn.FeedForwardNetwork.create(winner, config).activate
    train = score(predict)
    test = score(predict, TEST)
    blind = score(predict, TEST, uninformative_cue=True)
    return dict(seed=seed, generations=generations, genome_evaluations=evaluations,
                initial_best_fitness=first, best_training_fitness=train.fitness,
                training_accuracy=train.accuracy, test_fitness=test.fitness,
                test_accuracy=test.accuracy, uninformative_test_accuracy=blind.accuracy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 6)))
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--log", type=Path, help="optional per-seed CSV file")
    args = parser.parse_args()
    if args.generations < 1:
        parser.error("--generations must be positive")
    with TemporaryDirectory() as tmp:
        config = Path(tmp, "sensor.ini")
        config.write_text(CONFIG, encoding="utf-8")
        rows = []
        for seed in args.seeds:
            row = pilot(seed, args.generations, config)
            print(row, flush=True)
            rows.append(row)
    if args.log:
        with args.log.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
