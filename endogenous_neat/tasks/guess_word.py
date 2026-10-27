"""Guess-a-word task: fitness is negative Levenshtein distance.

The network spells a word with one output node per character. Letter nodes can
be added and deleted, so the output length is not fixed. Eight meta-outputs
stay a fixed tail.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import neat

from endogenous_neat.tasks.base import (
    NUM_META_OUTPUTS,
    apply_meta_theta,
    apply_random_theta,
    augment_inputs_with_theta,
    theta_for_genome,
)
from endogenous_neat.theta import Theta
from endogenous_neat.word_genome import WordEndogenousGenome, WordGenome

# a-z plus space. Space is the last symbol.
ALPHABET = "abcdefghijklmnopqrstuvwxyz "
TARGET_WORD = "neat"
FITNESS_THRESHOLD = 0.0


def default_config_path() -> Path:
    return Path("configs/guess-word-endogenous.cfg")


def fitness_threshold() -> float:
    return FITNESS_THRESHOLD


def genome_type_for_mode(mode: str):
    if mode == "standard":
        return WordGenome
    if mode in ("inherited-theta", "meta-theta", "random-theta"):
        return WordEndogenousGenome
    raise ValueError(
        f"Unknown mode {mode!r}; expected 'standard', 'inherited-theta', "
        "'meta-theta', or 'random-theta'"
    )


def normalize_answer(answer: str | None) -> str:
    """Lowercase a target word and reject characters outside the alphabet."""
    word = TARGET_WORD if answer is None else answer.lower()
    if not word:
        raise ValueError("answer must be a non-empty word")
    invalid = sorted({ch for ch in word if ch not in ALPHABET})
    if invalid:
        raise ValueError(
            f"answer {answer!r} has characters outside {ALPHABET!r}: {invalid}"
        )
    return word


def activation_to_char(activation: float) -> str:
    """Map a sigmoid activation in [0, 1] onto the alphabet."""
    index = int(activation * len(ALPHABET))
    index = min(len(ALPHABET) - 1, max(0, index))
    return ALPHABET[index]


def decode_word(outputs: list[float], n_letters: int) -> str:
    return "".join(activation_to_char(outputs[i]) for i in range(n_letters))


def edit_distance(source: str, target: str) -> int:
    """Levenshtein distance with unit insert, delete, and replace costs."""
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)
    previous = list(range(len(target) + 1))
    for i, src in enumerate(source, start=1):
        current = [i]
        for j, dst in enumerate(target, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (src != dst)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def network_for_genome(genome, config):
    """Feed-forward net whose outputs are letter nodes followed by meta nodes."""
    local = copy.copy(config)
    genome_config = copy.copy(config.genome_config)
    letters = [key for key in genome.letter_output_keys if key in genome.nodes]
    genome_config.output_keys = letters + list(config.genome_config.output_keys)
    local.genome_config = genome_config
    net = neat.nn.FeedForwardNetwork.create(genome, local)
    return net, len(letters)


def evaluate_guess_word(
    net,
    update_meta: bool,
    *,
    n_letters: int,
    theta: Theta | None = None,
    answer: str = TARGET_WORD,
) -> tuple[float, list[float] | None, str]:
    """Score one forward pass. Fitness is negative edit distance (0 is exact)."""
    if theta is None:
        theta = Theta.default()

    outputs = net.activate(augment_inputs_with_theta((), theta))
    if len(outputs) < n_letters + NUM_META_OUTPUTS:
        raise RuntimeError(
            f"expected at least {n_letters + NUM_META_OUTPUTS} outputs, got {len(outputs)}"
        )
    word = decode_word(outputs, n_letters)
    fitness = -float(edit_distance(word, answer))
    if update_meta:
        meta = list(outputs[n_letters : n_letters + NUM_META_OUTPUTS])
        return fitness, meta, word
    return fitness, None, word


def make_eval_genomes(
    mode: str,
    fixed_theta: Theta | None = None,
    answer: str | None = None,
) -> Callable:
    """Build an eval_genomes callback for the given experiment mode."""
    target = normalize_answer(answer)

    def _check_length(config) -> None:
        limit = int(config.genome_config.max_word_length)
        if len(target) > limit:
            raise ValueError(
                f"answer {target!r} is longer than max_word_length ({limit})"
            )

    if mode == "standard":

        def eval_genomes(genomes, config):
            _check_length(config)
            for _genome_id, genome in genomes:
                net, n_letters = network_for_genome(genome, config)
                fitness, _, word = evaluate_guess_word(
                    net,
                    update_meta=False,
                    theta=theta_for_genome(genome, fixed_theta=fixed_theta),
                    answer=target,
                    n_letters=n_letters,
                )
                genome.fitness = fitness
                genome.output_word = word

        return eval_genomes

    if mode == "inherited-theta":

        def eval_genomes(genomes, config):
            _check_length(config)
            for _genome_id, genome in genomes:
                net, n_letters = network_for_genome(genome, config)
                fitness, _, word = evaluate_guess_word(
                    net,
                    update_meta=False,
                    theta=theta_for_genome(genome),
                    answer=target,
                    n_letters=n_letters,
                )
                genome.fitness = fitness
                genome.output_word = word

        return eval_genomes

    if mode == "meta-theta":

        def eval_genomes(genomes, config):
            _check_length(config)
            for _genome_id, genome in genomes:
                net, n_letters = network_for_genome(genome, config)
                fitness, mean_meta, word = evaluate_guess_word(
                    net,
                    update_meta=True,
                    theta=theta_for_genome(genome),
                    answer=target,
                    n_letters=n_letters,
                )
                genome.fitness = fitness
                genome.output_word = word
                apply_meta_theta(genome, mean_meta, config)

        return eval_genomes

    if mode == "random-theta":

        def eval_genomes(genomes, config):
            _check_length(config)
            for _genome_id, genome in genomes:
                net, n_letters = network_for_genome(genome, config)
                fitness, _, word = evaluate_guess_word(
                    net,
                    update_meta=False,
                    theta=theta_for_genome(genome),
                    answer=target,
                    n_letters=n_letters,
                )
                genome.fitness = fitness
                genome.output_word = word
                apply_random_theta(genome)

        return eval_genomes

    raise ValueError(
        f"Unknown mode {mode!r}; expected 'standard', 'inherited-theta', "
        "'meta-theta', or 'random-theta'"
    )
