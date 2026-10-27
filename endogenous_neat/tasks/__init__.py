"""Task registry for endogenous NEAT experiments."""

from __future__ import annotations

from types import ModuleType

from endogenous_neat.tasks import dynamic_nk, guess_word, pole, sum_squares, xor

TASKS = {
    "xor": xor,
    "pole": pole,
    "sum-squares": sum_squares,
    "guess-word": guess_word,
    "dynamic-nk": dynamic_nk,
}


def get_task(name: str) -> ModuleType:
    if name not in TASKS:
        raise ValueError(
            f"Unknown task {name!r}; expected one of {sorted(TASKS)}"
        )
    return TASKS[name]
