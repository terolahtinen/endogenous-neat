"""Task registry for endogenous NEAT experiments."""

from __future__ import annotations

from types import ModuleType

from endogenous_neat.tasks import pole, xor

TASKS = {
    "xor": xor,
    "pole": pole,
}


def get_task(name: str) -> ModuleType:
    if name not in TASKS:
        raise ValueError(
            f"Unknown task {name!r}; expected one of {sorted(TASKS)}"
        )
    return TASKS[name]
