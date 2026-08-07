"""XOR fitness evaluation and meta-output theta updates.

Re-exports task helpers for backward compatibility with existing imports.
"""

from endogenous_neat.tasks.base import (
    NUM_META_OUTPUTS,
    TASK_OUTPUT_INDEX,
    genome_type_for_mode,
)
from endogenous_neat.tasks.xor import (
    XOR_INPUTS,
    XOR_OUTPUTS,
    evaluate_xor,
    make_eval_genomes,
)

__all__ = [
    "NUM_META_OUTPUTS",
    "TASK_OUTPUT_INDEX",
    "XOR_INPUTS",
    "XOR_OUTPUTS",
    "evaluate_xor",
    "genome_type_for_mode",
    "make_eval_genomes",
]
