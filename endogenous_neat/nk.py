"""Kauffman NK fitness landscapes with deterministic, evolution-independent randomness.

An ``NKLandscape`` has ``n`` binary loci. Locus ``i`` depends on ``k`` other
loci ``D_i`` and owns a lookup table ``f_i`` with ``2**(k+1)`` entries drawn
uniformly from ``[0, 1)``. Fitness is the mean of the ``f_i`` values selected by
each locus's own bit together with the bits of its dependencies, so fitness
lies in ``[0, 1]``.

All randomness comes from ``random.Random`` instances seeded from strings.
Python hashes string seeds with SHA-512, so the same seed gives the same
landscape in every process, and nothing here touches the global ``random``
module that NEAT uses for evolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from random import Random

import numpy as np


def seeded_rng(*parts) -> Random:
    """A ``Random`` seeded from a stable string built from ``parts``."""
    return Random(":".join(str(part) for part in parts))


def table_size(k: int) -> int:
    return 2 ** (k + 1)


class NKLandscape:
    """One static NK landscape.

    ``seed`` can be any value with a stable ``str`` form. Pass ``dependencies``
    and ``tables`` to construct a landscape from existing data (used by
    :meth:`perturbed`); otherwise both are generated from ``seed``.
    """

    def __init__(
        self,
        n: int,
        k: int,
        seed,
        *,
        dependencies: np.ndarray | None = None,
        tables: np.ndarray | None = None,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if k < 0 or k >= n:
            raise ValueError(f"k must satisfy 0 <= k < n, got k={k} with n={n}")
        self.n = int(n)
        self.k = int(k)
        self.seed = seed
        self._optimum: tuple[float, tuple[int, ...]] | None = None
        self._optimum_checked = False

        if dependencies is None or tables is None:
            rng = seeded_rng("nk", seed, n, k)
            dependencies = np.zeros((self.n, self.k), dtype=np.int64)
            tables = np.zeros((self.n, table_size(self.k)), dtype=float)
            for i in range(self.n):
                others = [j for j in range(self.n) if j != i]
                dependencies[i] = rng.sample(others, self.k)
                tables[i] = [rng.random() for _ in range(table_size(self.k))]
        self.dependencies = np.asarray(dependencies, dtype=np.int64).reshape(self.n, self.k)
        self.tables = np.asarray(tables, dtype=float).reshape(self.n, table_size(self.k))
        self._locus_index = np.arange(self.n)

    def _table_indices(self, bits: np.ndarray) -> np.ndarray:
        """Row index into each locus's table: own bit first, then dependency bits."""
        index = bits.copy()
        for column in range(self.k):
            index = index * 2 + bits[self.dependencies[:, column]]
        return index

    def evaluate(self, bits) -> float:
        """Mean table value over all loci for a 0/1 vector of length ``n``."""
        arr = np.asarray(bits, dtype=np.int64)
        if arr.shape != (self.n,):
            raise ValueError(f"expected {self.n} bits, got shape {arr.shape}")
        if np.any((arr != 0) & (arr != 1)):
            raise ValueError("bits must be 0 or 1")
        return float(self.tables[self._locus_index, self._table_indices(arr)].mean())

    def perturbed(self, fraction: float, seed) -> tuple[NKLandscape, list[int]]:
        """Copy with ``round(n * fraction)`` loci given new tables.

        Dependencies and ``k`` are unchanged. Returns the new landscape and the
        sorted loci whose tables were replaced.
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        n_changed = int(round(self.n * fraction))
        rng = seeded_rng("perturb", seed, self.n, self.k)
        loci = sorted(rng.sample(range(self.n), n_changed))
        tables = self.tables.copy()
        for locus in loci:
            tables[locus] = [rng.random() for _ in range(table_size(self.k))]
        landscape = NKLandscape(
            self.n,
            self.k,
            self.seed,
            dependencies=self.dependencies.copy(),
            tables=tables,
        )
        return landscape, loci

    def exact_optimum(self, max_n: int = 20) -> tuple[float, tuple[int, ...]] | None:
        """Global optimum by enumerating all ``2**n`` vectors when ``n <= max_n``.

        Returns ``None`` for larger ``n``. Cached after the first call.
        """
        if self.n > max_n:
            return None
        if self._optimum_checked:
            return self._optimum
        count = 2**self.n
        codes = np.arange(count, dtype=np.int64)
        bits = ((codes[:, None] >> np.arange(self.n)) & 1).astype(np.int64)
        total = np.zeros(count, dtype=float)
        for i in range(self.n):
            index = bits[:, i].copy()
            for dep in self.dependencies[i]:
                index = index * 2 + bits[:, dep]
            total += self.tables[i][index]
        best = int(np.argmax(total))
        self._optimum = (float(total[best] / self.n), tuple(int(b) for b in bits[best]))
        self._optimum_checked = True
        return self._optimum

    def describe(self) -> dict:
        return {"n": self.n, "k": self.k, "seed": str(self.seed)}


def parse_schedule(text: str | None, value_type=int) -> tuple[tuple[int, float], ...]:
    """Parse ``"0:0, 100:2, 200:4"`` into ``((0, 0), (100, 2), (200, 4))``.

    Entries are separated by commas or whitespace; each is ``generation:value``.
    """
    if text is None:
        return ()
    cleaned = text.replace(",", " ").split()
    entries = []
    for item in cleaned:
        if ":" not in item:
            raise ValueError(f"schedule entry {item!r} must look like generation:value")
        generation_text, value_text = item.split(":", 1)
        generation = int(generation_text)
        if generation < 0:
            raise ValueError(f"schedule generation must be >= 0, got {generation}")
        entries.append((generation, value_type(value_text)))
    return tuple(sorted(entries, key=lambda entry: entry[0]))


@dataclass
class DynamicNKTask:
    """Controls which NK landscape is in force at each generation.

    ``k_schedule`` lists ``(generation, k)`` pairs: from that generation on, a
    fresh landscape with that ``k`` is used. ``perturbations`` lists
    ``(generation, fraction)`` pairs: at that generation, that fraction of loci
    get new tables while ``k`` and dependencies stay. Both are optional; with
    neither, the landscape is static.

    Call :meth:`advance` once per generation, in order, before evaluating that
    generation's population. Landscape seeds are derived only from
    ``landscape_seed`` and the change index, so they do not depend on how many
    random numbers evolution consumed.
    """

    n: int
    initial_k: int = 0
    k_schedule: tuple[tuple[int, int], ...] = ()
    perturbations: tuple[tuple[int, float], ...] = ()
    landscape_seed: int | str = 12345
    exact_optimum_max_n: int = 20

    landscape: NKLandscape | None = field(default=None, init=False)
    version: int = field(default=0, init=False)
    generation: int = field(default=-1, init=False)
    events: list[dict] = field(default_factory=list, init=False)
    _k_by_generation: dict[int, int] = field(default_factory=dict, init=False)
    _perturb_by_generation: dict[int, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._k_by_generation = {0: int(self.initial_k)}
        for generation, k in self.k_schedule:
            self._k_by_generation[int(generation)] = int(k)
        for k in self._k_by_generation.values():
            if k < 0 or k >= self.n:
                raise ValueError(f"k must satisfy 0 <= k < n, got k={k} with n={self.n}")
        seen = set()
        for generation, fraction in self.perturbations:
            if generation in seen:
                raise ValueError(f"two perturbations at generation {generation}")
            seen.add(generation)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"perturbation fraction must be in [0, 1], got {fraction}")
            self._perturb_by_generation[int(generation)] = float(fraction)

    def k_at(self, generation: int) -> int:
        """The ``k`` in force at ``generation`` according to the schedule."""
        current = self._k_by_generation[0]
        for change_generation in sorted(self._k_by_generation):
            if change_generation <= generation:
                current = self._k_by_generation[change_generation]
        return current

    def stage_seed(self, kind: str, version: int) -> str:
        return f"{self.landscape_seed}:{kind}:{version}"

    @property
    def is_dynamic(self) -> bool:
        return len(self._k_by_generation) > 1 or bool(self._perturb_by_generation)

    def optimum(self) -> tuple[float, tuple[int, ...]] | None:
        if self.landscape is None:
            return None
        return self.landscape.exact_optimum(self.exact_optimum_max_n)

    def _optimum_fitness(self) -> float | None:
        optimum = self.optimum()
        return None if optimum is None else optimum[0]

    def _new_landscape(self, k: int) -> None:
        self.landscape = NKLandscape(self.n, k, self.stage_seed("k", self.version))

    def advance(self, generation: int) -> list[dict]:
        """Apply the changes scheduled for ``generation`` and return their events.

        Generations must be visited in order starting at 0. The events carry
        ``landscape_version``; the first call returns an ``initial_landscape``
        event.
        """
        if generation != self.generation + 1:
            raise ValueError(
                f"advance() must be called with consecutive generations; "
                f"got {generation} after {self.generation}"
            )
        self.generation = generation
        events: list[dict] = []

        if self.landscape is None:
            self._new_landscape(self.k_at(0))
            events.append(
                {
                    "generation": generation,
                    "event": "initial_landscape",
                    "k": self.landscape.k,
                    "landscape_version": self.version,
                    "stage_seed": self.stage_seed("k", self.version),
                    "global_optimum_fitness": self._optimum_fitness(),
                }
            )
        elif generation in self._k_by_generation:
            old_k = self.landscape.k
            new_k = self._k_by_generation[generation]
            self.version += 1
            self._new_landscape(new_k)
            events.append(
                {
                    "generation": generation,
                    "event": "k_change",
                    "old_k": old_k,
                    "new_k": new_k,
                    "landscape_version": self.version,
                    "stage_seed": self.stage_seed("k", self.version),
                    "global_optimum_fitness": self._optimum_fitness(),
                }
            )

        if generation in self._perturb_by_generation and self.landscape is not None:
            fraction = self._perturb_by_generation[generation]
            self.version += 1
            seed = self.stage_seed("perturb", self.version)
            self.landscape, loci = self.landscape.perturbed(fraction, seed)
            events.append(
                {
                    "generation": generation,
                    "event": "landscape_perturbation",
                    "fraction": fraction,
                    "changed_loci": len(loci),
                    "loci": loci,
                    "k": self.landscape.k,
                    "landscape_version": self.version,
                    "stage_seed": seed,
                    "global_optimum_fitness": self._optimum_fitness(),
                }
            )

        self.events.extend(events)
        return events

    def current_stage_seed(self) -> str:
        """Seed string of the landscape currently in force."""
        if not self.events:
            return ""
        return self.events[-1]["stage_seed"]

    def describe(self) -> dict:
        return {
            "n": self.n,
            "initial_k": self.initial_k,
            "k_schedule": [list(entry) for entry in self.k_schedule],
            "perturbations": [list(entry) for entry in self.perturbations],
            "landscape_seed": str(self.landscape_seed),
            "exact_optimum_max_n": self.exact_optimum_max_n,
        }


def events_to_jsonl(events: list[dict]) -> str:
    return "".join(json.dumps(event) + "\n" for event in events)
