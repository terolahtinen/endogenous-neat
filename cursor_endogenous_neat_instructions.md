# Cursor implementation instructions: Endogenous-Parameter NEAT in Python

We are implementing a research prototype of **Endogenous-Parameter NEAT** using Python.

The goal is not to implement a fully self-replicating neural network yet. The goal is a first, clean prototype where some NEAT mutation-control parameters are encoded in the genome, inherited, mutated, and modified by the neural network between generations.

Use **`neat-python`** as the base library. A **forked copy of neat-python is already in this workspace** at `neat-python/` (currently v2.1.0). Do not install neat-python from PyPI unless explicitly needed for comparison. Prefer the local fork so we can extend or patch the library if endogenous NEAT requires it.

References:

- neat-python documentation: https://neat-python.readthedocs.io/
- neat-python customisation: https://neat-python.readthedocs.io/en/latest/customization.html
- neat-python genome interface: https://neat-python.readthedocs.io/en/latest/genome-interface.html
- neat-python population implementation: https://neat-python.readthedocs.io/en/latest/_modules/population.html

---

## Research goal

Classical NEAT can be described as:

```text
G_t = (A_t, w_t)

G_{t+1} = M_NEAT(G_t; λ)
```

where:

- `A_t` is topology,
- `w_t` are connection weights,
- `λ` is a fixed set of mutation-control parameters defined externally by the experimenter.

We want to implement a weakly endogenous version:

```text
G_t = (A_t, w_t, θ_t)

G_{t+1} = M_NEAT(G_t; θ_t)
```

where:

- `θ_t` contains per-genome mutation-control parameters,
- `θ_t` is inherited,
- `θ_t` mutates,
- `θ_t` can be modified by the neural network’s meta-output after evaluation,
- reproduction uses the genome’s own `θ_t` instead of only global config values.

This is a stepping stone toward full endogenous NEAT, not the final full framework.

---

## Implementation scope

Implement **Phase 1 only**:

1. Use the local fork in `neat-python/` (install it editable from that path).
2. Add an `EndogenousGenome` class extending `neat.DefaultGenome`.
3. Add an `EndogenousReproduction` class extending or adapting `neat.DefaultReproduction`.
4. Add per-genome endogenous operator parameters `theta`.
5. Add meta-output handling: some network outputs are normal task outputs, and some outputs update `theta`.
6. Create a simple XOR experiment.
7. Add logging so we can observe how endogenous parameters evolve.
8. Add tests for inheritance, mutation, bounds, and meta-output transformation.

Do not implement TensorNEAT, HyperNEAT, CoDeepNEAT, quantum optimisation, crossover encodings, or full self-replication in this phase.

---

## Technology choices

Use:

```bash
python >= 3.11
neat-python   # local editable install from ./neat-python
numpy
pytest
matplotlib
```

Install the forked library from the workspace:

```bash
pip install -e ./neat-python
```

Keep the project simple and readable. Prioritise research clarity over performance.

If endogenous NEAT needs small hooks inside neat-python itself (for example, cleaner per-genome mutation config handling), change the fork in `neat-python/` and document why. Prefer extending via `EndogenousGenome` / `EndogenousReproduction` first; patch the fork only when subclassing is insufficient.

---

## Workspace layout

The repository already contains the forked NEAT implementation:

```text
endogenous-neat/
  neat-python/              # forked neat-python (v2.1.0); base library
    neat/                   # core library source (DefaultGenome, Population, etc.)
    examples/xor/           # reference XOR configs and scripts
    tests/                  # upstream neat-python tests
    pyproject.toml

  cursor_endogenous_neat_instructions.md

  pyproject.toml            # endogenous-neat package (to be created)
  README.md

  configs/
    xor-endogenous.cfg

  endogenous_neat/
    __init__.py
    theta.py
    genome.py
    reproduction.py
    evaluation.py
    experiment_xor.py
    logging_utils.py

  tests/
    test_theta.py
    test_genome.py
    test_reproduction.py
```

When reading neat-python internals, start in `neat-python/neat/` (`genome.py`, `reproduction.py`, `population.py`, `config.py`). The XOR example under `neat-python/examples/xor/` is a good template for configs and evaluation loops.

---

## Proposed project structure

Create the endogenous-neat package alongside the existing fork:

```text
endogenous-neat/
  neat-python/              # already present — do not re-clone

  pyproject.toml
  README.md

  configs/
    xor-endogenous.cfg

  endogenous_neat/
    __init__.py
    theta.py
    genome.py
    reproduction.py
    evaluation.py
    experiment_xor.py
    logging_utils.py

  tests/
    test_theta.py
    test_genome.py
    test_reproduction.py
```

In `pyproject.toml`, depend on the local fork via an editable path install during development:

```bash
pip install -e ./neat-python
pip install -e .
```

If the build tool supports path dependencies (for example uv or Poetry), point `neat-python` at `./neat-python` rather than PyPI.

---

## Endogenous parameters

Create a `Theta` dataclass in `endogenous_neat/theta.py`.

It should include at least these parameters:

```python
node_add_prob: float
node_delete_prob: float
conn_add_prob: float
conn_delete_prob: float
weight_mutate_rate: float
weight_mutate_power: float
bias_mutate_rate: float
enable_disable_rate: float
```

Each parameter must have:

- a default value,
- a minimum value,
- a maximum value,
- a method for clipping to bounds,
- a method for mutation,
- a method for crossover/inheritance,
- a method for conversion to and from a flat list or vector.

Use bounded values. No endogenous value may become negative or unreasonably large.

Suggested bounds:

```python
BOUNDS = {
    "node_add_prob": (0.0, 0.5),
    "node_delete_prob": (0.0, 0.5),
    "conn_add_prob": (0.0, 0.8),
    "conn_delete_prob": (0.0, 0.8),
    "weight_mutate_rate": (0.0, 1.0),
    "weight_mutate_power": (0.001, 5.0),
    "bias_mutate_rate": (0.0, 1.0),
    "enable_disable_rate": (0.0, 0.5),
}
```

Implement this kind of transformation for meta-outputs:

```python
value = min_value + sigmoid(raw_output) * (max_value - min_value)
```

Do not directly assign raw neural output to mutation probabilities.

---

## Genome implementation

Create `EndogenousGenome` in `endogenous_neat/genome.py`.

It should extend `neat.DefaultGenome`.

Responsibilities:

1. Add `self.theta`.
2. Initialise `theta` in `configure_new`.
3. Inherit/crossover `theta` in `configure_crossover`.
4. Mutate `theta` in `mutate`.
5. Include `theta` in compatibility distance, with a small coefficient.
6. Support copying/pickling/checkpointing.
7. Avoid breaking existing `neat-python` behaviour.

Pseudo-design:

```python
class EndogenousGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.theta = Theta.default()

    def configure_new(self, config):
        super().configure_new(config)
        self.theta = Theta.default()

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        self.theta = Theta.crossover(genome1.theta, genome2.theta)

    def mutate(self, config):
        # Use current theta to influence mutation behaviour.
        # Then mutate theta itself slightly.
        ...
```

Important: do not simply overwrite global config values permanently. We need per-genome mutation parameters.

A practical first implementation may temporarily create a shallow per-genome mutation config object derived from the normal genome config, then replace selected mutation probabilities using `self.theta`.

Example concept:

```python
local_config = copy.copy(config)
local_config.node_add_prob = self.theta.node_add_prob
local_config.node_delete_prob = self.theta.node_delete_prob
local_config.conn_add_prob = self.theta.conn_add_prob
local_config.conn_delete_prob = self.theta.conn_delete_prob
...
super().mutate(local_config)
self.theta = self.theta.mutated()
```

Be careful: if `neat-python` stores mutation values inside `config.genome_config`, then copy and adapt the correct object. Do not mutate the global config object in place.

---

## Reproduction implementation

Create `EndogenousReproduction` in `endogenous_neat/reproduction.py`.

The first version may adapt the existing `DefaultReproduction` behaviour. Keep the changes minimal.

Main requirement:

- child genomes must use the parent genome’s own `theta` during mutation,
- sexual reproduction must inherit `theta` from both parents,
- asexual reproduction must inherit `theta` from the parent,
- then `theta` mutates slightly.

If the standard `DefaultReproduction` already calls `child.configure_crossover(...)` and `child.mutate(...)`, we can rely on our `EndogenousGenome` methods. Avoid unnecessary rewrites unless required.

Add comments explaining where endogenous control enters reproduction.

---

## Meta-output update

Create meta-output handling in `endogenous_neat/evaluation.py`.

The network should have:

```text
normal task outputs + meta outputs
```

For XOR:

- task outputs: 1
- meta outputs: 8
- total outputs: 9

During evaluation:

1. Activate the network for each XOR input.
2. Use the first output as the task output.
3. Collect the remaining 8 outputs as meta-outputs.
4. Average meta-outputs over all XOR cases.
5. Transform averaged meta-outputs into bounded `theta`.
6. Assign the resulting theta back to the genome after evaluation.

Important: this update happens between generations. It should not update `theta` after every individual activation in a way that makes evaluation order matter.

Example concept:

```python
outputs = net.activate(inputs)

task_output = outputs[0]
meta_outputs = outputs[1:]
```

Then:

```python
genome.theta = Theta.from_meta_outputs(mean_meta_outputs)
```

For the first version, use complete replacement:

```python
theta_next = Theta.from_meta_outputs(mean_meta_outputs)
```

Later we can test residual update:

```python
theta_next = blend(old_theta, proposed_theta, alpha=0.2)
```

But do not implement residual update unless it is behind a clear option.

---

## XOR experiment

Create `endogenous_neat/experiment_xor.py`.

The experiment should:

1. Load `configs/xor-endogenous.cfg`.
2. Use `neat.Config` with:
   - `EndogenousGenome`,
   - `EndogenousReproduction`,
   - default species set,
   - default stagnation.
3. Run XOR for a configurable number of generations.
4. Save:
   - best genome,
   - fitness history,
   - theta history,
   - final report.

Fitness:

```python
error = sum((expected - output) ** 2 for each XOR case)
fitness = 4.0 - error
```

Use standard XOR inputs:

```python
(0, 0) -> 0
(0, 1) -> 1
(1, 0) -> 1
(1, 1) -> 0
```

---

## Config file

Create `configs/xor-endogenous.cfg`.

Base it on the XOR configs in `neat-python/examples/xor/` (for example `config-feedforward`), but remember:

- `num_inputs = 2`
- `num_outputs = 9`

The first output is the XOR task output.

The remaining 8 outputs are meta-outputs controlling endogenous mutation parameters.

Use a population size around 150 for the first version.

Keep recurrent networks disabled initially unless there is a strong reason to enable them.

---

## Logging

Create `endogenous_neat/logging_utils.py`.

Log at least:

- generation number,
- best fitness,
- mean fitness,
- best genome’s theta,
- population mean theta,
- population standard deviation theta.

Save CSV files:

```text
runs/<timestamp>/fitness.csv
runs/<timestamp>/theta_best.csv
runs/<timestamp>/theta_population.csv
```

Also generate simple plots:

```text
runs/<timestamp>/fitness.png
runs/<timestamp>/theta_best.png
```

Do not over-engineer logging.

---

## Tests

Create pytest tests.

### `test_theta.py`

Test:

1. Default theta values are within bounds.
2. Mutation preserves bounds.
3. Meta-output transformation preserves bounds.
4. Crossover returns values within parent ranges or valid bounds.
5. Vector conversion round-trips correctly.

### `test_genome.py`

Test:

1. `EndogenousGenome` has theta after construction/configuration.
2. Mutation does not remove theta.
3. Crossover creates theta.
4. Distance includes a theta component.

### `test_reproduction.py`

Test:

1. Children receive theta.
2. Children’s theta remains bounded.
3. Different parent theta values can affect child mutation behaviour.

---

## Baseline support

Include a simple way to compare:

1. Standard NEAT with fixed global mutation parameters.
2. Endogenous NEAT with inherited but not meta-updated theta.
3. Endogenous NEAT with network-updated theta.

This can be done with a command-line option:

```bash
python -m endogenous_neat.experiment_xor --mode standard
python -m endogenous_neat.experiment_xor --mode inherited-theta
python -m endogenous_neat.experiment_xor --mode meta-theta
```

If full baseline support is too much for the first commit, implement `meta-theta` first but keep the code structured so the other modes can be added easily.

---

## Command-line interface

Add CLI options:

```bash
python -m endogenous_neat.experiment_xor \
  --config configs/xor-endogenous.cfg \
  --generations 100 \
  --runs 10 \
  --mode meta-theta \
  --output-dir runs/
```

Required options:

- `--generations`
- `--runs`
- `--mode`
- `--output-dir`

Reasonable defaults are acceptable.

---

## Code quality requirements

Use:

- type hints,
- dataclasses where appropriate,
- clear names,
- comments explaining research logic,
- simple functions,
- no unnecessary abstractions,
- no hidden global state,
- deterministic seeds where possible.

Add a `README.md` explaining:

1. What endogenous NEAT means here.
2. What is implemented.
3. What is not implemented.
4. How to install the local `neat-python/` fork and run XOR.
5. How to interpret logs.
6. How this differs from standard NEAT.

---

## Important conceptual constraints

Do not claim this is full endogenous NEAT.

Use this terminology:

- “weak endogeneity”
- “self-adaptive NEAT”
- “endogenous operator-control parameters”
- “stepping stone toward full endogenous NEAT”

Avoid claiming:

- “fully self-replicating neural networks”
- “complete endogenous evolution”
- “the network implements its own reproduction”
- “full internal mutation algorithm”

The correct claim is:

> The reproduction machinery is still externally implemented, but selected operator-control parameters are encoded in the genome, inherited, mutated, selected, and modified by the neural network’s meta-outputs between generations.

---

## First milestone

The first milestone is complete when:

1. XOR runs successfully.
2. The project can run standard NEAT and endogenous NEAT modes.
3. Each genome has its own `theta`.
4. `theta` changes over generations.
5. `theta` values are logged.
6. The code has tests for theta bounds, inheritance, mutation, and meta-output transformation.
7. The README explains the research interpretation accurately.

---

## Suggested commit sequence

### Commit 1

Set up project structure, wire in the local `neat-python/` fork (editable install or path dependency), README skeleton, and XOR config.

### Commit 2

Implement `Theta` dataclass, bounds, mutation, crossover, vector conversion, and tests.

### Commit 3

Implement `EndogenousGenome`.

### Commit 4

Implement `EndogenousReproduction` or confirm that `DefaultReproduction` can be used safely with `EndogenousGenome`.

### Commit 5

Implement XOR evaluation and meta-output update.

### Commit 6

Implement CLI and logging.

### Commit 7

Add baseline modes.

### Commit 8

Add plots and final documentation.

---

## Development warning

Be very careful not to mutate the shared NEAT config object in place when applying per-genome mutation parameters.

Bad:

```python
config.node_add_prob = genome.theta.node_add_prob
```

Good:

```python
local_config = copy.copy(config)
local_config.node_add_prob = genome.theta.node_add_prob
```

or better, create a helper that safely creates a per-genome mutation config.

---

## Final expected result

After implementation, this command should work:

```bash
python -m endogenous_neat.experiment_xor \
  --config configs/xor-endogenous.cfg \
  --generations 100 \
  --runs 10 \
  --mode meta-theta \
  --output-dir runs/
```

It should produce:

```text
runs/<timestamp>/
  fitness.csv
  theta_best.csv
  theta_population.csv
  fitness.png
  theta_best.png
  best_genome.pkl
  summary.txt
```

The summary should clearly state whether XOR was solved and how the endogenous parameters evolved.
