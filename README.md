# Endogenous-Parameter NEAT

A research prototype of **weak endogenous NEAT** (self-adaptive NEAT with endogenous operator-control parameters) built on a local fork of [neat-python](neat-python/).

## What this is

Classical NEAT evolves topology and weights using fixed mutation parameters λ defined in the config file. Here each genome also carries **theta (θ)** — per-individual mutation-control parameters that are:

- encoded in the genome,
- inherited at crossover,
- mutated between generations,
- optionally updated by network **meta-outputs** after evaluation (in `meta-theta` mode),
- used during reproduction instead of only global config values.

This is a **stepping stone** toward full endogenous NEAT, not a self-replicating neural network. Reproduction machinery is still implemented externally; only selected operator-control parameters are endogenous.

## What is implemented (Phase 1)

- `Theta` dataclass with bounds, mutation, crossover, and meta-output mapping
- `EndogenousGenome` extending `neat.DefaultGenome`
- `EndogenousReproduction` (thin wrapper over `DefaultReproduction`)
- Tasks: **XOR** (default) and **single-pole balancing**, selected with `--task`
- Three comparison modes per task (`standard` / `inherited-theta` / `meta-theta`)
- `--mode compare` overlay plots and solve-rate summary across modes
- CSV logging, plots, and pytest tests

## What is not implemented

- Full self-replication or internal reproduction algorithms
- TensorNEAT, HyperNEAT, CoDeepNEAT
- Residual theta blending (only full meta-output replacement)
- Full neat-python checkpoint integration
- Double pendulum, memory, or CTRNN tasks

## Installation

This project depends on [neat-python](https://github.com/terolahtinen/neat-python) as a **git submodule** at `neat-python/`.

### Clone

Clone with submodules in one step:

```bash
git clone --recurse-submodules <repository-url>
cd endogenous-neat
```

If you already cloned without submodules:

```bash
cd endogenous-neat
git submodule update --init --recursive
```

### Install dependencies

```bash
pip install -e ./neat-python
pip install -e ".[dev]"
```

### Updating the neat-python fork

To move the submodule to a newer commit on the fork:

```bash
cd neat-python
git fetch origin
git checkout <commit-or-tag>   # e.g. a release tag or main
cd ..
git add neat-python
git commit -m "Bump neat-python submodule"
```

To see which commit the parent repo currently pins:

```bash
git submodule status neat-python
```

After updating the submodule, reinstall the editable package:

```bash
pip install -e ./neat-python
```

## Tasks

Select a task with `--task` (default `xor`). Prefer the unified entry point:

```bash
python -m endogenous_neat.experiment --task xor ...
python -m endogenous_neat.experiment --task pole ...
```

`python -m endogenous_neat.experiment_xor` still works and defaults to XOR.

### XOR

Config: `configs/xor-endogenous.cfg`

- Inputs: **10** (2 XOR bits + 8 normalized current θ)
- Outputs: 9 (output 0 = XOR answer; outputs 1–8 = meta / θ)
- Solved when fitness ≥ 3.9

```bash
python -m endogenous_neat.experiment --task xor --mode standard --generations 100 --runs 1
python -m endogenous_neat.experiment --task xor --mode inherited-theta --generations 100 --runs 1
python -m endogenous_neat.experiment --task xor --mode meta-theta --generations 100 --runs 10 --output-dir runs/xor/
```

### Single-pole balancing

Config: `configs/pole-endogenous.cfg`

- Inputs: **12** (4 scaled cart-pole state + 8 normalized current θ)
- Outputs: 9 (output 0 = discrete force; outputs 1–8 = meta / θ)
- Fitness: worst balance time (seconds) across 5 random starts, capped at 60 s
- Solved when fitness ≥ 60.0

All modes share the same I/O layout (matched topology capacity): task inputs + θ inputs, and 9 outputs. In `standard`, θ inputs are the fixed config defaults. In `inherited-theta` / `meta-theta`, θ inputs track `genome.theta`; after a meta rewrite, the next evaluation uses the updated values. In `standard` and `inherited-theta`, meta heads exist but are ignored for θ updates.

```bash
python -m endogenous_neat.experiment --task pole --mode standard --generations 100 --runs 1
python -m endogenous_neat.experiment --task pole --mode meta-theta --generations 200 --runs 5 \
  --output-dir runs/pole/
```

### Compare modes (standard vs weak endogenous)

Run all three modes with the same settings and write overlay comparison plots. Use a **task-specific** `--output-dir` so XOR and pole runs do not mix:

```bash
python -m endogenous_neat.experiment \
  --task xor \
  --mode compare \
  --generations 100 \
  --runs 10 \
  --output-dir runs/xor_compare/

python -m endogenous_neat.experiment \
  --task pole \
  --mode compare \
  --generations 200 \
  --runs 10 \
  --output-dir runs/pole_compare/
```

This runs `standard`, `inherited-theta`, and `meta-theta`, then writes comparison artifacts under `<output-dir>/comparison/`.

To rebuild comparison plots from existing run directories without re-evolving:

```bash
python -m endogenous_neat.experiment --task pole --compare-only --output-dir runs/pole_compare/
```

| Comparison | Question it answers |
|------------|---------------------|
| standard vs inherited-theta | Does per-genome θ help vs fixed global λ? |
| inherited-theta vs meta-theta | Do network meta-output θ updates help or hurt? |
| standard vs meta-theta | Overall weak endogenous NEAT vs classical NEAT |

## Output files

Each run writes to `runs/<timestamp>_run<N>_<mode>/` (or under your `--output-dir`):

| File | Contents |
|------|----------|
| `fitness.csv` | Best and mean fitness per generation |
| `theta_best.csv` | Best genome θ per generation (endogenous modes) |
| `theta_population.csv` | Population mean/std θ per generation |
| `fitness.png` | Fitness plot |
| `theta_best.png` | Best θ trajectories |
| `best_genome.pkl` | Pickled best genome |
| `summary.txt` | Whether the task was solved and final θ |
| `run_meta.txt` | task, mode, generations, config |

`--mode compare` / `--compare-only` also write to `<output-dir>/comparison/`:

| File | Contents |
|------|----------|
| `compare_fitness.png` | Median best-fitness ± IQR overlay for all modes |
| `compare_theta.png` | Best-genome θ trajectories for endogenous modes |
| `compare_summary.txt` | Solve rates and final fitness stats per mode |
| `compare_stats.csv` | Same stats in CSV form |

Solve thresholds: XOR ≥ 3.9; pole ≥ 60.0.

## Interpreting logs

- **fitness.csv**: task performance; compare across modes with the same matched-capacity topology.
- **theta_best.csv / theta_population.csv**: how endogenous parameters drift under selection and (in meta-theta mode) meta-output pressure.
- If population θ std collapses toward bounds, meta-outputs may be saturating — compare with `inherited-theta` to isolate meta-update effects.
- **compare_fitness.png**: side-by-side learning curves; higher median and earlier threshold crossing favor a mode.

## Tests

```bash
pytest tests/ -v
```

## How this differs from standard NEAT

| Aspect | Standard NEAT | Endogenous NEAT |
|--------|---------------|-----------------|
| Mutation rates | Global config λ | Per-genome θ (inherited, mutated) |
| θ as inputs | Fixed defaults (matched capacity) | Live `genome.theta`, normalized to [0, 1] |
| Meta-outputs | N/A | 8 sigmoid outputs mapped linearly into θ bounds (meta-theta mode) |
| Speciation distance | Topology/weights only | Adds small θ component |
| Reproduction | External | Still external |

