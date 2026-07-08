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
- XOR experiment with three comparison modes
- CSV logging, plots, and pytest tests

## What is not implemented

- Full self-replication or internal reproduction algorithms
- TensorNEAT, HyperNEAT, CoDeepNEAT
- Residual theta blending (only full meta-output replacement)
- Full neat-python checkpoint integration

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

## Running the XOR experiment

All modes use the same 9-output config (`configs/xor-endogenous.cfg`): output 0 is XOR fitness; outputs 1–8 are meta-outputs (used only in `meta-theta` mode).

```bash
# Standard NEAT baseline (fixed global λ, meta outputs ignored)
python -m endogenous_neat.experiment_xor --mode standard --generations 100 --runs 1

# Endogenous theta: inherited and mutated, no meta-update
python -m endogenous_neat.experiment_xor --mode inherited-theta --generations 100 --runs 1

# Endogenous theta + network meta-output update
python -m endogenous_neat.experiment_xor --mode meta-theta --generations 100 --runs 10 --output-dir runs/
```

## Output files

Each run writes to `runs/<timestamp>_run<N>_<mode>/`:

| File | Contents |
|------|----------|
| `fitness.csv` | Best and mean fitness per generation |
| `theta_best.csv` | Best genome θ per generation (endogenous modes) |
| `theta_population.csv` | Population mean/std θ per generation |
| `fitness.png` | Fitness plot |
| `theta_best.png` | Best θ trajectories |
| `best_genome.pkl` | Pickled best genome |
| `summary.txt` | Whether XOR was solved and final θ |

XOR is considered solved when fitness ≥ 3.9.

## Interpreting logs

- **fitness.csv**: task performance; compare across modes with the same 9-output topology.
- **theta_best.csv / theta_population.csv**: how endogenous parameters drift under selection and (in meta-theta mode) meta-output pressure.
- If population θ std collapses toward bounds, meta-outputs may be saturating — compare with `inherited-theta` to isolate meta-update effects.

## Tests

```bash
pytest tests/ -v
```

## How this differs from standard NEAT

| Aspect | Standard NEAT | Endogenous NEAT |
|--------|---------------|-----------------|
| Mutation rates | Global config λ | Per-genome θ (inherited, mutated) |
| Meta-outputs | N/A | 8 outputs map to θ via sigmoid (meta-theta mode) |
| Speciation distance | Topology/weights only | Adds small θ component |
| Reproduction | External | Still external |

