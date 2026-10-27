# Endogenous-Parameter NEAT

A research prototype of **weak endogenous NEAT** (self-adaptive NEAT with endogenous operator-control parameters) built on a local fork of [neat-python](neat-python/).

## What this is

Classical NEAT evolves topology and weights using fixed mutation parameters λ defined in the config file. Here each genome also carries **theta (θ)** — per-individual mutation-control parameters that are:

- encoded in the genome,
- inherited at crossover,
- mutated between generations,
- optionally updated by network **meta-outputs** after evaluation (in `meta-theta` mode),
- or replaced by a uniform random draw after evaluation (in `random-theta` mode), the random-change control for those meta updates,
- used during reproduction instead of only global config values.

This is a **stepping stone** toward full endogenous NEAT, not a self-replicating neural network. Reproduction machinery is still implemented externally; only selected operator-control parameters are endogenous.

## What is implemented (Phase 1)

- `Theta` dataclass with bounds, mutation, crossover, and meta-output mapping
- `EndogenousGenome` extending `neat.DefaultGenome`
- `EndogenousReproduction` (thin wrapper over `DefaultReproduction`)
- Tasks: **XOR**, **single-pole balancing**, **sum-of-squares** fitting, **guess-word**, and **dynamic-nk** (NK landscape with scheduled K changes / perturbations), selected with `--task`
- Comparison modes per task (`standard` / `inherited-theta` / `meta-theta` / `random-theta`)
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
pip install -e ".[ui]"    # optional local run browser
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
python -m endogenous_neat.experiment --task sum-squares ...
python -m endogenous_neat.experiment --task guess-word --answer neat ...
python -m endogenous_neat.experiment --task dynamic-nk --config configs/nk-smoke.cfg ...
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

All modes share the same I/O layout (matched topology capacity): task inputs + θ inputs, and 9 outputs. In `standard`, θ inputs are the fixed config defaults. In `inherited-theta` / `meta-theta` / `random-theta`, θ inputs track `genome.theta`. After a meta rewrite, the next evaluation uses the updated values. In `random-theta` the rewrite is a uniform draw within the θ bounds instead of the meta outputs, so the network is the same and only the source of the between-generation θ change differs. In `standard`, `inherited-theta`, and `random-theta`, meta heads exist but are ignored for θ updates.

```bash
python -m endogenous_neat.experiment --task pole --mode standard --generations 100 --runs 1
python -m endogenous_neat.experiment --task pole --mode meta-theta --generations 200 --runs 5 \
  --output-dir runs/pole/
```

### Sum-of-squares (function fitting)

Config: `configs/sum-squares-endogenous.cfg`

- Inputs: **10** (2 continuous \(x_1,x_2 \in [-1,1]\) + 8 normalized current θ)
- Outputs: 9 (output 0 = fitted value; outputs 1–8 = meta / θ)
- Target: \(t = (x_1^2 + x_2^2) / 2\) on a fixed \(5\times 5\) grid (25 cases)
- Fitness: \(F = 25 - \sum(\hat y - t)^2\) (maximize; perfect = 25)
- Solved when fitness ≥ 24.5

Same matched I/O layout as XOR (10-in / 9-out).

```bash
python -m endogenous_neat.experiment --task sum-squares --mode standard --generations 100 --runs 1
python -m endogenous_neat.experiment --task sum-squares --mode compare \
  --generations 100 --runs 10 --output-dir runs/sos_compare/
python -m endogenous_neat.experiment --task sum-squares --mode compare-random-lambda \
  --generations 100 --runs 10 --seed 0 --output-dir runs/sos_random_lambda/
python -m endogenous_neat.experiment --task sum-squares \
  --config configs/sum-squares-endogenous.cfg \
  --mode meta-theta-ablation \
  --generations 2000 --runs 50 --seed 0 \
  --output-dir final_runs/sos_ablation_2000_50/
```

`--mode meta-theta-ablation` is the single-component sufficiency test. It launches full meta-theta plus one arm per θ field (nine arms). In each single-field arm that field is rewritten from its meta output and the other seven stay at the hand-tuned defaults.

### Guess a word

Config: `configs/guess-word-endogenous.cfg`

- Inputs: **8** (normalized current θ only; the spelled word does not depend on a task input)
- Outputs: a variable number of letter nodes, then 8 fixed meta outputs
- Alphabet: `a`–`z` and space. Each letter activation maps onto that alphabet
- Fitness: negative Levenshtein distance to the target word (insert, delete, or replace costs 1). Exact spelling scores 0
- Solved when fitness ≥ 0
- Letter nodes start at length 1 and are added or deleted with `output_add_prob` / `output_delete_prob` (config constants, not part of θ), up to `max_word_length`
- Target word: `--answer` (default `neat`). Letters `a`–`z` and spaces; longer than `max_word_length` is rejected

```bash
python -m endogenous_neat.experiment --task guess-word --mode standard --answer neat --generations 100 --runs 1
python -m endogenous_neat.experiment --task guess-word --mode compare --answer "hello world" \
  --generations 200 --runs 5 --output-dir runs/guess_word/
python -m endogenous_neat.experiment --task guess-word --mode random-theta \
  --answer "michael soumen ingo" --generations 250 --runs 20 \
  --config configs/guess-word-endogenous.cfg \
  --output-dir final_runs/guess_word_random_theta/
```

### Dynamic NK landscape

Configs: `configs/nk-smoke.cfg`, `configs/nk-dynamic-v1.cfg` (default), `configs/nk-perturb.cfg`, `configs/nk-combined.cfg`

An NK landscape (Kauffman) is a tunable rugged fitness function over `N` binary loci. Each locus `i` contributes `f_i(bit_i, bits of its K dependencies)`, looked up in a table of `2^(K+1)` uniform `[0, 1)` values drawn once per landscape; fitness is the mean contribution, so it lies in `[0, 1]`. `K = 0` is a smooth additive landscape with one optimum; raising `K` increases epistasis (how much one bit's contribution depends on others) and the number of local optima, until `K = N - 1` is fully random. Dependencies are sampled without replacement and never include the locus itself.

- Inputs: **8** (normalized current θ only, like guess-word)
- Outputs: `N` bit outputs then 8 meta outputs, so `num_outputs = N + 8`
- Encoding: each bit output's sigmoid activation is thresholded at 0.5 (`bit = 1 if value >= 0.5 else 0`)
- `fitness_threshold = 1.0` is never reached; the configs set `no_fitness_termination = True` so every run uses the full `--generations` budget
- For `N <= exact_optimum_max_n` (default 20) the exact optimum is found by enumeration and logged per landscape version; for larger `N` it is `null`

The landscape can change during a run on a fixed, generation-indexed schedule (no performance-triggered curriculum, no population reset):

- `k_schedule = 0:0, 100:2, 200:4` — at generation 100 a **new** landscape with `K = 2` is drawn, and so on. A static control uses `initial_k` with no schedule.
- `perturbations = 100:0.10, 200:0.20` — at generation 100, `round(N * 0.10)` loci get fresh contribution tables; `K` and all dependencies stay the same.
- A change is applied before that generation is evaluated, so the generation-100 fitness already reflects the new landscape. Each change is logged as an event (`k_change` or `landscape_perturbation`) together with the previous generation's genomes re-scored on the new landscape (`same_population_*`), so the pre/post response of the same population is recorded.

Landscape randomness is independent of evolution: landscapes are drawn from `hash(landscape_seed, --seed, run_index, kind, version)` only, so all four compare arms of one run index see identical landscapes, and evolutionary RNG use cannot shift them. Set `landscape_seed` in `[NK]` or `--nk-landscape-seed` to get a different sequence.

CLI overrides (for sweeps; `--nk-n` also resizes the output layer): `--nk-n`, `--nk-k` (static, clears any schedule), `--nk-k-schedule "0:0, 100:2"`, `--nk-perturbations "150:0.10"`, `--nk-landscape-seed`.

```bash
# smoke test (N=16, K 0 -> 2 -> 4 at generations 20/40, 60 generations, ~30 s)
python -m endogenous_neat.experiment --task dynamic-nk --config configs/nk-smoke.cfg \
  --mode compare --generations 60 --runs 1 --seed 0 --output-dir runs/nk_smoke/

# main Dynamic NK v1 (N=64, K 0 -> 2 -> 4 -> 8 -> 12 every 100 generations)
python -m endogenous_neat.experiment --task dynamic-nk --config configs/nk-dynamic-v1.cfg \
  --mode compare --generations 500 --runs 10 --seed 0 --output-dir runs/nk_dynamic_v1/

# static control at a fixed K, same landscape seed
python -m endogenous_neat.experiment --task dynamic-nk --config configs/nk-dynamic-v1.cfg \
  --nk-k 4 --mode compare --generations 500 --runs 10 --seed 0 --output-dir runs/nk_static_k4/

# dynamic environment: fixed K=6, 10%/10%/20%/20% perturbations
python -m endogenous_neat.experiment --task dynamic-nk --config configs/nk-perturb.cfg \
  --mode compare --generations 500 --runs 10 --seed 0 --output-dir runs/nk_perturb/

# combined K ramp plus perturbations
python -m endogenous_neat.experiment --task dynamic-nk --config configs/nk-combined.cfg \
  --mode compare --generations 500 --runs 10 --seed 0 --output-dir runs/nk_combined/

# complexity sweep over N and K (static), one output dir per cell
for n in 16 32 64 128; do for k in 0 2 4 8 12; do
  python -m endogenous_neat.experiment --task dynamic-nk --config configs/nk-dynamic-v1.cfg \
    --nk-n $n --nk-k $k --mode compare --generations 300 --runs 5 --seed 0 \
    --output-dir runs/nk_sweep/n${n}_k${k}/
done; done
```

Per run, NK adds `nk.csv` (generation, `n`, `k`, `landscape_version`, `stage_seed`, `global_optimum_fitness`, best/mean/median fitness), `nk_events.jsonl` and `nk_settings.json`. The comparison adds change markers to `compare_fitness.png` and `compare_theta.png`, plus `compare_mean_fitness.png`, `compare_nk_schedule.png`, `compare_recovery.csv` and `compare_recovery_summary.csv` (per change and mode: pre/post-change best, drop, and `recovery_generations_95`, the generations until best fitness is back to 95% of the pre-change best; empty/censored if never within the run). The recovery table is also appended to `compare_summary.txt`.

### Compare modes

`--mode compare` runs the same five-arm comparison for whichever `--task` you pass. Use a **task-specific** `--output-dir` so tasks do not mix. `--seed` fixes the random-λ draw as `seed + run_index`.

```bash
python -m endogenous_neat.experiment \
  --task xor \
  --mode compare \
  --generations 100 \
  --runs 10 \
  --seed 0 \
  --output-dir runs/xor_compare/

python -m endogenous_neat.experiment \
  --task guess-word \
  --mode compare \
  --answer neat \
  --generations 200 \
  --runs 10 \
  --seed 0 \
  --output-dir runs/guess_word_compare/
```

Arms:

- `standard` — classical NEAT frozen at the known-good config λ
- `standard-random` — classical NEAT frozen at a uniform-random λ, one draw per run
- `inherited-theta` and `meta-theta` — usual initialization
- `random-theta` — the meta-theta network, with θ replaced by a uniform draw after each evaluation

The solve threshold is the task's own (`xor` 3.9, `pole` 60, `sum-squares` 24.5, `guess-word` 0). Pass the same `--task` to `--compare-only` so the threshold matches the runs.

To rebuild comparison plots from existing run directories without re-evolving:

```bash
python -m endogenous_neat.experiment --task pole --compare-only --output-dir runs/pole_compare/
```

| Comparison | Question it answers |
|------------|---------------------|
| standard vs standard-random | Does the hand-tuned λ beat a frozen random λ? |
| standard vs inherited-theta | Does per-genome θ help vs fixed global λ? |
| inherited-theta vs meta-theta | Do network meta-output θ updates help or hurt? |
| meta-theta vs random-theta | Do network-chosen θ updates differ from a uniform redraw each generation? |
| standard vs meta-theta | Overall weak endogenous NEAT vs classical NEAT at the known-good λ |
| compare-random-lambda | Can meta-θ recover from a shared bad random λ that standard freezes? |

### Random-λ stress test (standard vs meta-theta)

Each paired run draws one uniform-in-bounds θ vector. **Standard** freezes it as global λ (and θ inputs) for the whole run. **Meta-theta** starts all genomes from the same vector, then may rewrite θ from network meta-outputs after each eval.

```bash
python -m endogenous_neat.experiment \
  --task pole \
  --mode compare-random-lambda \
  --generations 200 \
  --runs 20 \
  --seed 0 \
  --output-dir runs/pole_random_lambda/
```

With `--seed`, θ for `run_index` is deterministic (`seed + run_index`), so standard and meta share the same start. `run_meta.txt` records the initial θ fields and L1 distance from defaults; the comparison summary reports whether meta left that start.

### Start from a saved genome

`--seed-genomes` takes one or more `best_genome.pkl` files. The genome's class must match the arm (`DefaultGenome` or `WordGenome` for classical NEAT, `EndogenousGenome` or `WordEndogenousGenome` for the endogenous arms). In `--mode compare`, a file is used only on the arms it matches; the others start as usual. A single-mode run rejects a file of the wrong class.

`--seed-fill elite` inserts one copy of each saved genome and leaves the rest of the population random. `--seed-fill clones` fills the population with copies (split evenly; any remainder goes to the earliest files). Copies are evaluated again at generation 0. If a saved genome is already a solution, the run stops there unless you pass `--continue-after-solution`.

```bash
python -m endogenous_neat.experiment \
  --task xor \
  --mode meta-theta \
  --seed-genomes runs/xor_compare/20260101_120000_run0_meta-theta/best_genome.pkl \
  --seed-fill elite \
  --continue-after-solution \
  --generations 100 \
  --output-dir runs/xor_from_seed/
```

### Run browser

The optional UI launches the same command, lists saved genomes (task, mode, fitness, size, and which modes each genome can seed), and plots fitness, θ, and network structure for a finished run.

```bash
pip install -e ".[ui]"
python -m endogenous_neat.ui
```

## Output files

Each run writes to `runs/<timestamp>_run<N>_<mode>/` (or under your `--output-dir`):

| File | Contents |
|------|----------|
| `fitness.csv` | Best and mean fitness per generation |
| `theta_best.csv` | Best-genome θ per generation (endogenous modes) or fixed λ (standard) |
| `theta_population.csv` | Population mean/std θ per generation (constant in standard mode) |
| `fitness.png` | Fitness plot |
| `theta_best.png` | Best θ trajectories |
| `best_genome.pkl` | Pickled best genome (can be passed to `--seed-genomes`) |
| `structure.csv` | Best-genome size each generation: nodes, hidden, enabled and disabled connections, species, letter nodes |
| `structure_best.jsonl` | Best-genome graph each generation, for the run browser's network slider. Guess-word letter nodes are labeled with the character they spell |
| `words.csv` | Guess-word only: the best genome's spelled word and its fitness score each generation |
| `nk.csv`, `nk_events.jsonl`, `nk_settings.json` | Dynamic-nk only: per-generation K / landscape version / optimum, landscape-change events, and the resolved NK settings |
| `summary.txt` | Whether the task was solved and final θ |
| `run_meta.txt` | task, mode, generations, config; `theta_initial_*` is the starting θ and `theta_final_*` is the θ in force at the end; `seed_genomes` / `seed_fill` when a run was seeded |

`--mode compare` / `--compare-only` also write to `<output-dir>/comparison/`:

| File | Contents |
|------|----------|
| `compare_fitness.png` | Median best-fitness ± IQR overlay for all modes |
| `compare_theta.png` | Best-genome θ trajectories for endogenous modes |
| `compare_summary.txt` | Solve rates and final fitness stats per mode |
| `compare_stats.csv` | Same stats in CSV form |
| `compare_mean_fitness.png`, `compare_nk_schedule.png`, `compare_recovery*.csv` | Dynamic-nk only: mean population fitness with change markers, the K / version schedule, and recovery metrics |

Solve thresholds come from the task: XOR ≥ 3.9; pole ≥ 60.0; sum-squares ≥ 24.5; guess-word ≥ 0; dynamic-nk 1.0 (never reached; runs use the full generation budget).

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
| Mutation rates | Global config λ | Per-genome θ (inherited, mutated gently) |
| θ as inputs | Fixed defaults (matched capacity) | Live `genome.theta`, normalized to [0, 1] |
| θ mutation | N/A | log-normal on rates; cap on `weight_mutate_power` (default 1.5) |
| θ in speciation | N/A | Off by default (`theta_distance_coefficient = 0`) |
| Meta-outputs | N/A | 8 sigmoid outputs mapped linearly into θ bounds (meta-theta mode) |
| Speciation distance | Topology/weights only | Adds small θ component |
| Reproduction | External | Still external |

