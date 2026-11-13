# Sensor calibration from one labeled reference

A reference reading of +1 is labeled light or dark. Given a new signed reading, the network predicts whether it means light under the same wiring. The target is light when the reference label equals the indicator that the new reading is positive. Both wirings occur for every training and test reading. Weights and the eight control inputs are fixed throughout an individual's evaluation.

The [supplementary report (PDF)](results/sensor_experiment_report.pdf) gives the fitness trajectories, timing by seed, held-out scores, training data, network diagrams and worked predictions. The [manuscript figure](figures/sensor_case_study.pdf) places all three performance measures beside the two evolved prediction circuits.

## Experiment

The five variants used population seeds 1–10, 120 networks per generation and 1,000 generations without early stopping. All use the original fully connected feed-forward starting architecture with ten inputs, nine outputs, and no hidden nodes. A genome is scored on 32 cases by one minus the mean squared prediction error. The constant prediction 0.5 earns fitness 0.75. A run counts as crossing the plateau the first time its training fitness exceeds 0.80. The champion that set the training record is replayed with its original control inputs to classify 64 new cases.

| Condition | Crossed by 100 | Crossed by 1,000 | Median best fitness at 1,000 | All 64 test cases correct |
|---|---:|---:|---:|---:|
| Standard | 1/10 | 4/10 | 0.7500 | 2/10 |
| Standard-random, one vector per run | 1/10 | 4/10 | 0.7500 | 0/10 |
| Inherited θ | 0/10 | 4/10 | 0.7500 | 4/10 |
| Random θ, refreshed after evaluation | 0/10 | 1/10 | 0.7500 | 1/10 |
| Meta-θ | 0/10 | 7/10 | 0.9983 | 6/10 |

The experiment uses the implementation at [parent revision 12dcffd](https://github.com/terolahtinen/endogenous-neat/tree/12dcffd7ed95e980196e55a50ced14d9e23ec441) and [NEAT fork revision 7d9acae](https://github.com/terolahtinen/neat-python/tree/7d9acae0a1e5d7ed6199ffbd525dd15c5c87a86f). The driver uses the configuration in `configs/xor-endogenous.cfg` and checks both revisions' implementation content. Population, control initialisation/mutation and random control rewrites have separate deterministic streams. The control streams use seeds 1,000,000 + population seed and 2,000,000 + population seed, respectively. Each run uses 120,000 training genome evaluations.

## Reproduce

Start from a clone of the repository. The recorded experiment used Python 3.12.2, NumPy 1.26.4 and Matplotlib 3.8.3. From the repository root:

~~~sh
git submodule update --init --recursive
cd experiments/sensor_calibration
python -m pip install -r requirements.txt
~~~

Rebuild all result tables and the five-page supplementary report using the archived runs:

~~~sh
python make_report.py --runs checkpoints --ref reference100 --data data --out regenerated
~~~

Run evolution again with the same five conditions and seeds, and compare all 50 histories with the original 100-generation reference prefixes while building the report:

~~~sh
python run_sensor.py --repo ../.. --out rerun --workers 4
python extract_architecture.py --fork ../../neat-python --scorer . --runs rerun --at100 reference100/meta-theta_02 --out replay
python make_report.py --runs rerun --ref reference100 --data replay --out regenerated
~~~

For a short check of the normal evolutionary entry point, use one arm, one seed and 100 generations:

~~~sh
python run_sensor.py --repo ../.. --out quick --arms meta-theta --seeds 2 --generations 100 --workers 1
~~~

To re-evaluate the 50 saved networks, verify their recorded training and test scores, extract the two illustrated task circuits, and verify their output against the complete networks on all 96 cases:

~~~sh
python extract_architecture.py --fork ../../neat-python --scorer . --runs checkpoints --at100 reference100/meta-theta_02 --out replay
python extract_predictions.py --fork ../../neat-python --at100 reference100/meta-theta_02/champion.pkl --at1000 checkpoints/meta-theta_02/champion.pkl --out replay/seed2_predictions.csv
~~~

To rebuild the three architecture figures and the manuscript panel from the archived data:

~~~sh
python architecture_figures.py
python paper_figure.py
~~~

The supplement-only commit retains the implementation used by the experiments. If using a later repository revision with changed production code, run the adapter from this supplement and point its `--repo` argument at a separate checkout of the pinned parent revision with its submodule initialised.

## Files

- [Generation records](results/fitness_by_generation.csv), [seed-level endpoints and first crossings](results/fitness_by_seed.csv), [variant summaries](results/summary.csv) and [generation checkpoints](results/checkpoints.csv).
- [Fifty complete histories, run summaries and network checkpoints](checkpoints). Network checkpoints were produced by the experiment and contain the evaluated phenotype and its pre-rewrite control inputs. Only load trusted Python pickle files.
- [Training cases](data/training_cases.csv), [held-out cases](data/test_cases.csv), [four worked predictions](data/worked_sensor_cases.csv), [all 50 circuit sizes](data/champion_architectures.csv) and [full-precision illustrated networks](data/sensor_networks.json).
- [Original three-panel performance plot](results/sensor_fitness.pdf), [weighted prediction circuits](figures/champion_circuits.pdf), [training grid](figures/training_cases.pdf) and [entire Meta-θ seed-2 network](figures/full_champion_network.pdf).

In the network JSON, input −1 is the reading, −2 is the calibration label and −3 through −10 are the eight normalised control inputs. Output 0 is the predicted probability of light. Outputs 1–8 are mutation-control outputs. Hidden unit 43 is the illustrated Standard predictor. Units 7733 and 18945 are A and B of the illustrated Meta-θ predictor. The displayed circuits absorb frozen control inputs into the biases. A full-precision replay reproduces the original outputs within an absolute tolerance of 1e−13.
