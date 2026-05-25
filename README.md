# Automaton Distillation: Neuro-Symbolic Transfer Learning for Deep Reinforcement Learning

Source code for the TMLR camera-ready submission.

## Quick Start

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate autd

# 2. Verify installation
python -c "import src.control_env_debug; print('OK')"

# 3. Run a quick test (all envs, 500 steps)
python -m src.control_env_debug.run_all_benchmarks \
    --total_steps 500 --n_envs 1 --seeds 0

# 4. Run full benchmark for a single environment
./run_patrol.sh 100000 "0 1 2" 4 bench
```

## Environments

All environments are wrapped with an **LTLf automaton** that tracks task progress through a automaton. The teacher trains on an easier source configuration; the student trains on a harder target configuration.

| Environment | Source (Teacher) | Target (Student) | LTLf Formula |
|---|---|---|---|
| `patrol` | HalfCheetah a=5, b=−2 | a=8, b=−5 | F(a ∧ F(b)) |
| `flatworld_patrol` | Default circles, no walls | Shifted circles + cross obstacle | F(a ∧ F(b)) |
| `flatworld_sequence` | Default circles, no walls | Shifted circles + corridors | F(a ∧ F(b ∧ F(c))) |
| `zones_patrol` | PointLtl1-v0 | CarLtl1-v0 | F(a ∧ F(b)) |
| `zones_sequence` | PointLtl2-v0 | CarLtl2-v0 | F(b ∧ F(a ∧ F(c ∧ F(d)))) |

## Methods

| Method | Key | Description |
|---|---|---|
| Scratch | `td3_base` | TD3 trained from scratch on target env |
| CRM | `td3_crm` | TD3 + Counterfactual Reward Machine experiences |
| Static Distill | `td3_static` | Fixed-weight Q-automaton distillation |
| **Dynamic Distill** | `td3_dynamic` | **Proposed** — annealing Q-automaton distillation |
| Reward Shaping | `td3_shaped` | Teacher reward shaping during rollout |
| C-PREP | `td3_cprep` | Warm-start student weights from teacher model |

## Usage

### Per-Environment Scripts

Each script runs the full pipeline (teacher + all 6 student methods):

```bash
# Arguments: steps seeds n_envs prefix
./run_patrol.sh 100000 "0 1 2" 4 bench
./run_flatworld_patrol.sh 100000 "0 1 2" 4 bench
./run_flatworld_sequence.sh 100000 "0 1 2" 4 bench
./run_zones_patrol.sh 100000 "0 1 2" 4 bench
./run_zones_sequence.sh 100000 "0 1 2" 4 bench
```

Scripts can run in parallel across terminals — each environment is independent.

### Full Benchmark

```bash
# All environments, all methods, 3 seeds
python -m src.control_env_debug.run_all_benchmarks \
    --total_steps 100000 --seeds 0 1 2 --n_envs 4

# Specific environment and methods
python -m src.control_env_debug.run_all_benchmarks \
    --envs patrol flatworld_patrol \
    --methods td3_base td3_dynamic td3_cprep
```

### Single Training Run

```bash
python -m src.control_env_debug.train_vectorized_td3 \
    --env_type patrol \
    --total_steps 100000 \
    --n_envs 4 \
    --run_name my_experiment \
    --a_threshold 5 --b_threshold -2 \
    --distill_mode dynamic \
    --teacher_run_name my_teacher
```

### Plotting Results

```bash
python -m src.control_env_debug.plot_mean_std \
    --prefix bench --envs patrol flatworld_patrol
```

## Outputs

All runs save to `logs/<run_name>/`:
- `td3_model_{actor,critic_1,critic_2}.pth` — trained network weights
- `episode_returns.npy`, `episode_lengths.npy` — per-episode metrics
- `vecnormalize.pkl` — observation normalisation statistics
- `episodes.csv` — step-level diagnostics

Q-automata are saved to `automaton_q/<run_name>.json`.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- MuJoCo 2.3+ (for HalfCheetah and Zones)
- See `requirements.txt` for full list

## Citation

```bibtex
@article{automaton-distillation,
  title={Automaton Distillation: Neuro-Symbolic Transfer Learning for Deep Reinforcement Learning},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2026}
}
```
