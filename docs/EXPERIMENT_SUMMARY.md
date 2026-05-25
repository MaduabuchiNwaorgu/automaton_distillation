# Automaton Distillation Benchmark Results

## Experiment Overview

We conducted a comprehensive benchmarking study evaluating six teacher-to-student knowledge transfer methods across four environments. The study measures how effectively different distillation approaches can transfer knowledge from a teacher agent trained on an easier task to a student agent facing a more difficult variant of the same task.

## Environments

### 1. **FlatWorld Patrol** (1M steps)
- **Teacher Configuration**: Default circle layout, no obstacles
- **Student Configuration**: Shifted circles + cross wall obstacles
- **Domain**: 2D grid world with point navigation and temporal logic constraints
- **Objective**: Navigate between two target locations sequentially
- **Difficulty Difference**: Walls and shifted goal positions create spatial complexity

### 2. **FlatWorld Sequence** (1M steps)
- **Teacher Configuration**: Default circles, no walls
- **Student Configuration**: Shifted circles + corridor obstacles
- **Domain**: 2D grid world with temporal logic objectives
- **Objective**: Visit three target locations in sequence: F(a ∧ F(b ∧ F(c)))
- **Difficulty Difference**: Corridors create path constraints; more goals increase complexity

### 3. **HalfCheetah Patrol** (2M steps)
- **Teacher Configuration**: Easy thresholds (a=5.0 right, b=-2.0 left)
- **Student Configuration**: Harder thresholds (a=8.0 right, b=-5.0 left)
- **Domain**: Continuous control MuJoCo environment
- **Objective**: Satisfy two temporal logic predicates reachability bounds
- **Difficulty Difference**: Tighter bounds require longer/faster movement

### 4. **Zones Sequence** (2M steps)
- **Teacher Configuration**: PointLtl2-v0 (easier point robot dynamics)
- **Student Configuration**: CarLtl2-v0 (constrained car dynamics)
- **Domain**: Safety Gymnasium continuous control with spatial constraints
- **Objective**: Visit four zones in sequence: F(b ∧ F(a ∧ F(c ∧ F(d))))
- **Difficulty Difference**: Car dynamics are more constrained than point robot; longer horizon

---

## Methods Tested

### Baseline
- **Vanilla TD3**: Scratch training with no knowledge transfer (baseline)

### Proposed Methods
1. **CRM (Counterfactual Replay Mechanism)**
   - Uses teacher's behavior to generate counterfactual experiences
   - No explicit weight/output transfer
   
2. **Static Distill** (Static Q-Automaton Distillation)
   - Fixed weight (β=0.05) blending of student and teacher Q-functions
   - Teacher control: static throughout training
   
3. **Dyn Distill** (Dynamic Q-Automaton Distillation)
   - Decaying weight from w₀=0.5 → w_min=0.0 over 20% of training
   - Teacher influence gradually fades as student learns
   
4. **ProductMDP** (Reward Shaping)
   - Teacher reward shaping with scale factor (w=0.05)
   - No explicit distillation; teacher used to shape student rewards
   
5. **CPREP** (C-PREP: Initialization from Teacher Weights)
   - Student initialized from teacher's neural network weights
   - Fine-tuned end-to-end on harder task
   - No online teacher interaction

---

## Experimental Setup

- **Model Size**: Tiny networks (68K parameters actor, 68K parameters critic)
- **Training Algorithm**: TD3 (Twin Delayed DDPG)
- **Batch Size**: 100
- **Vectorized Environments**: 4 parallel environments per job
- **Training Frequency**: Every 2 steps (train_freq=2)
- **Seeds**: 8 independent random seeds (0-7) for main results
- **GPU**: Distributed across 4× NVIDIA A40 GPUs (46GB each)
- **Hardware**: 64 CPU cores, 1TB RAM

---

## Results Location

### Publication-Quality Plots
Individual environment plots are located in **`paper_plots/`**:
- `paper_plots/patrol.pdf` — HalfCheetah task (2M steps, 8 seeds)
- `paper_plots/flatworld_patrol.pdf` — FlatWorld 2-color task (1M steps, 8 seeds)
- `paper_plots/flatworld_sequence.pdf` — FlatWorld 3-color sequence task (1M steps, 4 completed seeds)
- `paper_plots/zones_sequence.pdf` — Zones sequence task (2M steps, 4 completed seeds)

### Combined Visualization
- `latest.pdf` — All four environments on one page for quick comparison

### Raw Data
Training logs available in:
- `logs/bench_<env>_td3_<method>_s<seed>/episodes.csv` — Episode returns and step counts
- `logs/bench_<env>_<method>_teacher/` — Teacher model directories

---

## Key Metrics Visualized

For each method and environment:
- **X-axis**: Training step count (up to environment maximum)
- **Y-axis**: Episode return (cumulative undiscounted reward)
- **Lines**: Mean return across seeds
- **Shaded region**: 90% confidence interval (t-distribution based)
- **Smoothing**: EMA window=200 for noise reduction

---

## Replicability

All code is contained in:
- `src/control_env_debug/run_all_benchmarks.py` — Experiment launcher
- `src/control_env_debug/train_vectorized_td3.py` — Training loop
- `plot_paper.py` — Publication plot generation script
- `plot_live.py` — Live progress monitoring script

Run all experiments with:
```bash
SEEDS="0 1 2 3 4 5 6 7" bash run_two_envs.sh      # Patrol + FlatWorld Patrol
SEEDS="0 1 2 3 4 5 6 7" bash run_two_sequences.sh # FlatWorld Sequence + Zones Sequence
```

Regenerate plots:
```bash
python plot_paper.py --out paper_plots --format pdf
```

---

## Status Summary

| Environment | Status | Seeds | Horizon |
|---|---|---|---|
| FlatWorld Patrol | ✅ Complete | 8 | 1M steps |
| FlatWorld Sequence | ✅ Complete | 4 | 1M steps |
| HalfCheetah Patrol | ✅ Complete | 8 | 2M steps |
| Zones Sequence | 🔄 In Progress | 4 completed, 4 running | 2M steps |

**Note**: FlatWorld Sequence and Zones Sequence use 4 completed seeds (0-3) in plots. Seeds 4-7 are currently running (launched on 2026-03-13, ~25% complete for Zones Sequence as of 2026-03-15).

---

## Next Steps

1. Complete Zones Sequence seeds 4-7 training
2. Regenerate plots with full 8-seed results when complete
3. Generate statistical significance tests (t-tests, confidence interval overlaps)
4. Produce paper-ready tables with summary statistics
