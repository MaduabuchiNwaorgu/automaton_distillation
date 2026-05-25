# TD3 Hyperparameter Configuration

## Overview

All experiments employ TD3 (Twin Delayed DDPG) as the base reinforcement learning algorithm, with symmetric actor and critic networks (256→ 256 hidden units, ReLU activations) that condition on both environment observations and one-hot encoded automaton states. We evaluate six distillation variants—vanilla TD3, counterfactual replay memory (CRM), static and dynamic target mixing with teacher Q-values, reward-shaped product MDP, and C-PREP weight initialization—across four environments with varying temporal logic complexities. Detailed hyperparameters and architectural specifications are provided in the tables below.

## Network Architecture

| Component | Configuration |
|-----------|---------------|
| **Actor Network** | Input: `obs_dim + num_rm_states` → Linear(256) → ReLU → Linear(256) → ReLU → Linear(action_dim) → Tanh |
| **Critic Network** | Input: `obs_dim + num_rm_states + action_dim` → Linear(256) → ReLU → Linear(256) → ReLU → Linear(1) |
| **Observation Module** | None (direct state concatenation with RM one-hot encoding) |

## Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Actor Learning Rate** | 1e-3 | Adam optimizer learning rate for actor network |
| **Critic Learning Rate** | 1e-3 | Adam optimizer learning rate for critic networks |
| **Discount Factor (γ)** | 0.99 | Temporal discount for future rewards |
| **Soft Update Rate (τ)** | 0.005 | Target network exponential moving average: `θ' ← τθ + (1-τ)θ'` |
| **Batch Size** | 100 | Samples per training step from replay buffer |
| **Buffer Size** | 1,000,000 | Maximum replay buffer capacity |

## TD3-Specific Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Policy Noise** | 0.5 | Action noise std for target policy smoothing |
| **Noise Clip** | 0.5 | Clip range for smoothing noise |
| **Policy Delay** | 2 | Train actor every N critic updates |

## Environment Configuration

| Environment | Observation Dim | Action Dim | RM States | Total Steps | Notes |
|-------------|-----------------|-----------|-----------|-------------|-------|
| **Patrol (HalfCheetah)** | 17 | 6 | 3 | 2M | Continuous locomotion + reach-then-reach task |
| **FlatWorld Patrol** | 2 | 2 | 3 | 1M | 2D navigation, visit blue→red circles |
| **FlatWorld Sequence** | 2 | 2 | 4 | 1M | 2D navigation, visit red→blue→green circles |
| **ZoneEnv Sequence** | 5 | 5 | 4 | 2M | Robot arm reaching task, 3 sequential targets |

## Distillation Methods

Each method reuses the same TD3 base with optional modifications:

| Method | Configuration | Teacher | Notes |
|--------|---------------|---------|-------|
| **td3_base** | Vanilla TD3 | None | Baseline single-task learning |
| **td3_crm** | TD3 + Counterfactual Replay Memory | Same task | Removes pessimistic bias from RM-driven rewards |
| **td3_static** | TD3 + Static Target Mixing (β=0.1) | Teacher | Mixes 10% teacher Q-values into target |
| **td3_dynamic** | TD3 + Dynamic Target Mixing (β_t = ρ^η) | Teacher | Exponential decay mixing: `β ← 0.999^{count}` |
| **td3_shaped** | TD3 + Product MDP Reward Shaping | Teacher | Shape reward: `R_shaped = R + V_teacher(s')` |
| **td3_cprep** | TD3 + C-PREP Weight Transfer | Teacher | Initialize student from teacher model weights |

## Multi-GPU Distributed Training

| Parameter | Value |
|-----------|-------|
| **GPUs** | 4× NVIDIA A40 (46GB each) |
| **CPU Cores** | 64 |
| **Training Frequency** | 2 (train every 2 environment steps for GPU efficiency) |
| **Schedule** | Round-robin GPU assignment across jobs |

## Implementation Details

- **Framework**: PyTorch
- **Continuous Actions**: Clipped to `[-1, 1]` with environment-specific scaling
- **RM Integration**: One-hot encoded RM state concatenated to observation
- **RM State Count**: Variable per task (3 for Patrol, 3-4 for Sequence, 2 for base HalfCheetah)
- **Reward**: Sparse automaton-based rewards + optional reward shaping/mixing
- **Normalization**: Per-environment observation normalization applied in wrapper
- **Buffer Storage**: Episodes stored with propositions and RM transitions

## Notes

- **No separate observation module**: Unlike some descriptions, this implementation uses direct concatenation of state + RM one-hot
- **Symmetrical learning rates**: Both actor and critic use 1e-3 (can be overridden per-experiment)
- **Exploration**: Policy noise 0.5, noise clip 0.5 for aggressive exploration in sparse reward setting
- **Stability**: Delayed policy updates (every 2 critic steps), soft target updates (τ=0.005)

---

# Discrete Environment Training (DQN/Dueling Networks)

## Overview

Discrete action space environments (Gold Mine, Dungeon Quest, etc.) employ dueling DQN architectures with convolutional feature extraction, combined with the same distillation methods as continuous control. The agent processes visual observations (stacked frames) through residual convolutional blocks, then branches into separate value and advantage streams to support distributed training across CPU cores.

## Network Architecture (Discrete)

| Component | Configuration |
|-----------|---------------|
| **Feature Extractor** | 3× Residual Conv2d blocks: Conv2d(32, kernel=3×3, padding=1) + BatchNorm + LeakyReLU |
| **Dueling Q-Network** | Feature Extractor → Split(feat/2) → Value Stream: Linear(1), Advantage Stream: Linear(num_actions) → Q = V + (Adv - mean(Adv)) |
| **Input Shape** | (num_channels, H, W) — stacked frames from visual observations |

## Training Hyperparameters (Discrete)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Actor Learning Rate** | 1e-4 | Adam optimizer learning rate for policy network |
| **Critic Learning Rate** | 1e-4 | Adam optimizer learning rate for Q-network |
| **Discount Factor (γ)** | 0.99 | Temporal discount for future rewards |
| **Soft Update Rate (τ)** | 0.005 | Target network soft update rate |
| **Batch Size** | 64 | Samples per training step from replay buffer |
| **Buffer Size** | 150,000 | Prioritized replay buffer capacity |
| **Buffer Priority Scale** | 0.7 | Importance sampling weight for prioritized experience replay |
| **Min Size Before Training** | 1,000 | Minimum buffer size before training begins |

## DQN-Specific Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Epsilon (ε-greedy)** | 0.1 | Fixed exploration rate |
| **Target Update Frequency** | 1,000 steps | Update target network every N environment steps |
| **Num Parallel Environments** | 8 | CPU cores for rollout collection |
| **Checkpoint Frequency** | 10,000 steps | Save model checkpoints |

## Discrete Environments

| Environment | Input Resolution | Num Actions | RM States | Total Steps | Notes |
|-------------|------------------|-------------|-----------|-------------|-------|
| **Gold Mine** | Varies | Discrete | 3+ | 1M-2M | Mining task with temporal logic constraints |
| **Dungeon Quest** | Varies | Discrete | 3+ | 1M-2M | Quest completion with LTL formula constraints |

## Discrete Distillation Methods

Same six variants as continuous, adapted for DQN:

| Method | Configuration | Teacher | Notes |
|--------|---------------|---------|-------|
| **dqn_base** | Vanilla Dueling DQN | None | Baseline discrete-action learning |
| **dqn_crm** | DQN + Counterfactual Replay | Same task | Remove RM-pessimism bias |
| **dqn_static** | DQN + Static Automaton Mixing (β=0.1) | Teacher | Mix teacher Q-automaton into targets |
| **dqn_dynamic** | DQN + Dynamic Automaton Mixing | Teacher | Exponential decay β = 0.999^{count} |
| **dqn_shaped** | DQN + Product MDP Reward Shaping | Teacher | Shape reward with teacher V-automaton |
| **dqn_cprep** | DQN + C-PREP Weight Transfer | Teacher | Initialize from teacher model |

## Implementation Details (Discrete)

- **Framework**: PyTorch
- **Visual Input**: Stacked frames (multi-channel tensors)
- **Network Input**: Feature extractor processes raw observations independently of RM state
- **RM Integration**: Automaton state tracked separately, used by reward_machine for shaped rewards
- **Exploration**: Fixed ε=0.1 (alternatives to epsilon-greedy decay used in other work)
- **Buffer**: Prioritized circular replay buffer with importance sampling (priority_scale=0.7)
- **Training**: Continuous learning from environment rollouts, no train_freq optimization (different from continuous control)
- **CPU Parallelization**: 8 parallel environments for simultaneous experience collection

## Notes

- **Dueling Architecture**: Value and advantage streams allow better credit assignment for state values vs. action advantages
- **Residual Convolutions**: Inspired by AlphaGo architecture with batch normalization and LeakyReLU
- **Visual Input**: Handles arbitrary input resolutions through FeatureExtractor flattening
- **Separate RM Tracking**: Unlike continuous control (which concatenates RM to state), discrete environments track RM state independently for automaton-based reward computation
- **Lower Learning Rates**: 1e-4 vs. 1e-3 for continuous, reflecting the discrete action exploration dynamics
- **Fixed Epsilon**: 0.1 across all episodes (no decay schedule), standard for DQN with prioritized replay
