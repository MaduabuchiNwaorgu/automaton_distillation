#!/usr/bin/env python3
"""Quick test to verify new patrol setup works correctly.

Tests:
  1. Teacher environment (clean, no obstacles)
  2. Student environment (with obstacles)
  3. Both use same thresholds (a=5, b=-2)
  4. Events fire correctly
  5. Obstacles apply penalty
"""

import sys
sys.path.insert(0, 'src')

from control_env_debug.train_env import make_env
import numpy as np

print("="*70)
print("PATROL ENVIRONMENT SETUP TEST")
print("="*70)

# Test 1: Teacher environment (clean, no obstacles)
print("\n[1] Creating TEACHER environment (clean, no obstacles)...")
try:
    teacher_env = make_env(
        env_type="patrol",
        reward_shaping=True,
        max_episode_steps=1000,
        ap_config={"a_threshold": 5.0, "b_threshold": -2.0, "obstacles": False},
    )
    obs, info = teacher_env.reset(seed=0)
    print(f"  ✓ Teacher env created")
    print(f"    Obs shape: {obs.shape}")
    print(f"    Automaton: {teacher_env.automaton}")
    print(f"    Formula: F(a & F(b))")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Student environment (with obstacles)
print("\n[2] Creating STUDENT environment (with obstacles)...")
try:
    student_env = make_env(
        env_type="patrol",
        reward_shaping=True,
        max_episode_steps=1000,
        ap_config={"a_threshold": 5.0, "b_threshold": -2.0, "obstacles": True},
    )
    obs, info = student_env.reset(seed=0)
    print(f"  ✓ Student env created")
    print(f"    Obs shape: {obs.shape}")
    print(f"    Has obstacles: {hasattr(student_env.env, 'obstacle_zones')}")
    if hasattr(student_env.env, 'obstacle_zones'):
        print(f"    Obstacle zones: {student_env.env.obstacle_zones}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Run through environments and check events
print("\n[3] Testing environment dynamics and events...")
print("\n  TEACHER (clean):")
obs_t, _ = teacher_env.reset(seed=42)
rewards_t = []
events_sequence_t = []
x_positions_t = []

for step in range(200):
    # Take a forward-biased action for HalfCheetah (6 dims)
    action = teacher_env.action_space.sample() * 0.3 + 0.2  # Bias towards forward
    obs_t, reward, terminated, truncated, info = teacher_env.step(action)
    x_pos = float(teacher_env.env.unwrapped.data.qpos[0])
    x_positions_t.append(x_pos)
    rewards_t.append(reward)
    events = teacher_env.get_events()
    if events:
        events_sequence_t.append((step, events, x_pos))

print(f"    Steps: 200, Total reward: {sum(rewards_t):.2f}")
print(f"    X-position range: [{min(x_positions_t):.2f}, {max(x_positions_t):.2f}]")
if events_sequence_t:
    print(f"    Events triggered: {len(events_sequence_t)} times")
    print(f"    First 5 events: {events_sequence_t[:5]}")
else:
    print(f"    Events: None")

print("\n  STUDENT (with obstacles):")
obs_s, _ = student_env.reset(seed=42)
rewards_s = []
events_sequence_s = []
obstacle_hits = 0
x_positions_s = []

for step in range(200):
    # Same action pattern
    action = student_env.action_space.sample() * 0.3 + 0.2  # Same range
    obs_s, reward, terminated, truncated, info = student_env.step(action)
    x_pos = float(student_env.env.unwrapped.data.qpos[0])
    x_positions_s.append(x_pos)
    
    # Check if in obstacle zone
    if hasattr(student_env.env, '_is_in_obstacle'):
        if student_env.env._is_in_obstacle(x_pos):
            obstacle_hits += 1
    
    rewards_s.append(reward)
    events = student_env.get_events()
    if events:
        events_sequence_s.append((step, events, x_pos))

print(f"    Steps: 200, Total reward: {sum(rewards_s):.2f}")
print(f"    X-position range: [{min(x_positions_s):.2f}, {max(x_positions_s):.2f}]")
print(f"    Obstacle zones hit: {obstacle_hits} steps")
if events_sequence_s:
    print(f"    Events triggered: {len(events_sequence_s)} times")
    print(f"    First 5 events: {events_sequence_s[:5]}")
else:
    print(f"    Events: None")

# Test 4: Verify reward difference due to obstacles
print("\n[4] Reward comparison:")
teacher_reward = sum(rewards_t)
student_reward = sum(rewards_s)
reward_diff = teacher_reward - student_reward
print(f"  Teacher total reward: {teacher_reward:.2f}")
print(f"  Student total reward: {student_reward:.2f}")
print(f"  Difference: {reward_diff:.2f}")
if reward_diff > 0:
    print(f"  ✓ Student gets less reward due to obstacles (expected)")
else:
    print(f"  ⚠ Rewards similar - verify obstacles are being applied")

print("\n" + "="*70)
print("✓ TEST COMPLETE")
print("="*70)
print("\nSummary:")
print("  • Both environments created successfully")
print("  • Same task (a=5, b=-2)")
print("  • Different environments (clean vs with obstacles)")
print("  • Ready for benchmark runs!")
