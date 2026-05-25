#!/usr/bin/env python3
"""
Check the quality of existing teacher automata by examining their training data.
Specifically, check what % of time they visited their goal states.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Environment-specific automaton state and goal state mapping
ENV_INFO = {
    "flatworld_sequence": {
        "teacher_dir": "dump/logs/bench_flatworld_sequence_teacher",
        "goal_state": 1,  # Final state in F(a & F(b))
        "formula": "F(a & F(b))",
    },
    "zones_sequence": {
        "teacher_dir": "logs/bench_zones_sequence_teacher",
        "goal_state": 2,  # Final state in F(a & F(b))
        "formula": "F(a & F(b))",
    },
}

def check_teacher(env_name):
    info = ENV_INFO[env_name]
    teacher_dir = Path(info["teacher_dir"])
    
    if not teacher_dir.exists():
        print(f"\n❌ {env_name}: Teacher directory not found: {teacher_dir}")
        return False
    
    # Load automaton state data - try episodes_ex.csv first
    episodes_file = teacher_dir / "episodes_ex.csv"
    if not episodes_file.exists():
        episodes_file = teacher_dir / "episodes.csv"
    if not episodes_file.exists():
        print(f"\n❌ {env_name}: episodes*.csv not found")
        return False
    
    try:
        df = pd.read_csv(episodes_file)
    except Exception as e:
        print(f"\n❌ {env_name}: Failed to load episodes file: {e}")
        return False
    
    # Check for final automaton state column
    state_column = None
    for col_name in ['final_rm_state', 'automaton_state', 'final_automaton_state']:
        if col_name in df.columns:
            state_column = col_name
            break
    
    if state_column is None:
        print(f"\n❌ {env_name}: No automaton state column found. Available: {df.columns.tolist()}")
        return False
    
    # Find all unique automaton states
    unique_states = sorted(df[state_column].unique())
    state_counts = df[state_column].value_counts().sort_index()
    
    goal_state = info['goal_state']
    goal_visits = state_counts.get(goal_state, 0)
    total_steps = len(df)
    goal_percentage = (goal_visits / total_steps * 100) if total_steps > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"TEACHER QUALITY CHECK: {env_name.upper()}")
    print(f"{'='*60}")
    print(f"Formula: {info['formula']}")
    print(f"Goal State: {goal_state} (final/accepting state)")
    print(f"\nAutomaton States Distribution:")
    for state in unique_states:
        count = state_counts.get(state, 0)
        pct = (count / total_steps * 100) if total_steps > 0 else 0
        marker = "← GOAL" if state == goal_state else ""
        print(f"  State {state}: {count:>7} steps ({pct:>5.1f}%) {marker}")
    
    print(f"\nSummary:")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Goal state visits: {goal_visits:,} ({goal_percentage:.1f}%)")
    
    if goal_visits == 0:
        print(f"\n⚠️  WARNING: Goal state NEVER visited! Teacher is broken!")
        return False
    elif goal_percentage < 5:
        print(f"\n⚠️  WARNING: Goal state visited <5% of time. Consider retraining!")
        return False
    else:
        print(f"\n✅ Goal state visited {goal_percentage:.1f}% of time. Quality acceptable.")
        return True

# Run checks
print("\n" + "="*60)
print("TEACHER QUALITY VERIFICATION REPORT")
print("="*60)

results = {}
for env_name in ENV_INFO.keys():
    results[env_name] = check_teacher(env_name)

print(f"\n{'='*60}")
print("RECOMMENDATIONS:")
print(f"{'='*60}")

all_good = all(results.values())
for env_name, quality_ok in results.items():
    status = "✅ SKIP RETRAIN" if quality_ok else "⚠️  RETRAIN NEEDED"
    print(f"  {env_name}: {status}")

if all_good:
    print(f"\n✅ All teachers acceptable. Proceed directly to benchmark runs.")
else:
    print(f"\n⚠️  Some teachers need retraining. Start teacher training before benchmarks.")

print()
