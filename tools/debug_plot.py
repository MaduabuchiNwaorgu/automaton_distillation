#!/usr/bin/env python3
from plot_paper import discover, METHOD_ORDER, METHOD_LABELS, ENV_MAX_STEPS
import numpy as np

tree = discover('logs', 'bench', n_envs=4)
env = 'flatworld_sequence'

print(f"Methods discovered for {env}:")
for method in sorted(tree[env].keys()):
    seeds = tree[env][method]
    print(f"  {method}: {len(seeds)} seeds")
    
print(f"\nMethods in METHOD_ORDER:")
for method in METHOD_ORDER:
    if method in tree[env]:
        print(f"  ✓ {method}")
    else:
        print(f"  ✗ {method} (NOT FOUND)")

print(f"\nNow checking aggregation for each method:")
from plot_paper import aggregate_method

max_steps = ENV_MAX_STEPS.get(env)
print(f"Max steps for {env}: {max_steps}")

for method in METHOD_ORDER:
    seeds = tree[env].get(method, [])
    if not seeds:
        print(f"  {method}: NO DATA")
        continue
    
    agg = aggregate_method(seeds, window=50, max_steps=max_steps, n_interp=500)
    if agg is None:
        print(f"  {method}: AGGREGATION FAILED")
    else:
        n = agg['n']
        print(f"  {method}: OK ({n} seeds, mean return range [{agg['mean'].min():.1f}, {agg['mean'].max():.1f}])")
