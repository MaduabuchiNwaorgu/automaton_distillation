#!/usr/bin/env python3
"""Aggregate mean±std final returns from multiple run folders.

Example:
  python -m src.control_env_debug.aggregate_results \
    --runs logs/bench_td3_base_s0 \
           logs/bench_td3_base_s1 \
           logs/bench_td3_base_s2
"""

import argparse
import os
import numpy as np


def load_final_returns(run_dirs):
    finals = []
    for d in run_dirs:
        try:
            arr = np.load(os.path.join(d, 'episode_returns.npy'))
            if arr.size > 0:
                finals.append(np.mean(arr[-min(100, len(arr)):]))
        except Exception:
            pass
    return np.array(finals, dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=str, nargs='+', required=True)
    args = parser.parse_args()
    vals = load_final_returns(args.runs)
    if vals.size == 0:
        print('No data found')
    else:
        print(f'n={len(vals)} | mean={vals.mean():.2f} | std={vals.std():.2f}')
