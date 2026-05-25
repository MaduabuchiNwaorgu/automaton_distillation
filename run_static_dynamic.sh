#!/bin/bash
# Run static + dynamic distillation for patrol and flatworld_patrol
# Uses existing teacher Q-automaton (SKIP_TEACHER=1)
set -euo pipefail

PYTHON="${PYTHON:-$(which python 2>/dev/null || echo python3)}"
PREFIX="${PREFIX:-bench}"
SEEDS="${SEEDS:-0 1 2 3}"
N_ENVS=4

# flatworld_patrol: 1M steps (converges around 100k but we match full benchmark)
# patrol (HalfCheetah): 2M steps (needs ~1M to start converging)
declare -A ENV_STEPS
ENV_STEPS[flatworld_patrol]=1000000
ENV_STEPS[patrol]=2000000

cd "$(dirname "$0")"

ALL_PIDS=()
ALL_LABELS=()

for ENV in flatworld_patrol patrol; do
    STEPS=${ENV_STEPS[$ENV]}
    echo "═══════════════════════════════════════════════"
    echo "  $ENV — ${STEPS} steps — seeds: $SEEDS"
    echo "═══════════════════════════════════════════════"

    TEACHER_Q="automaton_q/${PREFIX}_${ENV}_teacher.json"
    if [[ ! -f "$TEACHER_Q" ]]; then
        echo "ERROR: Teacher Q-automaton not found: $TEACHER_Q"
        echo "  Run the teacher first or set PREFIX to match existing files."
        exit 1
    fi

    for METHOD in td3_static td3_dynamic; do
        for SEED in $SEEDS; do
            RUN="${PREFIX}_${ENV}_${METHOD}_s${SEED}"
            LOGFILE="logs/parallel_${ENV}/${METHOD}_s${SEED}.log"
            mkdir -p "logs/parallel_${ENV}"

            echo "  Launching $RUN → $LOGFILE"
            PYTHONPATH=. $PYTHON -m src.control_env_debug.run_all_benchmarks \
                --envs "$ENV" \
                --methods "$METHOD" \
                --total_steps "$STEPS" \
                --n_envs "$N_ENVS" \
                --seeds "$SEED" \
                --prefix "$PREFIX" \
                --skip_teacher \
                > "$LOGFILE" 2>&1 &
            ALL_PIDS+=($!)
            ALL_LABELS+=("$RUN")
        done
    done
done

echo ""
echo "  All ${#ALL_PIDS[@]} jobs launched in parallel. Waiting..."
FAIL=0
for i in "${!ALL_PIDS[@]}"; do
    if ! wait "${ALL_PIDS[$i]}"; then
        echo "  WARNING: ${ALL_LABELS[$i]} (PID ${ALL_PIDS[$i]}) failed"
        FAIL=$((FAIL+1))
    fi
done
echo "  All done ($FAIL failures)"

echo ""
echo "═══════════════════════════════════════════════"
echo "  RESULTS"
echo "═══════════════════════════════════════════════"
$PYTHON -c "
import numpy as np
for env in ['flatworld_patrol', 'patrol']:
    print(f'\n  {env}:')
    for m in ['td3_static', 'td3_dynamic']:
        rets = []
        for s in '${SEEDS}'.split():
            try:
                r = np.load(f'logs/${PREFIX}_{env}_{m}_s{s}/episode_returns.npy')
                rets.append(r[-100:].mean())
            except: pass
        if rets:
            print(f'    {m}: mean={np.mean(rets):.1f} ± {np.std(rets):.1f}  (seeds: {len(rets)})')
        else:
            print(f'    {m}: no results')
"
