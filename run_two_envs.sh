#!/bin/bash
# ───────────────────────────────────────────────────────────────────────
#  Run ALL 6 methods × 4 seeds for flatworld_patrol and patrol.
#  Teacher Q-automata must already exist (SKIP_TEACHER=1 by default).
#
#  Spreads jobs round-robin across NUM_GPUS GPUs.
#  Set DEVICE=cpu to force CPU-only.
#
#  Usage:
#    ./run_two_envs.sh                   # 3 GPUs, train_freq=2
#    DEVICE=cpu ./run_two_envs.sh        # CPU-only
#    NUM_GPUS=2 ./run_two_envs.sh        # 2 GPUs
#    TRAIN_FREQ=4 ./run_two_envs.sh      # train every 4 steps
#    SEEDS="0 1 2 3 4" ./run_two_envs.sh
# ───────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON="${PYTHON:-$(which python 2>/dev/null || echo python3)}"
PREFIX="${PREFIX:-bench}"
SEEDS="${SEEDS:-0 1 2 3}"
N_ENVS=4
METHODS="td3_base td3_crm td3_static td3_dynamic td3_shaped td3_cprep"
NUM_GPUS="${NUM_GPUS:-4}"
TRAIN_FREQ="${TRAIN_FREQ:-2}"
DEVICE="${DEVICE:-}"  # empty = auto GPU round-robin; "cpu" = CPU-only

# Step counts (same as previous experiments)
declare -A ENV_STEPS
ENV_STEPS[flatworld_patrol]=1000000
ENV_STEPS[patrol]=2000000

cd "$(dirname "$0")"

# Verify teachers exist
for ENV in flatworld_patrol patrol; do
    TEACHER_Q="automaton_q/${PREFIX}_${ENV}_teacher.json"
    if [[ ! -f "$TEACHER_Q" ]]; then
        echo "ERROR: Teacher Q-automaton not found: $TEACHER_Q"
        echo "  Train teachers first, or set PREFIX to match existing files."
        exit 1
    fi
    echo "✓ Teacher found: $TEACHER_Q"
done

ALL_PIDS=()
ALL_LABELS=()
JOB_IDX=0

for ENV in flatworld_patrol patrol; do
    STEPS=${ENV_STEPS[$ENV]}
    echo ""
    echo "═══════════════════════════════════════════════"
    echo "  $ENV — ${STEPS} steps — seeds: $SEEDS"
    echo "  Methods: $METHODS"
    if [[ "$DEVICE" == "cpu" ]]; then
        echo "  Device: CPU"
    else
        echo "  Devices: round-robin across $NUM_GPUS GPUs"
    fi
    echo "  train_freq: $TRAIN_FREQ"
    echo "═══════════════════════════════════════════════"

    for METHOD in $METHODS; do
        for SEED in $SEEDS; do
            RUN="${PREFIX}_${ENV}_${METHOD}_s${SEED}"
            LOGDIR="logs/parallel_${ENV}"
            LOGFILE="${LOGDIR}/${METHOD}_s${SEED}.log"
            mkdir -p "$LOGDIR"

            # Determine device for this job
            if [[ "$DEVICE" == "cpu" ]]; then
                DEV_ARG="cpu"
            else
                GPU_ID=$((JOB_IDX % NUM_GPUS))
                DEV_ARG="cuda:${GPU_ID}"
            fi

            echo "  Launching $RUN → $LOGFILE  [device=$DEV_ARG]"
            PYTHONPATH=. $PYTHON -m src.control_env_debug.run_all_benchmarks \
                --envs "$ENV" \
                --methods "$METHOD" \
                --total_steps "$STEPS" \
                --n_envs "$N_ENVS" \
                --seeds "$SEED" \
                --prefix "$PREFIX" \
                --skip_teacher \
                --device "$DEV_ARG" \
                --train_freq "$TRAIN_FREQ" \
                > "$LOGFILE" 2>&1 &

            ALL_PIDS+=($!)
            ALL_LABELS+=("$RUN")
            JOB_IDX=$((JOB_IDX + 1))
        done
    done
done

N_JOBS=${#ALL_PIDS[@]}
echo ""
echo "═══════════════════════════════════════════════"
echo "  $N_JOBS jobs launched in parallel"
echo "  Monitor: tail -f logs/parallel_*/*.log"
echo "═══════════════════════════════════════════════"

# Wait for all
FAILED=0
for i in "${!ALL_PIDS[@]}"; do
    pid=${ALL_PIDS[$i]}
    label=${ALL_LABELS[$i]}
    if wait "$pid"; then
        echo "  ✓ $label (PID $pid)"
    else
        echo "  ✗ $label FAILED (PID $pid)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "═══════════════════════════════════════════════"
echo "  DONE: $((N_JOBS - FAILED))/$N_JOBS succeeded"
if [[ $FAILED -gt 0 ]]; then
    echo "  $FAILED jobs failed — check logs in logs/parallel_*/"
fi
echo "═══════════════════════════════════════════════"

# Generate plots
echo ""
echo "▶ Generating publication plots ..."
$PYTHON -m src.control_env_debug.plot_results \
    --logs_root logs \
    --envs flatworld_patrol patrol \
    --ci 0.90 \
    --window 50 \
    --format pdf \
    --combined \
    2>&1 || true

echo "✓ All done."
