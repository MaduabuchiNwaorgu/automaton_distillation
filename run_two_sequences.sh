#!/bin/bash
# ───────────────────────────────────────────────────────────────────────
#  Run ALL 6 methods × 4 seeds for flatworld_sequence and zones_sequence.
#  Reuses existing teacher assets and restores archived teacher model dirs
#  into logs/ when needed so CPREP can initialize correctly.
#
#  Spreads jobs round-robin across NUM_GPUS GPUs.
#  Set DEVICE=cpu to force CPU-only.
#
#  Usage:
#    ./run_two_sequences.sh
#    DEVICE=cpu ./run_two_sequences.sh
#    NUM_GPUS=2 ./run_two_sequences.sh
#    TRAIN_FREQ=4 ./run_two_sequences.sh
#    SEEDS="0 1 2 3 4" ./run_two_sequences.sh
# ───────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON="${PYTHON:-$(which python 2>/dev/null || echo python3)}"
PREFIX="${PREFIX:-bench}"
SEEDS="${SEEDS:-4 5 6 7}"
N_ENVS=4
METHODS="td3_base td3_crm td3_static td3_dynamic td3_shaped td3_cprep"
NUM_GPUS="${NUM_GPUS:-4}"
TRAIN_FREQ="${TRAIN_FREQ:-2}"
DEVICE="${DEVICE:-}"  # empty = auto GPU round-robin; "cpu" = CPU-only

declare -A ENV_STEPS
ENV_STEPS[flatworld_sequence]=1000000
ENV_STEPS[zones_sequence]=2000000

cd "$(dirname "$0")"

ensure_teacher_assets() {
    local env="$1"
    local teacher_name="${PREFIX}_${env}_teacher"
    local teacher_q="automaton_q/${teacher_name}.json"
    local teacher_dir="logs/${teacher_name}"
    local teacher_actor="${teacher_dir}/td3_model_actor.pth"

    if [[ ! -f "$teacher_q" ]]; then
        echo "ERROR: Teacher Q-automaton not found: $teacher_q"
        echo "  Train teachers first, or set PREFIX to match existing files."
        exit 1
    fi

    if [[ ! -f "$teacher_actor" ]]; then
        local archived_dir=""
        for candidate in \
            "dump/logs/${teacher_name}" \
            "dump/logs/dump/${teacher_name}"; do
            if [[ -f "${candidate}/td3_model_actor.pth" ]]; then
                archived_dir="$candidate"
                break
            fi
        done

        if [[ -n "$archived_dir" ]]; then
            echo "↺ Restoring teacher model dir for ${env} from ${archived_dir}"
            mkdir -p logs
            rm -rf "$teacher_dir"
            cp -r "$archived_dir" "$teacher_dir"
        else
            echo "ERROR: Teacher model dir missing for ${env}: $teacher_dir"
            echo "  Expected ${teacher_actor} for CPREP initialization."
            exit 1
        fi
    fi

    echo "✓ Teacher Q found: $teacher_q"
    echo "✓ Teacher model dir ready: $teacher_dir"
}

for ENV in flatworld_sequence zones_sequence; do
    ensure_teacher_assets "$ENV"
done

ALL_PIDS=()
ALL_LABELS=()
JOB_IDX=0

for ENV in flatworld_sequence zones_sequence; do
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

echo ""
echo "▶ Generating publication plots ..."
$PYTHON -m src.control_env_debug.plot_results \
    --logs_root logs \
    --envs flatworld_sequence zones_sequence \
    --ci 0.90 \
    --window 50 \
    --format pdf \
    --combined \
    2>&1 || true

echo "✓ All done."