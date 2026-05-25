#!/bin/bash
# ───────────────────────────────────────────────────────────────────────
#  Parallel benchmark launcher (single environment)
#
#  1. Trains the teacher (single seed, overwrites old Q-avg)
#     — skipped if SKIP_TEACHER=1
#  2. Launches every method × seed as a separate background process
#  3. Waits for all, reports results, generates plots
#
#  Usage:
#    ./run_parallel.sh <env> <steps> [seeds] [n_envs] [methods]
#
#  Environment variables:
#    SKIP_TEACHER=1  — skip teacher training (reuse existing Q-automaton)
#    PYTHON=...      — path to python executable
#    PREFIX=...      — run name prefix (default: bench)
#
#  Examples:
#    ./run_parallel.sh flatworld_patrol 100000 "0 1 2 3" 4
#    ./run_parallel.sh flatworld_patrol 100000 "0 1 2" 4 "td3_base td3_crm"
#    SKIP_TEACHER=1 ./run_parallel.sh flatworld_patrol 100000 "0 1 2 3" 4
# ───────────────────────────────────────────────────────────────────────
set -euo pipefail

ENV="${1:?Usage: $0 <env> <steps> [seeds] [n_envs] [methods]}"
STEPS="${2:?Usage: $0 <env> <steps> [seeds] [n_envs] [methods]}"
SEEDS="${3:-0 1 2}"
N_ENVS="${4:-4}"
METHODS="${5:-td3_base td3_crm td3_static td3_dynamic td3_shaped td3_cprep}"
PREFIX="${PREFIX:-bench}"
SKIP_TEACHER="${SKIP_TEACHER:-0}"

PYTHON="${PYTHON:-$(which python 2>/dev/null || echo python)}"
LOG_DIR="logs/parallel_${ENV}"
mkdir -p "$LOG_DIR"

TEACHER_Q="automaton_q/${PREFIX}_${ENV}_teacher.json"

echo "═══════════════════════════════════════════════════════════"
echo "  PARALLEL BENCHMARK"
echo "  Env     : $ENV"
echo "  Steps   : $STEPS"
echo "  Seeds   : $SEEDS"
echo "  N_envs  : $N_ENVS"
echo "  Methods : $METHODS"
echo "  Python  : $PYTHON"
echo "═══════════════════════════════════════════════════════════"

# ── Phase 1: Teacher ────────────────────────────────────────────────
if [[ "$SKIP_TEACHER" == "1" ]]; then
    echo ""
    echo "▶ Phase 1: SKIP_TEACHER=1 — reusing $TEACHER_Q"
    if [[ ! -f "$TEACHER_Q" ]]; then
        echo "ERROR: $TEACHER_Q not found. Run without SKIP_TEACHER first."
        exit 1
    fi
else
    echo ""
    echo "▶ Phase 1: Training teacher (seed=0, overwrites old Q-avg) ..."
    TEACHER_LOG="$LOG_DIR/teacher.log"
    "$PYTHON" -m src.control_env_debug.run_all_benchmarks \
        --envs "$ENV" \
        --total_steps "$STEPS" \
        --n_envs "$N_ENVS" \
        --seeds 0 \
        --prefix "$PREFIX" \
        --teacher_only \
        2>&1 | tee "$TEACHER_LOG"

    if [[ ! -f "$TEACHER_Q" ]]; then
        echo "ERROR: Teacher failed — $TEACHER_Q not created."
        echo "       See $TEACHER_LOG"
        exit 1
    fi
fi
echo "✓ Teacher Q-automaton: $TEACHER_Q"

# ── Phase 2: Launch each method × seed in parallel ─────────────────
echo ""
echo "▶ Phase 2: Launching methods in parallel ..."
PIDS=()
LABELS=()

for method in $METHODS; do
    for seed in $SEEDS; do
        run_name="${PREFIX}_${ENV}_${method}_s${seed}"
        log_file="$LOG_DIR/${method}_s${seed}.log"
        echo "  → $run_name (log: $log_file)"

        "$PYTHON" -m src.control_env_debug.run_all_benchmarks \
            --envs "$ENV" \
            --methods "$method" \
            --total_steps "$STEPS" \
            --n_envs "$N_ENVS" \
            --seeds "$seed" \
            --prefix "$PREFIX" \
            --skip_teacher \
            > "$log_file" 2>&1 &

        PIDS+=($!)
        LABELS+=("$run_name")
    done
done

N_JOBS=${#PIDS[@]}
echo ""
echo "  $N_JOBS jobs launched. Waiting for completion ..."
echo "  Monitor progress:  tail -f $LOG_DIR/*.log"
echo ""

# ── Wait and report ─────────────────────────────────────────────────
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    label=${LABELS[$i]}
    if wait "$pid"; then
        echo "  ✓ $label (PID $pid)"
    else
        echo "  ✗ $label FAILED (PID $pid) — see $LOG_DIR/${label#${PREFIX}_${ENV}_}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  DONE: $((N_JOBS - FAILED))/$N_JOBS succeeded"
if [[ $FAILED -gt 0 ]]; then
    echo "  $FAILED jobs failed — check logs in $LOG_DIR/"
fi
echo "═══════════════════════════════════════════════════════════"

# ── Generate plots ──────────────────────────────────────────────────
echo ""
echo "▶ Generating learning-curve plots ..."
"$PYTHON" -m src.control_env_debug.plot_results \
    --logs_root logs --envs "$ENV" --ci 0.90 --window 10 2>&1 || true

echo "▶ Evaluating final policies and generating trajectory plots ..."
"$PYTHON" -m src.control_env_debug.eval_trajectories \
    --env "$ENV" --logs_root logs --prefix "$PREFIX" \
    --n_episodes 8 --plot 2>&1 || true

echo "✓ All done for $ENV"
