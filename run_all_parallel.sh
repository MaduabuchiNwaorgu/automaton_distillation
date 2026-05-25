#!/bin/bash
# ───────────────────────────────────────────────────────────────────────
#  Master parallel benchmark — all 5 environments
#
#  Phase 1: Train all 5 teachers sequentially (single seed each)
#  Phase 2: Launch run_parallel.sh for each env in a screen session
#           (each env trains 6 methods × N seeds in parallel)
#
#  Usage:
#    ./run_all_parallel.sh [steps] [seeds] [n_envs]
#
#  Examples:
#    ./run_all_parallel.sh 100000 "0 1 2 3" 4
#    SKIP_TEACHER=1 ./run_all_parallel.sh 100000 "0 1 2 3" 4
# ───────────────────────────────────────────────────────────────────────
set -euo pipefail

STEPS="${1:-100000}"
SEEDS="${2:-0 1 2}"
N_ENVS="${3:-4}"
PREFIX="${PREFIX:-bench}"
SKIP_TEACHER="${SKIP_TEACHER:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$(which python 2>/dev/null || echo python3)}"

ENVS="patrol flatworld_patrol flatworld_sequence zones_patrol zones_sequence"

echo "╔═════════════════════════════════════════════════════════════╗"
echo "║              FULL PARALLEL BENCHMARK                       ║"
echo "╠═════════════════════════════════════════════════════════════╣"
echo "║  Steps   : $STEPS"
echo "║  Seeds   : $SEEDS"
echo "║  N_envs  : $N_ENVS"
echo "║  Python  : $PYTHON"
echo "║  Skip teacher: $SKIP_TEACHER"
echo "╚═════════════════════════════════════════════════════════════╝"

# ── Phase 1: Train teachers sequentially ────────────────────────────
if [[ "$SKIP_TEACHER" != "1" ]]; then
    echo ""
    echo "═══ Phase 1: Training teachers (sequentially) ═══"
    for env in $ENVS; do
        TEACHER_Q="automaton_q/${PREFIX}_${env}_teacher.json"
        echo ""
        echo "▶ Training teacher for $env ..."
        mkdir -p "logs/parallel_${env}"
        "$PYTHON" -m src.control_env_debug.run_all_benchmarks \
            --envs "$env" \
            --total_steps "$STEPS" \
            --n_envs "$N_ENVS" \
            --seeds 0 \
            --prefix "$PREFIX" \
            --teacher_only \
            2>&1 | tee "logs/parallel_${env}/teacher.log"

        if [[ ! -f "$TEACHER_Q" ]]; then
            echo "ERROR: Teacher failed for $env — $TEACHER_Q not created"
            exit 1
        fi
        echo "✓ $env teacher done → $TEACHER_Q"
    done
    echo ""
    echo "═══ All 5 teachers trained ═══"
else
    echo ""
    echo "═══ Phase 1: SKIP_TEACHER=1 — verifying existing Q-automata ═══"
    for env in $ENVS; do
        TEACHER_Q="automaton_q/${PREFIX}_${env}_teacher.json"
        if [[ ! -f "$TEACHER_Q" ]]; then
            echo "ERROR: $TEACHER_Q not found. Run without SKIP_TEACHER first."
            exit 1
        fi
        echo "  ✓ $TEACHER_Q"
    done
fi

# ── Phase 2: Launch each env in a screen session ────────────────────
echo ""
echo "═══ Phase 2: Launching parallel methods (5 screen sessions) ═══"

declare -A SCREEN_NAMES=(
    [patrol]=bench_patrol
    [flatworld_patrol]=bench_fw_patrol
    [flatworld_sequence]=bench_fw_seq
    [zones_patrol]=bench_z_patrol
    [zones_sequence]=bench_z_seq
)

for env in $ENVS; do
    sname="${SCREEN_NAMES[$env]}"
    screen -S "$sname" -X quit 2>/dev/null || true
    echo "  → Launching $sname  ($env)"
    screen -dmS "$sname" bash -c \
        "cd '$SCRIPT_DIR' && SKIP_TEACHER=1 PYTHON='$PYTHON' PREFIX='$PREFIX' \
         ./run_parallel.sh '$env' '$STEPS' '$SEEDS' '$N_ENVS'; exec bash"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  5 screen sessions launched!"
echo ""
echo "  Monitor:"
echo "    screen -ls                          # list sessions"
echo "    screen -r bench_patrol              # attach to one"
echo "    tail -f logs/parallel_patrol/*.log   # follow logs"
echo ""
echo "  All methods × seeds run in parallel within each env."
echo "═══════════════════════════════════════════════════════════"
