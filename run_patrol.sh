#!/bin/bash
# HalfCheetah Patrol — parallel benchmark
# Teacher: a=5, b=-2  →  Student: a=8, b=-5
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_parallel.sh" patrol "${1:-100000}" "${2:-0 1 2}" "${3:-4}" "${4:-}"
