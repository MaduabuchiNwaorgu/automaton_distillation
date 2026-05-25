#!/bin/bash
# Zones Patrol (Safety Gymnasium) — parallel benchmark
# Teacher: PointLtl1 → Student: CarLtl1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_parallel.sh" zones_patrol "${1:-100000}" "${2:-0 1 2}" "${3:-4}" "${4:-}"
