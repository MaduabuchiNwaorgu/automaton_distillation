#!/bin/bash
# Zones Sequence (Safety Gymnasium) — parallel benchmark
# Teacher: PointLtl2 → Student: CarLtl2
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_parallel.sh" zones_sequence "${1:-100000}" "${2:-0 1 2}" "${3:-4}" "${4:-}"
