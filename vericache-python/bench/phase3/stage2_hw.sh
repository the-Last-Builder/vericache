#!/usr/bin/env bash
# Stage 2 cross-NUMA 2-GPU hardware wrapper with strict scheduler & cleanup hygiene.
set -e -o pipefail

phase3=/data/nvme2/zxw/vericache_phase3
scheduler=/usr/local/bin/scheduler
source "$phase3/env.sh"
set -u

mode=${1:?expected exp21 or exp22}
gpu0=${2:?expected NUMA-0 GPU id (0..3)}
gpu1=${3:?expected NUMA-1 GPU id (4..7)}

[[ "$gpu0" =~ ^[0-3]$ ]] || { echo "gpu0 must be 0..3 (NUMA 0)"; exit 2; }
[[ "$gpu1" =~ ^[4-7]$ ]] || { echo "gpu1 must be 4..7 (NUMA 1)"; exit 2; }

duration="${PHASE3_STAGE2_DURATION:-1200}"
run=$(mktemp -d "$phase3/g2.XXXXXX")
out=$(mktemp -d "$phase3/results/stage2_${mode}.XXXXXX")
lease_id=
worker_pid=

cleanup() {
    local rc=$?
    trap - EXIT
    trap '' HUP INT TERM
    if [[ -n "$worker_pid" ]]; then
        if kill -TERM -- "-$worker_pid" 2>/dev/null; then
            sleep 3
            kill -KILL -- "-$worker_pid" 2>/dev/null || true
        fi
        wait "$worker_pid" 2>/dev/null || true
    fi
    if [[ -n "${PHASE3_RESERVATION_ID:-}" ]]; then
        printf '{"reused_reservation": "%s"}\n' "$PHASE3_RESERVATION_ID" >"$out/release.json"
    elif [[ -n "$lease_id" ]]; then
        "$scheduler" claim release "$lease_id" >"$out/release.json" || true
    fi
    if find "$run" -depth -delete && [[ ! -e "$run" ]]; then
        printf 'Unlinked run directory %s\n' "$run" >"$out/cleanup.log"
    else
        rc=1
    fi
    printf '%d\n' "$rc" >"$out/exit-status"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Stage 2 Results: $out"

if [[ -n "${PHASE3_RESERVATION_ID:-}" ]]; then
    "$scheduler" status --gpus "$gpu0" "$gpu1" >"$out/claim.json"
    python - "$out/claim.json" "$PHASE3_RESERVATION_ID" "$gpu0" "$gpu1" <<'PY'
import json, sys
from pathlib import Path
status_file, res_id, g0, g1 = sys.argv[1:]
data = json.loads(Path(status_file).read_text())
leases = [l for l in data.get("my_allocations", []) if l.get("id") == res_id]
assert len(leases) == 1, f"reservation {res_id} not found"
lease = leases[0]
assert lease["kind"] == "reservation" and lease["state"] == "active", lease
assert {g0, g1}.issubset(set(lease["active_gpu_ids"])), lease
PY
else
    "$scheduler" claim acquire --gpus "$gpu0" "$gpu1" >"$out/claim.json"
    lease_id=$(python -c 'import json,sys; print(json.load(sys.stdin)["id"])' <"$out/claim.json")
fi

export STAGE2_GPU0="$gpu0"
export STAGE2_GPU1="$gpu1"
export PHASE3_TEST_DIR="$run"
export PHASE3_RESULT_DIR="$out"
export PYTHONPATH="$phase3/vericache-python/bench/phase3:$phase3/vericache-python/python:${PYTHONPATH:-}"

if [[ "$mode" == exp21 ]]; then
    script="$phase3/vericache-python/bench/phase3/exp21_safety_gate.py"
elif [[ "$mode" == exp22 ]]; then
    script="$phase3/vericache-python/bench/phase3/stage2.py"
else
    echo "Unknown mode: $mode"
    exit 2
fi

setsid timeout --foreground --kill-after=10s "$duration" \
    bash -o pipefail -c 'python "$1" 2>&1 | tee "$PHASE3_RESULT_DIR/driver.log"' _ "$script" &
worker_pid=$!
wait "$worker_pid"
