#!/usr/bin/env bash
set -e -o pipefail

phase3=/data/nvme2/zxw/vericache_phase3
scheduler=/usr/local/bin/scheduler
source "$phase3/env.sh"
set -u

mode=${1:?expected local, gdr, layout, correctness, calibration, or stage1}
shift
case "$mode:$#" in
    local:1 | gdr:2 | layout:1 | correctness:1 | calibration:1 | stage1:1 | stage1:2) ;;
    *) exit 2 ;;
esac

for gpu in "$@"; do
    [[ "$gpu" =~ ^[0-7]$ ]] || exit 2
    if [[ "$mode" == local ]]; then
        (( gpu >= 4 )) || exit 2
    else
        (( gpu / 4 == $1 / 4 )) || exit 2
    fi
done
if [[ $# == 2 ]]; then
    [[ "$1" != "$2" ]] || exit 2
fi
if [[ "$mode" == gdr ]]; then
    export VERICACHE_RDMA_DEVICE="${VERICACHE_RDMA_DEVICE:?select the local RoCE device}"
    export VERICACHE_RDMA_GID_INDEX=3 VERICACHE_REQUIRE_HW=1
    grep -Fx 'RoCE v2' "/sys/class/infiniband/$VERICACHE_RDMA_DEVICE/ports/1/gid_attrs/types/3"
fi
node=$(( $1 / 4 ))
duration=180
if [[ "$mode" == layout ]]; then
    duration=330
    export PHASE3_BLOCK_SIZE="${PHASE3_BLOCK_SIZE:?select 16 or 64}"
    export PHASE3_VERICACHE_SHA="${PHASE3_VERICACHE_SHA:?record the VeriCache commit}"
fi
if [[ "$mode" == correctness ]]; then
    duration=860
    export PHASE3_SCENARIO="${PHASE3_SCENARIO:?select vanilla, hbm, dram, or nvme}"
    export PHASE3_PROMPTS="${PHASE3_PROMPTS:?record the ShareGPT input set}"
    export PHASE3_VERICACHE_SHA="${PHASE3_VERICACHE_SHA:?record the VeriCache commit}"
fi
if [[ "$mode" == calibration ]]; then
    duration=480
    export PHASE3_VERICACHE_SHA="${PHASE3_VERICACHE_SHA:?record the VeriCache commit}"
fi
if [[ "$mode" == stage1 ]]; then
    duration="${PHASE3_STAGE1_DURATION:-760}"
    export PHASE3_STAGE1_JOB="${PHASE3_STAGE1_JOB:?provide the experiment JSON}"
    export PHASE3_VERICACHE_SHA="${PHASE3_VERICACHE_SHA:?record the VeriCache commit}"
    python - "$phase3/results/stage0_step2/summary.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as report:
    assert json.load(report)["accepted"]
PY
fi

preflight="$phase3/results/stage0_step2_preflight"
grep -q '23 passed, 2 skipped' "$preflight/pytest-cpu-short-path.log"
grep -q 'Median of 3 exact round trips' "$preflight/bench-phase2-cpu-pressure.log"
[[ $(grep -c ': OK$' "$preflight/model-sha256.log") == 16 ]]

run=$(mktemp -d "$phase3/g.XXXXXX")
if [[ "$mode" == stage1 ]]; then
    out=$(mktemp -d "$phase3/results/stage1_run.XXXXXX")
else
    out=$(mktemp -d "$phase3/results/stage0_hw_${mode}.XXXXXX")
fi
lease_id=
worker_pid=

cleanup() {
    local rc=$?
    trap - EXIT
    trap '' HUP INT TERM
    if [[ -n "$worker_pid" ]]; then
        if kill -TERM -- "-$worker_pid" 2>/dev/null; then
            sleep 2
            kill -KILL -- "-$worker_pid" 2>/dev/null || true
        fi
        wait "$worker_pid" 2>/dev/null || true
    fi
    if [[ -n "${PHASE3_RESERVATION_ID:-}" ]]; then
        printf '{"reused_reservation": "%s", "lease": {"id": "%s", "kind": "reservation", "state": "reused", "active_gpu_ids": []}}\n' \
            "$PHASE3_RESERVATION_ID" "$PHASE3_RESERVATION_ID" >"$out/release.json"
    else
        if [[ -z "$lease_id" && -s "$out/claim.json" ]]; then
            lease_id=$(python -c 'import json,sys; print(json.load(sys.stdin)["id"])' <"$out/claim.json") || rc=1
        fi
        if [[ -n "$lease_id" ]]; then
            if ! "$scheduler" claim release "$lease_id" >"$out/release.json"; then
                "$scheduler" claim release "$lease_id" >"$out/release.json" || rc=1
            fi
        fi
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

echo "Results: $out"
mkdir -p "$run/tests" "$run/bench" "$run/target"
cp "$phase3/vericache-python/tests/test_local_vllm_connector.py" \
   "$phase3/vericache-python/tests/test_vllm_connector.py" "$run/tests/"
cp "$phase3/vericache-python/bench/bench_phase2.py" "$run/bench/"
cmp -s "$run/tests/test_local_vllm_connector.py" "$phase3/vericache-python/tests/test_local_vllm_connector.py"
printf '%s\n' "$node" >"$out/numa-node"

if [[ -n "${PHASE3_RESERVATION_ID:-}" ]]; then
    "$scheduler" status --gpus "$@" >"$out/claim.json"
    python - "$out/claim.json" "$PHASE3_RESERVATION_ID" "$duration" "$@" <<'PY'
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

status_file, res_id, duration, *expected_ids = sys.argv[1:]
data = json.loads(Path(status_file).read_text())
leases = [l for l in data.get("my_allocations", []) if l.get("id") == res_id]
assert len(leases) == 1, f"reservation {res_id} not found in my_allocations"
lease = leases[0]
assert lease["kind"] == "reservation" and lease["state"] == "active", lease
assert set(expected_ids).issubset(set(lease["active_gpu_ids"])), lease
minimum_end = datetime.now(timezone.utc) + timedelta(seconds=int(duration) + 15)
assert datetime.fromisoformat(lease["end"]) >= minimum_end, lease
PY
else
    "$scheduler" claim acquire --gpus "$@" >"$out/claim.json"
    lease_id=$(python -c 'import json,sys; print(json.load(sys.stdin)["id"])' <"$out/claim.json")
    python - "$out/claim.json" "$duration" "$@" <<'PY'
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

claim_file, duration, *expected_ids = sys.argv[1:]
lease = json.loads(Path(claim_file).read_text())["lease"]
assert lease["kind"] == "claim" and lease["state"] == "active", lease
assert sorted(lease["active_gpu_ids"]) == sorted(expected_ids), lease
minimum_end = datetime.now(timezone.utc) + timedelta(seconds=int(duration) + 15)
assert datetime.fromisoformat(lease["end"]) >= minimum_end, lease
PY
fi

printf -v gpu_list '%s,' "$@"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${gpu_list%,}
export TMPDIR="$run"
export PHASE3_TEST_DIR="$run" PHASE3_RESULT_DIR="$out"
export PHASE3_GPU_COUNT="$#"

workload() {
    set -e -o pipefail
    cd "$PHASE3_TEST_DIR"
    python - <<'PY' | tee "$PHASE3_RESULT_DIR/cuda-preflight.log"
import os

import torch

assert torch.cuda.device_count() == int(os.environ["PHASE3_GPU_COUNT"])
print("CUDA_VISIBLE_DEVICES =", os.environ["CUDA_VISIBLE_DEVICES"])
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
    assert "H20" in torch.cuda.get_device_name(index)
PY
    if [[ $1 == local ]]; then
        python -m pytest -v -p no:cacheprovider --basetemp "$PHASE3_TEST_DIR/p" \
            tests/test_local_vllm_connector.py 2>&1 | tee "$PHASE3_RESULT_DIR/pytest.log"
        grep -Eq '\[cuda-module\] PASSED' "$PHASE3_RESULT_DIR/pytest.log"
        grep -Eq '\[cuda-registry\] PASSED' "$PHASE3_RESULT_DIR/pytest.log"
        if grep -Eq 'skipped|SKIPPED' "$PHASE3_RESULT_DIR/pytest.log"; then
            return 1
        fi
        python bench/bench_phase2.py --device cuda --pressure --repeats 3 2>&1 \
            | tee "$PHASE3_RESULT_DIR/bench-phase2-cuda-pressure.log"
        if compgen -G "$PHASE3_TEST_DIR/target/bench-phase2-*" >/dev/null; then
            return 1
        fi
    elif [[ $1 == gdr ]]; then
        python -m pytest -v -s -p no:cacheprovider --basetemp "$PHASE3_TEST_DIR/p" \
            tests/test_vllm_connector.py 2>&1 | tee "$PHASE3_RESULT_DIR/pytest.log"
        grep -q '3 passed' "$PHASE3_RESULT_DIR/pytest.log"
        grep -q 'PASS test_vllm_connector_2node_prefill_decode_gdr' "$PHASE3_RESULT_DIR/pytest.log"
        [[ -z $(find "$PHASE3_TEST_DIR/target/py-e2e-nvme" -mindepth 1 -print -quit) ]]
    elif [[ $1 == layout ]]; then
        export PYTHONPATH="$phase3/vericache-python/bench/phase3:$PYTHONPATH"
        python "$phase3/vericache-python/bench/phase3/stage0_layout.py"
    elif [[ $1 == correctness ]]; then
        export PYTHONPATH="$phase3/vericache-python/bench/phase3:$PYTHONPATH"
        python "$phase3/vericache-python/bench/phase3/stage0_correctness.py"
    elif [[ $1 == calibration ]]; then
        export PYTHONPATH="$phase3/vericache-python/bench/phase3:$PYTHONPATH"
        python "$phase3/vericache-python/bench/phase3/stage0_calibration.py"
    else
        export PYTHONPATH="$phase3/vericache-python/bench/phase3:$PYTHONPATH"
        python "$phase3/vericache-python/bench/phase3/stage1.py"
    fi
}
export -f workload
export phase3

setsid numactl --cpunodebind="$node" --membind="$node" \
    timeout --foreground --kill-after=5s "$duration" bash -o pipefail -c \
    'workload "$1" 2>&1 | tee "$PHASE3_RESULT_DIR/driver.log"' _ "$mode" &
worker_pid=$!
wait "$worker_pid"
