#!/usr/bin/env bash
set -euo pipefail

IMAGE="${VERICACHE_IMAGE:-localhost/vericache-dev:13.0.2}"
BENCH_IMAGE="${VERICACHE_BENCH_IMAGE:-localhost/vericache-bench:13.0.2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NETWORK="${VERICACHE_BUILD_NETWORK:-}"
PROXY="${VERICACHE_BUILD_PROXY:-}"
GID_INDEX="${VERICACHE_RDMA_GID_INDEX:-3}"

build_args=()
if [[ -n "$NETWORK" ]]; then
  build_args+=(--network="$NETWORK")
fi
if [[ -n "$PROXY" ]]; then
  build_args+=(
    --build-arg "http_proxy=$PROXY"
    --build-arg "https_proxy=$PROXY"
    --build-arg "HTTP_PROXY=$PROXY"
    --build-arg "HTTPS_PROXY=$PROXY"
  )
fi

cmd="${1:-help}"
shift || true

case "$cmd" in
  build)
    exec podman build "${build_args[@]}" -t "$IMAGE" "$ROOT/image"
    ;;
  exec)
    exec podman run --rm \
      --http-proxy=false \
      --security-opt=label=disable \
      -v "$ROOT:/workspace" \
      -w /workspace \
      -e CARGO_NET_OFFLINE=true \
      "$IMAGE" "$@"
    ;;
  shell)
    exec podman run --rm -it \
      --http-proxy=false \
      --security-opt=label=disable \
      -v "$ROOT:/workspace" \
      -w /workspace \
      -e CARGO_NET_OFFLINE=true \
      "$IMAGE" bash
    ;;
  hw)
    # Usage: image/dev.sh hw <gpu0,gpu1> <uverbs_dev> <rdma_dev> <cmd...>
    gpus="$1"; uverbs="$2"; rdma_dev="$3"; shift 3
    dev_args=()
    IFS=',' read -ra gpu_list <<< "$gpus"
    for g in "${gpu_list[@]}"; do
      dev_args+=(--device "nvidia.com/gpu=$g")
    done
    exec podman run --rm \
      --http-proxy=false \
      --net=host \
      "${dev_args[@]}" \
      --device "/dev/infiniband/$uverbs" \
      --device /dev/infiniband/rdma_cm \
      --ulimit memlock=-1 \
      --security-opt seccomp=unconfined \
      --security-opt=label=disable \
      -v "$ROOT:/workspace" \
      -w /workspace \
      -e CARGO_NET_OFFLINE=true \
      -e VERICACHE_REQUIRE_HW=1 \
      -e "VERICACHE_RDMA_DEVICE=$rdma_dev" \
      -e "VERICACHE_RDMA_GID_INDEX=$GID_INDEX" \
      "$IMAGE" "$@"
    ;;
  bench-build)
    exec podman build \
      -f "$ROOT/image/Containerfile.bench" \
      "${build_args[@]}" \
      -t "$BENCH_IMAGE" "$ROOT/image"
    ;;
  bench-exec)
    exec podman run --rm \
      --http-proxy=false \
      --security-opt=label=disable \
      -v "$ROOT:/workspace" \
      -w /workspace \
      -e CARGO_NET_OFFLINE=true \
      "$BENCH_IMAGE" "$@"
    ;;
  bench-shell)
    exec podman run --rm -it \
      --http-proxy=false \
      --security-opt=label=disable \
      -v "$ROOT:/workspace" \
      -w /workspace \
      -e CARGO_NET_OFFLINE=true \
      "$BENCH_IMAGE" bash
    ;;
  bench-hw)
    # Usage: image/dev.sh bench-hw <gpu0,gpu1> <uverbs_dev> <rdma_dev> <cmd...>
    gpus="$1"; uverbs="$2"; rdma_dev="$3"; shift 3
    dev_args=()
    IFS=',' read -ra gpu_list <<< "$gpus"
    for g in "${gpu_list[@]}"; do
      dev_args+=(--device "nvidia.com/gpu=$g")
    done
    extra_mounts=()
    if [[ -n "${VERICACHE_EXTRA_MOUNTS:-}" ]]; then
      IFS=':' read -ra mount_list <<< "$VERICACHE_EXTRA_MOUNTS"
      for d in "${mount_list[@]}"; do
        if [[ -n "$d" && -d "$d" ]]; then
          extra_mounts+=(-v "$d:$d")
        fi
      done
    fi
    # Podman forbids combining --shm-size with --ipc=host; --ipc=host shares host /dev/shm directly.
    exec podman run --rm \
      --http-proxy=false \
      --net=host \
      --ipc=host \
      "${dev_args[@]}" \
      --device "/dev/infiniband/$uverbs" \
      --device /dev/infiniband/rdma_cm \
      --ulimit memlock=-1 \
      --security-opt seccomp=unconfined \
      --security-opt=label=disable \
      -v "$ROOT:/workspace" \
      "${extra_mounts[@]}" \
      -w /workspace \
      -e CARGO_NET_OFFLINE=true \
      -e VERICACHE_REQUIRE_HW=1 \
      -e "VERICACHE_RDMA_DEVICE=$rdma_dev" \
      -e "VERICACHE_RDMA_GID_INDEX=$GID_INDEX" \
      "$BENCH_IMAGE" "$@"
    ;;
  *)
    echo "Usage: image/dev.sh {build|exec <cmd...>|shell|hw <gpus> <uverbs> <rdma_dev> <cmd...>|bench-build|bench-exec <cmd...>|bench-shell|bench-hw <gpus> <uverbs> <rdma_dev> <cmd...>}" >&2
    exit 2
    ;;
esac
