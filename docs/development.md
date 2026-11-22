# Development & Hardware Testing Guide

## 1. Container Toolchain (`/opt/vericache`)

VeriCache builds and verifies offline inside the container image `localhost/vericache-dev:13.0.2` (defined in `image/Containerfile`, pinned versions in `image/artifacts.env` and `verus.pin`):

- **Base**: `nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04` (CUDA toolkit, `ibverbs-providers`, `libibverbs-dev`)
- **Rust / Verus**: Rust `1.99.1`, Verus `0.2026.11.02.8bf68ee`
- **SMT Solvers**: Z3 `4.16.0` (`VERUS_Z3_PATH`), CVC5 `1.3.4` (`VERUS_CVC5_PATH`)
- **Tooling**: `verusfmt` `v0.7.4`, `ast-grep` `0.51.3` + `tree-sitter-verus` (`0c939ef2`), `just` `1.59.0`

Use `image/dev.sh` from the host:

```bash
./image/dev.sh exec <cmd...>   # Run offline CPU command inside container
./image/dev.sh shell           # Interactive shell in /workspace
./image/dev.sh build           # Rebuild container image from image/Containerfile
```

VS Code / Cursor DevContainer configuration is provided in `.devcontainer/devcontainer.json` with `CARGO_NET_OFFLINE=true`, `rust-analyzer`, and `ast-grep-vscode`.

---

## 2. Common `just` Recipes

Root `Justfile` (run via `./image/dev.sh exec just <recipe>`):

| Recipe | Purpose |
|---|---|
| `verify` | Verify all 3 crates under Z3 (`vericache-core` with `--no-cheating`) |
| `verify-cvc5` | Verify all 3 crates under CVC5 in isolated `target/cvc5` (`--fwd-verus-args-to roots -- -V cvc5`, `--no-cheating` on `core`) |
| `verify-dual` | Run both `verify` and `verify-cvc5` (`Z3 ∧ CVC5`) |
| `audit-rules` | Run `ast-grep test` on all `rule-tests/*-test.yml` |
| `audit` | Run `audit-rules`, `./scripts/audit-tcb.sh`, and per-crate `ast-grep` scans |
| `fmt` / `fmt-check` | Format or check all Rust/Verus sources and tests via `verusfmt` |
| `check` | Fast daily gate: `verify audit fmt-check` |
| `check-dual` | Release gate: `verify-dual audit fmt-check` |
| `cargo-check` | Run erased compilation (`cargo check --workspace --tests`) |

Per-crate checks can also be invoked inside each crate directory (e.g., `just --justfile vericache-core/Justfile audit`).

---

## 3. Real-Hardware E2E Tests (GPU + ConnectX-7 RDMA + NVMe)

`image/dev.sh hw <gpu0,gpu1> <uverbs_dev> <rdma_dev> <cmd...>` mounts the selected NVIDIA GPUs, `/dev/infiniband/<uverbs_dev>`, `/dev/infiniband/rdma_cm`, sets `--ulimit memlock=-1`, and exports:

- `VERICACHE_REQUIRE_HW=1` — **forbids silent test skips** if hardware init fails.
- `VERICACHE_RDMA_DEVICE=<rdma_dev>` and `VERICACHE_RDMA_GID_INDEX` (default `3`, override via environment variable to match your RoCEv2 IPv4 GID index from `show_gids`).

Before running hardware tests, inspect GPU/NIC PCIe affinity with `nvidia-smi topo -m` and `ibv_devinfo` / `ibdev2netdev`, and select a PIX-paired GPU pair + InfiniBand HCA on the same PCIe switch (for example, GPUs `0,1` with `uverbs0` / `mlx5_0`):

```bash
./image/dev.sh hw <gpu0,gpu1> <uverbs_dev> <rdma_dev> cargo test --workspace --offline
# Example:
VERICACHE_RDMA_GID_INDEX=3 ./image/dev.sh hw 0,1 uverbs0 mlx5_0 cargo test --workspace --offline
```
