# VeriCache

Verus-verified, location-transparent distributed KV cache spanning GPU VRAM, pinned Host DRAM, local NVMe (`io_uring`), and RDMA / GPUDirect RDMA pools. Every proof across the workspace is cross-verified under both **Z3 4.16.0** and **CVC5 1.3.4** (`Z3 ∧ CVC5`), backed by strict AST/TCB audits and real-hardware E2E tests.

## Workspace Crates

| Crate | Layer | Role | TCB Status |
|---|---|---|---|
| `vericache-foundation` | Layer 0 | Abstract asynchronous access `History`, race-freedom & read-value theorems, `HistoryTokens` & `WriteOnceTokens` PCMs | **Zero TCB** (92 verified) |
| `vericache-transfer` | Layer 1 | `SlotLoc`/`SlotRef`/`Seq<u8>` domain, linear single-key slot tokens, Host/CUDA/NVMe/RDMA wrappers, Option-A IBTA/GDR conformance, `Channel<T, M>` | Confined strictly to 6 `src/*/ffi.rs` + 2 C shims + `build.rs` (343 verified) |
| `vericache-core` | Layer 2 | Hardware-free `CacheCore`, pluggable `backends` (`host`/`cuda`/`nvme`/`rdma`/`compose`), `PeerSession<'a, T>`, `LocalEngine<'a>`, 4-tier `DistributedEngine<'a, T>` (`HBM + DRAM + NVMe + Peer RDMA/GDR`) | **Zero TCB**, `--no-cheating` enforced (775 verified) |

## Quickstart

All verification, audits, formatting, and builds run inside the pinned `/opt/vericache` container (`localhost/vericache-dev:13.0.2`) via `image/dev.sh`:

```bash
# Daily fast gate (Z3 verify + --no-cheating on core + AST/TCB audit + verusfmt)
./image/dev.sh exec just check

# High-assurance dual-SMT gate (Z3 AND CVC5 + AST/TCB audit + verusfmt)
./image/dev.sh exec just check-dual

# Audit TCB whitelist and tree-sitter-verus ast-grep rules only
./image/dev.sh exec just audit

# Real-hardware E2E suite (GPU + ConnectX RDMA + NVMe, zero silent skips)
./image/dev.sh hw <gpu0,gpu1> <uverbs_dev> <rdma_dev> cargo test --workspace --offline
```

For erased Rust compilation only:

```bash
./image/dev.sh exec just cargo-check
```

## Documentation

The source code and Verus `requires`/`ensures` contracts are the canonical specification. The docs below cover architecture, trust boundaries, tooling, and upcoming milestones:

- [Technical Report (PDF)](docs/tech-report/vericache-tech-report.pdf) ([LaTeX source](docs/tech-report/vericache-tech-report.tex)) — comprehensive formal model, Option-A Type-2B MW isolation protocol, 9-file TCB audit, and Suites 1--5 evaluation (up to 600 GiB 2-node 4-tier cluster).
- [Architecture & Memory Model](docs/architecture.md) — 3-layer split, `SlotLoc + SlotRef + Value = Seq<u8>`, service guarantees, zero-`unsafe` borrow graph, and `core_transition`.
- [TCB & Audit Gates](docs/tcb-and-audit.md) — dual-SMT + `--no-cheating`, all 10 `tree-sitter-verus` `ast-grep` rules, and the exact FFI/C TCB whitelist.
- [Development & Hardware Testing](docs/development.md) — container toolchain, `just` recipes, DevContainer setup, and GPU/RDMA device pairing.
- [Roadmap](docs/roadmap.md) — completed local multi-tier and Milestone 3 distributed RDMA/GDR stack (`PeerSession`, `DistributedEngine`, `distributed_e2e`; `policy.rs` and `engine/shard.rs` decoupled outside the verified core).
