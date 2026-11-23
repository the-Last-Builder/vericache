# Trusted Computing Base & Verification Audit Gates

VeriCache enforces verification integrity through three independent layers executed by `just check` / `just check-dual` and `just audit`.

---

## 1. Layer 1: Dual-SMT AND Gate & `--no-cheating`

- **Dual-SMT (`just verify-dual`)**: Every proof item across all three verified crates (`vericache-foundation`, `vericache-transfer`, `vericache-core`) must verify with **both** `Z3 4.16.0` (with `-V spinoff-all`) and `CVC5 1.3.4` (`target/cvc5` with `-V cvc5 -V spinoff-all`; `vericache-core` runs CVC5 with `--rlimit 80`).
- **Compiler `--no-cheating`**: Enabled on `vericache-core` (`just verify`, `just verify-cvc5`), where Verus itself rejects `assume`, `admit`, `external_body`, `assume_specification`, and check-weakening flags.
  - `vericache-foundation` uses `tokenized_state_machine!`, whose macro expansion triggers Verus's `--no-cheating` syntactic check on macro-generated internals; it is audited by Layer 2 instead.
  - `vericache-transfer` isolates its hardware/network axioms inside the 6 whitelisted `src/*/ffi.rs` files and is audited by Layers 2 and 3.
  - `vericache-bench` constructs all linear capabilities via `ShardAllocator` / `ShardCapabilityBundle` with zero `Tracked::assume_new()` / `Ghost::assume_new()` calls.

---

## 2. Layer 2: `tree-sitter-verus` `ast-grep` Rules (`rules/`)

All 10 rules are regression-tested (`rule-tests/*-test.yml`, run via `ast-grep test --custom-languages allow`) using the pinned `tree-sitter-verus` parser (`sgconfig.yml`). Attribute rules inspect `kind: attribute` (covering `#[...]`, `#![...]`, and `#[cfg_attr(..., ...)]`).

| Rule ID | Scope | Enforced Constraint |
|---|---|---|
| `no-assume-admit-truncate` | All `src` (all crates) + `transfer/tests`, `core/tests` | Forbids `assume`, `assume_`, `admit`, `assume_specification`, and `verifier::` / `verus::` `{truncate, rlimit(infinity), inline_air_stmt, skip, axiom, internal}` (bounded numeric `#[verifier::rlimit(N)]` is permitted when a cohesive state-machine proof needs extra solver budget) |
| `no-external-escape` | Same as above (including `ffi.rs`) | Forbids `verifier::{external, external_fn_specification, external_type_specification, external_trait_specification, external_trait_extension, allow_in_spec}` everywhere |
| `no-external-body` | `foundation/src`, `core/{src,tests}`, non-FFI `transfer/{src,tests}` | Confines `#[verifier::external_body]` strictly to the 6 `transfer/src/*/ffi.rs` files |
| `no-unsafe` | `foundation/src`, `core/{src,tests}`, non-FFI `transfer/{src,tests}` | Confines `unsafe` blocks/fn/trait/impl and `extern "C"` blocks strictly to the 6 `transfer/src/*/ffi.rs` files |
| `no-unverified-code` | `foundation/src`, `core/src`, non-FFI `transfer/src` | Forbids `fn`/`impl`/`trait`/`extern`/`static` outside `verus!{}` (except `#[cfg(test)]`), runtime-only `cfg(not(verus_only))` logic (only `use` permitted), `macro_rules!`, and `include*!` |
| `no-panicking-stubs` | `src` across all 3 crates | Forbids `todo!`, `unimplemented!`, `panic!`, and `.expect(...)` outside `#[cfg(test)]` |
| `no-hardware-import-in-core` | `vericache-core/src/{spec.rs,provenance,timeline,frame,catalog,kernel}` | Prevents hardware-free core from importing `transfer::{host,cuda,nvme,rdma,transport}` |
| `no-cuda-dep` | `vericache-core/src/backends/{host,nvme}` | Keeps CUDA isolated from Host and NVMe backends |
| `no-nvme-dep` | `vericache-core/src/backends/{host,cuda,rdma}` | Keeps NVMe isolated from Host, CUDA, and RDMA backends |
| `no-remote-dep` | `vericache-core/src/backends/{host,cuda,nvme}` | Keeps RDMA and `transfer::transport` isolated from local storage backends |

**Explicitly permitted non-soundness attributes**: `#[verifier::exec_allows_no_decreases_clause]`, `#[verifier::when_used_as_spec(...)]`, `#[verifier::loop_isolation(false)]`, and bounded numeric `#[verifier::rlimit(N)]`.

---

## 3. Layer 3: Exact TCB Whitelist (`scripts/audit-tcb.sh`)

`just audit` runs `scripts/audit-tcb.sh` to guarantee zero silent TCB expansion. The **entire** trusted boundary consists of these exact 9 files plus standard toolchain/OS assumptions (`vstd`, `verus_state_machines_macros`, Verus, Z3/CVC5, Linux kernel, CUDA driver, OFED verbs):

| File | Assumed Hardware / Platform Premise |
|---|---|
| `vericache-transfer/src/context/ffi.rs` | Cluster-unique `NodeId` process bootstrap (`AtomicBool` single-init guard) and initial linear token partition (further subdivided in verified code via `ShardAllocator`) |
| `vericache-transfer/src/host/ffi.rs` | Aligned host heap allocation/deallocation and synchronous CPU read/write/copy semantics |
| `vericache-transfer/src/cuda/ffi.rs` | `libcudart` device/host allocation, `cudaHostRegister`, RAII stream/event lifecycle guards, stream FIFO execution, `CudaEvent` completion, GPUDirect RDMA visibility |
| `vericache-transfer/src/nvme/ffi.rs` | `O_DIRECT` temporary file, persistent named file, and raw block device (`BLKGETSIZE64`) RAII lifecycle plus Linux `io_uring` SQ/CQ completion semantics |
| `vericache-transfer/src/rdma/ffi.rs` | `libibverbs` RAII-guarded PD/CQ/QP/MR/Type-2B MW resource coverage, RC ordering, inline 62-bit control WR IDs (`< CONTROL_WR_ID_LIMIT`), RDMA read/send completion |
| `vericache-transfer/src/transport/ffi.rs` | Protocol-magic and CRC32-verified byte-frame delivery, non-blocking `try_recv_frame`, and linear token attachment conservation over `ByteTransport` |
| `vericache-transfer/src/nvme/uring.c` | Raw `io_uring_setup` / `io_uring_enter` syscall & shared ring barrier C shim |
| `vericache-transfer/src/rdma/verbs.c` | `libibverbs` struct layout adapters, QP state transitions (`INIT→RTR→RTS`, `ERR→RESET`), `wide_cookie` zero-alloc inline control WR posting & CQ polling |
| `vericache-transfer/build.rs` | Static compilation (`-std=c11 -O2 -fPIC -Wall -Werror`) of `uring.c` and `verbs.c` and linking `cudart` / `ibverbs` |

`scripts/audit-tcb.sh` also verifies that no other `*ffi*.rs`, `.c`/`.h`/`.cu`/`.cpp`/`.s`/`.S`, or `build.rs` exists in the workspace, and that `--globs` exclusions in `Justfile`s never widen beyond `!src/{context,host,cuda,rdma,nvme,transport}/ffi.rs` inside `vericache-transfer/Justfile`.
