# All commands run inside the vericache-dev container (/opt/vericache on PATH).
# From the host, wrap with: image/dev.sh exec just <recipe>

default:
    @just --list

# Verify every proof across the entire workspace (Z3 — fast daily check).
verify:
    cargo verus verify --fwd-verus-args-to roots -p vericache-foundation -- -V spinoff-all
    cargo verus verify --fwd-verus-args-to roots -p vericache-transfer -- -V spinoff-all
    cargo verus verify --fwd-verus-args-to roots -p vericache-core -- -V spinoff-all --no-cheating

# Verify every proof across the entire workspace using CVC5 (isolated target dir + spinoff-all multi-core parallelism).
verify-cvc5:
    cargo verus verify --target-dir target/cvc5 --fwd-verus-args-to roots -p vericache-foundation -- -V cvc5 -V spinoff-all
    cargo verus verify --target-dir target/cvc5 --fwd-verus-args-to roots -p vericache-transfer -- -V cvc5 -V spinoff-all
    cargo verus verify --target-dir target/cvc5 --fwd-verus-args-to roots -p vericache-core -- -V cvc5 -V spinoff-all --rlimit 160 --no-cheating

# Verify every proof with BOTH Z3 and CVC5 (AND cross-verification).
verify-dual: verify verify-cvc5

# Fast inner-loop Z3 check on a single module (and optional extra args such as `--verify-function <pat>`).
focus crate mod *args:
    cargo verus focus --target-dir target/focus -p {{crate}} -- -V spinoff-all --time-expanded --verify-module {{mod}} {{args}}

# Fast inner-loop CVC5 check on a single module (and optional `--verify-function <pat>`).
focus-cvc5 crate mod *args:
    cargo verus focus --target-dir target/focus-cvc5 -p {{crate}} -- -V cvc5 -V spinoff-all --rlimit 160 --time-expanded --verify-module {{mod}} {{args}}

# Quantifier instantiation profiler for a slow CVC5 function.
profile-cvc5 crate mod fn:
    cargo verus focus --target-dir target/focus-cvc5 -p {{crate}} -- -V cvc5 --rlimit 160 --profile --time-expanded --verify-only-module {{mod}} --verify-function {{fn}}

# Run tree-sitter-verus ast-grep rule regression tests.
audit-rules:
    ast-grep test --custom-languages allow --skip-snapshot-tests

# Run full AST-level TCB, whitelist, and architecture boundary audits across all 3 crates.
audit: audit-rules
    ./scripts/audit-tcb.sh
    just --justfile vericache-foundation/Justfile audit
    just --justfile vericache-transfer/Justfile audit
    just --justfile vericache-core/Justfile audit

# Format every Rust/Verus source and test file across the workspace.
fmt:
    find vericache-foundation/src vericache-transfer/src vericache-transfer/tests vericache-core/src vericache-core/tests -name '*.rs' -exec verusfmt {} +

# Check formatting across the workspace.
fmt-check:
    find vericache-foundation/src vericache-transfer/src vericache-transfer/tests vericache-core/src vericache-core/tests -name '*.rs' -exec verusfmt --check {} +

# Fast daily gate: Z3 verify + audit + fmt-check.
check: verify audit fmt-check

# High-assurance dual-SMT gate: (Z3 AND CVC5) verify + audit + fmt-check.
check-dual: verify-dual audit fmt-check

# Check erased-build compilation across all targets.
cargo-check:
    cargo check --workspace --tests

# Compile the English technical report PDF using Tectonic.
report:
    tectonic docs/tech-report/vericache-tech-report.tex
