#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail=0
verified_crates=(
  vericache-foundation
  vericache-transfer-core
  vericache-transfer
  vericache-kernel
  vericache-core
)

check_exact_list() {
  local label="$1"
  local actual="$2"
  local expected="$3"
  if [[ "$actual" != "$expected" ]]; then
    echo "TCB AUDIT FAILED [$label]:" >&2
    diff -u <(printf '%s\n' "$expected") <(printf '%s\n' "$actual") >&2 || true
    fail=1
  fi
}

# 1. Exact FFI Rust file inventory across all workspace crates
actual_ffi="$(
  find "${verified_crates[@]}" \
    -type f -name '*ffi*.rs' | LC_ALL=C sort
)"
expected_ffi="$(cat <<'EOF' | LC_ALL=C sort
vericache-transfer-core/src/context/ffi.rs
vericache-transfer/src/cuda/ffi.rs
vericache-transfer/src/host/ffi.rs
vericache-transfer/src/nvme/ffi.rs
vericache-transfer/src/rdma/ffi.rs
vericache-transfer-core/src/transport/ffi.rs
EOF
)"
check_exact_list "ffi-rust-files" "$actual_ffi" "$expected_ffi"

# 2. Exact native C/C++/header/assembly file inventory across workspace (excluding third-party baseline harnesses)
actual_native="$(
  find . \
    \( -path './target' -o -path './.git' -o -path './.clice' -o -path './experiments/baselines' \) -prune -o \
    -type f \( -name '*.c' -o -name '*.h' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cu' -o -name '*.s' -o -name '*.S' \) \
    -print | sed 's|^\./||' | LC_ALL=C sort
)"
expected_native="$(cat <<'EOF' | LC_ALL=C sort
vericache-transfer/src/nvme/uring.c
vericache-transfer/src/rdma/verbs.c
EOF
)"
check_exact_list "native-c-files" "$actual_native" "$expected_native"

# 3. Exact build.rs file inventory across workspace (excluding third-party baseline harnesses)
actual_build="$(
  find . \
    \( -path './target' -o -path './.git' -o -path './.clice' -o -path './experiments/baselines' \) -prune -o \
    -type f -name 'build.rs' -print | sed 's|^\./||' | LC_ALL=C sort
)"
expected_build="vericache-transfer/build.rs"
check_exact_list "build-scripts" "$actual_build" "$expected_build"

# 4. Native sources referenced in vericache-transfer/build.rs must match the exact 2 C shims
actual_build_c="$(
  grep -oE '"[^"]+\.(c|h|cc|cpp|cu|s|S)"' vericache-transfer/build.rs \
    | tr -d '"' | LC_ALL=C sort -u
)"
expected_build_c="$(cat <<'EOF' | LC_ALL=C sort
src/nvme/uring.c
src/rdma/verbs.c
EOF
)"
check_exact_list "build-rs-c-sources" "$actual_build_c" "$expected_build_c"

# 5. No Cargo.toml in the workspace may configure a custom `build = ...` script
actual_custom_build="$(
  python3 - <<'PY'
import pathlib, re
bad = []
for toml in sorted(pathlib.Path(".").glob("**/Cargo.toml")):
    if any(p in ("target", ".git", ".clice") for p in toml.parts):
        continue
    for i, line in enumerate(toml.read_text().splitlines(), 1):
        stripped = line.split("#", 1)[0].strip()
        if re.match(r"^build\s*=", stripped):
            bad.append(f"{toml}:{i}: custom build key not allowed: {stripped}")
if bad:
    print("\n".join(bad))
PY
)"
if [[ -n "$actual_custom_build" ]]; then
  echo "TCB AUDIT FAILED [cargo-custom-build]:" >&2
  echo "$actual_custom_build" >&2
  fail=1
fi

# 6. Require the exact AST scans, including the only allowed exclusion globs.
actual_just_scans="$(
  python3 - "${verified_crates[@]}" <<'PY'
import difflib
import pathlib
import re
import sys

scan = "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/"
security_rules = (
    "no-assume-admit-truncate",
    "no-external-escape",
    "no-external-body",
    "no-unsafe",
    "no-unverified-code",
    "no-panicking-stubs",
)
source_only = {"no-unverified-code", "no-panicking-stubs"}
exclude_ffi = {"no-external-body", "no-unsafe", "no-unverified-code"}
ffi_globs = {
    "vericache-transfer-core": "'!src/{context,transport}/ffi.rs'",
    "vericache-transfer": "'!src/{host,cuda,nvme,rdma}/ffi.rs'",
}
drivers = ("host", "cuda", "nvme", "rdma")
architecture = {
    "vericache-transfer": [
        f"no-transfer-{driver}-dep.yml " + " ".join(f"src/{other}" for other in drivers if other != driver)
        for driver in drivers
    ],
    "vericache-kernel": ["no-hardware-import-in-core.yml src"],
    "vericache-core": [
        "no-cuda-dep.yml src/backends/host src/backends/nvme src/engine",
        "no-nvme-dep.yml src/backends/host src/backends/cuda src/backends/rdma",
        "no-remote-dep.yml src/backends/host src/backends/cuda src/backends/nvme",
    ],
}

for crate in ("", *sys.argv[1:]):
    justfile = pathlib.Path(crate, "Justfile")
    actual = []
    for line in justfile.read_text().splitlines():
        line = line.strip()
        if not line.startswith("#") and re.search(r"\bast-grep\s+scan\b|--globs\b", line):
            actual.append(line)

    expected = []
    if crate:
        for rule in security_rules:
            glob = f" --globs {ffi_globs[crate]}" if crate in ffi_globs and rule in exclude_ffi else ""
            targets = "src tests" if crate in {"vericache-transfer", "vericache-core"} and rule not in source_only else "src"
            expected.append(f"{scan}{rule}.yml{glob} {targets}")
        expected.extend(scan + rule for rule in architecture.get(crate, ()))
    if actual != expected:
        print("\n".join(difflib.unified_diff(expected, actual, "required scans", str(justfile), lineterm="")))
PY
)"
if [[ -n "$actual_just_scans" ]]; then
  echo "TCB AUDIT FAILED [justfile-scans]:" >&2
  echo "$actual_just_scans" >&2
  fail=1
fi

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

echo "TCB whitelist OK: 6 ffi.rs, 2 C shims, 1 build.rs"
