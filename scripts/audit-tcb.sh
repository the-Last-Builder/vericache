#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail=0

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
  find vericache-foundation vericache-transfer vericache-core \
    -type f -name '*ffi*.rs' | LC_ALL=C sort
)"
expected_ffi="$(cat <<'EOF' | LC_ALL=C sort
vericache-transfer/src/context/ffi.rs
vericache-transfer/src/cuda/ffi.rs
vericache-transfer/src/host/ffi.rs
vericache-transfer/src/nvme/ffi.rs
vericache-transfer/src/rdma/ffi.rs
vericache-transfer/src/transport/ffi.rs
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

# 5. Audit exclusion globs in Justfiles may not be widened beyond the 6 ffi.rs files
actual_globs="$(
  python3 - <<'PY'
import pathlib, re
expected_glob = "'!src/{context,host,cuda,rdma,nvme,transport}/ffi.rs'"
bad = []
for jf in ["Justfile", "vericache-foundation/Justfile", "vericache-transfer/Justfile", "vericache-core/Justfile"]:
    txt = pathlib.Path(jf).read_text()
    for i, line in enumerate(txt.splitlines(), 1):
        if "--globs" in line:
            if jf != "vericache-transfer/Justfile":
                bad.append(f"{jf}:{i}: unexpected --globs outside vericache-transfer/Justfile: {line.strip()}")
            else:
                globs = re.findall(r"--globs\s+(\S+)", line)
                for g in globs:
                    if g != expected_glob:
                        bad.append(f"{jf}:{i}: unexpected glob {g} (expected {expected_glob})")
if bad:
    print("\n".join(bad))
PY
)"
if [[ -n "$actual_globs" ]]; then
  echo "TCB AUDIT FAILED [justfile-globs]:" >&2
  echo "$actual_globs" >&2
  fail=1
fi

# 6. No Cargo.toml in the workspace may configure a custom `build = ...` script
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

# 7. Every crate Justfile must retain all expected ast-grep scan invocations
actual_just_scans="$(
  python3 - <<'PY'
import pathlib

expected = {
    "vericache-foundation/Justfile": [
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-assume-admit-truncate.yml src",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-external-escape.yml src",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-external-body.yml src",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-unsafe.yml src",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-unverified-code.yml src",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-panicking-stubs.yml src",
    ],
    "vericache-transfer/Justfile": [
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-assume-admit-truncate.yml src tests",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-external-escape.yml src tests",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-external-body.yml --globs '!src/{context,host,cuda,rdma,nvme,transport}/ffi.rs' src tests",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-unsafe.yml --globs '!src/{context,host,cuda,rdma,nvme,transport}/ffi.rs' src tests",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-unverified-code.yml --globs '!src/{context,host,cuda,rdma,nvme,transport}/ffi.rs' src",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-panicking-stubs.yml src",
    ],
    "vericache-core/Justfile": [
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-assume-admit-truncate.yml src tests",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-external-escape.yml src tests",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-external-body.yml src tests",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-unsafe.yml src tests",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-unverified-code.yml src",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-panicking-stubs.yml src",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-hardware-import-in-core.yml src/spec.rs src/provenance src/timeline src/frame src/catalog src/kernel",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-cuda-dep.yml src/backends/host src/backends/nvme",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-nvme-dep.yml src/backends/host src/backends/cuda src/backends/rdma",
        "ast-grep scan --custom-languages allow -c ../sgconfig.yml -r ../rules/no-remote-dep.yml src/backends/host src/backends/cuda src/backends/nvme",
    ],
}

missing = []
for jf, lines in expected.items():
    present = {line.strip() for line in pathlib.Path(jf).read_text().splitlines()}
    for req in lines:
        if req not in present:
            missing.append(f"{jf}: missing required scan: {req}")
if missing:
    print("\n".join(missing))
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
