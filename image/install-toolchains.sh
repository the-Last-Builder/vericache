#!/usr/bin/env bash
set -euo pipefail

source /opt/vericache/image/artifacts.env

WORK="/tmp/vericache-build"
mkdir -p "$WORK" /opt/vericache/{bin,lib,verus,z3,cvc5,rustup,cargo}
trap 'rm -rf "$WORK"' EXIT

fetch() {
    local url="$1"
    local sha="$2"
    local out="$WORK/$3"
    curl -fsSL --retry 3 --retry-delay 2 -o "$out" "$url"
    printf '%s  %s\n' "$sha" "$out" | sha256sum -c - >&2
    echo "$out"
}

echo "=== 1. Install pinned Rust ${RUST_CHANNEL} ==="
rustup_init=$(fetch "$RUSTUP_URL" "$RUSTUP_SHA256" "rustup-init")
chmod +x "$rustup_init"
"$rustup_init" -y --no-modify-path --profile minimal \
    --default-toolchain "$RUST_CHANNEL" \
    --component rustfmt --component clippy --component rustc-dev --component llvm-tools
# Best-effort rust-analyzer for DevContainer IDE support.
/opt/vericache/cargo/bin/rustup component add rust-analyzer || true

echo "=== 2. Install pinned Verus ${VERUS_RELEASE} + matching source ==="
verus_zip=$(fetch "$VERUS_URL" "$VERUS_SHA256" "verus.zip")
verus_src=$(fetch "$VERUS_SOURCE_URL" "$VERUS_SOURCE_SHA256" "verus-src.tar.gz")
unzip -q "$verus_zip" -d "$WORK/verus-bin"
tar --no-same-owner --no-same-permissions -xzf "$verus_src" --strip-components=1 -C /opt/vericache/verus
mv "$WORK/verus-bin/verus-x86-linux" /opt/vericache/verus/bin
echo "$VERUS_REVISION" > /opt/vericache/verus/.revision

echo "=== 3. Install pinned Z3 + CVC5 (${CVC5_VERSION}) ==="
z3_zip=$(fetch "$Z3_URL" "$Z3_SHA256" "z3.zip")
unzip -q "$z3_zip" -d "$WORK/z3"
mv "$WORK"/z3/z3-*/* /opt/vericache/z3/
ln -sfn /opt/vericache/z3/bin/z3 /opt/vericache/verus/bin/z3

cvc5_zip=$(fetch "$CVC5_URL" "$CVC5_SHA256" "cvc5.zip")
unzip -q "$cvc5_zip" -d "$WORK/cvc5"
mv "$WORK"/cvc5/cvc5-*/* /opt/vericache/cvc5/
ln -sfn /opt/vericache/cvc5/bin/cvc5 /opt/vericache/verus/bin/cvc5

echo "=== 4. Install verusfmt, ast-grep, just, ripgrep ==="
verusfmt_tar=$(fetch "$VERUSFMT_URL" "$VERUSFMT_SHA256" "verusfmt.tar.xz")
tar -xJf "$verusfmt_tar" -C "$WORK"
find "$WORK" -maxdepth 3 -name verusfmt -type f -exec install -m 755 {} /opt/vericache/bin/verusfmt \;

ast_grep_zip=$(fetch "$AST_GREP_URL" "$AST_GREP_SHA256" "ast-grep.zip")
mkdir -p "$WORK/ast-grep"
unzip -q "$ast_grep_zip" -d "$WORK/ast-grep"
install -m 755 "$WORK/ast-grep/ast-grep" /opt/vericache/bin/ast-grep
install -m 755 "$WORK/ast-grep/sg" /opt/vericache/bin/sg

just_tar=$(fetch "$JUST_URL" "$JUST_SHA256" "just.tar.gz")
mkdir -p "$WORK/just"
tar -xzf "$just_tar" -C "$WORK/just"
install -m 755 "$WORK/just/just" /opt/vericache/bin/just

rg_tar=$(fetch "$RG_URL" "$RG_SHA256" "rg.tar.gz")
mkdir -p "$WORK/rg"
tar -xzf "$rg_tar" -C "$WORK/rg"
find "$WORK/rg" -maxdepth 3 -name rg -type f -exec install -m 755 {} /opt/vericache/bin/rg \;

echo "=== 5. Build secure-foundations/tree-sitter-verus (${TREE_SITTER_VERUS_REV}) ==="
ts_tar=$(fetch "$TREE_SITTER_VERUS_URL" "$TREE_SITTER_VERUS_SHA256" "tree-sitter-verus.tar.gz")
mkdir -p "$WORK/ts-verus"
tar -xzf "$ts_tar" --strip-components=1 -C "$WORK/ts-verus"
cc -shared -fPIC -O2 -I "$WORK/ts-verus/src" \
    "$WORK/ts-verus/src/parser.c" "$WORK/ts-verus/src/scanner.c" \
    -o /opt/vericache/lib/libtree-sitter-verus.so
echo "$TREE_SITTER_VERUS_REV" > /opt/vericache/lib/tree-sitter-verus.rev

chmod -R a+rX /opt/vericache
chmod -R a+rwX /opt/vericache/cargo

echo "=== 6. Pre-populate Cargo cache for vstd + state_machines_macros (offline builds) ==="
export PATH="/opt/vericache/bin:/opt/vericache/verus/bin:/opt/vericache/z3/bin:/opt/vericache/cvc5/bin:/opt/vericache/cargo/bin:$PATH"
export VERUS_Z3_PATH="/opt/vericache/z3/bin/z3"
export VERUS_CVC5_PATH="/opt/vericache/cvc5/bin/cvc5"
mkdir -p "$WORK/prefetch/src"
cat > "$WORK/prefetch/Cargo.toml" <<'EOF'
[package]
name = "prefetch"
version = "0.1.0"
edition = "2024"
[dependencies]
vstd = { path = "/opt/vericache/verus/source/vstd" }
verus_state_machines_macros = { path = "/opt/vericache/verus/source/state_machines_macros" }
EOF
touch "$WORK/prefetch/src/lib.rs"
cargo fetch --manifest-path "$WORK/prefetch/Cargo.toml"
chmod -R a+rwX /opt/vericache/cargo

echo "=== 7. Smoke check installed versions ==="
rustc --version
cargo --version
verus --version
cargo-verus --help | head -n 3
z3 --version
cvc5 --version | head -n 2
verusfmt --version
ast-grep --version
just --version
rg --version
ls -la /opt/vericache/lib/libtree-sitter-verus.so
