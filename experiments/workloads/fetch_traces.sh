#!/usr/bin/env bash
set -euo pipefail

# Official Mooncake FAST'25 KV-cache traces (pre-downloaded in /workspace/target/bench-traces).
TRACE_DIR="${1:-/workspace/target/bench-traces}"
mkdir -p "$TRACE_DIR"

CONV="$TRACE_DIR/conversation_trace.jsonl"
TOOL="$TRACE_DIR/toolagent_trace.jsonl"

if [[ -s "$CONV" && -s "$TOOL" ]]; then
  echo "Mooncake FAST'25 traces present:"
  wc -l "$CONV" "$TOOL"
  exit 0
fi

BASE_URL="https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces"
curl -fL "$BASE_URL/conversation_trace.jsonl" -o "$CONV"
curl -fL "$BASE_URL/toolagent_trace.jsonl" -o "$TOOL"
wc -l "$CONV" "$TOOL"
