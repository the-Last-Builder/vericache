"""Check vLLM usage and prefix reuse against request-sim's exact inflated prompts."""

import json
import os
from pathlib import Path
from struct import Struct
import subprocess
from urllib.request import urlopen

from transformers import AutoTokenizer

from prepare_traces import TRACES, read_rows, write_rows
from stage0_correctness import MODEL, PHASE3, request
from stage0_layout import model_command, reserve_port, start_server, write_manifest


COUNT = Struct("<Q")
HEADER = Struct("<QQBQ")


def read_cache(path):
    entries = []
    with path.open("rb") as source:
        count, = COUNT.unpack(source.read(COUNT.size))
        for _ in range(count):
            input_length, output_length, kind, length = HEADER.unpack(source.read(HEADER.size))
            text = source.read(length)
            assert kind == 0 and len(text) == length
            entries.append((input_length, output_length, text))
        assert not source.read(1)
    return entries


def calibration_slice(name, preparation, run, tokenizer):
    rows = read_rows(preparation / f"{name}.jsonl")
    cache = read_cache(preparation / f"{name}.prompts.bin")
    assert len(cache) == len(rows)
    # Preserve six consecutive prompts and add the longest safe prompt.
    longest = max(range(len(rows)), key=lambda i: rows[i]["input_length"])
    indices = list(dict.fromkeys([*range(6), longest]))
    probe_rows, prompts = [], []
    dataset = run / f"{name}.jsonl"
    cache_file = run / f"{name}.prompts.bin"
    with cache_file.open("wb") as output:
        output.write(COUNT.pack(len(indices)))
        for sequence, index in enumerate(indices):
            row = rows[index]
            input_length, _, raw_text = cache[index]
            assert input_length == row["input_length"]
            output_length = min(16, row["output_length"])
            if index == longest and 32768 - input_length <= 384:
                # Exercise the max_tokens clamp with an input very close to the context limit.
                output_length = 32768 - input_length + 32
            probe_rows.append({**row, "timestamp": 4500 * sequence, "output_length": output_length})
            message = [{"role": "user", "content": raw_text.decode("utf-8")}]
            prompts.append(tokenizer.apply_chat_template(
                message, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            ))
            output.write(HEADER.pack(input_length, output_length, 0, len(raw_text)))
            output.write(raw_text)
    write_rows(dataset, probe_rows)
    return dataset, cache_file, probe_rows, prompts


def common_prefix(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def analyze(rows, requests, prompts):
    responses = sorted(requests, key=lambda row: float(row["s_time"]))
    assert len(responses) == len(prompts)
    details = []
    have_reuse = False
    for index, (row, response, tokens) in enumerate(zip(rows, responses, prompts, strict=True)):
        assert response["status"] == "200" and "stream_error" not in response, response
        assert row["input_length"] == int(response["input_length"]), response
        # Earlier requests define an upper bound even when their decode overlaps
        # this dispatch. Equality still requires the shared blocks to be ready.
        best = max((common_prefix(tokens, before) for before in prompts[:index]), default=0)
        bound = min(best, len(tokens) - 1) // 64 * 64
        actual = int(response.get("cached_tokens", 0))
        assert actual == bound, (index, actual, bound, response)
        assert int(response["prompt_tokens"]) == len(tokens), response
        output_cap = max(1, min(row["output_length"], 32768 - row["input_length"] - 32))
        assert int(response["actual_output_tokens"]) == output_cap, response
        have_reuse |= actual > 0
        details.append({
            "index": index,
            "trace_input_tokens": row["input_length"],
            "prompt_tokens": len(tokens),
            "chat_template_delta": len(tokens) - row["input_length"],
            "cached_tokens": actual,
            "prefix_sharing_upper_bound": bound,
            "output_length": row["output_length"],
            "completion_tokens": output_cap,
        })
    assert have_reuse, "no reused prefix observed in calibration"
    return details


def main():
    directory = Path(os.environ["PHASE3_RESULT_DIR"])
    run = Path(os.environ["PHASE3_TEST_DIR"])
    preparation = PHASE3 / "results/stage0_trace_preparation"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    port = reserve_port()
    endpoint = f"http://127.0.0.1:{port}"
    argv = model_command(MODEL, 64, port)
    write_manifest(directory, MODEL, argv)
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    summary = {}
    with start_server(directory, argv, endpoint):
        for name in TRACES:
            request(endpoint, "/reset_prefix_cache", {})
            dataset, cache, rows, prompts = calibration_slice(name, preparation, run, tokenizer)
            output = directory / f"{name}.requests.jsonl"
            replay = [
                str(PHASE3 / "bin/request-sim"),
                "--dataset", "mooncake", "--dataset-path", str(dataset),
                "--api", "openai", "--endpoint", endpoint + "/v1/chat/completions",
                "--model-name", "Qwen3-30B-A3B",
                "--tokenizer", str(MODEL / "tokenizer.json"),
                "--tokenizer-config", str(MODEL / "tokenizer_config.json"),
                "--cache", "file", "--cache-path", str(cache),
                "--stream", "--ignore-eos", "--disable-thinking", "--context-length", "32768",
                "--mode", "trace-replay", "--scale-factor", "1", "--time-in-secs", "45",
                "--num-producer", "8", "--channel-capacity", "1", "--threads", "4",
                "--output-path", str(output),
                "--summary-path", str(directory / f"{name}.request-sim-summary.json"),
            ]
            (directory / f"{name}.replay-argv.json").write_text(json.dumps(replay, indent=2) + "\n")
            with (directory / f"{name}.request-sim.log").open("w") as log:
                subprocess.run(replay, check=True, timeout=95, stdout=log, stderr=subprocess.STDOUT)
            summary[name] = analyze(rows, read_rows(output), prompts)
        with urlopen(endpoint + "/metrics", timeout=10) as response:
            (directory / "server-metrics.prom").write_bytes(response.read())
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Trace calibration passed for both workloads.", flush=True)


if __name__ == "__main__":
    main()
