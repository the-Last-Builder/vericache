"""Greedy token and top-1 logprob checks on the real Qwen3 model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from struct import pack
from urllib.request import Request, urlopen

from stage0_layout import model_command, reserve_port, start_server, write_manifest


PHASE3 = Path("/data/nvme2/zxw/vericache_phase3")
MODEL = PHASE3.parent / "models/Qwen3-30B-A3B"
SHAREGPT = Path(
    "/home/zxw/.cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered"
) / "snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json"


def select_prompts(output):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    prompts = []
    for row in json.loads(SHAREGPT.read_text()):
        conversation = row["conversations"]
        last_user = next((i for i in range(len(conversation) - 1, -1, -1)
                          if conversation[i]["from"] == "human"), -1)
        if last_user < 2:
            continue
        prefix = conversation[:last_user + 1]
        if any(message["from"] not in roles for message in prefix):
            continue
        messages = [{"role": roles[message["from"]], "content": message["value"]}
                    for message in prefix]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        if not 1024 <= len(ids) <= 3072:
            continue
        prompts.append({
            "sharegpt_id": row["id"],
            "turns": len(messages),
            "prompt_token_ids": ids,
            "sha256": hashlib.sha256(pack(f"!{len(ids)}I", *ids)).hexdigest(),
        })
        if len(prompts) == 20:
            break
    assert len(prompts) == 20
    output.write_text(json.dumps(prompts) + "\n")


def request(endpoint, path, body):
    req = Request(endpoint + path, data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=90) as response:
        return response.read()


def sample(endpoint, token_ids, salt, max_tokens=16):
    response = json.loads(request(endpoint, "/v1/completions", {
        "model": "Qwen3-30B-A3B",
        "prompt": token_ids,
        "temperature": 0,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "return_token_ids": True,
        "logprobs": 1,
        "cache_salt": salt,
    }))
    choice, = response["choices"]
    probabilities = choice["logprobs"]["top_logprobs"]
    assert len(choice["token_ids"]) == max_tokens == len(probabilities)
    return {
        "response_id": response["id"],
        "token_ids": choice["token_ids"],
        "top1_logprobs": [max(token_logprobs.values()) for token_logprobs in probabilities],
        "usage": response["usage"],
    }


def load_events(path):
    return [json.loads(row) for row in path.read_text().splitlines()] if path.is_file() else []


def same_output(a, b):
    return {
        "token_ids_exact": a["token_ids"] == b["token_ids"],
        "top1_logprobs_exact": (
            b"".join(pack("!d", p) for p in a["top1_logprobs"])
            == b"".join(pack("!d", p) for p in b["top1_logprobs"])
        ),
        "max_top1_logprob_delta": max(abs(a - b) for a, b in zip(
            a["top1_logprobs"], b["top1_logprobs"], strict=True,
        )),
    }


def check(endpoint, directory, scenario, prompts, golden):
    rows = []
    loads = directory / "connector-loads.jsonl"
    event_index = 0
    flag = Path(os.environ["PHASE3_TEST_DIR"]) / "inject-late-eviction"
    with (directory / "requests.jsonl").open("w") as output:
        for ordinal, entry in enumerate(prompts):
            tokens = entry["prompt_token_ids"]
            salt = f"stage0-check-{ordinal}"
            request(endpoint, "/reset_prefix_cache?reset_external=false", {})
            cold = sample(endpoint, tokens, salt)
            row = {"prompt_index": ordinal, "prompt_sha256": entry["sha256"], "cold": cold}
            if scenario == "hbm":
                warm = sample(endpoint, tokens, salt)
                row.update({
                    "cold_matches_golden": same_output(cold, golden[ordinal]["cold"]),
                    "warm": warm,
                    "warm_matches_golden": same_output(warm, golden[ordinal]["cold"]),
                    "warm_matches_same_server_cold": same_output(warm, cold),
                })
            elif scenario != "vanilla":
                if scenario == "nvme":
                    # Fill more DRAM than the configured slab with disjoint salted KV.
                    sample(endpoint, tokens * (10240 // len(tokens) + 1),
                           salt + "-nvme-pressure", max_tokens=1)
                request(endpoint, "/reset_prefix_cache?reset_external=false", {})
                warm = sample(endpoint, tokens, salt)
                events = load_events(loads)
                warm_loads = events[event_index:]
                event_index = len(events)
                tier = "nvme_hits" if scenario == "nvme" else "dram_hits"
                row.update({
                    "cold_matches_golden": same_output(cold, golden[ordinal]["cold"]),
                    "warm": warm,
                    "warm_matches_golden": same_output(warm, golden[ordinal]["cold"]),
                    "warm_tier_hits": sum(event["stats_delta"][tier] for event in warm_loads),
                    "warm_load_errors": [error for event in warm_loads for error in event["load_errors"]],
                })
                if scenario == "dram":
                    fault_salt = salt + "-fault"
                    request(endpoint, "/reset_prefix_cache?reset_external=false", {})
                    sample(endpoint, tokens, fault_salt)
                    request(endpoint, "/reset_prefix_cache?reset_external=false", {})
                    flag.touch()
                    after_fault = sample(endpoint, tokens, fault_salt)
                    events = load_events(loads)
                    fault_loads = events[event_index:]
                    event_index = len(events)
                    row.update({
                        "fault": after_fault,
                        "fault_matches_golden": same_output(after_fault, golden[ordinal]["cold"]),
                        "late_eviction_reported": any(
                            event["late_evicted_id"] in event["load_errors"]
                            for event in fault_loads
                        ),
                    })
                    flag.unlink(missing_ok=True)
            output.write(json.dumps(row) + "\n")
            output.flush()
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path)
    args = parser.parse_args()
    if args.prepare:
        select_prompts(args.prepare)
        return

    directory = Path(os.environ["PHASE3_RESULT_DIR"])
    scenario = os.environ["PHASE3_SCENARIO"]
    assert scenario in {"vanilla", "hbm", "dram", "nvme"}
    prompts = json.loads(Path(os.environ["PHASE3_PROMPTS"]).read_text())
    assert len(prompts) == 20
    golden = None
    if scenario != "vanilla":
        golden = [json.loads(row) for row in Path(os.environ["PHASE3_GOLDEN"]).read_text().splitlines()]
        assert [row["prompt_sha256"] for row in golden] == [row["sha256"] for row in prompts]
    port = reserve_port()
    argv = model_command(MODEL, 64, port, scenario)
    write_manifest(directory, MODEL, argv)
    endpoint = f"http://127.0.0.1:{port}"
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    with start_server(directory, argv, endpoint):
        rows = check(endpoint, directory, scenario, prompts, golden)
        with urlopen(endpoint + "/metrics", timeout=10) as response:
            (directory / "server-metrics.prom").write_bytes(response.read())
    summary = {"scenario": scenario, "n_prompts": len(rows)}
    if scenario != "vanilla":
        modes = ["cold_matches_golden", "warm_matches_golden"]
        if scenario == "dram":
            modes.append("fault_matches_golden")
            summary["late_eviction_reported"] = sum(row["late_eviction_reported"] for row in rows)
        for mode in modes:
            summary[mode] = {
                metric: sum(row[mode][metric] for row in rows)
                for metric in ("token_ids_exact", "top1_logprobs_exact")
            }
        if scenario == "hbm":
            mode = "warm_matches_same_server_cold"
            summary[mode] = {
                metric: sum(row[mode][metric] for row in rows)
                for metric in ("token_ids_exact", "top1_logprobs_exact")
            }
            summary["warm_hbm_hit_prompts"] = sum(
                (row["warm"]["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0) > 0
                for row in rows
            )
        else:
            summary["warm_tier_hit_prompts"] = sum(row["warm_tier_hits"] > 0 for row in rows)
            summary["warm_load_error_prompts"] = sum(bool(row["warm_load_errors"]) for row in rows)
            summary["gate_pass"] = (
                all(count == 20 for mode in modes for count in summary[mode].values())
                and summary["warm_tier_hit_prompts"] == 20
                and summary["warm_load_error_prompts"] == 0
                and (scenario != "dram" or summary["late_eviction_reported"] == 20)
            )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if not summary.get("gate_pass", True):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
