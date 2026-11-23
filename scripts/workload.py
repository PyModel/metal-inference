"""Streamed OpenAI-compatible workload for an already identified local server."""
import hashlib
import json
import random
import secrets
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

PROMPTS = (
    "Implement an LRU cache in Python and explain its complexity.",
    "Explain how TCP slow start and congestion avoidance work.",
    "Write a short story about a lighthouse keeper who finds a message.",
    "Compare B-trees and LSM trees for write-heavy workloads.",
    "Explain stable sorting and give an example with repeated keys.",
    "Summarize the causes and consequences of the 1929 stock market crash.",
)
CONCURRENT_PROMPTS = (
    "Explain why a hash table sometimes needs resizing.",
    "Describe how a compiler performs register allocation.",
    "Write a story about a violin maker learning an instrument.",
    "Compare delta encoding and run-length encoding.",
    "Explain how a ferry operator would build a seasonal timetable.",
    "Describe the difference between a lunar and a solar eclipse.",
)
WORDS = "river stone paper cloud engine garden silver market winter orange signal harbor candle forest copper meadow".split()


def chat(cfg, prompt, limit):
    body = {**cfg.get("extra_body", {}), "model": cfg["model"],
            "messages": [{"role": "user", "content": prompt}], "temperature": cfg.get("temperature", 0),
            "max_tokens": limit, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(cfg["base_url"] + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.monotonic()
    first = visible = None
    pieces = []
    answer = []
    usage = {}
    done = False
    with urllib.request.urlopen(req, timeout=cfg.get("timeout_s", 180)) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise ValueError("server returned a streamed error")
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                piece = (delta.get("reasoning_content") or delta.get("reasoning") or "") + content
                if piece:
                    if first is None:
                        first = time.monotonic()
                    if content and visible is None:
                        visible = time.monotonic()
                    pieces.append(piece)
                    answer.append(content)
    end = time.monotonic()
    counts = [usage.get(k) for k in ("prompt_tokens", "completion_tokens")]
    if not done or first is None or any(type(v) is not int or v < 1 for v in counts):
        raise ValueError("incomplete stream, empty output, or missing positive token usage")
    words = "".join(pieces).split()
    shingles = [tuple(words[i:i + 8]) for i in range(len(words) - 7)]
    if len(shingles) > 50 and len(set(shingles)) < len(shingles) / 2:
        raise ValueError("looping output")
    return {"start": start, "end": end, "ttft_s": first - start, "prompt_tokens": counts[0],
            "completion_tokens": counts[1], "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "visible_ttft_s": visible - start if visible is not None else None,
            "output_tps": counts[1] / (end - start),
            "answer": "".join(answer), "output_sha256": hashlib.sha256("".join(pieces).encode()).hexdigest()}


def document(n_words, seed):
    words = random.Random(seed).choices(WORDS, k=n_words)
    codes = [secrets.token_hex(4).upper() for _ in range(3)]
    for i, code in enumerate(codes):
        words.insert(n_words * (i + 1) // 4, f". Code {i + 1} is {code}.")
    prompt = f"Document {secrets.token_hex(8)}. " + " ".join(words)
    return prompt + "\nReturn the three codes in order, separated only by spaces.", codes


def context_run(cfg, target, seed, qualification, barrier=None):
    reserve = cfg.get("answer_tokens", 128) if qualification else 1
    prompt_budget = target - reserve
    if prompt_budget <= 0:
        raise ValueError("context must leave space for the answer")
    prompt, codes = document(int(prompt_budget * cfg.get("words_per_token", 0.92)), seed)
    if barrier is not None:
        barrier.wait(timeout=cfg.get("timeout_s", 180))
    r = chat(cfg, prompt, reserve)
    result = {"target": target, "prompt_tokens": r["prompt_tokens"], "ttft_s": r["ttft_s"],
              "cached_tokens": r["cached_tokens"], "prompt_tokens_per_ttft_s": r["prompt_tokens"] / r["ttft_s"]}
    result["length_ok"] = 0.95 * prompt_budget <= r["prompt_tokens"] <= prompt_budget
    if qualification:
        result["exact"] = r["answer"].strip().split() == codes
    return result


def measure(cfg):
    limit, n = cfg.get("max_tokens", 128), cfg.get("concurrency", 1)
    chat(cfg, "Say hello.", 8)
    short = chat(cfg, "Reply with the single word ready.", 8)
    samples = []
    for i in range(cfg.get("reps", 6)):
        r = chat(cfg, PROMPTS[i % len(PROMPTS)], limit)
        if not r["answer"].strip():
            raise ValueError("no visible benchmark answer")
        samples.append({k: r[k] for k in ("completion_tokens", "prompt_tokens", "ttft_s", "visible_ttft_s", "cached_tokens", "output_sha256", "output_tps")})
    out = {"ttft_short_s": short["ttft_s"], "c1": samples,
           "c1_output_tps": statistics.median(x["output_tps"] for x in samples)}
    if n > 1:
        prompts = [f"Task {i + 1}. {CONCURRENT_PROMPTS[i % len(CONCURRENT_PROMPTS)]}" for i in range(n)]
        with ThreadPoolExecutor(n) as pool:
            rs = list(pool.map(lambda prompt: chat(cfg, prompt, limit), prompts))
        wall = max(r["end"] for r in rs) - min(r["start"] for r in rs)
        out["concurrency"] = n
        out["aggregate_tps"] = sum(r["completion_tokens"] for r in rs) / wall
        out["aggregate_wall_s"] = wall
        out["streams"] = [{k: r[k] for k in ("prompt_tokens", "completion_tokens", "ttft_s", "cached_tokens")}
                          | {"output_tps": r["output_tps"], "visible_ttft_s": r["visible_ttft_s"]} for r in rs]
    out["prefill"] = [context_run(cfg, size, size, False) for size in cfg.get("prefill_lengths", [])]
    if "qualify_context" in cfg:
        size = cfg["qualify_context"]
        out["needles"] = [context_run(cfg, size // 2, 10, True), context_run(cfg, size, 11, True)]
        if n > 1:
            barrier = Barrier(n)
            with ThreadPoolExecutor(n) as pool:
                out["soak"] = list(pool.map(lambda i: context_run(cfg, size, 100 + i, True, barrier), range(n)))
    return out


def worker(cfg, pipe):
    try:
        pipe.send(measure(cfg))
    except Exception as e:
        pipe.send({"error": str(e)})
    finally:
        pipe.close()
