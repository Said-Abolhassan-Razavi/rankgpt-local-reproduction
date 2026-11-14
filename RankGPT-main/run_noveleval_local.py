"""RankGPT listwise re-ranking on NovelEval with a local model served by Ollama.

The prompt and the answer parsing are the paper's own code, imported unchanged from rank_gpt.py
(create_permutation_instruction, receive_permutation). Only the model call is replaced: instead of
OpenAI it calls the local Ollama server directly, so that we get the real prompt token count and
separate timings for prompt reading and generation.

NovelEval has 20 passages per query = exactly one window of 20, so there is one model call per query.

Usage (from the RankGPT-main folder):
    python run_noveleval_local.py                 # ONE query (index 0)
    python run_noveleval_local.py --start 3       # one query, index 3
    python run_noveleval_local.py --limit 21      # all 21 queries
    python run_noveleval_local.py --limit 21 --shuffle-seed 0     # passages randomly shuffled first (paper, Section 6.1)

Without --shuffle-seed the passages keep the order stored in data.json, which is NOT random (see NOTES.md).
An existing log file is never overwritten unless --overwrite is given.
"""
import argparse
import copy
import ctypes
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import pytrec_eval

from rank_gpt import clean_response, create_permutation_instruction, receive_permutation, remove_duplicate

HERE = Path(__file__).resolve().parent
DATA = HERE / "NovelEval"
OUT_DIR = HERE.parent / "results"
OLLAMA_URL = "http://localhost:11434/api/chat"
MAX_NEW_TOKENS = 300      # a ranking of 20 identifiers is about 100 tokens; this only stops a runaway answer


def ollama_chat(messages, model, num_ctx):
    """One chat call to the local Ollama server. Returns the answer text and Ollama's own counters."""
    body = {"model": model, "messages": messages, "stream": False, "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx, "num_predict": MAX_NEW_TOKENS}}
    request = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        out = json.loads(response.read().decode("utf-8"))
    wall = time.perf_counter() - t0
    prompt_s = out.get("prompt_eval_duration", 0) / 1e9
    gen_s = out.get("eval_duration", 0) / 1e9
    stats = {
        "prompt_tokens": out.get("prompt_eval_count", 0),       # real token count, from the model's tokenizer
        "gen_tokens": out.get("eval_count", 0),
        "prompt_s": round(prompt_s, 3),                         # time to read the prompt
        "gen_s": round(gen_s, 3),                               # time to write the answer
        "load_s": round(out.get("load_duration", 0) / 1e9, 3),  # > 0 only when the model had to be loaded
        "wall_s": round(wall, 3),
        "prompt_tok_per_s": round(out.get("prompt_eval_count", 0) / prompt_s, 1) if prompt_s else None,
        "gen_tok_per_s": round(out.get("eval_count", 0) / gen_s, 1) if gen_s else None,
        "done_reason": out.get("done_reason"),
    }
    return out["message"]["content"], stats


def answer_diagnostics(raw, n):
    """How well-formed is the answer? Uses the paper's own cleaning, then counts what is wrong."""
    numbers = [int(x) for x in clean_response(raw).split()]
    in_range = [x for x in numbers if 1 <= x <= n]
    distinct = remove_duplicate(in_range)
    return {
        "n_numbers": len(numbers),
        "n_valid_distinct": len(distinct),
        "missing_ids": [i for i in range(1, n + 1) if i not in distinct],
        "n_duplicates": len(in_range) - len(distinct),
        "out_of_range": [x for x in numbers if not 1 <= x <= n],
        "stopped_early": len(numbers) < n,      # the answer ended before listing n identifiers
    }


def plugged_in():
    """True if the laptop is on mains power, False on battery, None if Windows cannot tell."""
    class Status(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                    ("saver", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
    status = Status()
    try:
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return None
    except AttributeError:      # not Windows
        return None
    return {0: False, 1: True}.get(status.ac)


def shuffled(item, seed):
    """The item with its passages in a random order. Returns (new item, order), where order[i] is the
    0-based position in data.json of the passage now shown as [i + 1]. Fixed by (seed, query id)."""
    qid = int(item["hits"][0]["qid"])
    order = list(range(len(item["hits"])))
    random.Random(seed * 1000 + qid).shuffle(order)
    new = copy.deepcopy(item)
    new["hits"] = [copy.deepcopy(item["hits"][i]) for i in order]
    for rank, hit in enumerate(new["hits"], 1):     # placeholder rank and score, as in data.json
        hit["rank"], hit["score"] = rank, 1 / rank
    return new, order


def load_qrels():
    qrels = {}
    with open(DATA / "qrels.txt", encoding="utf-8") as f:
        for line in f:
            qid, _, docid, grade = line.split()
            qrels.setdefault(qid, {})[docid] = int(grade)
    return qrels


def ndcg(qrels, qid, docids):
    """nDCG@1/5/10 of one ordered list of document ids."""
    run = {qid: {d: float(len(docids) - i) for i, d in enumerate(docids)}}
    scores = pytrec_eval.RelevanceEvaluator({qid: qrels[qid]}, {"ndcg_cut.1,5,10"}).evaluate(run)[qid]
    return {k: round(scores[f"ndcg_cut_{k}"], 4) for k in (1, 5, 10)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--num-ctx", type=int, default=8192, help="context window given to the model, in tokens")
    parser.add_argument("--start", type=int, default=0, help="index of the first query")
    parser.add_argument("--limit", type=int, default=1, help="number of queries to run (default: ONE)")
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="shuffle the passages of each query with this seed before re-ranking")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing log file")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    items = json.load(open(DATA / "data.json", encoding="utf-8"))
    qrels = load_qrels()
    chosen = items[args.start: args.start + args.limit]
    OUT_DIR.mkdir(exist_ok=True)
    order_tag = "" if args.shuffle_seed is None else f"_shuffle-seed{args.shuffle_seed}"
    out_path = OUT_DIR / f"noveleval_{args.model.replace(':', '-')}{order_tag}_start{args.start}_n{len(chosen)}.jsonl"
    if out_path.exists() and not args.overwrite:
        sys.exit(f"{out_path} already exists: not overwritten (use --overwrite to replace it)")
    print(f"model {args.model} | num_ctx {args.num_ctx} | {len(chosen)} query(ies) from index {args.start}"
          f" | initial order: {'data.json' if args.shuffle_seed is None else f'shuffle, seed {args.shuffle_seed}'}"
          f" | plugged in: {plugged_in()}\n")

    records = []
    with open(out_path, "w", encoding="utf-8") as log:
        for item in chosen:
            qid = str(item["hits"][0]["qid"])
            n = len(item["hits"])
            order = list(range(n))
            if args.shuffle_seed is not None:
                item, order = shuffled(item, args.shuffle_seed)
            before = [h["docid"] for h in item["hits"]]

            # (1) the paper's prompt, unchanged
            messages = create_permutation_instruction(item=item, rank_start=0, rank_end=n, model_name=args.model)
            prompt_words = sum(len(m["content"].split()) for m in messages)
            # (2) the local model instead of OpenAI
            raw, stats = ollama_chat(messages, args.model, args.num_ctx)
            # (3) the paper's parsing, unchanged
            new_item = receive_permutation(copy.deepcopy(item), raw, rank_start=0, rank_end=n)
            after = [h["docid"] for h in new_item["hits"]]

            record = {
                "qid": qid, "query": item["query"], "window": [1, n],
                "model": args.model, "num_ctx": args.num_ctx,
                "shuffle_seed": args.shuffle_seed,
                "initial_order": order,             # order[i] = 0-based data.json position of the passage shown as [i + 1]
                "docids_before": before,
                "plugged_in": plugged_in(),
                "prompt_words": prompt_words, **stats,
                "context_ok": stats["prompt_tokens"] + MAX_NEW_TOKENS <= args.num_ctx,
                "raw_answer": raw,
                "parsed_ranking": [before.index(d) + 1 for d in after],     # original identifiers, new order
                **answer_diagnostics(raw, n),
                "grades_before": [qrels[qid].get(d, 0) for d in before],
                "grades_after": [qrels[qid].get(d, 0) for d in after],
                "ndcg_before": ndcg(qrels, qid, before),
                "ndcg_after": ndcg(qrels, qid, after),
                "docids_after": after,
            }
            records.append(record)
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()

            print(f"=== query {qid}: {item['query']}")
            print(f"window: passages 1-{n} | prompt: {record['prompt_tokens']} tokens ({prompt_words} words)"
                  f" | fits in context: {record['context_ok']}")
            print(f"prompt reading: {record['prompt_s']} s ({record['prompt_tok_per_s']} tok/s)"
                  f" | generation: {record['gen_s']} s for {record['gen_tokens']} tokens ({record['gen_tok_per_s']} tok/s)"
                  f" | model load: {record['load_s']} s | wall: {record['wall_s']} s | stop: {record['done_reason']}")
            print(f"RAW ANSWER:\n{raw}")
            print(f"PARSED RANKING: {record['parsed_ranking']}")
            print(f"valid distinct identifiers: {record['n_valid_distinct']}/{n} | missing: {record['missing_ids']}"
                  f" | duplicates: {record['n_duplicates']} | out of range: {record['out_of_range']}")
            print(f"relevance grades before: {record['grades_before']}")
            print(f"relevance grades after : {record['grades_after']}")
            print(f"nDCG@10 before {record['ndcg_before'][10]} -> after {record['ndcg_after'][10]}\n")

    if len(records) > 1:
        mean = lambda xs: sum(xs) / len(xs)
        print("=== summary over", len(records), "queries")
        for k in (1, 5, 10):
            print(f"nDCG@{k}: given order {100 * mean([r['ndcg_before'][k] for r in records]):.2f}"
                  f" -> re-ranked {100 * mean([r['ndcg_after'][k] for r in records]):.2f}")
        print(f"prompt tokens: mean {mean([r['prompt_tokens'] for r in records]):.0f},"
              f" max {max(r['prompt_tokens'] for r in records)}")
        print(f"seconds per call: prompt {mean([r['prompt_s'] for r in records]):.2f},"
              f" generation {mean([r['gen_s'] for r in records]):.2f}, wall {mean([r['wall_s'] for r in records]):.2f}")
        print(f"answers with all {n} identifiers: {sum(r['n_valid_distinct'] == n for r in records)}/{len(records)}")
        print(f"answers with a duplicate: {sum(r['n_duplicates'] > 0 for r in records)}"
              f" | with a missing identifier: {sum(len(r['missing_ids']) > 0 for r in records)}"
              f" | with an out-of-range identifier: {sum(len(r['out_of_range']) > 0 for r in records)}"
              f" | stopped early: {sum(r['stopped_early'] for r in records)}"
              f" | hit the {MAX_NEW_TOKENS}-token cap: {sum(r['done_reason'] == 'length' for r in records)}")
        print(f"plugged in for all calls: {all(r['plugged_in'] is True for r in records)}")
        print(f"answers that did not fit the context: {sum(not r['context_ok'] for r in records)}")
    print(f"log written to {out_path}")


if __name__ == "__main__":
    main()
