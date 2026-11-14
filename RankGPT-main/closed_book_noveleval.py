"""Contamination check: ask the local model the 21 NovelEval questions with NO passages (closed book).

If the model already knows the answers, NovelEval is not a "novel" test for it.
The script does not judge the answers: it prints question | reference answer | model answer, to be marked by hand.

Usage (from the RankGPT-main folder):
    python closed_book_noveleval.py
An existing log file is never overwritten unless --overwrite is given.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUERIES = HERE / "NovelEval" / "queries.tsv"
OUT_DIR = HERE.parent / "results"
OLLAMA_URL = "http://localhost:11434/api/chat"
SYSTEM = "Answer the question in one short sentence."

# Reference answers from the paper's Table 8, as copied by us, matched by hand to the query ids of queries.tsv.
REFERENCE = {
    "0": "280 variations",
    "1": "4K per eye",
    "2": "Anatomy of a Fall",
    "3": "Linda Yaccarino",
    "4": "0",
    "5": "Saudi Arabia",
    "6": "Wembley Stadium",
    "7": "Google DeepMind",
    "8": "Stade de France",
    "9": "Hiroshima",
    "10": "Visual Programming and another paper (the table is cut off)",
    "11": "May 19, 2023",
    "12": "Denver Nuggets",
    "13": "Dante",
    "14": "$50M-$60M",
    "15": "Notorious B.I.G",
    "16": "144TB",
    "17": "faster, low memory, dynamic shapes",
    "18": "Eobard Thawne / Professor Zoom",
    "19": "Lionel Messi",
    "20": "$163.8 million worldwide",
}


def ask(question, model, num_ctx):
    body = {"model": model, "stream": False, "keep_alive": "30m",
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
            "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx, "num_predict": 150}}
    request = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        out = json.loads(response.read().decode("utf-8"))
    return out["message"]["content"].strip(), {
        "prompt_tokens": out.get("prompt_eval_count", 0), "gen_tokens": out.get("eval_count", 0),
        "wall_s": round(time.perf_counter() - t0, 3), "done_reason": out.get("done_reason")}


def cell(text):
    return " ".join(text.split()).replace("|", "/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing log file")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    queries = [line.rstrip("\n").split("\t", 1) for line in open(QUERIES, encoding="utf-8") if line.strip()]
    unmatched = [qid for qid, _ in queries if qid not in REFERENCE] + [q for q in REFERENCE if q not in dict(queries)]
    if unmatched:
        sys.exit(f"questions and reference answers do not match: {unmatched}")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"noveleval_closedbook_{args.model.replace(':', '-')}.jsonl"
    if out_path.exists() and not args.overwrite:
        sys.exit(f"{out_path} already exists: not overwritten (use --overwrite to replace it)")

    print("| # | Question | Reference answer (paper, Table 8) | Model answer, no passages |")
    print("|:-:|---|---|---|")
    with open(out_path, "w", encoding="utf-8") as log:
        for qid, question in queries:
            answer, stats = ask(question, args.model, args.num_ctx)
            record = {"qid": qid, "question": question, "reference": REFERENCE[qid], "model": args.model,
                      "system_prompt": SYSTEM, "temperature": 0, "raw_answer": answer, **stats}
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            print(f"| {qid} | {cell(question)} | {cell(REFERENCE[qid])} | {cell(answer)} |")
    print(f"\nlog written to {out_path}")


if __name__ == "__main__":
    main()
