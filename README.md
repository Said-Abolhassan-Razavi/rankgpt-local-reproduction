# RankGPT with a local model

A small reproduction of the main method of the paper
["Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents"](https://arxiv.org/abs/2304.09542)
(Sun et al., EMNLP 2023), done for a university seminar on information retrieval.

The paper re-ranks search results by showing an LLM 20 passages at once and asking it to output them in order
of relevance, for example `[3] > [1] > [2]`. The paper uses ChatGPT and GPT-4. Here the same prompt and the same
answer parser are run with a free, open-source model on a laptop GPU instead, with no paid API.

## Result so far

Dataset: NovelEval, the paper's own test set of 21 questions with 20 candidate passages each.
Model: Qwen2.5-7B (4-bit) served by Ollama on an 8 GB laptop GPU.
The 20 passages are randomly shuffled before re-ranking, as in the paper; scores are the mean over 5 shuffles.

| | nDCG@1 | nDCG@5 | nDCG@10 |
|---|:-:|:-:|:-:|
| Shuffled passages, before re-ranking (ours) | 33.81 | 31.81 | 41.57 |
| **Re-ranked by Qwen2.5-7B (ours, mean ± sd over 5 seeds)** | **81.43 ± 1.99** | **76.92 ± 2.61** | **79.67 ± 2.80** |
| gpt-3.5-turbo (published) | 76.19 | 74.15 | 75.71 |
| gpt-4 (published) | 85.71 | 87.49 | 90.45 |

Things to keep in mind when reading this table:

- Only 32 of the 105 model answers were a complete, valid ranking. The others repeated or left out identifiers,
  or stopped early. The paper's parser fills the gaps with the original order, so the scores hide these errors.
- The score depends on the shuffle: the five seeds range from 76.76 to 83.18 nDCG@10.
- The paper's shuffle is unknown and its numbers look like a single run, so the comparison is indicative only.
- NovelEval was built in 2023 to be unseen by GPT-4. Qwen2.5 is more recent, so a closed-book check was run
  (the model answers the 21 questions with no passages); the answers are in the notes.

All numbers, timings, problems and their details are in **[NOTES.md](NOTES.md)**.

## What is in this repository

| Path | What it is |
|---|---|
| [NOTES.md](NOTES.md) | Lab notes: setup, every result, timings and problems found |
| [RankGPT-main/run_noveleval_local.py](RankGPT-main/run_noveleval_local.py) | Our script: re-ranks NovelEval with the local model |
| [RankGPT-main/closed_book_noveleval.py](RankGPT-main/closed_book_noveleval.py) | Our script: asks the model the 21 questions with no passages |
| [results/](results/) | One log file per run, with the raw model answer of every call |
| [RankGPT-main/](RankGPT-main/) | The paper authors' code, unmodified (see Credits) |

## How to run it

Requirements: Windows or Linux, an NVIDIA GPU with about 6 GB of free memory, Python 3.10 or later, and [Ollama](https://ollama.com).

```
ollama pull qwen2.5:7b
pip install tqdm pytrec-eval-terrier
cd RankGPT-main
python run_noveleval_local.py                              # one query, to check that everything works
python run_noveleval_local.py --limit 21 --shuffle-seed 0  # all 21 queries, shuffled with seed 0
python closed_book_noveleval.py                            # the closed-book check
```

Each run writes a log to `results/` and refuses to overwrite an existing one. On the laptop used here one
re-ranking call takes about 7 seconds, so 21 queries take about two and a half minutes.

## Status

Work in progress. Still to do: the TREC DL19 benchmark with the full sliding window, and the seminar's shared
test datasets (SciFact, NFCorpus, ArguAna).

## Credits

The folder `RankGPT-main` is the official code of the paper, from
[github.com/sunnweiwei/RankGPT](https://github.com/sunnweiwei/RankGPT), by Weiwei Sun and co-authors, under the
Apache 2.0 licence ([LICENSE.txt](RankGPT-main/LICENSE.txt)). The only files added to that folder are the two
scripts listed above; the authors' files are unchanged.
