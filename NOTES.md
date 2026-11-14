# RankGPT reproduction notes

Paper 13: "Is ChatGPT Good at Search?" (RankGPT). Presentation Fri Nov 27, 10:20; hand-in Wed Nov 25, 18:00.
Everything below was measured on this laptop unless marked "published".

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 (laptop), 8 GB, 50 W limit |
| Model | `qwen2.5:7b` through Ollama 0.40.16 (4.7 GB, 4-bit), 100 % on GPU, about 4.9 GB used at context 8192 |
| Settings | temperature 0, seed 0, context 8192 tokens, answer capped at 300 new tokens |
| Python | 3.12.10, virtual environment in `RankGPT\.venv` |
| Packages added | litellm, tqdm, rank_bm25, scipy, pytrec-eval-terrier |
| Code | `RankGPT-main/run_noveleval_local.py` (new). `rank_gpt.py` is unmodified: its prompt builder and answer parser are imported as they are; only the model call is replaced (Ollama instead of OpenAI) |
| Not installed | Java and Pyserini (needed only for TREC DL19) |

## Speed

| Condition | Prompt reading | Generation |
|---|:-:|:-:|
| On battery (18 %), GPU at 30 W | not measured on a long prompt | 27 tokens/s |
| Plugged in, GPU at about 40 W | about 1,750 tokens/s | about 40 tokens/s |

Per call on NovelEval (20 passages, plugged in, mean of 21 calls): prompt 4,590 tokens on average (max 5,684),
prompt reading 2.50 s, generation 2.14 s, wall clock 6.90 s. About 2.3 s per call is not counted by Ollama as either
reading or generating; cause not found yet.

- First request after starting Ollama: about 2.5 minutes (model load). Changing the context size reloads the model (9.3 s).
- Ollama's default context is 4,096 tokens, too small for 20 passages. It must be set explicitly; all 21 prompts fit in 8,192.
- Extrapolated cost of the full sliding window (9 calls per query, about 7 s per call): about 5 hours for SciFact
  (300 queries), about 35 hours for all three testbed datasets (2,029 queries). Not measured, only extrapolated.

## NovelEval, run 1: passages in the order stored in the repo (Nov 14)

Log: `results/noveleval_qwen2.5-7b_start0_n21.jsonl` (one-query test: `results/noveleval_qwen2.5-7b_start0_n1.jsonl`).

| nDCG | Stored order | Re-ranked by Qwen2.5-7B |
|---|:-:|:-:|
| @1 | 64.29 | 85.71 |
| @5 | 58.24 | 72.21 |
| @10 | 65.03 | 78.11 |

- nDCG@10 gain +13.08 points, bootstrap 95 % interval +3.9 to +21.9, paired t = 2.75 (20 degrees of freedom),
  sign-flip permutation p = 0.013.
- 17 queries better, 4 worse: query 1 (0.78 to 0.61), query 3 (0.94 to 0.64), query 7 (0.77 to 0.57), query 18 (0.75 to 0.50).
- **This run is not comparable with the paper.** The stored order is neither random nor BM25 (see next section),
  so the model started from an unusually good list.

Published (repo README, nDCG@10): BM25 55.77, gpt-3.5-turbo 75.71, gpt-4 90.45.

## What the stored order is (Nov 14, read-only analysis)

- The repo has no NovelEval evaluation script. `run_evaluation.py` covers only TREC and BEIR/Mr.TyDi through Pyserini;
  nothing in the repo reads `NovelEval/data.json`, and no code shuffles passages. So the repo does not show which
  initial order the authors used. The random shuffle is stated in the paper (Section 6.1, as read by us), not in the code.
- `data.json` lists the passages of each query in the same order as `corpus.tsv` and `qrels.txt` (document ids
  `q-0` to `q-19`), with placeholder scores 1/rank. It is the order in which the dataset was built.
- That order is strongly correlated with relevance. Mean relevance grade by position: 0.94 for positions 1-5,
  0.48 for 6-10, 0.34 for 11-15, 0.33 for 16-20 (position 1 alone: 1.29 out of 2).
- A random shuffle scores nDCG@10 38.9 on average (standard deviation 3.8 over 2,000 shuffles; @1 25.9, @5 28.7).
  None of the 2,000 shuffles reached 65.03. The reversed order scores 23.7.
- Likely reason (not confirmed): the README says the 20 candidates came from Google search, and the file keeps
  them in collected order, so Google's ranking is still in it.
- 80 of the 420 passages differ by a few characters between `data.json` and `corpus.tsv`; this looks like quote
  escaping in the TSV file, not different text.

## NovelEval, run 2: random shuffle, 5 seeds (Nov 14) — the comparable setting

Setting: the 20 passages of each query are randomly shuffled before re-ranking (paper, Section 6.1), then one
window of 20. Command: `python run_noveleval_local.py --limit 21 --shuffle-seed S` for S = 0 to 4.
Logs: `results/noveleval_qwen2.5-7b_shuffle-seed{0..4}_start0_n21.jsonl`. Each line holds the exact shuffle
(`initial_order`), the raw answer, per-query nDCG@1/5/10 before and after, the format errors, token counts,
timings and `plugged_in`. The laptop was plugged in for all 105 calls; all prompts fit in the context.

| Seed | nDCG@1 before | after | nDCG@5 before | after | nDCG@10 before | after |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 28.57 | 83.33 | 27.96 | 72.90 | 40.41 | 77.63 |
| 1 | 40.48 | 80.95 | 36.26 | 79.70 | 43.78 | 83.18 |
| 2 | 38.10 | 83.33 | 34.19 | 78.27 | 42.97 | 82.04 |
| 3 | 35.71 | 80.95 | 29.96 | 76.00 | 40.84 | 78.77 |
| 4 | 26.19 | 78.57 | 30.70 | 77.72 | 39.86 | 76.76 |
| **Mean ± sd** | 33.81 ± 6.16 | **81.43 ± 1.99** | 31.81 ± 3.35 | **76.92 ± 2.61** | 41.57 ± 1.71 | **79.67 ± 2.80** |

Standard deviation is over the 5 seeds (sample sd, n − 1).

- "Before" nDCG@10 is 41.57 on average, far below the 65.03 of the stored order and in line with the 38.9 ± 3.8
  simulated for random shuffles.
- Published nDCG@1 / @5 / @10: BM25 33.33 / 45.96 / 55.77, gpt-3.5-turbo 76.19 / 74.15 / 75.71, gpt-4 85.71 / 87.49 / 90.45.
  Our Qwen2.5-7B: 81.43 / 76.92 / 79.67. Above the published gpt-3.5-turbo at all three cut-offs, below gpt-4.
  Caveats: the paper's shuffle seed is unknown, its numbers look like a single run, and Qwen2.5 may have seen
  these 2023 facts (problem 5 below).
- Per query (nDCG@10): 99 of 105 query-seed pairs improved, 6 got worse (seed 0: 3, seed 3: 2, seed 4: 1).
  Averaging each query over the 5 seeds: mean gain +38.1 points, paired t = 10.9 (20 degrees of freedom),
  sign-flip permutation p < 0.00001. Only query 4 (Haaland goals) is worse on average.
- Result depends on the shuffle: the after-score of a single query varies a lot between seeds (sd up to 33 points
  for query 8, 23 for query 15, 18 for query 4), and the seed means range from 76.76 to 83.18.

Format errors per seed (number of answers out of 21; one answer can have several errors):

| Seed | Complete (20 distinct) | With a duplicate | With a missing identifier | Out of range | Stopped early | Hit 300-token cap |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 4 | 13 | 17 | 0 | 5 | 0 |
| 1 | 6 | 7 | 15 | 0 | 9 | 0 |
| 2 | 9 | 6 | 12 | 0 | 6 | 0 |
| 3 | 9 | 6 | 12 | 0 | 7 | 0 |
| 4 | 4 | 10 | 17 | 0 | 7 | 0 |
| **Total of 105** | 32 | 42 | 73 | 0 | 34 | 0 |

"Stopped early" means the answer listed fewer than 20 identifiers in total (most often 19; the shortest had 6).
178 identifiers were missing in all (1.7 per answer). Identifiers [2], [4] and [8] are the ones most often left
out (36, 25 and 25 times of 178); cause not investigated.

Timing per call (mean over seeds): prompt reading 2.60 s, generation 2.14 s, wall clock 7.05 s
(sd between seeds 0.04 s). The 2.3 s of uncounted overhead per call is still there.

## NovelEval contamination check: closed book (Nov 14)

Question: does Qwen2.5-7B already know the NovelEval answers without reading any passage?
Script: `RankGPT-main/closed_book_noveleval.py` (new). Log: `results/noveleval_closedbook_qwen2.5-7b.jsonl`.
Setting: each of the 21 questions alone, no passages, system prompt "Answer the question in one short sentence.",
temperature 0, seed 0. One run. Reference answers are the paper's Table 8 as copied by us; all 21 were matched to
a question in `queries.tsv` (none left over). **Not judged automatically: the "Mark" column is to be filled by hand.**

| # | Question | Reference answer (paper, Table 8) | Qwen2.5-7B answer, no passages | Mark |
|:-:|---|---|---|:-:|
| 0 | How many different Spider-Men are there in Across the Spider-Verse? | 280 variations | There are 11 different Spider-Men in "Across the Spider-Verse". | |
| 1 | What is the screen resolution of vision pro? | 4K per eye | The Vision Pro features a screen resolution of 2304 x 2304 pixels per eye. | |
| 2 | Which film was the 2023 Palme d'Or winner? | Anatomy of a Fall | "El Cielo en la Tierra" won the 2023 Palme d'Or. | |
| 3 | Who will be the CEO of Twitter after Elon Musk is no longer the CEO? | Linda Yaccarino | The next CEO of Twitter after Elon Musk is no longer the CEO has not been announced yet. | |
| 4 | How many goals did Haaland scored in the 2023 Champions League Final | 0 | Haaland scored 3 goals in the 2023 Champions League Final. | |
| 5 | Where did Benzema go after leaving Real Madrid? | Saudi Arabia | After leaving Real Madrid, Benzema played for Juventus and Lyon. | |
| 6 | Where was the 2023 Premier League FA Cup Final held? | Wembley Stadium | The 2023 Premier League FA Cup Final was held at Wembley Stadium in London. | |
| 7 | What is the name of the combined Deepmind and Google Brain? | Google DeepMind | The combined entity is simply called DeepMind. | |
| 8 | Where will Blackpink's 2023 world tour concert in France be held? | Stade de France | Blackpink's 2023 world tour concert in France will be held at Stade de France in Paris. | |
| 9 | Where did the G7 Summit 2023 take place? | Hiroshima | The G7 Summit 2023 took place in Hiroshima, Japan. | |
| 10 | What are the best papers of CVPR 2023? | Visual Programming and another paper (the table is cut off) | Identifying the best papers from CVPR 2023 requires reviewing the conference proceedings and expert opinions, as the specific top papers are not publicly announced until the conference. | |
| 11 | What is the release date of song Middle Ground? | May 19, 2023 | The release date of the song "Middle Ground" by Portugal. The Man is March 19, 2019. | |
| 12 | Who wins NBA Finals 2023? | Denver Nuggets | Predicting the exact winner of the 2023 NBA Finals is not possible, as it depends on the teams and players performing at that time. | |
| 13 | Who does Momoa play in Fast X? | Dante | In "Fast X," Momoa plays the character of Aquaman. | |
| 14 | What is Messi's annual income after transferring to Miami? | $50M-$60M | Lionel Messi's annual income after transferring to Miami is estimated to be around $80 million. | |
| 15 | Who sang the theme song of Transformers Rise of the Beasts? | Notorious B.I.G | Shawn Mendes sang the theme song for Transformers Rise of the Beasts. | |
| 16 | How much video memory does the DGX GH200 have? | 144TB | The DGX GH200 has 800 GB of video memory. | |
| 17 | What are the new features of PyTorch 2? | faster, low memory, dynamic shapes | PyTorch 2 introduces TorchScript 2.0, a new API called TorchCore, and a just-in-time (JIT) compiler for faster execution. | |
| 18 | Who is the villain in The Flash? | Eobard Thawne / Professor Zoom | The main villain in The Flash can vary, but popular choices include Reverse-Flash and Zoom. | |
| 19 | Who win 2023 Laureus World Sportsman Of The Year Award? | Lionel Messi | Cristiano Ronaldo won the 2023 Laureus World Sportsman of the Year Award. | |
| 20 | The Little Mermaid first week box office? | $163.8 million worldwide | The Little Mermaid's first-week box office was approximately $44.9 million in the United States and Canada. | |

Limits of this check: one prompt wording and one run; a short closed-book answer shows what the model can recall,
not everything it may have seen in training.

## Problems found

1. **Format errors.** With shuffled passages only 32 of 105 answers were complete (table above). In run 1
   (stored order) only 5 of 21 answers were a complete ranking of 20 identifiers. 13 answers had 19 valid
   identifiers (one missing, often one repeated), one had 17 (three repeated), and two stopped early with 14 and
   7 identifiers. No out-of-range
   identifier. The paper's parser fills the gaps with the original order, so the score hides these errors.
2. **Not fully repeatable.** Query 0 gave a slightly different answer in two runs with temperature 0 and seed 0
   (same nDCG). The second time the prompt was already cached by Ollama.
3. **The repo's evaluation code is broken as shipped.** `trec_eval.py` line 67 calls `os.path.exists()` with no
   argument, and needs `pytrec_eval`, which is missing from `requirements.txt`. `rank_gpt.py`'s `main()` uses
   undefined names (`THE_TOPICS`, `data`). We score with `pytrec_eval` directly.
4. **scipy blocked by Windows, once.** On Nov 14 at 19:45 `from scipy import stats` failed in the RankGPT venv
   (Python 3.12.10, scipy 1.18.1, numpy 2.5.4) with
   `ImportError: DLL load failed while importing _trlib: An Application Control policy has blocked this file.`
   - Blocked file: `RankGPT\.venv\Lib\site-packages\scipy\optimize\_trlib\_trlib.cp312-win_amd64.pyd`
     (207,872 bytes, not digitally signed, SHA-256 `5A24CA0F7984CFCE5AB7B0DE61BD4D4F61593D1DA1817D4CD168061CEE3BC99C`).
   - Windows event log (CodeIntegrity, events 3077 and 3033, 19:45:00): python.exe "attempted to load ...
     `_trlib.cp312-win_amd64.pyd` that did not meet the Enterprise signing level requirements or violated code
     integrity policy (Policy ID {0283ac0f-fff1-49ae-ada1-8a933130cad6})".
   - Smart App Control is ON on this laptop (state 1). It was not changed, and no Windows security setting was touched.
   - **The block did not repeat.** About an hour later, with nothing changed, the same file loads and
     `scipy.stats`, `scipy.optimize`, `scipy.linalg`, `scipy.sparse`, `scipy.special` and `scipy.spatial` all import.
     Likely explanation (not confirmed): Smart App Control decides on unsigned files with a cloud reputation
     lookup, and the first lookup for this freshly installed file failed or had not finished.
   - Consequence: a freshly installed unsigned library can be blocked the first time it is loaded, so every new
     environment should be import-tested well before the demo. Statistics of run 1 were computed with NumPy.
   - **The testbed is not blocked** (checked Nov 14, see "Testbed baselines check" below).

## Testbed baselines check (Nov 14)

Separate environment, outside both repos: `IR2\testbed-venv` (Python 3.12.10, 1.0 GB), installed with the
testbed's own `requirements.txt`: numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, nltk 3.10.4, rank-bm25 0.2.2,
sentence-transformers 6.3.0, transformers 5.23.0, torch 2.15.0 (CPU-only build, no CUDA), pytrec-eval-terrier.

Commands, run from `ir2-testbed`:

```
py -3.12 -m venv "..\testbed-venv"
..\testbed-venv\Scripts\python.exe -m pip install -r requirements.txt
..\testbed-venv\Scripts\python.exe -c "import scipy.stats, sklearn, nltk, rank_bm25, pytrec_eval, sentence_transformers, torch"
..\testbed-venv\Scripts\python.exe testbed.py baselines --systems bm25 minilm
```

No Application Control block at any step: install, import test (including `scipy.stats`) and scoring all ran.

| Dataset | System | nDCG@10 | Recall@100 | Time | Same as `baselines/scores.json` |
|---|---|:-:|:-:|:-:|:-:|
| SciFact | BM25 | 0.676 | 0.918 | 17 s | yes |
| SciFact | all-MiniLM-L6-v2 | 0.646 | 0.925 | 84 s | yes |
| NFCorpus | BM25 | 0.321 | 0.246 | 6 s | yes |
| NFCorpus | all-MiniLM-L6-v2 | 0.316 | 0.311 | 49 s | yes |
| ArguAna | BM25 | 0.516 | 0.969 | 352 s | yes |
| ArguAna | all-MiniLM-L6-v2 | 0.503 | 0.977 | 112 s | yes |

- Times include the first download of each dataset and of the MiniLM model; MiniLM ran on CPU.
- Written inside `ir2-testbed`: `data/` (three datasets) and `runs/baselines/<dataset>/{bm25,minilm}.json`.
  Both folders are git-ignored; `git status` shows no change. `baselines/scores.json` was not rewritten.
- The BM25 top-100 runs that a re-ranker starts from now exist locally for all three datasets.
- Not yet tested here: `testbed.py score` on a run of ours (it is the part that calls `scipy.stats.ttest_rel`;
  `scipy.stats` itself imports in this environment).
5. **NovelEval may not be "novel" for our model.** It was built in June 2023 to be unseen by GPT-4; Qwen2.5 is
   more recent and may know the answers. Checked closed book (section above); the table still has to be marked by hand.
6. **Battery.** On battery the GPU ran at 30 W and generation was about a third slower.

## Open items

- NovelEval BM25 ordering: is it close to the published 55.77? (Part 3, not started)
- scipy diagnosis in `ir2-testbed`. (Part 4, not started)
- TREC DL19 reproduction (needs Java and Pyserini).
- Testbed row for `pairs/13-rankgpt` (folder not created yet).
