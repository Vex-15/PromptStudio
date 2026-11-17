# Prompt Studio

Two prompt tools that were separate projects, working as one:

| Engine | Origin | What it does |
|---|---|---|
| **Neural** | Prompt Enhancer (`prompt-enhancer-2-main`) | BiLSTM + attention classifies the task (6 classes) and detects 5 missing parts, with attention weights as an explanation and a rule safety net. |
| **Rule rubric** | Prompt Optimizer (`Prompt-Optimizer-main`) | LangChain LCEL pipeline grades a prompt out of 100 across 6 dimensions, diagnoses issues, rebuilds it with sections, guards against intent drift, and batches. |

New in this project is the **fusion layer**: one merged prompt that closes the gaps
both engines found, plus a re-grade with each engine after the rewrite.

```
raw prompt
   |-- rule engine (LCEL)   -> score /100, diagnosis, requirements, constraints, drift
   |-- neural engine        -> task class, missing parts, attention, per-task templates
   v
fusion: rule rubric drives coding prompts, neural templates drive the rest,
        one spec per section, near-duplicate lines dropped
   v
merged prompt + score before/after + residual gaps
```

## Quick start

```bash
pip install -r requirements.txt
python run.py            # http://127.0.0.1:8000
```

The header nav switches between the **Studio** and a full **How it works** reference
(`/how-it-works`): both engines, the fusion rules, the rubric, voting, evaluation
numbers, the API, commands and limits.

The trained model (`artifacts/model.pt`, 1.2 MB) is included, so the neural engine
works immediately — no training step needed. Requires Python 3.13+.

Everything runs locally. No API keys, no cloud calls; the optional Ollama panel is
the only network feature and it talks to `localhost:11434`.

## Frontend

A single-page studio in `web/` served by FastAPI — no build step — plus a detailed
reference page linked from the header nav. Light and dark themes
share one set of CSS variables; the toggle follows `prefers-color-scheme` until you pick
a theme, then remembers the choice in `localStorage`, and it is applied before first paint
so a dark visit never flashes light. It uses four typefaces on purpose:

- **Fraunces** (variable serif) — wordmark, scores, metrics
- **Space Grotesk** — body copy and controls
- **JetBrains Mono** — prompt text, code, labels
- **Caveat** — handwritten accents and the "only you can answer these" heading

What it shows: score before → after with the delta, detected task and confidence,
per-gap probabilities, attention heatmap, rubric issues, clarifying questions,
drift/token safety, the merged prompt (copyable), provenance cards per section,
a side-by-side old-vs-new comparison with a saved preference vote, the rule-only and
neural-only rewrites, version history, a batch benchmark with CSV download, and the
optional Ollama comparison.

## What the fusion actually does

1. **Rule engine first**: `optimize()` grades and rebuilds, returning structured
   sections (`TASK`, `CONTEXT`, `REQUIREMENTS`, `CONSTRAINTS`, `OUTPUT FORMAT`, `VALIDATION`).
2. **Neural engine second**: `enhance()` predicts task + missing parts and drafts
   per-task additions.
3. **Choose a backbone**: if the neural classifier says `coding` (or the rubric intent
   is debug/refactor/test), the rubric structure wins — that is the only case its
   templates are written for. Otherwise the neural templates win, so a story prompt
   never gets "state the function signature".
4. **Merge with rules**: `CONTEXT` and `AUDIENCE` may take lines from both engines;
   `REQUIREMENTS` / `CONSTRAINTS` / `OUTPUT FORMAT` / `VALIDATION` are owned by one
   engine so the final prompt never contains two conflicting specs. Near-duplicates
   are dropped with token-set Jaccard similarity.
5. **TASK is always the user's prompt verbatim.** Only missing parts are appended, so
   intent cannot drift.
6. **Re-grade**: the merged text runs through the rubric analyzer and the BiLSTM
   again to report score gain and any gaps still open.

`mode` (auto/basic/standard/advanced) and `level` (developer/beginner) are passed to
the rule engine; `basic` compresses only, `advanced` adds step decomposition and a
validation section.

## Motion (reversible)

Motion is a layer you can switch off, not a baked-in style.

- A **Motion on / Motion off** pill sits next to the theme toggle. It turns every
  animation off instantly and remembers the choice; it starts from your OS
  `prefers-reduced-motion` setting.
- Off sets `html[data-motion="off"]`, which collapses all durations to 0.001 ms — the
  score counter jumps straight to its final number and nothing slides.
- All motion lives in one labelled `/* motion */` block at the end of `web/styles.css`
  plus four helpers (`countUp`, `playReveal`, `applyMotion`, `initMotion`) in
  `web/app.js`. Delete those and the UI is static with no other change.

What animates: masthead and composer entrance, a staggered reveal of each fresh
analysis, tab cross-fades, a 2px hover lift on cards, the theme cross-fade, a single
pulse on the card you pick, and the score counting from the old value to the new one.

## Human in the loop: old vs new

Every analysis ends with the two prompts side by side: the original and the merged one,
each with its rubric score, estimated tokens and open gaps, plus a "what changed" list.
You then say which one you would actually send.

- **"I'd use this"** saves your pick to `artifacts/preferences.jsonl` — one JSON object
  per line with the prompt, the merged text, the choice, both scores, the engine that
  drove the rewrite, and a timestamp.
- The running tally (e.g. "merged 3 · original 1") appears under the comparison and
  survives a restart. One vote per analysis, so repeated clicks cannot inflate it.
- Votes stay local: nothing is retrained or re-scored from them.
- The tally answers a different question than the automatic evaluation. The rubric says
  the merged prompt scores better; the votes say whether you agree.

Throwing the votes away is safe: delete `artifacts/preferences.jsonl` to reset the tally.

## Fusion evaluation (measured, not assumed)

```bash
python -m src.evaluate_fusion                                  # 40 hand-written prompts
python -m src.evaluate_fusion --csv data/test.csv --limit 100  # any CSV with a `text` column
python -m src.evaluate_fusion --mode advanced --level beginner
```

It runs original → merged through both engines for every prompt and writes
`artifacts/fusion_results.md` and `artifacts/fusion_results.json`, plus a console
summary. Measured on `data/manual_test.csv` (40 hand-written prompts, mode=auto,
level=developer):

| Metric | Original | Merged |
|---|---|---|
| Average rubric score | 35.45 / 100 | 63.17 / 100 |
| Average gain | – | **+27.73** (median +27.5, range +11 to +52) |
| Prompts improved / unchanged / regressed | – | 40 / 0 / 0 |
| Intent preserved (drift guard) | – | 40/40 (100%) |
| Estimated tokens | 16.1 | 155.0 (20.35×) |
| Gaps addressed | – | 3.02 / 5, 1.02 still open |

By detected task (average gain): coding +31.3 (n=7), analysis +29.1 (n=7),
brainstorming +27.7 (n=6), explanation +27.3 (n=8), writing +25.7 (n=7),
summarization +24.4 (n=5). The rule engine drove 9 of 40 rewrites (the coding ones),
the neural templates the other 31.

Read these numbers for what they are:

- The rubric that scores "after" is the same one whose diagnosis drives the rewrite,
  so part of the gain is self-fulfilling — it measures rubric satisfaction, not
  independent quality. A blind human rating would be the honest next step.
- Key-term coverage is 100% by construction, because the merged prompt always starts
  with the original as its `TASK` line. It proves intent was not dropped, not that the
  result reads better.
- The 20× token ratio is a real cost; short prompts inflate it. Check gain against
  cost before making the merged prompt your default.
- All 40 prompts improved, which is suspiciously uniform — the rubric rewards exactly
  the sections the rewriter adds. The weakest cases in the report (gains of +11 to +14)
  are prompts that were already fairly specific.

## Project layout

```
prompt-studio/
├── server.py              FastAPI: /api/analyze, /api/batch, /api/ollama, /api/health
├── run.py                 uvicorn launcher
├── web/                   index.html, how-it-works.html, styles.css, docs.css,
│                          theme.js, app.js (custom UI, no build step)
├── src/
│   ├── combined.py        fusion layer + public analyze_prompt / batch_analyze
│   ├── rubric.py          rule engine (LCEL pipeline, rubric, rebuild, drift guard)
│   ├── enhancer.py        neural enhancer + CLI
│   ├── model.py           BiLSTM + attention, two heads
│   ├── data.py            tokenizer, vocab, dataset, batching
│   ├── slots.py           regex slot extraction / rule safety net
│   ├── templates.py       per-task rewrite templates
│   ├── config.py          shared constants
│   ├── train.py           multi-task training (CE + BCE, early stopping)
│   ├── baseline.py        TF-IDF + logistic regression baseline
│   ├── evaluate.py        baseline vs BiLSTM vs hybrid, plots, results.md
│   ├── evaluate_fusion.py fusion evaluation: score gain, intent, tokens
│   ├── preferences.py     local old-vs-new vote store (JSONL) + tally
│   └── generate_dataset.py  synthetic dataset generator
├── data/                  train/val/test + 40 hand-written manual_test prompts
├── artifacts/             trained model, vocab, plots, evaluation results
└── tests/                 sanity tests
```

## Retraining and evaluation

```bash
python -m src.generate_dataset   # rebuild synthetic data
python -m src.train              # train BiLSTM, saves artifacts/model.pt
python -m src.evaluate           # compare models, writes artifacts/results.md
python -m src.evaluate_fusion    # fusion gain/intent report, artifacts/fusion_results.md
python -m src.enhancer "write a python function to check if a number is prime"
pytest -q
```

Numbers from the included `artifacts/results.md`:

| Test set | Model | Task acc | Gap macro-F1 |
|---|---|---|---|
| synthetic test | TF-IDF + LogReg | 1.000 | 0.999 |
| synthetic test | BiLSTM + attention | 1.000 | 1.000 |
| hand-written (40) | TF-IDF + LogReg | 0.975 | 0.943 |
| hand-written (40) | BiLSTM + attention | 0.950 | 0.939 |

The honest reading is unchanged from the original project: synthetic data is too easy,
and 40 hand-written prompts are too few to separate the models. The BiLSTM's real
advantages are word-order handling, shared learning and attention-based explanations —
not a better score here.

## Known limitations

- Rewriting is template-based, not generated. Output is consistent, not creative.
- The rule rubric is written for programming prompts; the neural classifier gates it,
  but non-coding prompts therefore lose the rubric's structural rewrite.
- The rubric scores are heuristic, not validated against human ratings.
- Sentiment of "gap probabilities" comes from synthetic training data; treat them as
  signals, not calibrated probabilities.
- Training data, plots and metrics are the originals, not re-derived for the fusion.

## API

| Method | Path | Body |
|---|---|---|
| GET | `/api/examples` | – |
| GET | `/api/health` | – |
| POST | `/api/analyze` | `{prompt, mode, level}` |
| POST | `/api/batch` | `{prompts: [...], mode, level}` |
| GET | `/api/preferences` | – (returns `{tally, recent}`) |
| POST | `/api/preference` | `{prompt, merged, choice: "original"|"merged", score_before, score_after, engine}` |
| POST | `/api/ollama` | `{prompt, optimized, model, base_url}` |

`/api/analyze` returns `combined` (text, sections with source, addressed/residual gaps),
`score` (before/after/delta and both dimension vectors), `neural` (task, confidence,
task probabilities, gap probabilities, attention, tokens, enhanced text) and `rubric`
(dimensions, issues, questions, changes, drift, tokens, versions, rule-only rewrite).
