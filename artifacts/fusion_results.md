# Fusion evaluation

- Prompt set: `data/manual_test.csv` (40 prompts)
- Mode / level: `auto` / `developer`
- Generated: 2026-11-17 16:48 UTC

| Metric | Value |
|---|---|
| Average score before | 35.45 / 100 |
| Average score after | 63.17 / 100 |
| Average gain | +27.73 (median +27.5, range +11 to +52) |
| Improved / unchanged / regressed | 40 / 0 / 0 (100.0% improved) |
| Intent preserved | 40/40 (100.0%) |
| Key terms kept | 100.0% |
| Tokens | 16.12 → 154.97 (20.35×) |
| Gaps addressed | 3.02 / 5 (remaining 1.02) |

## By detected task

| Task | n | Before | After | Gain | Intent kept |
|---|---|---|---|---|---|
| analysis | 7 | 30.86 | 60.0 | +29.14 | 100.0% |
| brainstorming | 6 | 29.0 | 56.67 | +27.67 | 100.0% |
| coding | 7 | 48.29 | 79.57 | +31.29 | 100.0% |
| explanation | 8 | 35.75 | 63.0 | +27.25 | 100.0% |
| summarization | 5 | 31.4 | 55.8 | +24.4 | 100.0% |
| writing | 7 | 35.29 | 61.0 | +25.71 | 100.0% |

## Weakest cases (lowest gain)

| Prompt | Before | After | Delta | Intent kept |
|---|---|---|---|---|
| summarise the meeting notes for my manager in 3 bullet points | 35 | 46 | +11 | yes |
| Compare LSTM and GRU for a team of beginners. Present it as a table. | 37 | 51 | +14 | yes |
| Explain how transformers work. I am preparing for an interview next… | 43 | 57 | +14 | yes |
| Give pros and cons of electric cars in a table, aimed at first-time… | 37 | 51 | +14 | yes |
| Hi! Could you explain overfitting to a non-technical manager in sim… | 42 | 56 | +14 | yes |

> The rubric that scores "after" is the same one that drives the rewrite, so part of the gain is self-fulfilling. Read the gain as "the merged prompt satisfies the rubric much better", not as an independent quality measure.
>
> Key-term coverage is 100% by construction: the merged prompt always starts with the original prompt as its TASK line. It proves intent was not dropped, not that the result is better prose.
>
> The token ratio is a real cost: merging roughly twenty times the input tokens of a very short prompt. Compare it against the gain before deciding it is worth it.
