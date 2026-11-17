"""
Small evaluation of the fusion layer.

Measures whether the merged prompt actually scores better than the original and
whether the original intent survived, over a fixed prompt set. No LLM, no network.

    python -m src.evaluate_fusion
    python -m src.evaluate_fusion --csv data/test.csv --limit 100
    python -m src.evaluate_fusion --mode standard --level beginner

Writes artifacts/fusion_results.json and artifacts/fusion_results.md.

Caveat to keep in mind: the rubric that grades "after" is the same one whose
diagnosis drives the rewrite, so part of the gain is self-fulfilling. The honest
signals here are the size of the gain across tasks, the share of prompts that
regress, and the drift guard (intent preserved / key terms kept).
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .combined import analyze_prompt
from .config import ART_DIR, DATA_DIR, ROOT


def _mean(xs) -> float:
    return round(statistics.fmean(xs), 2) if xs else 0.0


def _median(xs) -> float:
    return round(statistics.median(xs), 2) if xs else 0.0


def _pct(part: int, total: int) -> float:
    return round(100 * part / total, 1) if total else 0.0


def _display_path(path: str | Path) -> str:
    """Relative path in reports so they stay portable between machines."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def evaluate(csv_path: str | Path, limit: int | None = None, mode: str = "auto",
             level: str = "developer") -> dict:
    df = pd.read_csv(csv_path)
    prompts = [str(t).strip() for t in df["text"].dropna().tolist() if str(t).strip()]
    if limit:
        prompts = prompts[:limit]

    rows = []
    for prompt in prompts:
        r = analyze_prompt(prompt, mode, level)
        rows.append({
            "prompt": prompt,
            "task": r["neural"]["task"],
            "engine": r["combined"]["engine"],
            "before": r["score"]["before"],
            "after": r["score"]["after"],
            "delta": r["score"]["delta"],
            "intent_preserved": bool(r["rubric"]["drift"]["ok"]),
            "coverage": r["rubric"]["drift"]["coverage"],
            "token_ratio": r["rubric"]["tokens"]["ratio"],
            "tokens_before": r["meta"]["tokens_before"],
            "tokens_after": r["meta"]["tokens_after"],
            "gaps_addressed": len(r["combined"]["addressed"]),
            "gaps_remaining": len(r["combined"]["remaining"]),
        })

    deltas = [x["delta"] for x in rows]
    improved = sum(1 for d in deltas if d > 0)
    unchanged = sum(1 for d in deltas if d == 0)
    regressed = sum(1 for d in deltas if d < 0)
    kept = sum(1 for x in rows if x["intent_preserved"])

    by_task: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for x in rows:
        grouped[x["task"]].append(x)
    for task, items in sorted(grouped.items()):
        by_task[task] = {
            "n": len(items),
            "before": _mean([x["before"] for x in items]),
            "after": _mean([x["after"] for x in items]),
            "gain": _mean([x["delta"] for x in items]),
            "intent_preserved_pct": _pct(sum(1 for x in items if x["intent_preserved"]), len(items)),
        }

    engine_counts: dict[str, int] = defaultdict(int)
    for x in rows:
        engine_counts[x["engine"]] += 1

    result = {
        "meta": {
            "prompt_set": _display_path(csv_path),
            "n": len(rows),
            "mode": mode,
            "level": level,
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        },
        "summary": {
            "avg_before": _mean([x["before"] for x in rows]),
            "avg_after": _mean([x["after"] for x in rows]),
            "avg_gain": _mean(deltas),
            "median_gain": _median(deltas),
            "min_gain": min(deltas) if deltas else 0,
            "max_gain": max(deltas) if deltas else 0,
            "improved": improved,
            "unchanged": unchanged,
            "regressed": regressed,
            "improved_pct": _pct(improved, len(rows)),
            "regressed_pct": _pct(regressed, len(rows)),
            "intent_preserved": kept,
            "intent_preserved_pct": _pct(kept, len(rows)),
            "avg_coverage_pct": round(100 * _mean([x["coverage"] for x in rows]), 1),
            "avg_tokens_before": _mean([x["tokens_before"] for x in rows]),
            "avg_tokens_after": _mean([x["tokens_after"] for x in rows]),
            "avg_token_ratio": _mean([x["token_ratio"] for x in rows]),
            "avg_gaps_addressed": _mean([x["gaps_addressed"] for x in rows]),
            "avg_gaps_remaining": _mean([x["gaps_remaining"] for x in rows]),
            "engine_split": dict(sorted(engine_counts.items())),
        },
        "by_task": by_task,
        "rows": rows,
    }
    return result


def write_reports(result: dict) -> tuple[Path, Path]:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    json_path = ART_DIR / "fusion_results.json"
    md_path = ART_DIR / "fusion_results.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    meta, s, by_task = result["meta"], result["summary"], result["by_task"]
    worst = sorted(result["rows"], key=lambda x: x["delta"])[:5]

    lines = [
        "# Fusion evaluation",
        "",
        f"- Prompt set: `{meta['prompt_set']}` ({meta['n']} prompts)",
        f"- Mode / level: `{meta['mode']}` / `{meta['level']}`",
        f"- Generated: {meta['generated']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Average score before | {s['avg_before']} / 100 |",
        f"| Average score after | {s['avg_after']} / 100 |",
        f"| Average gain | +{s['avg_gain']} (median +{s['median_gain']}, "
        f"range {s['min_gain']:+d} to {s['max_gain']:+d}) |",
        f"| Improved / unchanged / regressed | {s['improved']} / {s['unchanged']} / {s['regressed']}"
        f" ({s['improved_pct']}% improved) |",
        f"| Intent preserved | {s['intent_preserved']}/{meta['n']} ({s['intent_preserved_pct']}%) |",
        f"| Key terms kept | {s['avg_coverage_pct']}% |",
        f"| Tokens | {s['avg_tokens_before']} → {s['avg_tokens_after']} ({s['avg_token_ratio']}×) |",
        f"| Gaps addressed | {s['avg_gaps_addressed']} / 5 (remaining {s['avg_gaps_remaining']}) |",
        "",
        "## By detected task",
        "",
        "| Task | n | Before | After | Gain | Intent kept |",
        "|---|---|---|---|---|---|",
    ]
    for task, t in by_task.items():
        lines.append(f"| {task} | {t['n']} | {t['before']} | {t['after']} | +{t['gain']} | {t['intent_preserved_pct']}% |")

    lines += ["", "## Weakest cases (lowest gain)", "",
              "| Prompt | Before | After | Delta | Intent kept |", "|---|---|---|---|---|"]
    for x in worst:
        prompt = x["prompt"] if len(x["prompt"]) <= 70 else x["prompt"][:67] + "…"
        lines.append(f"| {prompt} | {x['before']} | {x['after']} | {x['delta']:+d} | "
                     f"{'yes' if x['intent_preserved'] else 'no'} |")

    lines += ["",
              "> The rubric that scores \"after\" is the same one that drives the rewrite, "
              "so part of the gain is self-fulfilling. Read the gain as \"the merged prompt "
              "satisfies the rubric much better\", not as an independent quality measure.",
              ">",
              "> Key-term coverage is 100% by construction: the merged prompt always starts with "
              "the original prompt as its TASK line. It proves intent was not dropped, not that "
              "the result is better prose.",
              ">",
              "> The token ratio is a real cost: merging roughly twenty times the input tokens "
              "of a very short prompt. Compare it against the gain before deciding it is worth it."]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the fusion layer on a prompt set.")
    parser.add_argument("--csv", default=str(DATA_DIR / "manual_test.csv"),
                        help="CSV with a 'text' column (default: data/manual_test.csv)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N prompts")
    parser.add_argument("--mode", default="auto", choices=["auto", "basic", "standard", "advanced"])
    parser.add_argument("--level", default="developer", choices=["developer", "beginner"])
    args = parser.parse_args(argv)

    result = evaluate(args.csv, limit=args.limit or None, mode=args.mode, level=args.level)
    json_path, md_path = write_reports(result)

    m, s = result["meta"], result["summary"]
    print(f"\nFusion evaluation - {m['n']} prompts from {m['prompt_set']} "
          f"(mode={m['mode']}, level={m['level']})\n")
    print(f"  score before      {s['avg_before']:>6} / 100")
    print(f"  score after       {s['avg_after']:>6} / 100")
    print(f"  average gain      {s['avg_gain']:>+6}   (median {s['median_gain']:+})")
    print(f"  improved          {s['improved']:>3}/{m['n']}  ({s['improved_pct']}%)")
    print(f"  regressed         {s['regressed']:>3}/{m['n']}  ({s['regressed_pct']}%)")
    print(f"  intent preserved  {s['intent_preserved']:>3}/{m['n']}  ({s['intent_preserved_pct']}%)")
    print(f"  key terms kept    {s['avg_coverage_pct']:>6}%")
    print(f"  tokens            {s['avg_tokens_before']} -> {s['avg_tokens_after']}  ({s['avg_token_ratio']}x)")
    print(f"  gaps addressed    {s['avg_gaps_addressed']} / 5  (remaining {s['avg_gaps_remaining']})")
    print(f"  engines           {s['engine_split']}")

    print("\n  by task:")
    for task, t in result["by_task"].items():
        print(f"    {task:<14} n={t['n']:<3} {t['before']:>5} -> {t['after']:<5} "
              f"(+{t['gain']})  intent {t['intent_preserved_pct']}%")

    print(f"\n  wrote {json_path}")
    print(f"  wrote {md_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
