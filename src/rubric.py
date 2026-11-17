"""
Rule engine (merged from Prompt-Optimizer): a deterministic grader + improver for
programming prompts.

LangChain (LCEL) is the orchestration layer: every stage is a Runnable, stages are
chained with `|`, independent detectors run in parallel via RunnablePassthrough.assign,
and many prompts are scored at once with `.batch()`.

NO LLM is used to rewrite or grade prompts. An LLM is only used (optionally) by
run_llm() to execute the original vs. optimized prompt for comparison.
"""
from __future__ import annotations

import math
import re

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# ----------------------------------------------------------------------------
# 1. RUBRIC: maximum points per dimension (sums to 100). Tune these.
# ----------------------------------------------------------------------------
MAX = {"clarity": 20, "context": 20, "specificity": 20,
       "output": 15, "constraints": 15, "efficiency": 10}

INTENT_NAME = {"generate": "Code generation", "debug": "Debugging",
               "refactor": "Refactor / review", "test": "Test writing",
               "explain": "Code explanation", "general": "General programming Q&A"}

# ----------------------------------------------------------------------------
# 2. Pattern banks (the "knowledge" of the system). Extend these first.
# ----------------------------------------------------------------------------
STOP = set("a an the and or of to in on for with is are be it this that my me i you your "
           "as at by from do does can could would please".split())

FILLER = [re.compile(p, re.I) for p in (
    r"\bplease\b", r"\bkindly\b", r"\bi (?:want|need|would like) you to\b",
    r"\b(?:can|could|would) you\b", r"\btry to\b", r"\bif possible\b",
    r"\b(?:very|really|actually|basically|simply)\b")]

LANGS = [(n, re.compile(p, re.I)) for n, p in (
    ("Python", r"\b(python|py)\b"), ("JavaScript", r"\b(javascript|js|node(?:\.?js)?)\b"),
    ("TypeScript", r"\b(typescript|ts)\b"), ("Java", r"\bjava\b"),
    ("C++", r"\bc\+\+|\bcpp\b"), ("C#", r"c#|\bcsharp\b"), ("Go", r"\bgolang\b|\bin go\b"),
    ("Rust", r"\brust\b"), ("SQL", r"\b(sql|mysql|postgres(?:ql)?)\b"),
    ("PHP", r"\bphp\b"), ("Kotlin", r"\bkotlin\b"), ("Bash", r"\b(bash|shell script)\b"))]

LIBS = re.compile(r"\b(pandas|numpy|scikit-learn|sklearn|matplotlib|pytorch|tensorflow|flask|"
                  r"fastapi|django|express|react|spring|junit|pytest|requests|sqlalchemy|streamlit)\b", re.I)
VAGUE = re.compile(r"\b(something|stuff|thing|things|better|good|nice|etc|whatever|some|proper(?:ly)?|best)\b", re.I)
VERBS = re.compile(r"\b(write|create|build|make|develop|implement|generate|code|fix|debug|refactor|"
                   r"optimi[sz]e|review|explain|convert|train|parse|design|add|test|calculate|sort|find)\b", re.I)
FORMAT_CUES = [re.compile(p, re.I) for p in (
    r"\b(single|one) (file|script|function|class|module)\b",
    r"\b(function|class|method|signature)\b",
    r"\b(explain|explanation|comments?|docstrings?|step[- ]by[- ]step)\b",
    r"\b(example|usage|demo|test cases?|unit tests?)\b",
    r"\b(json|markdown|table|diff|csv|code block)\b",
    r"\b(report|return|print|plot|output)\b")]
INJECTION = re.compile(r"ignore (all )?(previous|prior|above) instructions|reveal (the |your )?system prompt|"
                       r"override (the )?instructions|print (hidden|your) (config|instructions)|"
                       r"disregard .{0,30}rules", re.I)
UNSUPPORTED = re.compile(r"\b(guarantee|100% (correct|secure|bug[- ]free)|no bugs|perfect)\b", re.I)
VERSION_RISK = re.compile(r"\b(latest|newest|any library|best library)\b", re.I)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def words(s: str) -> list[str]:
    return re.findall(r"[\w+#.\-]+", s)


def tokens(s: str) -> int:
    """Rough token estimate (chars/4). Swap for tiktoken for exact counts."""
    return math.ceil(len(s) / 4)


def clamp(v: float, hi: float, lo: float = 0) -> float:
    return max(lo, min(hi, v))


def detect_intent_text(p: str) -> str:
    if re.search(r"\b(bug|error|exception|traceback|crash|not working|doesn'?t work|fails?|fix|debug)\b", p, re.I):
        return "debug"
    if re.search(r"\b(unit tests?|write tests?|test cases?)\b", p, re.I):
        return "test"
    if re.search(r"\b(refactor|optimi[sz]e|clean up|review|improve (my )?code)\b", p, re.I):
        return "refactor"
    if re.search(r"\b(explain|what does|how does|walk me through)\b", p, re.I) and \
            re.search(r"\b(code|function|snippet|this|class)\b", p, re.I):
        return "explain"
    if re.search(r"\b(write|create|build|make|develop|implement|generate|code|program|script|function|api|app|convert|parse)\b", p, re.I):
        return "generate"
    return "general"


# ----------------------------------------------------------------------------
# 3. PIPELINE STAGES (each takes and returns a state dict)
# ----------------------------------------------------------------------------
def s_security(st: dict) -> bool:
    return bool(INJECTION.search(st["prompt"]))


def s_intent(st: dict) -> str:
    return detect_intent_text(st["prompt"])


def s_features(st: dict) -> dict:
    """Turn raw text into checkable booleans/counts. Scoring only ever sees these."""
    p = st["prompt"]
    w = words(p)
    lang = next((n for n, r in LANGS if r.search(p)), None)
    lib = LIBS.search(p)
    seen: dict[str, int] = {}
    for x in (t.lower() for t in w if len(t) > 4):
        seen[x] = seen.get(x, 0) + 1
    f = {
        "n_words": len(w),
        "lang": lang,
        "lib": lib.group(0) if lib else None,
        "has_code": bool(re.search(r"```|\bdef \w+\(|\bfunction\s*\w*\(|[{};]\s*$|\btraceback\b|\w+Error\b", p, re.M | re.I)),
        "n_filler": sum(len(r.findall(p)) for r in FILLER),
        "vague": VAGUE.findall(p),
        "tasks": len(re.findall(r"\b(and then|and also|also|additionally)\b|;", p, re.I))
                 + (1 if re.search(r"\b(write|create|build|fix|explain|test|deploy)\b.*\band\b.*\b(write|create|build|fix|explain|test|deploy)\b", p, re.I) else 0),
        "conflict": bool((re.search(r"\b(short|brief|concise|one[- ]liner|\d+ words?)\b", p, re.I)
                          and re.search(r"\b(detailed|comprehensive|in[- ]depth|fully|everything)\b", p, re.I))
                         or (re.search(r"\b(no|without) (external )?(libraries|libs|packages)\b", p, re.I) and lib)),
        "repetition": any(v >= 3 for v in seen.values()),
        "env": bool(re.search(r"\b(v?\d+\.\d+|version|linux|windows|macos|docker|python ?3|node ?\d+|jdk|colab)\b", p, re.I)),
        "data": bool(re.search(r"\b(dataset|csv|json|database|table|file|api|input|supplied|data)\b", p, re.I)),
        "purpose": bool(re.search(r"\b(for (a|an|my|our|the) \w+|so that|in order to|because|which will)\b", p, re.I)),
        "io": bool(re.search(r"\b(input|output|returns?|takes?|accepts?|reads?|prints?|parameters?|arguments?|csv|json|file|dataset|database)\b", p, re.I)),
        "concrete": len(re.findall(r"\b(\d+|sort|binary search|linked list|hash ?map|dataframe|regression|classif\w+|rest|crud|jwt|oauth|algorithm|queue|stack|tree|graph|recursion|random_state|stratified)\b", p, re.I)),
        "list": bool(re.search(r"(^|\n)\s*(\d+[.)]|[-*•])\s", p)) or p.count(",") >= 3,
        "fmt": sum(1 for r in FORMAT_CUES if r.search(p)),
        "err": bool(re.search(r"\b(error handling|exceptions?|validate|validation|invalid|edge cases?)\b", p, re.I)),
        "perf": bool(re.search(r"\b(complexity|O\(|performance|efficient|memory|fast)\b", p, re.I)),
        "style": bool(re.search(r"\b(pep ?8|type hints?|clean|readable|naming|lint|solid)\b", p, re.I)),
        "neg": bool(re.search(r"\b(do not|don'?t|without|no external|avoid|never)\b", p, re.I)),
        "repro": bool(re.search(r"\b(random_state|seed|stratified|deterministic|reproducib\w+)\b", p, re.I)),
        "tests": bool(re.search(r"\b(tests?|pytest|junit|unit)\b", p, re.I)),
    }
    return f


def s_score(st: dict) -> dict:
    """Rubric: every point is earned by a specific, checkable feature."""
    f = st["feat"]
    D = {
        "clarity": clamp((6 if f["n_words"] >= 3 else 2)            # long enough to carry a task
                         + (6 if VERBS.search(st["prompt"]) else 0)  # has an action verb
                         + clamp(4 - 2 * len(f["vague"]), 4)         # few vague words
                         + (4 if f["tasks"] < 2 else 0)              # one task, not many
                         - (6 if f["conflict"] else 0), 20),         # contradictions hurt
        "context": clamp((8 if f["lang"] else 0) + (5 if f["lib"] else 0)
                         + (3 if f["env"] else 0) + (4 if (f["purpose"] or f["data"]) else 0), 20),
        "specificity": clamp((6 if f["io"] else 0)
                             + (5 if f["concrete"] >= 2 else 3 if f["concrete"] else 0)
                             + (5 if f["list"] else 0)
                             + (4 if f["n_words"] >= 8 else 2 if f["n_words"] >= 5 else 0), 20),
        "output": clamp(f["fmt"] * 5, 15),
        "constraints": clamp((4 if f["err"] else 0) + (3 if f["perf"] else 0) + (3 if f["style"] else 0)
                             + (3 if f["neg"] else 0) + (3 if f["repro"] else 0) + (2 if f["tests"] else 0), 15),
        "efficiency": clamp(10 - 2 * f["n_filler"] - (2 if f["repetition"] else 0)
                            - (2 if f["n_words"] > 120 else 0), 10),
    }
    D = {k: int(v) for k, v in D.items()}
    return {**st, "dims": D, "total": sum(D.values())}


def s_diagnose(st: dict) -> dict:
    f, intent = st["feat"], st["intent"]
    issues, qs = [], []
    if st["injection"]:
        issues.append(("high", "Possible prompt-injection pattern. Review before forwarding."))
    if f["n_words"] < 3:
        issues.append(("high", "Too short to infer a task."))
        qs.append("What exactly should the code do?")
    if not f["lang"]:
        issues.append(("medium", "No programming language named."))
        if intent in ("generate", "test", "general"):
            qs.append("Which language or framework should be used?")
    if not f["io"] and intent in ("generate", "test", "general"):
        issues.append(("medium", "Inputs and outputs not described."))
    if not f["fmt"]:
        issues.append(("medium", "No output format requested."))
    if not (f["err"] or f["neg"] or f["perf"]):
        issues.append(("medium", "No constraints (errors, performance, style, exclusions)."))
    if f["vague"]:
        issues.append(("medium", "Vague words: " + ", ".join(f["vague"])))
    if f["tasks"] >= 2:
        issues.append(("medium", "Several tasks mixed in one request."))
    if f["conflict"]:
        issues.append(("high", "Contradictory requirements."))
    if f["n_filler"] or f["repetition"]:
        issues.append(("low", "Filler or repetition wastes tokens."))
    if UNSUPPORTED.search(st["prompt"]):
        issues.append(("medium", "Unsupported expectation (guarantees)."))
    if VERSION_RISK.search(st["prompt"]):
        issues.append(("medium", "Version-sensitive request: risk of invented APIs."))
    if intent in ("debug", "refactor", "explain") and not f["has_code"]:
        issues.append(("high", "No code or error message supplied."))
        qs.append("Paste the code" + (" and the exact error or wrong output." if intent == "debug" else "."))
    return {**st, "issues": issues, "questions": qs}


def s_decide(st: dict) -> dict:
    mode, t = st.get("mode", "auto"), st["total"]
    if mode == "auto":
        mode = "none" if t > 85 else "standard" if t >= 60 else "advanced"   # <30 also advanced (+ questions)
    return {**st, "strategy": mode}


def compress(p: str) -> str:
    parts = re.split(r"(```[\s\S]*?```)", p)
    out = []
    for seg in parts:
        if not seg.startswith("```"):
            for r in FILLER:
                seg = r.sub("", seg)
            seg = re.sub(r"\s{2,}", " ", seg)
            seg = re.sub(r"\s+([,.?!])", r"\1", seg).strip()
        out.append(seg)
    s = "".join(out)
    return s[:1].upper() + s[1:]


REQ = {
    "generate": ["State the function or module signature with input and output types before the code.",
                 "Include one short usage example."],
    "debug": ["Identify the root cause before proposing a fix.",
              "Show the corrected code and explain what changed and why.",
              "If code or the error message is missing, say what is needed instead of guessing."],
    "refactor": ["Preserve existing behaviour.", "List each change with its reason.",
                 "Note any change in time or space complexity."],
    "test": ["Cover normal, edge and invalid inputs.", "Name each test after the behaviour it checks."],
    "explain": ["Walk through the code in execution order.", "Point out edge cases and likely pitfalls."],
    "general": ["Give a direct answer first, then supporting detail."],
}


def s_rebuild(st: dict) -> dict:
    """Template-based rebuild. Only weak dimensions get new sections."""
    p, D, intent, m = st["prompt"], st["dims"], st["intent"], st["strategy"]
    changes: list[str] = []
    task = compress(p)
    if task != p.strip():
        changes.append("Removed filler words and fixed spacing (V1)")
    if m == "none":
        changes.append("Prompt already strong: left unchanged apart from trimming")
        return {**st, "optimized": task, "changes": changes, "task_only": task,
                "sections": [("TASK", task)]}
    if m == "basic":
        if not re.search(r"[.?!:]$|```$", task):
            task += "."
        return {**st, "optimized": task, "changes": changes, "task_only": task,
                "sections": [("TASK", task)]}

    ctx, req, con, out, val = [], [], [], [], []
    if not st["feat"]["lang"]:
        ctx.append("Language: Python 3 (assumed; replace if you need another)")
        changes.append("Added assumed language (flagged so you can change it)")
    ctx.append("Reader: beginner; explain each step in plain language, define terms"
               if st.get("level") == "beginner"
               else "Reader: working developer; keep explanation brief and precise")
    if D["specificity"] < 14:
        req += REQ[intent]
        changes.append(f"Added requirements for {INTENT_NAME[intent].lower()}")
    if D["output"] < 8:
        out.append("Corrected code as a single code block, then a short change list (max ~150 words)."
                   if intent in ("debug", "refactor")
                   else "One runnable code block, then a short explanation (max ~150 words).")
        changes.append("Defined the output format")
    if D["constraints"] < 6:
        con += ["Handle invalid input and errors explicitly.",
                "Use only real, documented APIs. If unsure a function exists, say so instead of inventing it.",
                "Add concise comments; avoid unexplained external dependencies."]
        changes.append("Added constraints incl. hallucination control")
    if m == "advanced":
        req.append("Work in steps: restate the problem and assumptions, outline the approach, "
                   "write the code, then check it against 2-3 examples.")
        val.append("Re-check the result against every requirement above and list any you could not meet.")
        if intent == "generate" and not st["feat"]["tests"]:
            req.append("Include 3 test cases.")
        changes.append("Added task decomposition and a validation step (advanced)")

    sections = [("TASK", task)]
    if ctx: sections.append(("CONTEXT", "\n".join("- " + x for x in ctx)))
    if req: sections.append(("REQUIREMENTS", "\n".join(f"{i}. {x}" for i, x in enumerate(req, 1))))
    if con: sections.append(("CONSTRAINTS", "\n".join("- " + x for x in con)))
    if out: sections.append(("OUTPUT FORMAT", " ".join(out)))
    if val: sections.append(("VALIDATION", " ".join(val)))
    text = "\n\n".join(f"{h}:\n{b}" for h, b in sections)
    return {**st, "optimized": text, "changes": changes, "task_only": task,
            "sections": sections}


def s_rescore(st: dict) -> dict:
    after = analyzer.invoke({"prompt": st["optimized"]})
    return {**st, "after_dims": after["dims"], "after_total": after["total"]}


def s_validate(st: dict) -> dict:
    """Drift guard: keep the user's key terms and the same intent."""
    core = [w for w in words(compress(st["prompt"]).lower()) if w not in STOP and len(w) > 2]
    low = st["optimized"].lower()
    coverage = sum(1 for w in core if w in low) / len(core) if core else 1.0
    same = detect_intent_text(st["task_only"]) == st["intent"]
    ok = coverage >= 0.9 and same and not st["injection"]
    t0, t1 = tokens(st["prompt"]), tokens(st["optimized"])
    return {**st,
            "drift": {"coverage": round(coverage, 2), "intent_same": same, "ok": ok},
            "tokens": {"before": t0, "after": t1, "ratio": round(t1 / max(t0, 1), 2)},
            "versions": [("V0 original", st["prompt"]), ("V1 compressed", compress(st["prompt"])),
                         (f"V2 {st['strategy']}", st["optimized"])]}


# ----------------------------------------------------------------------------
# 4. LANGCHAIN WIRING
# ----------------------------------------------------------------------------
def _add(key, fn):
    return lambda st: {**st, key: fn(st)}


# assign() runs security / intent / feature detectors in parallel and merges their outputs
analyzer = (
    RunnablePassthrough.assign(
        injection=RunnableLambda(s_security),
        intent=RunnableLambda(s_intent),
        feat=RunnableLambda(s_features),
    )
    | RunnableLambda(s_score)
    | RunnableLambda(s_diagnose)
)

pipeline = (
    analyzer
    | RunnableLambda(s_decide)
    | RunnableLambda(s_rebuild)
    | RunnableLambda(s_rescore)
    | RunnableLambda(s_validate)
)


def optimize(prompt: str, mode: str = "auto", level: str = "developer") -> dict:
    return pipeline.invoke({"prompt": prompt.strip(), "mode": mode, "level": level})


def score_many(prompts: list[str], mode: str = "auto", level: str = "developer") -> list[dict]:
    return pipeline.batch([{"prompt": p.strip(), "mode": mode, "level": level} for p in prompts])


# ----------------------------------------------------------------------------
# 5. OPTIONAL: run original vs optimized on a real model (execution only, not rewriting)
# ----------------------------------------------------------------------------
def run_llm(prompt: str, model: str = "llama3.2", temperature: float = 0.0,
            base_url: str = "http://localhost:11434") -> str:
    """Run a prompt on a local Ollama model (start Ollama and `ollama pull <model>` first)."""
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, temperature=temperature, base_url=base_url).invoke(prompt).content


if __name__ == "__main__":
    r = optimize("write python code for student marks")
    print(r["total"], "->", r["after_total"], "\n")
    print(r["optimized"])