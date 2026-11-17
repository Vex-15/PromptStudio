"""
Rule-based slot extraction.

Two jobs:
 1. Pull concrete facts the user already stated (programming language, word limit)
    so the rewritten prompt can reuse them instead of asking again.
 2. Provide "evidence" that a part is already present (e.g. a word limit means
    constraints exist). This is a safety net that corrects false "missing" calls
    from the neural model.
"""
import re

LANGUAGES = ["python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
             "sql", "kotlin", "swift", "php", "ruby", "r"]
_LANG_RE = re.compile(r"(?<![\w+#])(" + "|".join(re.escape(l) for l in LANGUAGES) + r")(?![\w+#])",
                      re.I)
_LIMIT_RE = re.compile(
    r"\b(under|within|max(?:imum)?|at most|no more than|less than|limit(?:ed)?(?: to)?)\s+\d+\s*"
    r"(words?|sentences?|points?|lines?|characters?|minutes?|seconds?)", re.I)
# Format evidence must be high-precision: "read a json file" is NOT an output format,
# but "return it in json" is. So we need either an unambiguous phrase, or a cue word
# ("as", "in", "into", "with", "be", "using") shortly before a format term.
_FORMAT_TERMS = (r"(?:bullet(?:ed)? points?|bullets|numbered list|table|json|markdown|csv|"
                 r"paragraphs?|outline|plain text|code block|list)")
_FORMAT_RE = re.compile(
    r"\b(?:bullet(?:ed)? points?|numbered list|step[- ]by[- ]step|code block|plain text)\b|"
    r"\b(?:as|in|into|with|be|using)\s+(?:\w+\s+){0,2}" + _FORMAT_TERMS + r"\b", re.I)
_TONE_RE = re.compile(r"\b(formal|friendly|professional|casual|persuasive|neutral|humorous)\s+tone\b|"
                      r"\buse a \w+ tone\b|\bavoid jargon\b|\bno jargon\b", re.I)


def extract_slots(text: str) -> dict:
    lang = _LANG_RE.search(text)
    return {
        "language": lang.group(1) if lang else None,
        "has_limit": bool(_LIMIT_RE.search(text)),
        "has_format_words": bool(_FORMAT_RE.search(text)),
        "has_tone": bool(_TONE_RE.search(text)),
    }


def evidence_present(text: str) -> set[str]:
    """Gap names the rules can PROVE are already present."""
    s = extract_slots(text)
    present = set()
    if s["has_limit"] or s["has_tone"]:
        present.add("constraints")
    if s["has_format_words"]:
        present.add("output_format")
    return present
