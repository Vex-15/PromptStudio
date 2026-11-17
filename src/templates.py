"""
Rewrite templates.

For every (task, missing part) we store either
  - a DEFAULT: a sensible assumption we can safely add (flagged as an assumption), or
  - a PLACEHOLDER: something only the user knows (context, purpose) -> [in brackets]
    and listed as a question to answer.

Editing this file is the easiest way to improve the enhancer's output quality.
"""

DEFAULT, PLACEHOLDER = "default", "placeholder"

TEMPLATES = {
    "coding": {
        "context": (PLACEHOLDER, "[Describe your project and where this code will run]"),
        "audience": (DEFAULT, "Write it so a student with basic programming knowledge can follow it; explain any non-obvious step."),
        "constraints": (DEFAULT, "Handle edge cases and invalid input. Prefer the standard library. Keep the code clean and readable."),
        "output_format": (DEFAULT, "Return the complete code in one code block, then a short explanation and an example run."),
        "purpose": (PLACEHOLDER, "[What should this code achieve, and how will you use it?]"),
    },
    "writing": {
        "context": (PLACEHOLDER, "[Who is this from, and what is the situation or background?]"),
        "audience": (DEFAULT, "A general reader; keep the language clear and natural."),
        "constraints": (DEFAULT, "Keep it focused and under 300 words. Avoid cliches and filler."),
        "output_format": (DEFAULT, "A clear structure: a title, short paragraphs and a closing line."),
        "purpose": (PLACEHOLDER, "[What should the reader think or do after reading this?]"),
    },
    "summarization": {
        "context": (PLACEHOLDER, "[Where does the text come from and what is it about?]"),
        "audience": (DEFAULT, "A busy reader who has not seen the original."),
        "constraints": (DEFAULT, "Stay faithful to the source: do not add anything that is not in the text. Keep it under 150 words."),
        "output_format": (DEFAULT, "3 to 5 bullet points, followed by a one-sentence takeaway."),
        "purpose": (PLACEHOLDER, "[What will you use the summary for?]"),
    },
    "explanation": {
        "context": (PLACEHOLDER, "[What do you already know about this topic?]"),
        "audience": (DEFAULT, "A beginner with basic background knowledge; define technical terms when first used."),
        "constraints": (DEFAULT, "Use plain language and one concrete example. Keep it under 250 words."),
        "output_format": (DEFAULT, "A one-sentence definition, then a step-by-step explanation, then a short example."),
        "purpose": (PLACEHOLDER, "[What do you need this for? exam, project, interview...]"),
    },
    "analysis": {
        "context": (PLACEHOLDER, "[Share the data, document or situation to analyse, plus any background]"),
        "audience": (DEFAULT, "A decision-maker who wants clear conclusions rather than raw detail."),
        "constraints": (DEFAULT, "Base every point on the evidence provided; state assumptions and limitations explicitly."),
        "output_format": (DEFAULT, "A short summary of findings, then a table or bullet list of key points, then recommendations."),
        "purpose": (PLACEHOLDER, "[What decision or question should this analysis support?]"),
    },
    "brainstorming": {
        "context": (PLACEHOLDER, "[Describe your situation, resources and limits: time, budget, skills]"),
        "audience": (DEFAULT, "A general audience; ideas should be easy to explain in one sentence."),
        "constraints": (DEFAULT, "Make the ideas realistic and clearly different from each other; avoid generic ones."),
        "output_format": (DEFAULT, "A numbered list of 5 ideas, each with a one-line description and why it could work."),
        "purpose": (PLACEHOLDER, "[What are you trying to achieve with these ideas?]"),
    },
}

# Used when the task classifier is not confident, so we never apply a wrong task's template.
TEMPLATES["generic"] = {
    "context": (PLACEHOLDER, "[Give the background: who you are, what this is for]"),
    "audience": (DEFAULT, "A general reader; keep the language clear."),
    "constraints": (DEFAULT, "Be specific and concise. Say so if something is unclear instead of guessing."),
    "output_format": (DEFAULT, "A clear, well-structured answer."),
    "purpose": (PLACEHOLDER, "[What do you want to do with the answer?]"),
}

LABELS = {
    "context": "Context",
    "audience": "Audience",
    "constraints": "Constraints",
    "output_format": "Output format",
    "purpose": "Purpose",
}
