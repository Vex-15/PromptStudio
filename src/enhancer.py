"""
The full enhancer: neural analysis -> rule safety net -> template rewrite.

    python -m src.enhancer "write a python function to check if a number is prime"
"""
import sys

import torch

from .config import ART_DIR, GAPS, MAX_LEN, TASK_CONF_MIN, TASKS, THRESHOLD
from .data import Vocab, pad_batch, tokenize
from .model import PromptNet
from .slots import evidence_present, extract_slots
from .templates import DEFAULT, LABELS, TEMPLATES


class PromptEnhancer:
    def __init__(self, art_dir=ART_DIR, threshold: float = THRESHOLD):
        self.threshold = threshold
        self.vocab = Vocab.load(art_dir / "vocab.json")
        self.model = PromptNet(len(self.vocab))
        self.model.load_state_dict(torch.load(art_dir / "model.pt", map_location="cpu"))
        self.model.eval()

    # ---------------------------------------------------------------- analysis
    @torch.no_grad()
    def predict_batch(self, texts: list[str]):
        """Returns task probs (B, 6), gap 'missing' probs (B, 5), attention (list of lists)."""
        ids = [self.vocab.encode(t) for t in texts]
        t_logits, g_logits, attn = self.model(pad_batch(ids))
        weights = [attn[i, : len(ids[i])].tolist() for i in range(len(ids))]
        return torch.softmax(t_logits, 1), torch.sigmoid(g_logits), weights

    def analyse(self, text: str, use_rules: bool = True) -> dict:
        t_p, g_p, attn = self.predict_batch([text])
        task = TASKS[int(t_p[0].argmax())]
        probs = {g: float(g_p[0, i]) for i, g in enumerate(GAPS)}
        missing = [g for g in GAPS if probs[g] > self.threshold]
        overridden = []
        if use_rules:  # safety net: rules can prove a part is present
            proven = evidence_present(text)
            overridden = [g for g in missing if g in proven]
            missing = [g for g in missing if g not in proven]
        tokens = tokenize(text)[:MAX_LEN] or ["<unk>"]
        return {
            "task": task,
            "task_probs": {t: float(t_p[0, i]) for i, t in enumerate(TASKS)},
            "gap_probs": probs,
            "missing": missing,
            "rule_overrides": overridden,
            "tokens": tokens,
            "attention": attn[0][: len(tokens)],
        }

    # ----------------------------------------------------------------- rewrite
    def enhance(self, text: str) -> dict:
        a = self.analyse(text)
        slots = extract_slots(text)
        added, questions = [], []
        confident = a["task_probs"][a["task"]] >= TASK_CONF_MIN
        template_task = a["task"] if confident else "generic"
        lines = [f"Task: {text.strip()}"]
        for gap in GAPS:                       # fixed, readable order
            if gap not in a["missing"]:
                continue
            kind, content = TEMPLATES[template_task][gap]
            if gap == "context" and slots["language"]:
                content = f"Language: {slots['language']}. {content}"
            lines.append(f"{LABELS[gap]}: {content}")
            added.append({"part": gap, "kind": kind, "text": content})
            if kind != DEFAULT:
                questions.append(f"{LABELS[gap]}: {content}")
        enhanced = "\n".join(lines) if added else text.strip()
        return {"original": text, "enhanced": enhanced, "added": added,
                "questions": questions, "analysis": a, "template": template_task}


def main():
    if len(sys.argv) < 2:
        print('usage: python -m src.enhancer "your prompt here"')
        return
    enh = PromptEnhancer()
    out = enh.enhance(" ".join(sys.argv[1:]))
    a = out["analysis"]
    print(f"\nDetected task : {a['task']}  ({a['task_probs'][a['task']]:.0%})")
    print("Missing parts :", ", ".join(a["missing"]) or "none")
    if a["rule_overrides"]:
        print("Rule override :", ", ".join(a["rule_overrides"]), "(already present)")
    print("\n--- Enhanced prompt ---")
    print(out["enhanced"])
    if out["questions"]:
        print("\nFill in these placeholders:")
        for q in out["questions"]:
            print(" *", q)


if __name__ == "__main__":
    main()
