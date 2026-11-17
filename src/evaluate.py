"""
Evaluate baseline vs BiLSTM+attention vs hybrid (BiLSTM + rules) on:
  - test.csv        synthetic held-out set (same generator as training -> optimistic)
  - manual_test.csv hand-written prompts (the honest number)

Run:  python -m src.evaluate
Outputs: artifacts/results.json, results.md, confusion_matrix.png, attention_examples.png
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .baseline import train_baseline
from .config import ART_DIR, DATA_DIR, GAP_COLS, GAPS, TASKS, THRESHOLD
from .enhancer import PromptEnhancer
from .slots import evidence_present


def deep_predict(enh: PromptEnhancer, texts, use_rules: bool):
    t_p, g_p, _ = enh.predict_batch(list(texts))
    task_pred = [TASKS[i] for i in t_p.argmax(1).tolist()]
    gap_pred = (g_p > THRESHOLD).int().numpy()
    if use_rules:
        for i, text in enumerate(texts):
            for gap in evidence_present(text):
                gap_pred[i, GAPS.index(gap)] = 0
    return task_pred, gap_pred


def score(df, task_pred, gap_pred):
    y = df[GAP_COLS].values
    per_label = f1_score(y, gap_pred, average=None, zero_division=0)
    return {
        "task_acc": float(accuracy_score(df["task"], task_pred)),
        "gap_macro_f1": float(f1_score(y, gap_pred, average="macro", zero_division=0)),
        "gap_exact_match": float((y == gap_pred).all(axis=1).mean()),
        "per_label_f1": {g: float(v) for g, v in zip(GAPS, per_label)},
    }


def main():
    enh = PromptEnhancer()
    base = train_baseline()
    results = {}
    for split in ["test", "manual_test"]:
        df = pd.read_csv(DATA_DIR / f"{split}.csv")
        texts = df["text"].tolist()
        b_task, b_gap = base.predict(texts)
        results[split] = {
            "TF-IDF + LogReg": score(df, b_task, b_gap),
            "BiLSTM + attention": score(df, *deep_predict(enh, texts, use_rules=False)),
            "BiLSTM + rules (hybrid)": score(df, *deep_predict(enh, texts, use_rules=True)),
        }
    with open(ART_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # human-readable table
    lines = []
    for split, models in results.items():
        lines += [f"\n### {split}", "| model | task acc | gap macro-F1 | gap exact match |", "|---|---|---|---|"]
        for name, m in models.items():
            lines.append(f"| {name} | {m['task_acc']:.3f} | {m['gap_macro_f1']:.3f} | {m['gap_exact_match']:.3f} |")
    table = "\n".join(lines)
    (ART_DIR / "results.md").write_text(table)
    print(table)

    # plots
    manual = pd.read_csv(DATA_DIR / "manual_test.csv")
    plot_confusion(manual, enh)
    plot_attention(enh, [
        "write a python function to check if a number is prime",
        "Explain backpropagation to a first year student in under 200 words.",
        "I'm a student building a chatbot. Summarize this paper in five bullet points.",
    ])
    print("\nsaved results.json, results.md, confusion_matrix.png, attention_examples.png")


def plot_confusion(df, enh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred, _ = deep_predict(enh, df["text"].tolist(), use_rules=False)
    cm = confusion_matrix(df["task"], pred, labels=TASKS)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(TASKS)), yticks=range(len(TASKS)), xticklabels=TASKS,
           yticklabels=TASKS, xlabel="predicted", ylabel="true",
           title="Task classifier: confusion matrix (manual test)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    for i in range(len(TASKS)):
        for j in range(len(TASKS)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(ART_DIR / "confusion_matrix.png", dpi=130)


def plot_attention(enh, prompts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(prompts), 1, figsize=(10, 1.4 * len(prompts)))
    for ax, p in zip(np.atleast_1d(axes), prompts):
        a = enh.analyse(p)
        w = np.array(a["attention"])[None, :]
        ax.imshow(w, cmap="Reds", aspect="auto")
        ax.set_xticks(range(len(a["tokens"])))
        ax.set_xticklabels(a["tokens"], rotation=60, ha="right", fontsize=8)
        ax.set_yticks([])
        ax.set_title(f"task={a['task']} | missing={a['missing']}", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(ART_DIR / "attention_examples.png", dpi=130)


if __name__ == "__main__":
    main()
