"""
Train the BiLSTM + attention model.

Run:  python -m src.train
Outputs (in artifacts/): model.pt, vocab.json, history.json, training_curves.png
"""
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from .config import (ART_DIR, BATCH_SIZE, DATA_DIR, EPOCHS, GAP_COLS, LR, PATIENCE,
                     SEED, THRESHOLD)
from .data import PromptDataset, Vocab, collate
from .model import PromptNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, task_loss_fn, gap_loss_fn):
    model.eval()
    total, n = 0.0, 0
    task_true, task_pred, gap_true, gap_pred = [], [], [], []
    for x, y_task, y_gap in loader:
        t_logits, g_logits, _ = model(x)
        loss = task_loss_fn(t_logits, y_task) + gap_loss_fn(g_logits, y_gap)
        total += loss.item() * len(x)
        n += len(x)
        task_true += y_task.tolist()
        task_pred += t_logits.argmax(1).tolist()
        gap_true += y_gap.int().tolist()
        gap_pred += (torch.sigmoid(g_logits) > THRESHOLD).int().tolist()
    return {
        "loss": total / n,
        "task_acc": accuracy_score(task_true, task_pred),
        "gap_f1": f1_score(gap_true, gap_pred, average="macro", zero_division=0),
    }


def main():
    set_seed(SEED)
    ART_DIR.mkdir(exist_ok=True)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

    vocab = Vocab.build(train_df["text"])
    vocab.save(ART_DIR / "vocab.json")
    print(f"vocab size: {len(vocab)}")

    train_dl = DataLoader(PromptDataset(train_df, vocab, train=True), batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate)
    val_dl = DataLoader(PromptDataset(val_df, vocab), batch_size=256, collate_fn=collate)

    model = PromptNet(len(vocab))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    task_loss_fn, gap_loss_fn = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()

    history, best, bad_epochs = [], -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for x, y_task, y_gap in train_dl:
            opt.zero_grad()
            t_logits, g_logits, _ = model(x)
            loss = task_loss_fn(t_logits, y_task) + gap_loss_fn(g_logits, y_gap)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # exploding-gradient guard
            opt.step()
            run += loss.item() * len(x)
            n += len(x)
        val = evaluate(model, val_dl, task_loss_fn, gap_loss_fn)
        row = {"epoch": epoch, "train_loss": run / n, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(f"epoch {epoch:2d} | train {row['train_loss']:.4f} | val {val['loss']:.4f} | "
              f"task acc {val['task_acc']:.3f} | gap F1 {val['gap_f1']:.3f} | "
              f"{time.time() - t0:.0f}s")

        score = val["task_acc"] + val["gap_f1"]
        if score > best:
            best, bad_epochs = score, 0
            torch.save(model.state_dict(), ART_DIR / "model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("early stopping")
                break

    with open(ART_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_curves(history)
    print("saved best model to artifacts/model.pt")


def plot_curves(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[0].legend()
    ax[1].plot(ep, [h["val_task_acc"] for h in history], label="task accuracy")
    ax[1].plot(ep, [h["val_gap_f1"] for h in history], label="gap macro-F1")
    ax[1].set(title="Validation metrics", xlabel="epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(ART_DIR / "training_curves.png", dpi=130)


if __name__ == "__main__":
    main()
