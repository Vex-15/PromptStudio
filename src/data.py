"""Tokenizer, vocabulary, dataset and batching."""
import json
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import GAP_COLS, MAX_LEN, TASKS, WORD_DROPOUT

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase, keep words/numbers, drop punctuation."""
    return _TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, texts, min_freq: int = 2):
        counts = Counter(tok for t in texts for tok in tokenize(t))
        words = [w for w, c in counts.most_common() if c >= min_freq]
        return cls([PAD, UNK] + words)

    def encode(self, text: str) -> list[int]:
        ids = [self.stoi.get(tok, UNK_ID) for tok in tokenize(text)][:MAX_LEN]
        return ids or [UNK_ID]

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))


class PromptDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocab, train: bool = False):
        self.texts = df["text"].tolist()
        self.tasks = [TASKS.index(t) for t in df["task"]]
        self.gaps = df[GAP_COLS].values.astype("float32")
        self.vocab, self.train = vocab, train

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        ids = self.vocab.encode(self.texts[i])
        if self.train and WORD_DROPOUT > 0:  # randomly hide words -> regularisation
            ids = [UNK_ID if random.random() < WORD_DROPOUT else t for t in ids]
        return ids, self.tasks[i], self.gaps[i]


def pad_batch(seqs: list[list[int]]) -> torch.Tensor:
    longest = max(len(s) for s in seqs)
    out = torch.full((len(seqs), longest), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out


def collate(batch):
    ids, tasks, gaps = zip(*batch)
    return (
        pad_batch(list(ids)),
        torch.tensor(tasks, dtype=torch.long),
        torch.from_numpy(np.stack(gaps)).float(),
    )
