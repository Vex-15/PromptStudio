"""
BiLSTM + attention with two heads (multi-task learning):

    tokens -> Embedding -> BiLSTM -> Attention pooling -> +-> task head  (6-way softmax)
                                                          +-> gap head   (5 independent sigmoids)

The attention weights double as an explanation: they show which words the model
used to decide "task = coding" or "output format is missing".
"""
import torch
import torch.nn as nn

from .config import DROPOUT, EMB_DIM, GAPS, HIDDEN, TASKS
from .data import PAD_ID


class PromptNet(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(EMB_DIM, HIDDEN, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * HIDDEN, 1)
        self.drop = nn.Dropout(DROPOUT)
        self.task_head = nn.Linear(2 * HIDDEN, len(TASKS))
        self.gap_head = nn.Linear(2 * HIDDEN, len(GAPS))

    def forward(self, x):
        mask = x != PAD_ID                                   # (B, T)
        h, _ = self.lstm(self.drop(self.emb(x)))             # (B, T, 2H)
        scores = self.attn(h).squeeze(-1)                    # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))    # ignore padding
        weights = torch.softmax(scores, dim=1)               # (B, T), sums to 1
        ctx = (h * weights.unsqueeze(-1)).sum(dim=1)         # (B, 2H)
        ctx = self.drop(ctx)
        return self.task_head(ctx), self.gap_head(ctx), weights
