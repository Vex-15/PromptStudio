"""Shared constants. Change things here, not scattered across files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"

# What the model predicts
TASKS = ["coding", "writing", "summarization", "explanation", "analysis", "brainstorming"]
GAPS = ["context", "audience", "constraints", "output_format", "purpose"]
GAP_COLS = [f"missing_{g}" for g in GAPS]  # 1 = this part is MISSING from the prompt

# Model / training
MAX_LEN = 64
EMB_DIM = 100
HIDDEN = 128
DROPOUT = 0.3
WORD_DROPOUT = 0.1   # randomly hide words during training -> better generalisation
BATCH_SIZE = 64
EPOCHS = 25
LR = 2e-3
PATIENCE = 5         # early stopping
THRESHOLD = 0.5      # probability above which a part is called "missing"
SEED = 42
TASK_CONF_MIN = 0.70   # below this task confidence, use the generic template
