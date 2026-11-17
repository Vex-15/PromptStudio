"""
Classical baseline: TF-IDF (1-2 grams) + logistic regression.
The deep model has to beat this to justify itself.

Run:  python -m src.baseline
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from .config import DATA_DIR, GAP_COLS, TASKS


class Baseline:
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.task_clf = LogisticRegression(max_iter=1000, C=5)
        self.gap_clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=5))

    def fit(self, df: pd.DataFrame):
        X = self.vec.fit_transform(df["text"])
        self.task_clf.fit(X, df["task"])
        self.gap_clf.fit(X, df[GAP_COLS].values)
        return self

    def predict(self, texts):
        """Returns (task_names, gap_predictions as 0/1 matrix)."""
        X = self.vec.transform(texts)
        return self.task_clf.predict(X), self.gap_clf.predict(X)


def train_baseline() -> Baseline:
    return Baseline().fit(pd.read_csv(DATA_DIR / "train.csv"))


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, f1_score

    base = train_baseline()
    for name in ["test", "manual_test"]:
        df = pd.read_csv(DATA_DIR / f"{name}.csv")
        task_pred, gap_pred = base.predict(df["text"])
        print(f"{name:12s} task acc {accuracy_score(df['task'], task_pred):.3f} | "
              f"gap macro-F1 {f1_score(df[GAP_COLS].values, gap_pred, average='macro', zero_division=0):.3f}")
