"""Fast sanity tests (no trained model needed except where noted).  Run: pytest -q"""
import pandas as pd
import pytest

from src.config import DATA_DIR, GAP_COLS, TASKS
from src.data import Vocab, tokenize
from src.slots import evidence_present, extract_slots


def test_tokenizer_lowercases_and_drops_punctuation():
    assert tokenize("Write a Python function, please!") == ["write", "a", "python", "function", "please"]


def test_vocab_handles_unknown_words():
    v = Vocab.build(["write a function", "write a function"], min_freq=1)
    ids = v.encode("write a zzzunknown")
    assert ids[-1] == 1  # UNK id


def test_slots_find_language_and_limits():
    s = extract_slots("Write a Python script, under 100 words, as a table")
    assert s["language"].lower() == "python" and s["has_limit"] and s["has_format_words"]
    assert evidence_present("keep it under 50 words") == {"constraints"}


def test_dataset_files_are_valid():
    for split in ["train", "val", "test", "manual_test"]:
        df = pd.read_csv(DATA_DIR / f"{split}.csv")
        assert set(df["task"]) <= set(TASKS)
        assert set(GAP_COLS) <= set(df.columns)
        assert df[GAP_COLS].isin([0, 1]).all().all()


def test_no_train_test_leakage():
    train = set(pd.read_csv(DATA_DIR / "train.csv")["text"])
    test = set(pd.read_csv(DATA_DIR / "test.csv")["text"])
    assert not train & test


def test_fusion_keeps_the_prompt_verbatim_and_reports_both_engines():
    from src.combined import analyze_prompt

    prompt = "write python code for student marks"
    r = analyze_prompt(prompt)
    assert r["combined"]["text"].startswith(f"TASK:\n{prompt}")
    assert r["combined"]["addressed"]
    assert r["combined"]["engine"] == "rubric"
    assert 0 <= r["score"]["before"] <= 100 and 0 <= r["score"]["after"] <= 100
    assert set(r["neural"]["gaps"]) == {"Context", "Audience", "Constraints",
                                          "Output format", "Purpose"}


def test_writing_prompt_never_gets_code_requirements():
    from src.combined import analyze_prompt

    r = analyze_prompt("write something about ai in education")
    text = r["combined"]["text"].lower()
    assert r["combined"]["engine"] == "neural"
    assert "function signature" not in text
    assert "usage example" not in text


def test_preference_tally_ignores_unknown_choices():
    from src.preferences import tally

    t = tally([{"choice": "merged"}, {"choice": "merged"},
               {"choice": "original"}, {"choice": "maybe"}])
    assert t == {"original": 1, "merged": 2, "total": 3, "merged_share": 66.7}


def test_preference_record_appends_jsonl(tmp_path):
    from src.preferences import load, record, summary

    path = tmp_path / "votes.jsonl"
    record({"prompt": "a", "choice": "merged"}, path=path)
    record({"prompt": "b", "choice": "original"}, path=path)
    assert len(load(path)) == 2
    assert summary(path)["tally"]["total"] == 2
    with pytest.raises(ValueError):
        record({"prompt": "c", "choice": "maybe"}, path=path)
