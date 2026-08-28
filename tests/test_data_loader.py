import pandas as pd

from src.data_loader import SupportDataset, _clean


def _fake_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instruction": [
                "  I want to cancel my order  ",
                "I want to cancel my order",  # duplicate after stripping
                "",  # empty, should be dropped
                "How do I reset my password?",
                "Where is my refund?",
            ],
            "category": ["ORDER", "ORDER", "ORDER", "ACCOUNT", "REFUND"],
            "intent": ["cancel_order", "cancel_order", "cancel_order", "reset_password", "track_refund"],
            "response": ["...", "...", "...", "...", "..."],
        }
    )


def test_clean_strips_dedupes_and_drops_empty():
    cleaned = _clean(_fake_raw_df())

    assert list(cleaned["instruction"]) == [
        "I want to cancel my order",
        "How do I reset my password?",
        "Where is my refund?",
    ]
    assert (cleaned["instruction"].str.strip() == cleaned["instruction"]).all()
    assert cleaned["id"].tolist() == [0, 1, 2]


def test_clean_is_idempotent_on_already_clean_input():
    once = _clean(_fake_raw_df())
    twice = _clean(once)
    assert once["instruction"].tolist() == twice["instruction"].tolist()


def test_support_dataset_keeps_text_and_labels_separate():
    cleaned = _clean(_fake_raw_df())
    texts = cleaned[["id", "instruction"]].rename(columns={"instruction": "text"})
    labels = cleaned[["id", "category", "intent"]]
    dataset = SupportDataset(texts=texts, labels=labels)

    assert list(dataset.texts.columns) == ["id", "text"]
    assert "category" not in dataset.texts.columns
    assert "intent" not in dataset.texts.columns
    assert list(dataset.labels.columns) == ["id", "category", "intent"]
    assert len(dataset.texts) == len(dataset.labels)
