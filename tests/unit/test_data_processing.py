"""
test_data_processing.py
"""

import pandas as pd
import builtins
import runpy
import torch
from transformers import AutoTokenizer
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
import src.data_processing as dp  # noqa: E402
from src.data_processing import (  # noqa: E402
    clean_text,
    add_sentiment_label,
    clean_dataset,
    split_data,
    tokenize_data,
    ReviewDataset
)

# ---------- CLEAN TEXT ----------


def test_clean_text_lowercase():
    assert clean_text("HELLO WORLD!") == "hello world"


def test_clean_text_removes_urls_mentions_symbols():
    text = "Visit http://test.com @user #wow!!!"
    cleaned = clean_text(text)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "!" not in cleaned


def test_clean_text_non_string_input():
    """Test clean_text with non-string inputs (covers line 23)"""
    assert clean_text(None) == ""
    assert clean_text(123) == ""
    assert clean_text([]) == ""


# ---------- ADD SENTIMENT LABEL ----------

def test_add_sentiment_label_values():
    df = pd.DataFrame({"score": [1, 3, 5]})
    df = add_sentiment_label(df)
    assert list(df["sentiment"]) == ["negative", "neutral", "positive"]


# ---------- CLEAN DATASET ----------

def test_clean_dataset_structure():
    df = pd.DataFrame({
        "content": ["Great app!!!", "Terrible update :(", None],
        "score": [5, 1, 3]
    })
    cleaned = clean_dataset(df)
    assert "content" in cleaned.columns
    assert "sentiment" in cleaned.columns
    assert not cleaned["content"].isnull().any()
    assert cleaned["content"].iloc[0] == "great app"
    assert len(cleaned) <= len(df)


# ---------- SPLIT DATA ----------

def test_split_data_stratified():
    """Test stratified split with larger dataset (covers line 74)"""
    # Create larger dataset to trigger stratification
    df = pd.DataFrame({
        "content": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
                    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t"],
        "sentiment": ["positive"] * 7 + ["neutral"] * 7 + ["negative"] * 6
    })
    train_df, val_df = split_data(df, test_size=0.2, random_state=0)
    assert "content" in train_df.columns
    assert "sentiment" in val_df.columns
    assert len(train_df) + len(val_df) == 20


def test_split_data_no_stratify():
    """Test non-stratified split with small dataset"""
    df = pd.DataFrame({
        "content": ["a", "b"],
        "sentiment": ["positive", "negative"]
    })
    train_df, val_df = split_data(df, test_size=0.5, random_state=0)
    assert len(train_df) + len(val_df) == 2


# ---------- TOKENIZATION ----------

def test_tokenize_data_returns_tensors():
    df = pd.DataFrame({"content": ["this app is good", "bad performance"]})
    tokens = tokenize_data(df, tokenizer_name="prajjwal1/bert-tiny")
    assert isinstance(tokens["input_ids"], torch.Tensor)
    assert tokens["input_ids"].shape[0] == 2


# ---------- REVIEW DATASET ----------

def test_review_dataset_output_shape():
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    texts = ["good app", "bad app"]
    labels = ["positive", "negative"]
    ds = ReviewDataset(texts, labels, tokenizer, max_len=16)
    sample = ds[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["input_ids"].shape[0] == 16
    assert isinstance(sample["labels"], torch.Tensor)


def test_review_dataset_len():
    """Test _len_ method of ReviewDataset (covers line 109)"""
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    texts = ["text1", "text2", "text3"]
    labels = ["positive", "neutral", "negative"]
    ds = ReviewDataset(texts, labels, tokenizer, max_len=16)
    assert len(ds) == 3


# ---------- MAIN EXECUTION BLOCK ----------

def test_main_block(monkeypatch, tmp_path):
    """Couvre le bloc if __name__ == '__main__' (lignes 137–155)."""

    # Mock dataset
    df_mock = pd.DataFrame({
        "content": ["good app", "bad update"],
        "score": [5, 1],
        "sentiment": ["positive", "negative"]
    })

    # Mock des fonctions utilisées dans le main
    def mock_load_file(path):
        return df_mock

    def mock_check_columns(df):
        return True

    # Appliquer les patches
    monkeypatch.setattr(dp, "load_file", mock_load_file)
    monkeypatch.setattr(dp, "check_columns", mock_check_columns)
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)
    runpy.run_module("src.data_processing", run_name="__main__")
