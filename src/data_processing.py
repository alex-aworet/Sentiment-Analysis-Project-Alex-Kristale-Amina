"""
data_processing.py
"""

from data_extraction import load_file, check_columns
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==========================
# TEXT CLEANING
# ==========================

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_sentiment_label(df: pd.DataFrame) -> pd.DataFrame:
    def label_from_score(score):
        if score <= 2:
            return "negative"
        elif score == 3:
            return "neutral"
        else:
            return "positive"
    df["sentiment"] = df["score"].apply(label_from_score)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_sentiment_label(df)
    df = df[["content", "sentiment"]]
    df["content"] = df["content"].fillna("").astype(str).apply(clean_text)
    df.drop_duplicates(subset=["content"], inplace=True)
    return df


# ==========================
# SPLIT & TOKENIZATION
# ==========================

def split_data(
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42):
    # Check if stratification is possible
    # Stratification requires: test_size * len(df) >= number of classes
    min_samples_per_class = df["sentiment"].value_counts().min()
    n_classes = df["sentiment"].nunique()
    n_test = int(test_size * len(df))

    # If test set would be smaller than number of classes, don't stratify
    if n_test < n_classes or min_samples_per_class < 2:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state)
    else:
        return train_test_split(
            df,
            test_size=test_size,
            stratify=df["sentiment"],
            random_state=random_state)


def tokenize_data(
        df: pd.DataFrame,
        tokenizer_name: str = "bert-base-cased",
        max_len: int = 160):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encodings = tokenizer(
        list(df["content"]),
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return encodings


# ==========================
# TORCH DATASET
# ==========================

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {"negative": 0, "neutral": 1, "positive": 2}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.label_map[self.labels[idx]]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# ==========================
# MAIN EXECUTION PIPELINE
# ==========================

if __name__ == "__main__":
    path = "data/dataset.csv"
    print(">>> Loading dataset from data_extraction module...")
    df = load_file(path)
    if df is not None and check_columns(df):
        print(">>> Cleaning and preparing dataset...")
        df = clean_dataset(df)
        train_df, val_df = split_data(df)

        print(">>> Tokenizing training data...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        train_dataset = ReviewDataset(
            train_df["content"].to_numpy(),
            train_df["sentiment"].to_numpy(),
            tokenizer,
            max_len=160
        )
        print(f" Ready: {len(train_dataset)} training samples.")
    else:
        print(" Dataset invalid or missing columns.")
