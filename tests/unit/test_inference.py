# flake8: noqa
"""
test_model.py - Unit tests for model.py
"""

import os
import sys
import tempfile
from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

# Add project root to PYTHONPATH
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
)

from src.data_processing import ReviewDataset
from src.model import (
    MODEL_NAME,
    create_model,
    create_optimizer_and_scheduler,
    eval_model,
    get_device,
    train_epoch,
    train_model,
)


# ---------- GET DEVICE ----------


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cuda", "cpu"]


# ---------- CREATE MODEL ----------


def test_create_model_returns_model():
    model = create_model(n_classes=3)
    assert model is not None
    assert model.config.num_labels == 3


def test_create_model_custom_dropout():
    model = create_model(n_classes=3, dropout=0.5)
    assert model.config.hidden_dropout_prob == 0.5
    assert model.config.attention_probs_dropout_prob == 0.5


def test_create_model_custom_model_name():
    model = create_model(
        n_classes=3,
        model_name="prajjwal1/bert-tiny",
        dropout=0.1,
    )
    assert model.config.num_labels == 3


def test_create_model_on_device():
    model = create_model(n_classes=3)
    device = get_device()
    assert next(model.parameters()).device.type == device.type


# ---------- OPTIMIZER & SCHEDULER ----------


def test_create_optimizer_and_scheduler():
    model = create_model(n_classes=3)

    dummy_data = TensorDataset(
        torch.randint(0, 100, (10, 16)),
        torch.randint(0, 2, (10, 16)),
        torch.randint(0, 3, (10,)),
    )
    loader = DataLoader(dummy_data, batch_size=2)

    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        train_data_loader=loader,
        epochs=2,
        learning_rate=1e-5,
    )

    assert optimizer is not None
    assert scheduler is not None
    assert optimizer.param_groups[0]["lr"] == 1e-5


# ---------- TRAIN / EVAL ----------


def test_train_epoch():
    device = get_device()
    model = create_model(n_classes=3)
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    dataset = ReviewDataset(
        ["good", "bad", "ok"],
        ["positive", "negative", "neutral"],
        tokenizer,
        max_len=16,
    )

    loader = DataLoader(dataset, batch_size=2)

    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        train_data_loader=loader,
        epochs=1,
        learning_rate=2e-5,
    )

    acc, loss = train_epoch(
        model,
        loader,
        optimizer,
        device,
        scheduler,
        len(dataset),
    )

    assert 0 <= float(acc) <= 1
    assert loss >= 0


def test_eval_model():
    device = get_device()
    model = create_model(n_classes=3)
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    dataset = ReviewDataset(
        ["good", "bad"],
        ["positive", "negative"],
        tokenizer,
        max_len=16,
    )
    loader = DataLoader(dataset, batch_size=2)

    acc, loss = eval_model(model, loader, device, len(dataset))

    assert 0 <= float(acc) <= 1
    assert loss >= 0


# ---------- TRAIN MODEL ----------


def test_train_model_saves_best_model():
    model = create_model(n_classes=3)
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    dataset = ReviewDataset(
        ["good", "bad", "ok"],
        ["positive", "negative", "neutral"],
        tokenizer,
        max_len=16,
    )

    loader = DataLoader(dataset, batch_size=2)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        path = tmp.name

    try:
        history = train_model(
            model,
            loader,
            loader,
            len(dataset),
            len(dataset),
            epochs=2,
            model_save_path=path,
            verbose=False,
        )

        assert len(history["train_loss"]) == 2
        assert os.path.exists(path)

    finally:
        os.remove(path)


# ---------- MODEL NAME ----------


def test_model_name_constant():
    assert MODEL_NAME == "prajjwal1/bert-tiny"
    assert isinstance(MODEL_NAME, str)


# ---------- MAIN ----------


def test_main_function_with_mock_data():
    mock_df = pd.DataFrame(
        {
            "content": ["good app"] * 10 + ["bad app"] * 10 + ["ok app"] * 10,
            "score": [5] * 10 + [1] * 10 + [3] * 10,
        }
    )

    with (
        patch("src.model.load_file", return_value=mock_df),
        patch("src.model.check_columns", return_value=True),
        patch("src.model.split_data") as mock_split,
        patch("builtins.print"),
    ):
        cleaned_df = mock_df.copy()
        cleaned_df["sentiment"] = (
            ["positive"] * 10
            + ["negative"] * 10
            + ["neutral"] * 10
        )

        mock_split.return_value = (
            cleaned_df.iloc[:20],
            cleaned_df.iloc[20:],
        )

        from src.model import main

        main()
