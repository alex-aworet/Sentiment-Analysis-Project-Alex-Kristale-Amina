# flake8: noqa
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, List
import argparse
import sys
import os

# Import data processing functions and dataset class
from data_processing import (
    clean_dataset,
    split_data,
    ReviewDataset
)

from data_processing import load_file, check_columns

from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set the model name
MODEL_NAME = 'prajjwal1/bert-tiny'


def get_device() -> torch.device:
    """Get GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(
    n_classes: int,
    model_name: str = MODEL_NAME,
    dropout: float = 0.1
):
    device = get_device()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=n_classes
    )
    model.config.hidden_dropout_prob = dropout
    model.config.attention_probs_dropout_prob = dropout
    model.dropout.p = dropout

    return model.to(device)


def create_optimizer_and_scheduler(
    model: nn.Module,
    train_data_loader: DataLoader,
    epochs: int,
    learning_rate: float = 2e-5
) -> Tuple[optim.AdamW, any]:
    """
    Create optimizer and learning rate scheduler.
    """
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_data_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: any,
    n_examples: int
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    """
    model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def run_evaluation(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    return_metrics: bool = False,
    output_path: str | None = None,
    verbose: bool = True
):
    """
    Can return only (acc, loss) OR full metrics, OR write a report.

    Used both during training and from CLI.
    """

    model.eval()
    losses = []
    correct_predictions = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    n_examples = len(all_labels)
    acc_tensor = correct_predictions.double() / n_examples
    mean_loss = np.mean(losses)

    # --------------------------
    # BASIC MODE (training)
    # --------------------------
    if not return_metrics:
        return acc_tensor, mean_loss

    # --------------------------
    # FULL METRICS MODE (CLI)
    # --------------------------
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    target_names = ["negative", "neutral", "positive"]
    clf_report = classification_report(
        all_labels, all_preds, target_names=target_names, zero_division=0
    )

    # build report
    report = f"""
Classification Report:
{clf_report}
"""

    if verbose:
        print(report)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nSaved evaluation report → {output_path}")

    # ---- Threshold check for CI ----
    MINIMUM_ACCURACY = 0.60
    acc_value = float(acc_tensor)

    if acc_value < MINIMUM_ACCURACY:
        msg = (
            f"Warning: Model accuracy {acc_value:.4f} is below the minimum "
            f"threshold of {MINIMUM_ACCURACY:.4f}."
        )
        print(msg)
        # Non-zero exit code → GitHub Actions step fails
        sys.exit(1)

    return {
        "acc": acc_value,
        "loss": float(mean_loss),
        "report": clf_report
    }



def train_model(
    model: nn.Module,
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    n_train_examples: int,
    n_val_examples: int,
    epochs: int,
    learning_rate: float = 2e-5,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1,
    model_save_path: str = 'models/best_model_state.bin',
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train for several epochs, saving the best model.

    Args:
        model: The model to train
        train_data_loader: DataLoader for training data
        val_data_loader: DataLoader for validation data
        n_train_examples: Number of training examples
        epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        use_amp: Whether to use automatic mixed precision
            (currently ignored, for future use)
        gradient_accumulation_steps: Number of steps to accumulate
            gradients (currently ignored, for future use)
        model_save_path: Path to save the best model
        verbose: Whether to print progress
    """
    device = get_device()

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        train_data_loader,
        epochs,
        learning_rate
    )

    history = defaultdict(list)
    best_accuracy = 0
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(epochs):
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            optimizer,
            device,
            scheduler,
            n_train_examples
        )

        val_acc, val_loss = run_evaluation(
            model,
            val_data_loader,
            device,
            return_metrics=False
        )


        if verbose:
            print(f"Train loss {train_loss:.4f}, acc {train_acc:.4f}")
            print(f"Val   loss {val_loss:.4f}, acc {val_acc:.4f}")
            print()

        train_acc_val = (train_acc.item() if torch.is_tensor(train_acc)
                         else train_acc)
        history['train_acc'].append(train_acc_val)
        history['train_loss'].append(train_loss)
        val_acc_val = (val_acc.item() if torch.is_tensor(val_acc)
                       else val_acc)
        history['val_acc'].append(val_acc_val)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), model_save_path)
            best_accuracy = val_acc
            if verbose:
                print(f"Best model saved with accuracy: {best_accuracy:.4f}")

    return dict(history)


def main():

    """Main function to demonstrate the complete pipeline."""

    # ==========================
    # 1. LOAD AND PROCESS DATA
    # ==========================
    print("=" * 50)
    print("STEP 1: Loading and processing data")
    print("=" * 50)

    # Load the dataset
    path = "data/dataset.csv"
    print(f"Loading dataset from {path}...")
    df = load_file(path)

    if df is None or not check_columns(df):
        print("Error: Dataset invalid or missing columns.")
        return

    print(f"Dataset loaded: {len(df)} rows")

    # Clean the dataset
    # (removes duplicates, adds sentiment labels, cleans text)
    print("\nCleaning dataset...")
    df_clean = clean_dataset(df)
    print(f"Cleaned dataset: {len(df_clean)} rows")
    print(f"Sentiment distribution:\n{df_clean['sentiment'].value_counts()}")

    # Split into train and validation sets
    print("\nSplitting data into train/validation...")
    train_df, val_df = split_data(df_clean, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")

    # ==========================
    # 2. CREATE TORCH DATASETS
    # ==========================
    print("\n" + "=" * 50)
    print("STEP 2: Creating PyTorch datasets")
    print("=" * 50)

    # Initialize tokenizer
    tokenizer_name = "prajjwal1/bert-tiny"
    max_len = 128
    print(f"Using tokenizer: {tokenizer_name}")
    print(f"Max sequence length: {max_len}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Create training dataset
    print("\nCreating training dataset...")
    train_dataset = ReviewDataset(
        texts=train_df["content"].to_numpy(),
        labels=train_df["sentiment"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    print(f"Training dataset created: {len(train_dataset)} samples")

    # Create validation dataset
    print("Creating validation dataset...")
    val_dataset = ReviewDataset(
        texts=val_df["content"].to_numpy(),
        labels=val_df["sentiment"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    print(f"Validation dataset created: {len(val_dataset)} samples")

    # Example: inspect a sample
    print("\nExample sample from training dataset:")
    sample = train_dataset[0]
    print(f"  Text (truncated): {sample['text'][:100]}...")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Attention mask shape: {sample['attention_mask'].shape}")
    print(f"  Label: {sample['labels']}")

    # ==========================
    # 3. CREATE DATA LOADERS
    # ==========================
    print("\n" + "=" * 50)
    print("STEP 3: Creating data loaders")
    print("=" * 50)

    batch_size = 16
    print(f"Batch size: {batch_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for Windows, can increase on Linux/Mac
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # ==========================
    # 4. CREATE AND TRAIN MODEL
    # ==========================
    print("\n" + "=" * 50)
    print("STEP 4: Creating and training model")
    print("=" * 50)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Create model
    n_classes = 3  # negative, neutral, positive
    print(f"\nCreating model with {n_classes} classes...")
    model = create_model(n_classes=n_classes, dropout=0.3)
    print("Model created successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training configuration
    epochs = 3  # Use more epochs in production (e.g., 10)
    learning_rate = 2e-5
    use_amp = torch.cuda.is_available()  # Use mixed precision on GPU
    gradient_accumulation_steps = 1

    print("\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Mixed precision training: {use_amp}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")

    # Train the model
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    history = train_model(
        model=model,
        train_data_loader=train_loader,
        val_data_loader=val_loader,
        n_train_examples=len(train_dataset),
        n_val_examples=len(val_dataset),
        epochs=epochs,
        learning_rate=learning_rate,
        use_amp=use_amp,
        gradient_accumulation_steps=gradient_accumulation_steps,
        model_save_path='models/best_model_state.bin'
    )

    # ==========================
    # 5. DISPLAY RESULTS
    # ==========================
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)

    print("\nTraining history:")
    for epoch in range(len(history['train_loss'])):
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {history['train_loss'][epoch]:.4f}, "
              f"Train Acc: {history['train_acc'][epoch]:.4f}")
        print(f"  Val Loss: {history['val_loss'][epoch]:.4f}, "
              f"Val Acc: {history['val_acc'][epoch]:.4f}")

    best_val_acc = max(history['val_acc'])
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print("Best model saved to: models/best_model_state.bin")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation instead of training."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation report, e.g. results/evaluation_report.txt"
    )
    args = parser.parse_args()
    if args.evaluate:
        # load dataset
        df = load_file("data/dataset.csv")
        df = clean_dataset(df)
        _, val_df = split_data(df, test_size=0.2, random_state=42)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # IMPORTANT: use .to_numpy() so indexing is 0..N-1
        val_dataset = ReviewDataset(
            texts=val_df["content"].to_numpy(),
            labels=val_df["sentiment"].to_numpy(),
            tokenizer=tokenizer,
            max_len=128
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False
        )

        # load model
        model = create_model(3)
        model.load_state_dict(torch.load("models/best_model_state.bin"))

        run_evaluation(
            model=model,
            data_loader=val_loader,
            device=get_device(),
            return_metrics=True,
            output_path=args.output
        )
    else:
        main()
