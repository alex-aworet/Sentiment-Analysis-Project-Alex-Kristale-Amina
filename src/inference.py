"""
Inference module for sentiment analysis using trained BERT model.
Allows users to pass in new text and get sentiment predictions.
"""

import argparse
import logging
import os
from typing import Dict, List, Union

import torch
from transformers import AutoTokenizer

from src.model import MODEL_NAME, create_model, get_device
from src.database import log_inference


class SentimentPredictor:
    """Class for making sentiment predictions on new text."""

    def __init__(
        self,
        model_path: str = "models/best_model_state.bin",
        model_name: str = MODEL_NAME,
        n_classes: int = 3,
        max_len: int = 128,
    ):
        """
        Initialize the sentiment predictor.

        Args:
            model_path: Path to the saved model weights
            model_name: Name of the pre-trained model
            n_classes: Number of sentiment classes (default: 3)
            max_len: Maximum sequence length for tokenization
        """
        self.device = get_device()
        self.max_len = max_len
        self.n_classes = n_classes

        self.sentiment_map = {
            0: "negative",
            1: "neutral",
            2: "positive",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Train the model first using model.py"
            )

        self.model = create_model(
            n_classes=n_classes,
            model_name=model_name,
        )

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(
            f"Model loaded successfully on device: {self.device}"
        )

    def predict(
        self,
        text: str,
    ) -> Dict[str, Union[str, float, Dict[str, float]]]:

        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(
                probabilities,
                dim=1,
            )

        predicted_class = predicted_class.item()
        confidence = confidence.item()

        probs_dict = {
            self.sentiment_map[i]: probabilities[0][i].item()
            for i in range(self.n_classes)
        }

        try:
            log_inference(
                text=text,
                sentiment=self.sentiment_map[predicted_class],
                confidence=confidence,
            )
        except Exception as exc:  # pragma: no cover
            logging.error(f"DB logging failed: {exc}")

        return {
            "text": text,
            "sentiment": self.sentiment_map[predicted_class],
            "confidence": confidence,
            "probabilities": probs_dict,
        }

    def predict_batch(
        self,
        texts: List[str],
    ) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
        return [self.predict(text) for text in texts]


def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis Inference"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_model_state.bin",
        help="Path to the trained model weights",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to analyze, runs once if provided",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("SENTIMENT ANALYSIS INFERENCE")
    print("=" * 50)

    try:
        predictor = SentimentPredictor(
            model_path=args.model_path
        )
    except Exception as exc:
        print(f"Error loading model: {exc}")
        return

    if args.text:
        result = predictor.predict(args.text)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")

        for sentiment, prob in result["probabilities"].items():
            print(f"  {sentiment.capitalize()}: {prob:.2%}")
        return

    print("\nType 'quit' or 'exit' to stop")
    print("-" * 50)

    while True:
        try:
            text = input("\nEnter text to analyze: ").strip()
        except EOFError:
            break

        if text.lower() in {"quit", "exit", "q"}:
            break

        if not text:
            print("Please enter some text.")
            continue

        result = predictor.predict(text)
        print(
            f"\nSentiment: {result['sentiment'].upper()} "
            f"({result['confidence']:.2%})"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
