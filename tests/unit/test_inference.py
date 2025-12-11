import types
import os
import sys

import pytest
import torch

import src.inference as inference


class DummyTokenizer:
    def encode_plus(
        self,
        text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    ):
        # Return minimal tensors with a .to() method
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class DummyModel:
    def __init__(self, *args, **kwargs):
        self.state_dict = {}

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

    def eval(self):
        # No-op for testing
        return self

    def __call__(self, input_ids=None, attention_mask=None):
        # Deterministic "logits" for 3 classes
        logits = torch.tensor(
            [[-1.0, 0.0, 1.0]]
        )  # class 2 (positive) is max
        return types.SimpleNamespace(logits=logits)


class DummyAutoTokenizer:
    @classmethod
    def from_pretrained(cls, model_name):
        return DummyTokenizer()


@pytest.fixture
def predictor_with_mocks(monkeypatch):
    """
    Fixture that prepares a SentimentPredictor instance with
    all external dependencies mocked.
    """
    # Always pretend the model file exists
    monkeypatch.setattr(os.path, "exists", lambda path: True)

    # Device & model creation
    monkeypatch.setattr(inference, "get_device", lambda: "cpu")
    monkeypatch.setattr(
        inference,
        "create_model",
        lambda n_classes, model_name: DummyModel(),
    )

    # Tokenizer
    monkeypatch.setattr(inference, "AutoTokenizer", DummyAutoTokenizer)

    # torch.load -> return dummy state dict
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})

    # Capture calls to log_inference
    log_calls = []

    def fake_log_inference(text, sentiment, confidence):
        log_calls.append(
            {
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence,
            }
        )

    monkeypatch.setattr(inference, "log_inference", fake_log_inference)

    predictor = inference.SentimentPredictor(
        model_path="models/best_model_state.bin",
        model_name="dummy-model",
        n_classes=3,
        max_len=128,
    )
    return predictor, log_calls


def test_init_raises_if_model_missing(monkeypatch):
    """SentimentPredictor should raise if the model file does not exist."""
    # Make os.path.exists always return False so FileNotFoundError is triggered
    monkeypatch.setattr(os.path, "exists", lambda path: False)

    # Prevent AutoTokenizer.from_pretrained from reaching the network
    monkeypatch.setattr(inference, "AutoTokenizer", DummyAutoTokenizer)

    # (Optional safety) Avoid touching real device/model
    monkeypatch.setattr(inference, "get_device", lambda: "cpu")
    monkeypatch.setattr(
        inference,
        "create_model",
        lambda *args, **kwargs: DummyModel(),
    )

    with pytest.raises(FileNotFoundError):
        inference.SentimentPredictor(
            model_path="does_not_exist.bin"
        )


def test_predict_returns_expected_structure_and_logs(predictor_with_mocks):
    predictor, log_calls = predictor_with_mocks

    text = "This is amazing!"
    result = predictor.predict(text)

    # Structure
    assert set(result.keys()) == {
        "text",
        "sentiment",
        "confidence",
        "probabilities",
    }
    assert result["text"] == text
    assert isinstance(result["sentiment"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["probabilities"], dict)
    assert set(result["probabilities"].keys()) == {
        "negative",
        "neutral",
        "positive",
    }

    # From DummyModel logits, class 2 ("positive") should be chosen
    assert result["sentiment"] == "positive"

    # Probabilities should sum (approximately) to 1
    prob_sum = sum(result["probabilities"].values())
    assert pytest.approx(prob_sum, rel=1e-5) == 1.0

    # API should have logged the inference once
    assert len(log_calls) == 1
    assert log_calls[0]["text"] == text
    assert log_calls[0]["sentiment"] == "positive"
    assert isinstance(log_calls[0]["confidence"], float)


def test_predict_batch_uses_predict(predictor_with_mocks, monkeypatch):
    predictor, _ = predictor_with_mocks

    calls = []

    def fake_predict(self, text):
        calls.append(text)
        return {
            "text": text,
            "sentiment": "neutral",
            "confidence": 0.5,
            "probabilities": {},
        }

    # Patch the method on the class so predict_batch uses our fake
    monkeypatch.setattr(
        inference.SentimentPredictor,
        "predict",
        fake_predict,
        raising=False,
    )

    texts = ["a", "b", "c"]
    results = predictor.predict_batch(texts)

    assert len(results) == len(texts)
    assert calls == texts
    for t, r in zip(texts, results):
        assert r["text"] == t
        assert r["sentiment"] == "neutral"


def test_main_with_text_argument(monkeypatch, capsys):
    """
    Smoke test for main() when --text is provided.
    Ensures it runs end-to-end without crashing and prints expected parts.
    """
    # Reuse the mocking strategy above, but patch within the module used by main()
    monkeypatch.setattr(os.path, "exists", lambda path: True)
    monkeypatch.setattr(inference, "get_device", lambda: "cpu")
    monkeypatch.setattr(
        inference,
        "create_model",
        lambda n_classes, model_name: DummyModel(),
    )
    monkeypatch.setattr(inference, "AutoTokenizer", DummyAutoTokenizer)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(inference, "log_inference", lambda **kwargs: None)

    # Fake CLI args
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model_path",
            "models/best_model_state.bin",
            "--text",
            "hello world",
        ],
    )

    # Run main
    inference.main()

    captured = capsys.readouterr()
    out = captured.out

    # Basic sanity checks on output text
    assert "SENTIMENT ANALYSIS INFERENCE" in out
    assert "Text: hello world" in out
    assert "Sentiment:" in out
    assert "Confidence:" in out
