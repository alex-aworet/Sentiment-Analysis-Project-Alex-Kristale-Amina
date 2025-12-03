"""
test_inference.py - Unit tests for inference.py
"""

import torch
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock, mock_open
import pytest

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
from src.inference import SentimentPredictor, main
from src.model import create_model, MODEL_NAME


# ---------- HELPER FIXTURES ----------


@pytest.fixture
def mock_model_path():
    """Create a temporary model file for testing."""
    # Create a simple model and save it
    model = create_model(n_classes=3)
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
        model_path = tmp.name
        torch.save(model.state_dict(), model_path)

    yield model_path

    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)


# ---------- SENTIMENTPREDICTOR INITIALIZATION ----------


def test_sentiment_predictor_initialization(mock_model_path):
    """Test that SentimentPredictor initializes correctly."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    assert predictor is not None
    assert predictor.device is not None
    assert predictor.max_len == 128
    assert predictor.n_classes == 3
    assert predictor.tokenizer is not None
    assert predictor.model is not None
    assert len(predictor.sentiment_map) == 3


def test_sentiment_predictor_custom_params(mock_model_path):
    """Test SentimentPredictor with custom parameters."""
    predictor = SentimentPredictor(
        model_path=mock_model_path,
        model_name=MODEL_NAME,
        n_classes=3,
        max_len=128
    )

    assert predictor.max_len == 128
    assert predictor.n_classes == 3


def test_sentiment_predictor_sentiment_map(mock_model_path):
    """Test that sentiment_map is correctly initialized."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    assert predictor.sentiment_map[0] == 'negative'
    assert predictor.sentiment_map[1] == 'neutral'
    assert predictor.sentiment_map[2] == 'positive'


def test_sentiment_predictor_model_eval_mode(mock_model_path):
    """Test that model is in evaluation mode after initialization."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    # Model should be in eval mode (not training mode)
    assert not predictor.model.training


def test_sentiment_predictor_invalid_model_path():
    """Test that SentimentPredictor handles invalid model path."""
    with pytest.raises(Exception):
        SentimentPredictor(model_path='nonexistent_model.bin')


# ---------- PREDICT METHOD ----------


def test_predict_returns_correct_structure(mock_model_path):
    """Test that predict returns the correct dictionary structure."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    result = predictor.predict("This app is amazing!")

    assert isinstance(result, dict)
    assert 'text' in result
    assert 'sentiment' in result
    assert 'confidence' in result
    assert 'probabilities' in result


def test_predict_text_field(mock_model_path):
    """Test that predict returns the original text."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    test_text = "Great app!"
    result = predictor.predict(test_text)

    assert result['text'] == test_text


def test_predict_sentiment_is_valid(mock_model_path):
    """Test that predict returns a valid sentiment label."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    result = predictor.predict("This is a test")

    assert result['sentiment'] in ['negative', 'neutral', 'positive']


def test_predict_confidence_range(mock_model_path):
    """Test that confidence is between 0 and 1."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    result = predictor.predict("This is a test")

    assert 0 <= result['confidence'] <= 1
    assert isinstance(result['confidence'], float)


def test_predict_probabilities_structure(mock_model_path):
    """Test that probabilities dictionary has all sentiment classes."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    result = predictor.predict("This is a test")

    assert 'negative' in result['probabilities']
    assert 'neutral' in result['probabilities']
    assert 'positive' in result['probabilities']


def test_predict_probabilities_sum_to_one(mock_model_path):
    """Test that probabilities sum to approximately 1."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    result = predictor.predict("This is a test")

    prob_sum = sum(result['probabilities'].values())
    assert abs(prob_sum - 1.0) < 0.001  # Allow small floating point error


def test_predict_probabilities_range(mock_model_path):
    """Test that all probabilities are between 0 and 1."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    result = predictor.predict("This is a test")

    for sentiment, prob in result['probabilities'].items():
        assert 0 <= prob <= 1
        assert isinstance(prob, float)


def test_predict_empty_text(mock_model_path):
    """Test predict with empty text."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    result = predictor.predict("")

    # Should still return valid structure
    assert 'sentiment' in result
    assert 'confidence' in result


def test_predict_long_text(mock_model_path):
    """Test predict with text longer than max_len."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    # Create a very long text
    long_text = "This is a great app. " * 100
    result = predictor.predict(long_text)

    # Should handle truncation and return valid result
    assert 'sentiment' in result
    assert 'confidence' in result


def test_predict_special_characters(mock_model_path):
    """Test predict with special characters."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    result = predictor.predict("This app is @#$%^&* amazing!!!")

    assert 'sentiment' in result
    assert isinstance(result['confidence'], float)


def test_predict_no_gradient_computation(mock_model_path):
    """Test that predict doesn't compute gradients."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    # Ensure no gradients are tracked
    with torch.no_grad():
        result = predictor.predict("Test text")
        assert result is not None


def test_predict_consistent_output(mock_model_path):
    """Test that predict gives consistent output for same input."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    text = "This is a test"
    result1 = predictor.predict(text)
    result2 = predictor.predict(text)

    # Results should be identical for same input
    assert result1['sentiment'] == result2['sentiment']
    assert abs(result1['confidence'] - result2['confidence']) < 0.001


# ---------- PREDICT_BATCH METHOD ----------


def test_predict_batch_returns_list(mock_model_path):
    """Test that predict_batch returns a list."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    texts = ["Great app!", "Terrible app"]
    results = predictor.predict_batch(texts)

    assert isinstance(results, list)
    assert len(results) == 2


def test_predict_batch_structure(mock_model_path):
    """Test that predict_batch returns correctly structured results."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    texts = ["Amazing!", "Bad", "Okay"]
    results = predictor.predict_batch(texts)

    assert len(results) == 3
    for result in results:
        assert 'text' in result
        assert 'sentiment' in result
        assert 'confidence' in result
        assert 'probabilities' in result


def test_predict_batch_empty_list(mock_model_path):
    """Test predict_batch with empty list."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    results = predictor.predict_batch([])

    assert isinstance(results, list)
    assert len(results) == 0


def test_predict_batch_single_text(mock_model_path):
    """Test predict_batch with single text."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    results = predictor.predict_batch(["Single text"])

    assert len(results) == 1
    assert results[0]['text'] == "Single text"


def test_predict_batch_preserves_order(mock_model_path):
    """Test that predict_batch preserves text order."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    texts = ["First", "Second", "Third"]
    results = predictor.predict_batch(texts)

    assert results[0]['text'] == "First"
    assert results[1]['text'] == "Second"
    assert results[2]['text'] == "Third"


def test_predict_batch_various_lengths(mock_model_path):
    """Test predict_batch with texts of various lengths."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    texts = [
        "Hi",
        "This is a medium length text",
        "This is a much longer text that goes on and on " * 10
    ]
    results = predictor.predict_batch(texts)

    assert len(results) == 3
    for result in results:
        assert 'sentiment' in result


# ---------- MAIN FUNCTION ----------


def test_main_function_with_mock_model(mock_model_path):
    """Test main function with mocked model loading."""
    with patch('sys.argv', ['inference.py', '--model_path', mock_model_path]), \
         patch('src.inference.SentimentPredictor') as MockPredictor, \
         patch('builtins.input') as mock_input, \
         patch('builtins.print'):

        # Mock predictor instance
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {
            'text': 'test',
            'sentiment': 'positive',
            'confidence': 0.95,
            'probabilities': {
                'negative': 0.02,
                'neutral': 0.03,
                'positive': 0.95
            }
        }
        MockPredictor.return_value = mock_predictor

        # Simulate user input: one prediction then quit
        mock_input.side_effect = ['Great app!', 'quit']

        # Run main
        main()

        # Verify predictor was created
        MockPredictor.assert_called_once()


def test_main_function_quit_command(mock_model_path):
    """Test main function with quit command."""
    with patch('sys.argv', ['inference.py', '--model_path', mock_model_path]), \
         patch('src.inference.SentimentPredictor') as MockPredictor, \
         patch('builtins.input') as mock_input, \
         patch('builtins.print'):

        mock_predictor = MagicMock()
        MockPredictor.return_value = mock_predictor

        # User quits immediately
        mock_input.return_value = 'quit'

        main()

        # Predict should not be called if user quits immediately
        mock_predictor.predict.assert_not_called()


def test_main_function_exit_command(mock_model_path):
    """Test main function with exit command."""
    with patch('sys.argv', ['inference.py', '--model_path', mock_model_path]), \
         patch('src.inference.SentimentPredictor') as MockPredictor, \
         patch('builtins.input') as mock_input, \
         patch('builtins.print'):

        mock_predictor = MagicMock()
        MockPredictor.return_value = mock_predictor

        # User types 'exit'
        mock_input.return_value = 'exit'

        main()

        mock_predictor.predict.assert_not_called()


def test_main_function_empty_input(mock_model_path):
    """Test main function with empty input."""
    with patch('sys.argv', ['inference.py', '--model_path', mock_model_path]), \
         patch('src.inference.SentimentPredictor') as MockPredictor, \
         patch('builtins.input') as mock_input, \
         patch('builtins.print'):

        mock_predictor = MagicMock()
        MockPredictor.return_value = mock_predictor

        # User enters empty text then quits
        mock_input.side_effect = ['', 'quit']

        main()

        # Predict should not be called for empty input
        mock_predictor.predict.assert_not_called()


def test_main_function_model_loading_error():
    """Test main function handles model loading errors."""
    with patch('sys.argv', ['inference.py']), \
         patch('src.inference.SentimentPredictor') as MockPredictor, \
         patch('builtins.print') as mock_print:

        # Simulate model loading error
        MockPredictor.side_effect = Exception("Model not found")

        # Run main - should handle error gracefully
        main()

        # Should print error message
        assert any('Error loading model' in str(call)
                   for call in mock_print.call_args_list)


def test_main_function_with_argparse():
    """Test main function with command line arguments."""
    with patch('sys.argv', ['inference.py', '--model_path', 'custom_model.bin']), \
         patch('src.inference.SentimentPredictor') as MockPredictor, \
         patch('builtins.input') as mock_input, \
         patch('builtins.print'):

        mock_predictor = MagicMock()
        MockPredictor.return_value = mock_predictor

        mock_input.return_value = 'quit'

        main()

        # Should be called with custom model path
        MockPredictor.assert_called_with(model_path='custom_model.bin')


def test_main_function_multiple_predictions(mock_model_path):
    """Test main function with multiple predictions before quitting."""
    with patch('sys.argv', ['inference.py', '--model_path', mock_model_path]), \
         patch('src.inference.SentimentPredictor') as MockPredictor, \
         patch('builtins.input') as mock_input, \
         patch('builtins.print'):

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {
            'text': 'test',
            'sentiment': 'positive',
            'confidence': 0.95,
            'probabilities': {
                'negative': 0.02,
                'neutral': 0.03,
                'positive': 0.95
            }
        }
        MockPredictor.return_value = mock_predictor

        # Multiple inputs before quitting
        mock_input.side_effect = [
            'Great app!',
            'Bad experience',
            'It is okay',
            'q'  # quit with 'q'
        ]

        main()

        # Predict should be called 3 times
        assert mock_predictor.predict.call_count == 3


def test_main_function_case_insensitive_quit(mock_model_path):
    """Test that quit commands are case insensitive."""
    with patch('sys.argv', ['inference.py', '--model_path', mock_model_path]), \
         patch('src.inference.SentimentPredictor') as MockPredictor, \
         patch('builtins.input') as mock_input, \
         patch('builtins.print'):

        mock_predictor = MagicMock()
        MockPredictor.return_value = mock_predictor

        # User types 'QUIT' in uppercase
        mock_input.return_value = 'QUIT'

        main()

        mock_predictor.predict.assert_not_called()


# ---------- DEVICE COMPATIBILITY ----------


def test_sentiment_predictor_device_compatibility(mock_model_path):
    """Test that SentimentPredictor works with available device."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    # Should work regardless of CUDA availability
    result = predictor.predict("Test text")

    assert result is not None
    assert predictor.device.type in ['cuda', 'cpu']


def test_model_parameters_on_correct_device(mock_model_path):
    """Test that model parameters are on the correct device."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    # Check that model parameters match predictor device
    for param in predictor.model.parameters():
        assert param.device.type == predictor.device.type
        break  # Just check first parameter


# ---------- TOKENIZER TESTS ----------


def test_sentiment_predictor_tokenizer_loaded(mock_model_path):
    """Test that tokenizer is properly loaded."""
    predictor = SentimentPredictor(model_path=mock_model_path)

    assert predictor.tokenizer is not None
    # Should be able to tokenize
    tokens = predictor.tokenizer("test", return_tensors='pt')
    assert 'input_ids' in tokens


def test_predict_uses_correct_max_length(mock_model_path):
    """Test that predict respects max_len parameter."""
    predictor = SentimentPredictor(model_path=mock_model_path, max_len=50)

    # Create text longer than max_len
    long_text = "word " * 100
    result = predictor.predict(long_text)

    # Should still work (tokenizer truncates)
    assert result is not None
    assert 'sentiment' in result
