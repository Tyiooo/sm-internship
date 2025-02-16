import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from main import app, TranslationRequest, TranslationResponse

client = TestClient(app)


# Mock the transformers components
@pytest.fixture(autouse=True)
def mock_transformers():
    with (
        patch("transformers.T5Tokenizer") as mock_tokenizer,
        patch("transformers.T5ForConditionalGeneration") as mock_model,
    ):
        # Configure mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.batch_decode.return_value = [
            "Hallo Welt",
            "Wie sind Sie",
            "Test@#$%",  # Added translation for special chars
        ]
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.return_value = {
            "input_ids": Mock(),
            "attention_mask": Mock(),
        }
        mock_tokenizer_instance.pad_token_id = 0  # Mock pad_token_id

        # Configure mock model
        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = [
            1,
            2,
            3,
        ]  # Mock generated token IDs
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.config = Mock()
        mock_model_instance.config.decoder_start_token_id = 0
        mock_model_instance.config.eos_token_id = 1
        mock_model_instance.config.pad_token_id = 0

        yield mock_tokenizer_instance, mock_model_instance


def test_ping():
    """Test the health check endpoint"""
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Server is running"}


def test_predict_successful_translation(mock_transformers):
    """Test successful translation request"""
    mock_tokenizer_instance, _ = mock_transformers
    test_data = {"sentences": ["Hello world", "How are you"]}

    response = client.post("/predict", json=test_data)

    assert response.status_code == 200
    assert "translations" in response.json()
    assert len(response.json()["translations"]) == 2
    assert response.json() == {"translations": ["Hallo Welt", "Wie sind Sie"]}


def test_predict_empty_input():
    """Test translation with empty input"""
    test_data = {"sentences": []}

    response = client.post("/predict", json=test_data)

    assert response.status_code == 200
    assert response.json()["translations"] == []


def test_predict_invalid_input():
    """Test translation with invalid input format"""
    test_data = {"invalid_key": ["Hello world"]}

    response = client.post("/predict", json=test_data)

    assert response.status_code == 422  # Validation error


def test_predict_none_input():
    """Test translation with None input"""
    test_data = {"sentences": None}

    response = client.post("/predict", json=test_data)

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_translation_request_model():
    """Test TranslationRequest model validation"""
    request = TranslationRequest(sentences=["Hello world"])
    assert request.sentences == ["Hello world"]


@pytest.mark.asyncio
async def test_translation_response_model():
    """Test TranslationResponse model validation"""
    response = TranslationResponse(translations=["Hallo Welt"])
    assert response.translations == ["Hallo Welt"]


def test_predict_with_special_characters(mock_transformers):
    """Test translation with special characters"""
    mock_tokenizer_instance, _ = mock_transformers
    test_data = {"sentences": ["Hello world!", "How are you?", "Test@#$%"]}

    response = client.post("/predict", json=test_data)

    assert response.status_code == 200
    assert len(response.json()["translations"]) == 3
    assert response.json()["translations"] == [
        "Hallo Welt!",
        "Wie sind Sie?",
        "Test@#$%",
    ]
