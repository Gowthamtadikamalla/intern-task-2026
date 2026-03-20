"""Unit tests -- run without an API key using mocked LLM responses."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.feedback import get_feedback, _validate_and_fix, _cache
from app.models import FeedbackRequest, FeedbackResponse, VALID_ERROR_TYPES, VALID_DIFFICULTIES


def _mock_completion(response_data: dict) -> MagicMock:
    """Build a mock ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = json.dumps(response_data)
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the response cache before each test."""
    _cache.clear()
    yield
    _cache.clear()


# --------------------------------------------------------------------------
# Core feedback behaviour (mocked LLM)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feedback_with_single_error():
    mock_response = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You mixed two verb forms.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback._get_client") as mock_get:
        instance = MagicMock()
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )
        mock_get.return_value = instance

        request = FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.difficulty == "A2"


@pytest.mark.asyncio
async def test_feedback_correct_sentence():
    mock_response = {
        "corrected_sentence": "Ich habe gestern einen interessanten Film gesehen.",
        "is_correct": True,
        "errors": [],
        "difficulty": "B1",
    }

    with patch("app.feedback._get_client") as mock_get:
        instance = MagicMock()
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )
        mock_get.return_value = instance

        request = FeedbackRequest(
            sentence="Ich habe gestern einen interessanten Film gesehen.",
            target_language="German",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == request.sentence


@pytest.mark.asyncio
async def test_feedback_multiple_errors():
    mock_response = {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "gender_agreement",
                "explanation": "'Chat' is masculine.",
            },
            {
                "original": "le table",
                "correction": "la table",
                "error_type": "gender_agreement",
                "explanation": "'Table' is feminine.",
            },
        ],
        "difficulty": "A1",
    }

    with patch("app.feedback._get_client") as mock_get:
        instance = MagicMock()
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )
        mock_get.return_value = instance

        request = FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert len(result.errors) == 2
    assert all(e.error_type == "gender_agreement" for e in result.errors)


@pytest.mark.asyncio
async def test_feedback_japanese_non_latin():
    mock_response = {
        "corrected_sentence": "私は東京に住んでいます。",
        "is_correct": False,
        "errors": [
            {
                "original": "を",
                "correction": "に",
                "error_type": "grammar",
                "explanation": "The verb 住む takes the particle に for location.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback._get_client") as mock_get:
        instance = MagicMock()
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )
        mock_get.return_value = instance

        request = FeedbackRequest(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert any("に" in e.correction for e in result.errors)


@pytest.mark.asyncio
async def test_feedback_punctuation_only_error():
    mock_response = {
        "corrected_sentence": "Bonjour, comment allez-vous?",
        "is_correct": False,
        "errors": [
            {
                "original": "Bonjour comment",
                "correction": "Bonjour, comment",
                "error_type": "punctuation",
                "explanation": "A comma is needed after a greeting.",
            }
        ],
        "difficulty": "A1",
    }

    with patch("app.feedback._get_client") as mock_get:
        instance = MagicMock()
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )
        mock_get.return_value = instance

        request = FeedbackRequest(
            sentence="Bonjour comment allez-vous?",
            target_language="French",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert result.errors[0].error_type == "punctuation"


# --------------------------------------------------------------------------
# Cache behaviour
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_returns_same_result_without_second_api_call():
    mock_response = {
        "corrected_sentence": "Hola mundo.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }

    with patch("app.feedback._get_client") as mock_get:
        instance = MagicMock()
        create_mock = AsyncMock(return_value=_mock_completion(mock_response))
        instance.chat.completions.create = create_mock
        mock_get.return_value = instance

        request = FeedbackRequest(
            sentence="Hola mundo.",
            target_language="Spanish",
            native_language="English",
        )
        result1 = await get_feedback(request)
        result2 = await get_feedback(request)

    assert result1 == result2
    assert create_mock.await_count == 1


# --------------------------------------------------------------------------
# Post-validation (_validate_and_fix)
# --------------------------------------------------------------------------


def test_postval_is_correct_true_with_errors_sets_false():
    data = {
        "corrected_sentence": "Le chat noir.",
        "is_correct": True,
        "errors": [
            {
                "original": "La",
                "correction": "Le",
                "error_type": "gender_agreement",
                "explanation": "Masculine.",
            }
        ],
        "difficulty": "A1",
    }
    fixed = _validate_and_fix(data, "La chat noir.")
    assert fixed["is_correct"] is False
    assert len(fixed["errors"]) == 1


def test_postval_is_correct_true_forces_original_sentence():
    data = {
        "corrected_sentence": "Some other text",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }
    fixed = _validate_and_fix(data, "Original sentence.")
    assert fixed["corrected_sentence"] == "Original sentence."


def test_postval_not_correct_no_errors_matching_sentence_becomes_correct():
    data = {
        "corrected_sentence": "Hello world.",
        "is_correct": False,
        "errors": [],
        "difficulty": "A1",
    }
    fixed = _validate_and_fix(data, "Hello world.")
    assert fixed["is_correct"] is True
    assert fixed["errors"] == []


def test_postval_invalid_error_type_replaced_with_other():
    data = {
        "corrected_sentence": "Test.",
        "is_correct": False,
        "errors": [
            {
                "original": "x",
                "correction": "y",
                "error_type": "not_a_real_type",
                "explanation": "test",
            }
        ],
        "difficulty": "B1",
    }
    fixed = _validate_and_fix(data, "Test orig.")
    assert fixed["errors"][0]["error_type"] == "other"


def test_postval_invalid_difficulty_defaults_to_b1():
    data = {
        "corrected_sentence": "Test.",
        "is_correct": False,
        "errors": [
            {
                "original": "x",
                "correction": "y",
                "error_type": "grammar",
                "explanation": "test",
            }
        ],
        "difficulty": "Z9",
    }
    fixed = _validate_and_fix(data, "Test orig.")
    assert fixed["difficulty"] == "B1"


def test_postval_strips_errors_with_empty_original():
    data = {
        "corrected_sentence": "Corrected.",
        "is_correct": False,
        "errors": [
            {
                "original": "",
                "correction": "y",
                "error_type": "grammar",
                "explanation": "test",
            },
            {
                "original": "real",
                "correction": "fixed",
                "error_type": "spelling",
                "explanation": "test",
            },
        ],
        "difficulty": "A2",
    }
    fixed = _validate_and_fix(data, "Original.")
    assert len(fixed["errors"]) == 1
    assert fixed["errors"][0]["original"] == "real"


# --------------------------------------------------------------------------
# Model validation
# --------------------------------------------------------------------------


def test_response_model_rejects_invalid_error_type():
    with pytest.raises(Exception):
        FeedbackResponse(
            corrected_sentence="test",
            is_correct=False,
            errors=[
                {
                    "original": "x",
                    "correction": "y",
                    "error_type": "fake_type",
                    "explanation": "test",
                }
            ],
            difficulty="A1",
        )


def test_response_model_rejects_invalid_difficulty():
    with pytest.raises(Exception):
        FeedbackResponse(
            corrected_sentence="test",
            is_correct=True,
            errors=[],
            difficulty="Z9",
        )


def test_all_valid_error_types_accepted():
    for et in VALID_ERROR_TYPES:
        resp = FeedbackResponse(
            corrected_sentence="test",
            is_correct=False,
            errors=[
                {
                    "original": "x",
                    "correction": "y",
                    "error_type": et,
                    "explanation": "test",
                }
            ],
            difficulty="A1",
        )
        assert resp.errors[0].error_type == et


def test_all_valid_difficulties_accepted():
    for d in VALID_DIFFICULTIES:
        resp = FeedbackResponse(
            corrected_sentence="test",
            is_correct=True,
            errors=[],
            difficulty=d,
        )
        assert resp.difficulty == d
