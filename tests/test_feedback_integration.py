"""Integration tests -- require OPENAI_API_KEY to be set.

Run with: pytest tests/test_feedback_integration.py -v

These tests make real API calls. Skip them in CI or when no key is available.
"""

import os
import time

import pytest

from app.feedback import get_feedback, _cache
from app.models import FeedbackRequest, VALID_ERROR_TYPES, VALID_DIFFICULTIES

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set -- skipping integration tests",
)


@pytest.fixture(autouse=True)
def clear_cache():
    _cache.clear()
    yield
    _cache.clear()


def _assert_valid_response(result, expect_correct: bool | None = None):
    """Shared assertions for schema compliance."""
    assert result.difficulty in VALID_DIFFICULTIES
    for error in result.errors:
        assert error.error_type in VALID_ERROR_TYPES
        assert len(error.original) > 0
        assert error.correction is not None
        assert len(error.explanation) > 0
    if expect_correct is True:
        assert result.is_correct is True
        assert result.errors == []
    elif expect_correct is False:
        assert result.is_correct is False
        assert len(result.errors) >= 1


# --------------------------------------------------------------------------
# European languages (Latin script)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spanish_conjugation_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    assert "fui" in result.corrected_sentence.lower() or "fue" not in result.corrected_sentence.lower().replace("fui", "")


@pytest.mark.asyncio
async def test_french_gender_agreement():
    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    assert len(result.errors) >= 1


@pytest.mark.asyncio
async def test_correct_german_sentence():
    request = FeedbackRequest(
        sentence="Ich habe gestern einen interessanten Film gesehen.",
        target_language="German",
        native_language="English",
    )
    result = await get_feedback(request)
    _assert_valid_response(result, expect_correct=True)
    assert result.corrected_sentence == request.sentence


@pytest.mark.asyncio
async def test_portuguese_conjugation_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Eu gosta de comer frutas todos os dias.",
            target_language="Portuguese",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)


# --------------------------------------------------------------------------
# Non-Latin scripts
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_japanese_particle_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    assert any("に" in e.correction for e in result.errors)


@pytest.mark.asyncio
async def test_korean_particle_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="나는 학교을 갑니다.",
            target_language="Korean",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)


@pytest.mark.asyncio
async def test_russian_case_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Я читаю интересная книга.",
            target_language="Russian",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)


@pytest.mark.asyncio
async def test_chinese_correct_sentence():
    request = FeedbackRequest(
        sentence="我今天去了图书馆。",
        target_language="Chinese",
        native_language="English",
    )
    result = await get_feedback(request)
    _assert_valid_response(result, expect_correct=True)
    assert result.corrected_sentence == request.sentence


# --------------------------------------------------------------------------
# Native language explanations
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explanation_in_native_language_spanish():
    result = await get_feedback(
        FeedbackRequest(
            sentence="I goes to school every day.",
            target_language="English",
            native_language="Spanish",
        )
    )
    _assert_valid_response(result, expect_correct=False)


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_correct_simple_english():
    request = FeedbackRequest(
        sentence="The cat sat on the mat.",
        target_language="English",
        native_language="Spanish",
    )
    result = await get_feedback(request)
    _assert_valid_response(result, expect_correct=True)
    assert result.corrected_sentence == request.sentence


@pytest.mark.asyncio
async def test_response_time_under_30_seconds():
    start = time.time()
    await get_feedback(
        FeedbackRequest(
            sentence="Elle a mange les pommes hier soir avec ses amis.",
            target_language="French",
            native_language="English",
        )
    )
    elapsed = time.time() - start
    assert elapsed < 30, f"Response took {elapsed:.1f}s, exceeds 30s limit"
