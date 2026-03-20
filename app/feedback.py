"""System prompt, LLM interaction, caching, and post-validation for language feedback."""

import json
import logging

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError

from app.models import (
    FeedbackRequest,
    FeedbackResponse,
    VALID_DIFFICULTIES,
    VALID_ERROR_TYPES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-initialised OpenAI client (avoids issues with dotenv load order)
# ---------------------------------------------------------------------------
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(timeout=20.0)
    return _client


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a language-learning assistant that provides grammar feedback on learner-written sentences.

CORRECTION POLICY:
- Make the SMALLEST possible edits to fix genuine errors. Never rephrase, simplify, or improve style.
- Preserve the learner's word choices, sentence structure, and intended meaning.
- The corrected_sentence MUST be the original sentence with ALL identified errors actually fixed. Do not leave any identified error uncorrected in corrected_sentence.
- If the sentence is already correct, set is_correct to true, return an empty errors array, and set corrected_sentence to the EXACT original sentence with no changes whatsoever.
- Do not add or remove punctuation unless it is a clear error.

ERROR IDENTIFICATION:
- For each error, extract the exact original text as it appears in the learner's sentence.
- Provide the minimal correction that fixes only that error.
- Classify each error using exactly one of: grammar, spelling, word_choice, punctuation, word_order, missing_word, extra_word, conjugation, gender_agreement, number_agreement, tone_register, other.
- For missing_word errors, set "original" to the adjacent word(s) where the word should be inserted, and "correction" to those words with the missing word included.

EXPLANATION LANGUAGE -- THIS IS CRITICAL:
- Every "explanation" field MUST be written in the learner's NATIVE language, NOT in the target language.
- If the native language is English, write explanations in English.
- If the native language is Spanish, write explanations in Spanish.
- NEVER write explanations in the target language. The learner may not understand the target language well enough.
- Keep explanations 1-2 sentences, friendly, and educational. Briefly tell the learner WHY the correction is needed.

DIFFICULTY RATING:
- Rate the CEFR level (A1-C2) based on the vocabulary and grammar complexity of the sentence itself, NOT based on whether it contains errors.
- A1: basic memorised phrases and greetings.
- A2: simple everyday sentences with common verbs.
- B1: connected text on familiar topics, compound sentences.
- B2: complex arguments, abstract topics, relative clauses.
- C1: nuanced expression, implicit meaning, advanced idioms.
- C2: near-native precision, rare vocabulary, sophisticated structures.

MULTILINGUAL RULES:
- Support ALL writing systems: Latin, CJK, Cyrillic, Arabic, Devanagari, Hangul, and others.
- For languages without word spaces (Japanese, Chinese), use the smallest meaningful unit as the original span.
- Never translate the sentence into another language. Only correct errors in the target language.
- Explanations MUST be in the native language, not the target language."""

# ---------------------------------------------------------------------------
# Strict JSON schema for OpenAI Structured Outputs
# ---------------------------------------------------------------------------
RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["corrected_sentence", "is_correct", "errors", "difficulty"],
    "additionalProperties": False,
    "properties": {
        "corrected_sentence": {"type": "string"},
        "is_correct": {"type": "boolean"},
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "original",
                    "correction",
                    "error_type",
                    "explanation",
                ],
                "additionalProperties": False,
                "properties": {
                    "original": {"type": "string"},
                    "correction": {"type": "string"},
                    "error_type": {
                        "type": "string",
                        "enum": sorted(VALID_ERROR_TYPES),
                    },
                    "explanation": {"type": "string"},
                },
            },
        },
        "difficulty": {
            "type": "string",
            "enum": sorted(VALID_DIFFICULTIES),
        },
    },
}

# ---------------------------------------------------------------------------
# Exact-match response cache
# ---------------------------------------------------------------------------
_cache: dict[tuple[str, str, str], FeedbackResponse] = {}
_CACHE_MAX_SIZE = 128


# ---------------------------------------------------------------------------
# Post-validation
# ---------------------------------------------------------------------------
def _validate_and_fix(data: dict, original_sentence: str) -> dict:
    """Deterministic post-processing to fix common LLM inconsistencies."""

    is_correct = data.get("is_correct", False)
    errors = data.get("errors", [])
    corrected = data.get("corrected_sentence", original_sentence)

    if is_correct and errors:
        data["is_correct"] = False
        is_correct = False

    if is_correct:
        data["corrected_sentence"] = original_sentence
        data["errors"] = []
        return data

    if not is_correct and not errors and corrected == original_sentence:
        data["is_correct"] = True
        data["corrected_sentence"] = original_sentence
        data["errors"] = []
        return data

    valid_errors = []
    for err in errors:
        et = err.get("error_type", "")
        if et not in VALID_ERROR_TYPES:
            err["error_type"] = "other"
        orig = err.get("original", "").strip()
        corr = err.get("correction")
        if orig and corr is not None:
            valid_errors.append(err)
    data["errors"] = valid_errors

    if data.get("difficulty") not in VALID_DIFFICULTIES:
        data["difficulty"] = "B1"

    return data


# ---------------------------------------------------------------------------
# OpenAI call with one retry on transient errors
# ---------------------------------------------------------------------------
async def _call_openai(user_message: str) -> dict:
    """Call GPT-4o mini with Structured Outputs. Retries once on transient failure."""

    client = _get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "feedback_response",
                        "strict": True,
                        "schema": RESPONSE_SCHEMA,
                    },
                },
                temperature=0,
                max_tokens=1024,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except (APITimeoutError, RateLimitError, APIConnectionError) as exc:
            last_error = exc
            logger.warning(
                "OpenAI transient error (attempt %d/2): %s", attempt + 1, exc
            )
            if attempt == 0:
                continue
            raise

    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    cache_key = (request.sentence, request.target_language, request.native_language)

    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    user_message = (
        f"Target language: {request.target_language}\n"
        f"Native language: {request.native_language}\n"
        f"Sentence: {request.sentence}\n\n"
        f"Write all explanations in {request.native_language}. "
        f"The corrected_sentence must have all errors fixed."
    )

    data = await _call_openai(user_message)
    data = _validate_and_fix(data, request.sentence)
    result = FeedbackResponse(**data)

    if len(_cache) >= _CACHE_MAX_SIZE:
        oldest_key = next(iter(_cache))
        del _cache[oldest_key]
    _cache[cache_key] = result

    return result
