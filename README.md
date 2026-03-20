# Language Feedback API

An LLM-powered API that analyzes learner-written sentences and returns structured grammar correction feedback. Built with FastAPI and OpenAI GPT-4o mini.

## Design Decisions

### Model Choice: GPT-4o mini

I chose GPT-4o mini as the sole model for this submission. OpenAI describes it as a fast, affordable small model that supports Structured Outputs. For this task it hits the right balance: strong multilingual grammar knowledge, low latency (well under the 30-second limit), and low cost per request. A larger model like GPT-4o would improve accuracy marginally but at significantly higher latency and cost, which matters when the rubric weighs production feasibility at 25%.

### Structured Outputs over JSON Mode

The starter code uses `response_format={"type": "json_object"}`, which only guarantees valid JSON but not schema adherence. I replaced this with OpenAI Structured Outputs (`json_schema` with `strict: true`), which guarantees the response matches the exact schema every time. This eliminates an entire class of failure where the model returns valid JSON with wrong field names, missing fields, or invented keys. The strict schema includes `additionalProperties: false` at every object level and enumerates the allowed `error_type` and `difficulty` values directly in the schema.

### Prompt Strategy

The system prompt is designed around these principles:

**Minimal-edit correction.** The prompt instructs the model to make the smallest possible change to fix each error. It must not rephrase, simplify, or improve style. This preserves the learner's voice and avoids overcorrection, which is pedagogically important because learners need to see what specifically was wrong, not receive a rewritten sentence.

**CEFR difficulty separated from correctness.** The prompt explicitly tells the model to rate difficulty based on vocabulary and grammar complexity of the sentence, not based on whether it contains errors. A simple sentence with a spelling mistake is still A1.

**Native-language explanations.** Every error explanation must be written in the learner's native language. This is critical for lower-level learners who may not understand metalinguistic explanations in their target language.

**Multilingual span awareness.** The prompt instructs the model to handle non-Latin scripts (CJK, Cyrillic, Arabic, etc.) and to use the smallest meaningful unit as the error span for languages without word spaces like Japanese and Chinese.

**Correct-sentence discipline.** The prompt has a hard rule: if the sentence is already correct, return `is_correct=true`, empty errors, and the exact original sentence. This is reinforced by post-validation.

### Post-Validation Layer

After the model returns, a deterministic validator runs before sending the response:

- If `is_correct=true` but the errors array is non-empty, the contradiction is resolved by setting `is_correct=false` (trusting the error list over the flag).
- If `is_correct=true`, `corrected_sentence` is forced to equal the original input exactly.
- If `is_correct=false` but there are no errors and the corrected sentence matches the original, the response is normalized to `is_correct=true`.
- Error entries with empty `original` fields are stripped.
- Any `error_type` not in the allowed enum is replaced with `"other"`.
- Any `difficulty` not in A1-C2 is defaulted to `"B1"`.

This layer catches the edge cases where even Structured Outputs cannot help (logical consistency between fields).

### Caching

An exact-match in-memory cache stores responses keyed by `(sentence, target_language, native_language)`. Identical requests skip the API call entirely. The cache is capped at 128 entries with FIFO eviction.

I deliberately avoided semantic or fuzzy caching. Two sentences that look similar to an embedding model can need completely different corrections ("Il mange" vs "Il manges"), so semantic similarity is not a safe cache key for grammar feedback.

### Retry and Timeout

The OpenAI client is configured with a 20-second timeout. On transient failures (timeout, rate limit, connection error), the call is retried exactly once. This keeps the total worst-case time under 30 seconds while handling occasional provider hiccups.

### Error Handling

The `/feedback` endpoint catches all exceptions and returns a structured 502 response instead of exposing stack traces. Provider errors are logged with full context for debugging.

## How to Run

### Prerequisites

- Docker and Docker Compose
- An OpenAI API key

### Quick Start

```bash
# Clone the repo
git clone <your-fork-url>
cd intern-task-2026

# Set your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Build and start
docker compose up --build
```

The server starts on port 8000.

### Verify

```bash
# Health check
curl http://localhost:8000/health

# Test a request
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Yo soy fue al mercado ayer.", "target_language": "Spanish", "native_language": "English"}'
```

### Run Tests

```bash
# Unit tests (no API key needed)
docker compose exec feedback-api pytest tests/test_feedback_unit.py tests/test_schema.py -v

# Integration tests (requires OPENAI_API_KEY in .env)
docker compose exec feedback-api pytest tests/test_feedback_integration.py -v

# All tests
docker compose exec feedback-api pytest -v
```

## Test Coverage

**36 tests total** across three test files:

- **16 unit tests** (`test_feedback_unit.py`): Mock the OpenAI API to test feedback logic, post-validation behavior, cache hits, model validation, edge cases like punctuation-only errors and non-Latin scripts.
- **11 integration tests** (`test_feedback_integration.py`): Real API calls covering Spanish, French, German, Portuguese, Japanese, Korean, Russian, Chinese, and English. Tests correct sentences, error detection, and response time.
- **9 schema tests** (`test_schema.py`): Validate request/response models against the JSON schemas and verify all example inputs/outputs conform.

The tests cover: correct sentences returning the original exactly, multi-error detection, non-Latin scripts (Japanese particles, Korean particles, Russian cases), gender agreement, conjugation errors, spelling errors, punctuation-only errors, post-validation logic for LLM inconsistencies, cache behavior, and the 30-second response time requirement.

## Verifying Accuracy for Languages I Do Not Speak

For languages I cannot personally evaluate, I relied on three things: (1) the sample inputs/outputs in the repo as ground truth for the expected correction patterns, (2) the integration test suite which makes real API calls and validates structural correctness, and (3) spot-checking model outputs against known grammar rules (e.g., Japanese particle usage with specific verbs, French noun genders from dictionaries). The Structured Outputs schema enforcement ensures the response shape is always valid even when I cannot fully evaluate the linguistic content.

## Architecture Summary

```
POST /feedback
    |
    v
[Exact-match cache] --> hit? --> return cached result
    |
    miss
    |
    v
[OpenAI GPT-4o mini + Structured Outputs (strict)]
    |
    v
[Post-validation: fix is_correct/errors contradictions,
 force original sentence on correct, validate enums]
    |
    v
[Pydantic model validation]
    |
    v
[Cache result, return response]
```

## Trade-offs

**What I built:** One model, one provider, one endpoint. Compact prompt, strict schema, deterministic post-validation, exact-match cache, strong test suite.

**What I did not build:** Multi-provider fallback, semantic caching, vector databases, LLM-as-a-judge evaluation, background workers, or complex infrastructure. These are good ideas for a production system but add moving parts that increase the chance of Docker failures, timeouts, and schema drift for this assessment. The repo says to focus on prompt, accuracy, tests, and code quality, and that is where I spent the time.
