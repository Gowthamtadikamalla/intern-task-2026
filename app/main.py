"""FastAPI application -- language feedback endpoint."""

import logging

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.feedback import get_feedback
from app.models import FeedbackRequest, FeedbackResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Language Feedback API",
    description="Analyzes learner-written sentences and provides structured language feedback.",
    version="1.0.0",
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    try:
        return await get_feedback(request)
    except Exception as exc:
        logger.error("Feedback endpoint error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=502,
            content={"detail": "LLM provider error. Please try again."},
        )
