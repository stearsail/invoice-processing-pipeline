import json
from PIL import UnidentifiedImageError
import sqlalchemy.exc
import anthropic
import openai
from pydantic import ValidationError

# transient failures (API, extraction deadline, DB) where a later attempt can succeed
_RETRYABLE_EXCEPTIONS = (
    openai.APIConnectionError,  # covers APITimeoutError too (subclass)
    openai.RateLimitError,
    openai.InternalServerError,
    anthropic.APIConnectionError,  # covers APITimeoutError too (subclass)
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
    TimeoutError,
    sqlalchemy.exc.OperationalError,
    sqlalchemy.exc.TimeoutError,
)

_FILE_EXCEPTIONS = (
    FileNotFoundError,
    UnidentifiedImageError,
)


def categorize_error(exc: Exception) -> str:
    """Decide who handles a failure.

    "connectivity" (transient) and "unknown" (bugs, misconfiguration) propagate
    to the worker, which retries the former and fails the latter fast.
    "bad_input", "invalid_output" and "rejected_input" mean this model couldn't
    read this document: extractors return them as a parse_error so the graph
    can fall back or send the document to review.
    """
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return "connectivity"
    if isinstance(exc, _FILE_EXCEPTIONS):
        return "bad_input"
    if isinstance(exc, (ValidationError, json.JSONDecodeError)):
        return "invalid_output"
    if isinstance(exc, (anthropic.BadRequestError, openai.BadRequestError)):
        return "rejected_input"
    return "unknown"
