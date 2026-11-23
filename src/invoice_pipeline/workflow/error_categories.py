import json
import sqlalchemy.exc
import anthropic
import openai
from pydantic import ValidationError

# exceptions worth retrying the identical call for — network issues where a later attempt might succeed
_RETRYABLE_EXCEPTIONS = (
    openai.APIConnectionError,  # covers APITimeoutError too (subclass)
    openai.RateLimitError,
    openai.InternalServerError,
    openai.NotFoundError,
    anthropic.APIConnectionError,  # covers APITimeoutError too (subclass)
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
    TimeoutError,
    sqlalchemy.exc.OperationalError,
)


def categorize_error(exc: Exception) -> str:
    """Classify an extraction failure as retryable ("connectivity") or not.
    Retryable failures should propagate uncaught so Arq's own retry/backoff
    handles them, since a later attempt at the identical call can succeed.
    Everything else (bad input, malformed request, missing file, model output
    that fails validation) will fail identically no matter how many times you
    retry the same call — those get reported instead.
    """
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return "connectivity"
    if isinstance(exc, FileNotFoundError):
        return "missing_file"
    if isinstance(exc, (ValidationError, json.JSONDecodeError)):
        return "invalid_output"
    return "unknown"
