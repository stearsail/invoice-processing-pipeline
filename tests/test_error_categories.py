import json

import anthropic
import httpx
import openai
import pytest
from PIL import UnidentifiedImageError
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from invoice_pipeline.schema import Invoice
from invoice_pipeline.workflow.error_categories import categorize_error

_REQUEST = httpx.Request("POST", "http://api.test/v1")


def _status(sdk, cls_name: str, code: int):
    cls = getattr(sdk, cls_name)
    return cls(cls_name, response=httpx.Response(code, request=_REQUEST), body=None)


def _validation_error() -> ValidationError:
    try:
        Invoice.model_validate({})
    except ValidationError as e:
        return e


_CASES = [
    # transient: the worker retries
    ("openai_unreachable", openai.APIConnectionError(request=_REQUEST), "connectivity"),
    ("openai_timeout", openai.APITimeoutError(request=_REQUEST), "connectivity"),
    ("openai_429", _status(openai, "RateLimitError", 429), "connectivity"),
    ("openai_500", _status(openai, "InternalServerError", 500), "connectivity"),
    (
        "anthropic_unreachable",
        anthropic.APIConnectionError(request=_REQUEST),
        "connectivity",
    ),
    ("anthropic_529", _status(anthropic, "OverloadedError", 529), "connectivity"),
    ("deadline", TimeoutError(), "connectivity"),
    ("db_down", OperationalError("SELECT 1", {}, Exception("down")), "connectivity"),
    ("db_pool_exhausted", PoolTimeoutError("QueuePool limit reached"), "connectivity"),
    # this document can't be read by this model: fall back / review
    ("missing_image", FileNotFoundError("gone.png"), "bad_input"),
    ("undecodable_image", UnidentifiedImageError("cannot identify"), "bad_input"),
    ("bad_json", json.JSONDecodeError("Expecting value", "", 0), "invalid_output"),
    ("schema_mismatch", _validation_error(), "invalid_output"),
    ("openai_400", _status(openai, "BadRequestError", 400), "rejected_input"),
    ("anthropic_400", _status(anthropic, "BadRequestError", 400), "rejected_input"),
    # config errors and bugs: must surface, never fall back silently
    ("openai_wrong_key", _status(openai, "AuthenticationError", 401), "unknown"),
    ("anthropic_wrong_key", _status(anthropic, "AuthenticationError", 401), "unknown"),
    ("openai_403", _status(openai, "PermissionDeniedError", 403), "unknown"),
    ("openai_404_wrong_model", _status(openai, "NotFoundError", 404), "unknown"),
    ("bug_type", TypeError("bad call"), "unknown"),
    ("bug_key", KeyError("missing"), "unknown"),
]


@pytest.mark.parametrize(
    "exc, expected", [(e, c) for _, e, c in _CASES], ids=[i for i, _, _ in _CASES]
)
def test_categorize_error(exc, expected):
    assert categorize_error(exc) == expected
