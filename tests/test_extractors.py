import logging
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx
import openai
import pytest
from PIL import UnidentifiedImageError
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from invoice_pipeline.workflow import extractors
from invoice_pipeline.workflow.extractors import (
    FrontierExtractor,
    SpecialistExtractor,
)
from invoice_pipeline.schema import Invoice, LineItem, WireInvoice


def _invoice(**overrides) -> Invoice:
    defaults = dict(
        document_type="receipt",
        currency="USD",
        invoice_number="INV-1",
        grand_total=Decimal("100"),
        subtotal=Decimal("100"),
        line_items=[
            LineItem(
                description="Item",
                quantity=Decimal("1"),
                unit_price=Decimal("100"),
                line_total=Decimal("100"),
            ),
        ],
    )
    defaults.update(overrides)
    return Invoice(**defaults)


def _fake_specialist_client(content: str) -> MagicMock:
    message = MagicMock(content=content)
    choice = MagicMock(message=message)
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[choice])
    )
    return fake_client


def _fake_specialist_client_raising(exc: Exception) -> MagicMock:
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(side_effect=exc)
    return fake_client


def _text_message(text: str, stop_reason: str = "end_turn") -> AIMessage:
    # the shape Claude returns for json_schema structured output: JSON as a text block
    return AIMessage(
        content=[{"type": "text", "text": text}],
        response_metadata={"stop_reason": stop_reason},
    )


def _structured_reply(text: str, stop_reason: str = "end_turn") -> dict:
    # what with_structured_output(..., include_raw=True) returns, including langchain's own parse
    try:
        parsed, parsing_error = WireInvoice.model_validate_json(text), None
    except ValidationError as e:
        parsed, parsing_error = None, OutputParserException(str(e))
    return {
        "raw": _text_message(text, stop_reason),
        "parsed": parsed,
        "parsing_error": parsing_error,
    }


def _fake_frontier_client(
    invoice: Invoice | None = None, raises: bool = False
) -> MagicMock:
    structured = MagicMock()
    if raises:
        structured.ainvoke = AsyncMock(side_effect=RuntimeError("frontier boom"))
    else:
        structured.ainvoke = AsyncMock(
            return_value=_structured_reply(invoice.model_dump_json())
        )
    fake_client = MagicMock()
    fake_client.with_structured_output.return_value = structured
    return fake_client


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def no_real_image_io(monkeypatch):
    monkeypatch.setattr(
        extractors, "load_image_b64", lambda path: ("fakeb64", "image/png")
    )


def _specialist(client: MagicMock, seed: int | None = None) -> SpecialistExtractor:
    extractor = SpecialistExtractor(
        "http://vllm.test/v1", "key", "test-model", timeout=30, seed=seed
    )
    extractor._client = client
    return extractor


def _frontier(client: MagicMock) -> FrontierExtractor:
    extractor = FrontierExtractor("test-model", timeout=30)
    extractor._client = client
    return extractor


@pytest.mark.anyio
async def test_specialist_parses_valid_json_into_invoice():
    invoice = _invoice()
    extractor = _specialist(_fake_specialist_client(invoice.model_dump_json()))

    result = await extractor.extract_invoice("fake.png")

    assert result.parse_error is None
    # our own parse of the raw text, not langchain's WireInvoice
    assert type(result.invoice) is Invoice
    assert result.invoice == invoice


@pytest.mark.anyio
async def test_specialist_reports_parse_error_on_invalid_json():
    extractor = _specialist(_fake_specialist_client("not valid json"))

    result = await extractor.extract_invoice("fake.png")

    assert result.invoice is None
    assert result.parse_error.startswith("invalid_output:")


@pytest.mark.anyio
async def test_specialist_reraises_connectivity_errors_for_arq_retry():
    exc = openai.APIConnectionError(request=httpx.Request("POST", "http://vllm.test"))
    extractor = _specialist(_fake_specialist_client_raising(exc))

    with pytest.raises(openai.APIConnectionError):
        await extractor.extract_invoice("fake.png")


@pytest.mark.anyio
async def test_specialist_sends_wire_schema_via_response_format():
    invoice = _invoice()
    client = _fake_specialist_client(invoice.model_dump_json())
    extractor = _specialist(client)

    await extractor.extract_invoice("fake.png")

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "WireInvoice",
            "schema": WireInvoice.model_json_schema(),
            "strict": True,
        },
    }
    assert "extra_body" not in kwargs


@pytest.mark.anyio
async def test_specialist_keeps_three_decimal_amounts_exact():
    # the wire grammar only admits dot-decimal, so a 3-digit fraction is a
    # fraction — not a thousands separator (regression: 0.064 used to become 64)
    raw = """{
        "document_type": "receipt", "currency": "KWD",
        "grand_total": "12.500", "subtotal": "12.500",
        "line_items": [
            {"description": "Bananas", "unit_price": "1.879",
             "quantity": "0.064", "line_total": "0.120"},
            {"description": "Fuel", "unit_price": "33.415",
             "quantity": "3.000", "line_total": "100.245"}
        ]
    }"""
    extractor = _specialist(_fake_specialist_client(raw))

    result = await extractor.extract_invoice("fake.png")

    assert result.parse_error is None
    assert result.invoice.grand_total == Decimal("12.500")
    assert result.invoice.subtotal == Decimal("12.500")
    first, second = result.invoice.line_items
    assert first.quantity == Decimal("0.064")
    assert first.unit_price == Decimal("1.879")
    assert first.line_total == Decimal("0.120")
    assert second.unit_price == Decimal("33.415")
    assert second.quantity == Decimal("3.000")
    assert second.line_total == Decimal("100.245")


@pytest.mark.anyio
@pytest.mark.parametrize("amount", ["20.000,00", "1,234.56", "1 234.56", "$12.00"])
async def test_specialist_rejects_non_canonical_wire_amounts(amount):
    # if the grammar constraint is ever silently dropped, locale-formatted
    # output must fail loudly (-> frontier fallback), never be guessed at
    raw = f"""{{
        "document_type": "invoice", "currency": "USD",
        "grand_total": "{amount}", "line_items": []
    }}"""
    extractor = _specialist(_fake_specialist_client(raw))

    result = await extractor.extract_invoice("fake.png")

    assert result.invoice is None
    assert result.parse_error.startswith("invalid_output")


@pytest.mark.anyio
async def test_specialist_rejects_non_canonical_optional_amount():
    # previously an unparseable optional amount was silently nulled
    raw = """{
        "document_type": "invoice", "currency": "USD",
        "grand_total": "100.00", "tax": "19,00", "line_items": []
    }"""
    extractor = _specialist(_fake_specialist_client(raw))

    result = await extractor.extract_invoice("fake.png")

    assert result.invoice is None
    assert result.parse_error.startswith("invalid_output")


@pytest.mark.anyio
async def test_specialist_parses_negative_grand_total_from_wire_response():
    raw = """{
        "document_type": "invoice", "currency": "USD",
        "grand_total": "-1500.00", "line_items": []
    }"""
    extractor = _specialist(_fake_specialist_client(raw))

    result = await extractor.extract_invoice("fake.png")

    assert result.parse_error is None
    assert result.invoice.grand_total == Decimal("-1500.00")


@pytest.mark.anyio
@pytest.mark.parametrize("wire_discount", ['"25.00"', '"-25.00"'])
async def test_specialist_applies_schema_discount_rule(wire_discount):
    # the sign rule lives on Invoice, not in the extractor — this pins that the
    # specialist path still goes through it (and that only discount is affected)
    raw = f"""{{
        "document_type": "invoice", "currency": "USD",
        "grand_total": "-100.00", "discount": {wire_discount},
        "line_items": [{{"line_total": "-5.00"}}]
    }}"""
    extractor = _specialist(_fake_specialist_client(raw))

    result = await extractor.extract_invoice("fake.png")

    assert result.parse_error is None
    assert isinstance(result.invoice.discount, Decimal)
    assert result.invoice.discount == Decimal("25.00")
    assert result.invoice.grand_total == Decimal("-100.00")
    assert result.invoice.line_items[0].line_total == Decimal("-5.00")


@pytest.mark.anyio
async def test_specialist_keeps_null_amounts_null():
    raw = """{
        "document_type": "receipt", "currency": "USD",
        "grand_total": "9.99", "subtotal": null, "discount": null,
        "line_items": [{"description": "Tea", "unit_price": null,
                        "quantity": null, "line_total": "9.99"}]
    }"""
    extractor = _specialist(_fake_specialist_client(raw))

    result = await extractor.extract_invoice("fake.png")

    assert result.parse_error is None
    assert result.invoice.subtotal is None
    assert result.invoice.discount is None
    assert result.invoice.line_items[0].unit_price is None
    assert result.invoice.line_items[0].quantity is None


@pytest.mark.anyio
async def test_specialist_sends_configured_model_name():
    invoice = _invoice()
    client = _fake_specialist_client(invoice.model_dump_json())
    extractor = _specialist(client)

    await extractor.extract_invoice("fake.png")

    assert client.chat.completions.create.await_args.kwargs["model"] == "test-model"


@pytest.mark.anyio
async def test_specialist_sends_configured_seed():
    invoice = _invoice()
    client = _fake_specialist_client(invoice.model_dump_json())
    extractor = _specialist(client, seed=42)

    await extractor.extract_invoice("fake.png")

    assert client.chat.completions.create.await_args.kwargs["seed"] == 42


@pytest.mark.anyio
async def test_specialist_defaults_to_no_seed():
    invoice = _invoice()
    client = _fake_specialist_client(invoice.model_dump_json())
    extractor = _specialist(client)

    await extractor.extract_invoice("fake.png")

    assert client.chat.completions.create.await_args.kwargs["seed"] is None


@pytest.mark.anyio
async def test_frontier_returns_structured_invoice():
    invoice = _invoice()
    extractor = _frontier(_fake_frontier_client(invoice))

    result = await extractor.extract_invoice("fake.png")

    assert result.parse_error is None
    assert result.invoice == invoice


@pytest.mark.anyio
async def test_frontier_reraises_unexpected_errors_instead_of_falling_back():
    # an unknown failure is a bug or misconfiguration: it must reach the worker
    extractor = _frontier(_fake_frontier_client(raises=True))

    with pytest.raises(RuntimeError, match="frontier boom"):
        await extractor.extract_invoice("fake.png")


@pytest.mark.anyio
async def test_frontier_reraises_connectivity_errors_for_arq_retry():
    exc = anthropic.APIConnectionError(
        request=httpx.Request("POST", "http://anthropic.test")
    )
    structured = MagicMock()
    structured.ainvoke = AsyncMock(side_effect=exc)
    fake_client = MagicMock()
    fake_client.with_structured_output.return_value = structured
    extractor = _frontier(fake_client)

    with pytest.raises(anthropic.APIConnectionError):
        await extractor.extract_invoice("fake.png")


@pytest.mark.anyio
async def test_frontier_appends_extra_context_to_prompt():
    invoice = _invoice()
    client = _fake_frontier_client(invoice)
    extractor = _frontier(client)

    await extractor.extract_invoice("fake.png", "issues: grand total mismatch")

    sent = client.with_structured_output.return_value.ainvoke.await_args.args[0][0]
    prompt = sent["content"][0]["text"]
    assert "issues: grand total mismatch" in prompt


_REQ = httpx.Request("POST", "http://api.test/v1")


def _status_error(sdk, cls_name: str, code: int):
    cls = getattr(sdk, cls_name)
    return cls(cls_name, response=httpx.Response(code, request=_REQ), body=None)


def _frontier_raising(exc: Exception) -> FrontierExtractor:
    structured = MagicMock()
    structured.ainvoke = AsyncMock(side_effect=exc)
    client = MagicMock()
    client.with_structured_output.return_value = structured
    return _frontier(client)


def _frontier_replying(text: str, stop_reason: str = "end_turn") -> FrontierExtractor:
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=_structured_reply(text, stop_reason))
    client = MagicMock()
    client.with_structured_output.return_value = structured
    return _frontier(client)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "exc",
    [
        _status_error(openai, "AuthenticationError", 401),
        TypeError("bug building the request"),
    ],
    ids=["wrong_api_key", "bug"],
)
async def test_specialist_reraises_unknown_errors(exc):
    extractor = _specialist(_fake_specialist_client_raising(exc))

    with pytest.raises(type(exc)):
        await extractor.extract_invoice("fake.png")


@pytest.mark.anyio
async def test_specialist_rejected_request_becomes_a_logged_parse_error(caplog):
    # e.g. vLLM refusing a request over --max-model-len: the frontier may still read it
    exc = _status_error(openai, "BadRequestError", 400)
    extractor = _specialist(_fake_specialist_client_raising(exc))

    with caplog.at_level(logging.WARNING, logger=extractors.__name__):
        result = await extractor.extract_invoice("fake.png")

    assert result.invoice is None
    assert result.parse_error.startswith("rejected_input:")
    assert "Specialist failed before responding (rejected_input)" in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize(
    "exc",
    [FileNotFoundError("gone.png"), UnidentifiedImageError("cannot identify")],
    ids=["missing", "undecodable"],
)
async def test_specialist_unusable_image_becomes_a_logged_parse_error(
    monkeypatch, caplog, exc
):
    def _broken_image(path):
        raise exc

    monkeypatch.setattr(extractors, "load_image_b64", _broken_image)
    client = _fake_specialist_client("{}")
    extractor = _specialist(client)

    with caplog.at_level(logging.WARNING, logger=extractors.__name__):
        result = await extractor.extract_invoice("fake.png")

    assert result.parse_error.startswith("bad_input:")
    assert "Specialist failed before responding (bad_input)" in caplog.text
    # the model is never called for an image that can't be loaded
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.anyio
async def test_frontier_reraises_a_wrong_api_key():
    extractor = _frontier_raising(_status_error(anthropic, "AuthenticationError", 401))

    with pytest.raises(anthropic.AuthenticationError):
        await extractor.extract_invoice("fake.png")


@pytest.mark.anyio
async def test_frontier_rejected_request_becomes_a_logged_parse_error(caplog):
    extractor = _frontier_raising(_status_error(anthropic, "BadRequestError", 400))

    with caplog.at_level(logging.WARNING, logger=extractors.__name__):
        result = await extractor.extract_invoice("fake.png")

    assert result.invoice is None
    assert result.parse_error.startswith("rejected_input:")
    assert "Frontier failed before responding (rejected_input)" in caplog.text


@pytest.mark.anyio
async def test_frontier_requests_claude_structured_output_with_the_wire_schema():
    client = _fake_frontier_client(_invoice())

    await _frontier(client).extract_invoice("fake.png")

    client.with_structured_output.assert_called_once_with(
        WireInvoice, method="json_schema", include_raw=True
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "text",
    [
        pytest.param("not json at all", id="not-json"),
        pytest.param('{"document_type": "receipt"}', id="missing-fields"),
        pytest.param(
            '{"document_type": "receipt", "currency": "EUR", "grand_total": "12,50"}',
            id="locale-formatted-amount",
        ),
    ],
)
async def test_frontier_unparseable_reply_becomes_a_parse_error_not_an_exception(
    text, caplog
):
    # regression guard: langchain's OutputParserException categorizes as "unknown",
    # which would raise and error the job instead of sending it to review
    extractor = _frontier_replying(text)

    with caplog.at_level(logging.WARNING, logger=extractors.__name__):
        result = await extractor.extract_invoice("fake.png")

    assert result.invoice is None
    assert result.parse_error.startswith("invalid_output:")
    assert result.raw_output == text
    # pytest raises on a broken format string, so this also guards the %s
    assert "Extraction parse failure (invalid_output" in caplog.text


@pytest.mark.anyio
async def test_frontier_truncated_reply_logs_why_claude_stopped(caplog):
    truncated = '{"document_type": "receipt", "curr'
    extractor = _frontier_replying(truncated, stop_reason="max_tokens")

    with caplog.at_level(logging.WARNING, logger=extractors.__name__):
        result = await extractor.extract_invoice("fake.png")

    assert result.parse_error.startswith("invalid_output:")
    assert result.raw_output == truncated
    assert "stop_reason=max_tokens" in caplog.text


@pytest.mark.anyio
async def test_frontier_keeps_money_exact_from_string_amounts():
    text = (
        '{"document_type": "receipt", "currency": "KWD", "grand_total": "20.125",'
        ' "line_items": [{"description": "x", "line_total": "20.125"}]}'
    )

    result = await _frontier_replying(text).extract_invoice("fake.png")

    assert result.invoice.grand_total == Decimal("20.125")
    assert str(result.invoice.line_items[0].line_total) == "20.125"


# --- model name and raw output, stored on every Extraction row ---


@pytest.mark.anyio
async def test_specialist_keeps_raw_text_and_model_on_success():
    raw = _invoice().model_dump_json()
    extractor = _specialist(_fake_specialist_client(raw))

    result = await extractor.extract_invoice("fake.png")

    assert result.raw_output == raw
    assert result.model_name == "test-model"


@pytest.mark.anyio
async def test_specialist_keeps_raw_text_when_parsing_fails():
    # the raw reply is the only evidence of what went wrong
    extractor = _specialist(_fake_specialist_client("not valid json"))

    result = await extractor.extract_invoice("fake.png")

    assert result.invoice is None
    assert result.raw_output == "not valid json"
    assert result.model_name == "test-model"


@pytest.mark.anyio
async def test_specialist_has_no_raw_output_when_no_reply_arrived():
    extractor = _specialist(
        _fake_specialist_client_raising(_status_error(openai, "BadRequestError", 400))
    )

    result = await extractor.extract_invoice("fake.png")

    assert result.parse_error.startswith("rejected_input:")
    assert result.raw_output is None
    assert result.model_name == "test-model"


@pytest.mark.anyio
async def test_frontier_raw_output_is_exactly_the_text_claude_returned():
    # content is a list of blocks: str(content) would store a Python repr instead
    text = _invoice(vendor=None).model_dump_json()

    result = await _frontier_replying(text).extract_invoice("fake.png")

    assert result.raw_output == text
    assert result.model_name == "test-model"


@pytest.mark.anyio
async def test_frontier_has_no_raw_output_when_no_reply_arrived():
    extractor = _frontier_raising(_status_error(anthropic, "BadRequestError", 400))

    result = await extractor.extract_invoice("fake.png")

    assert result.raw_output is None
    assert result.model_name == "test-model"


@pytest.mark.anyio
async def test_frontier_connectivity_error_still_propagates_for_retry():
    # guards against touching `result` in the except block (UnboundLocalError)
    exc = anthropic.APIConnectionError(request=_REQ)

    with pytest.raises(anthropic.APIConnectionError):
        await _frontier_raising(exc).extract_invoice("fake.png")
