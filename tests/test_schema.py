import json
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from invoice_pipeline.schema import Invoice, WireInvoice

FULL_INVOICE = {
    "document_type": "invoice",
    "vendor": {
        "name": "SC Exemplu SRL",
        "address": "Str. Victoriei 12, Bucuresti",
        "tax_id": "RO12345678",
        "iban": "RO49AAAA1B31007593840000",
    },
    "customer": {
        "name": "Acme GmbH",
        "address": "Hauptstrasse 5, Berlin",
        "tax_id": "DE999999999",
    },
    "invoice_number": "INV-2026-0042",
    "issue_date": "2026-07-01",
    "due_date": "2026-07-31",
    "currency": "RON",
    "line_items": [
        {
            "description": "Consulting services",
            "quantity": "10",
            "unit_price": "150.00",
            "line_total": "1500.00",
        },
        {
            "description": None,
            "quantity": "1.5",
            "unit_price": "99.90",
            "line_total": "149.85",
        },
    ],
    "grand_total": "1649.85",
    "confidence_notes": ["due_date was printed faintly"],
}

MINIMAL_RECEIPT = {
    "document_type": "receipt",
    "currency": "EUR",
    "grand_total": "12.50",
}


def test_full_invoice_parses():
    invoice = Invoice.model_validate(FULL_INVOICE)

    assert invoice.document_type == "invoice"
    assert invoice.vendor is not None
    assert invoice.vendor.name == "SC Exemplu SRL"
    assert invoice.issue_date == date(2026, 7, 1)
    assert invoice.grand_total == Decimal("1649.85")
    assert len(invoice.line_items) == 2
    assert invoice.line_items[0].unit_price == Decimal("150.00")
    assert invoice.line_items[1].description is None


def test_minimal_receipt_parses_with_defaults():
    receipt = Invoice.model_validate(MINIMAL_RECEIPT)

    assert receipt.vendor is None
    assert receipt.customer is None
    assert receipt.invoice_number is None
    assert receipt.issue_date is None
    assert receipt.line_items == []
    assert receipt.confidence_notes == []


@pytest.mark.parametrize(
    "corruption",
    [
        pytest.param({"grand_total": None}, id="missing-grand-total"),
        pytest.param({"issue_date": "01/07/2026"}, id="non-iso-date"),
        pytest.param({"currency": "eur"}, id="lowercase-currency"),
        pytest.param({"currency": "EURO"}, id="four-letter-currency"),
        pytest.param({"document_type": "quote"}, id="unknown-document-type"),
        pytest.param({"total_amount": "10.00"}, id="hallucinated-extra-key"),
    ],
)
def test_rejects_malformed_input(corruption):
    data = {**MINIMAL_RECEIPT, **corruption}
    data = {key: value for key, value in data.items() if value is not None}

    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_json_round_trip_preserves_values_exactly():
    original = Invoice.model_validate(FULL_INVOICE)

    restored = Invoice.model_validate_json(original.model_dump_json())

    assert restored == original
    assert restored.grand_total == Decimal("1649.85")
    assert restored.line_items[1].line_total == Decimal("149.85")
    assert restored.issue_date == date(2026, 7, 1)


# --- discount sign convention -------------------------------------------------
# discount is always stored as a non-negative magnitude: reconcile() subtracts
# it, so a printed "-5.40" and "5.40" must mean the same thing whichever path
# (specialist wire string, frontier JSON number, human edit) the value came from.


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("5.00", "5.00", id="positive-string"),
        pytest.param("-5.00", "5.00", id="negative-string"),
        pytest.param(5.4, "5.4", id="positive-float"),
        pytest.param(-5.4, "5.4", id="negative-float"),
        pytest.param(-7, "7", id="negative-int"),
        pytest.param(Decimal("-12.000"), "12.000", id="negative-decimal"),
        pytest.param("0", "0", id="zero"),
    ],
)
def test_discount_is_stored_as_a_non_negative_decimal(raw, expected):
    invoice = Invoice.model_validate({**MINIMAL_RECEIPT, "discount": raw})

    # an after-validator's return value is not re-validated, so pin the type:
    # returning a str here would silently break reconcile()'s arithmetic
    assert isinstance(invoice.discount, Decimal)
    assert invoice.discount == Decimal(expected)
    assert not invoice.discount.is_signed()


def test_discount_normalization_preserves_scale():
    invoice = Invoice.model_validate({**MINIMAL_RECEIPT, "discount": "-12.000"})

    assert str(invoice.discount) == "12.000"


def test_negative_zero_discount_loses_its_sign():
    invoice = Invoice.model_validate({**MINIMAL_RECEIPT, "discount": "-0"})

    assert invoice.discount == Decimal("0")
    assert not invoice.discount.is_signed()


def test_explicit_null_discount_stays_none():
    invoice = Invoice.model_validate({**MINIMAL_RECEIPT, "discount": None})

    assert invoice.discount is None


def test_omitted_discount_defaults_to_none():
    invoice = Invoice.model_validate(MINIMAL_RECEIPT)

    assert invoice.discount is None


def test_discount_normalized_when_constructed_directly():
    # Invoice(...) — how tests, converters and LangChain's parser build it —
    # must apply the rule too, not only model_validate
    invoice = Invoice(
        document_type="receipt",
        currency="EUR",
        grand_total=Decimal("10"),
        discount=Decimal("-2.50"),
    )

    assert invoice.discount == Decimal("2.50")


def test_discount_normalized_from_tool_call_json_numbers():
    # the frontier path: Anthropic tool-use args arrive as JSON numbers, which
    # never went through the old string-only removeprefix("-") fix
    payload = json.loads(
        '{"document_type": "receipt", "currency": "USD",'
        ' "grand_total": 94.6, "subtotal": 100, "discount": -5.4}'
    )

    invoice = Invoice.model_validate(payload)

    assert invoice.discount == Decimal("5.4")


def test_discount_rule_does_not_touch_other_signed_amounts():
    # negative totals (refunds) and negative line items (promo lines) carry
    # real meaning and must keep their sign
    invoice = Invoice.model_validate(
        {
            **MINIMAL_RECEIPT,
            "grand_total": "-100.00",
            "subtotal": "-90.00",
            "tax": "-15.00",
            "service_charge": "-5.00",
            "discount": "-10.00",
            "line_items": [
                {"description": "Refund", "line_total": "-88.00"},
                {"description": "Promo", "unit_price": "-2.00", "line_total": "-2.00"},
            ],
        }
    )

    assert invoice.grand_total == Decimal("-100.00")
    assert invoice.subtotal == Decimal("-90.00")
    assert invoice.tax == Decimal("-15.00")
    assert invoice.service_charge == Decimal("-5.00")
    assert invoice.line_items[0].line_total == Decimal("-88.00")
    assert invoice.line_items[1].unit_price == Decimal("-2.00")
    assert invoice.line_items[1].line_total == Decimal("-2.00")
    assert invoice.discount == Decimal("10.00")


def test_discount_normalization_is_stable_across_json_round_trip():
    original = Invoice.model_validate({**MINIMAL_RECEIPT, "discount": "-3.25"})

    restored = Invoice.model_validate_json(original.model_dump_json())

    assert restored == original
    assert restored.discount == Decimal("3.25")


def test_discount_normalization_does_not_change_the_wire_schema():
    # the validator is runtime-only; the grammar sent to vLLM must still admit
    # a leading "-" so the model is never forced into a stall on a printed sign
    schema = WireInvoice.model_json_schema()
    discount_branches = schema["properties"]["discount"]["anyOf"]

    assert {"type": "string", "pattern": r"^-?\d+(\.\d+)?$"} in discount_branches
