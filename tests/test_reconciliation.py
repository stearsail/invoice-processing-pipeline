from decimal import Decimal

import pytest

from invoice_pipeline.util.reconciliation import reconcile
from invoice_pipeline.schema import Invoice, LineItem


def _invoice(**overrides) -> Invoice:
    defaults = {
        "document_type": "receipt",
        "currency": "USD",
        "grand_total": Decimal("110"),
        "subtotal": Decimal("100"),
        "tax": Decimal("10"),
        "line_items": [
            LineItem(
                description="Item A",
                quantity=Decimal("1"),
                unit_price=Decimal("100"),
                line_total=Decimal("100"),
            ),
        ],
    }
    defaults.update(overrides)
    return Invoice(**defaults)


def test_reconciles_when_everything_matches():
    assert reconcile(_invoice()) == []


def test_missing_subtotal_is_unverifiable():
    invoice = _invoice(subtotal=None)
    issues = reconcile(invoice)
    assert len(issues) == 1
    assert issues[0].category == "unverifiable"
    assert issues[0].message == "Missing subtotal"


def test_no_line_items_is_unverifiable():
    invoice = _invoice(line_items=[])
    issues = reconcile(invoice)
    assert len(issues) == 1
    assert issues[0].category == "unverifiable"
    assert issues[0].message == "No line items"


def test_missing_line_total_is_unverifiable_and_skips_math():
    invoice = _invoice(
        line_items=[
            LineItem(
                description="Item A",
                quantity=Decimal("1"),
                unit_price=Decimal("100"),
                line_total=None,
            ),
        ]
    )
    issues = reconcile(invoice)
    assert len(issues) == 1
    assert issues[0].category == "unverifiable"
    assert issues[0].message == "Missing line total for line 1"


def test_missing_line_total_index_is_one_based():
    invoice = _invoice(
        line_items=[
            LineItem(
                description="Item A",
                quantity=Decimal("1"),
                unit_price=Decimal("100"),
                line_total=Decimal("100"),
            ),
            LineItem(
                description="Item B",
                quantity=Decimal("1"),
                unit_price=Decimal("50"),
                line_total=None,
            ),
        ]
    )
    issues = reconcile(invoice)
    assert len(issues) == 1
    assert issues[0].message == "Missing line total for line 2"


def test_line_items_not_matching_subtotal_is_flagged():
    invoice = _invoice(
        line_items=[
            LineItem(
                description="Item A",
                quantity=Decimal("1"),
                unit_price=Decimal("50"),
                line_total=Decimal("50"),
            ),
        ],
    )
    issues = reconcile(invoice)
    assert len(issues) == 1
    assert issues[0].category == "mismatch"
    assert "Line items sum to 50" in issues[0].message
    assert "Subtotal is 100" in issues[0].message


def test_grand_total_not_matching_is_flagged():
    invoice = _invoice(grand_total=Decimal("500"))
    issues = reconcile(invoice)
    assert len(issues) == 1
    assert issues[0].category == "mismatch"
    assert "Grand total computation: 110" in issues[0].message
    assert "not 500" in issues[0].message


def test_small_rounding_difference_within_tolerance_passes():
    # 1 line item -> grand-total tolerance is (1 + 1) * 0.01 = 0.02
    invoice = _invoice(grand_total=Decimal("110.02"))
    assert reconcile(invoice) == []


def test_zero_subtotal_tolerates_small_rounding_in_line_items():
    # under the old tolerance (subtotal * 1%), a subtotal of 0 made the
    # tolerance exactly 0 regardless of invoice size, so even a genuine
    # 1-cent rounding difference was incorrectly flagged.
    invoice = _invoice(
        subtotal=Decimal("0"),
        tax=Decimal("0"),
        grand_total=Decimal("0"),
        line_items=[
            LineItem(
                description="Free sample",
                quantity=Decimal("1"),
                unit_price=Decimal("0.01"),
                line_total=Decimal("0.01"),
            ),
        ],
    )
    assert reconcile(invoice) == []


def test_negative_subtotal_matching_line_items_is_not_falsely_flagged():
    # under the old tolerance (subtotal * 1%), a negative subtotal made the
    # tolerance negative too, so abs(diff) > tolerance was true even for an
    # exact match.
    invoice = _invoice(
        subtotal=Decimal("-50"),
        tax=Decimal("0"),
        grand_total=Decimal("-50"),
        line_items=[
            LineItem(
                description="Discount adjustment",
                quantity=Decimal("1"),
                unit_price=Decimal("-50"),
                line_total=Decimal("-50"),
            ),
        ],
    )
    assert reconcile(invoice) == []


def test_tolerance_boundary_for_multiple_line_items_passes():
    # 3 line items -> line-item tolerance is 3 * 0.01 = 0.03
    invoice = _invoice(
        subtotal=Decimal("100.02"),
        tax=Decimal("0"),
        grand_total=Decimal("100.02"),
        line_items=[
            LineItem(
                description=f"Item {i}",
                quantity=Decimal("1"),
                unit_price=Decimal("33.33"),
                line_total=Decimal("33.33"),
            )
            for i in range(3)
        ],
    )
    # items sum to 99.99, subtotal is 100.02 -> a 0.03 difference, right at
    # the 3-line tolerance boundary
    assert reconcile(invoice) == []


def test_tolerance_scales_with_number_of_line_items():
    invoice = _invoice(
        subtotal=Decimal("100.03"),
        tax=Decimal("0"),
        grand_total=Decimal("100.03"),
        line_items=[
            LineItem(
                description=f"Item {i}",
                quantity=Decimal("1"),
                unit_price=Decimal("33.33"),
                line_total=Decimal("33.33"),
            )
            for i in range(3)
        ],
    )
    # items sum to 99.99, subtotal is 100.03 -> a 0.04 difference, past the
    # 3-line tolerance (0.03)
    issues = reconcile(invoice)
    assert len(issues) == 1
    assert issues[0].category == "mismatch"


def test_both_checks_can_fail_independently():
    invoice = _invoice(
        line_items=[
            LineItem(
                description="Item A",
                quantity=Decimal("1"),
                unit_price=Decimal("50"),
                line_total=Decimal("50"),
            ),
        ],
        grand_total=Decimal("500"),
    )
    assert len(reconcile(invoice)) == 2


@pytest.mark.parametrize(
    "discount",
    [
        pytest.param(Decimal("5"), id="positive-decimal"),
        pytest.param(Decimal("-5"), id="negative-decimal"),
        pytest.param("-5.00", id="negative-wire-string"),
        pytest.param(-5.0, id="negative-tool-call-float"),
    ],
)
def test_discount_is_subtracted_whatever_sign_it_was_given_with(discount):
    # 100 + 10 tax - 5 discount = 105. Before the schema rule, a negative
    # discount was *added* (100 + 10 - (-5) = 115) and falsely flagged.
    invoice = Invoice.model_validate(
        {
            "document_type": "receipt",
            "currency": "USD",
            "grand_total": "105",
            "subtotal": "100",
            "tax": "10",
            "discount": discount,
            "line_items": [{"line_total": "100"}],
        }
    )

    assert reconcile(invoice) == []


def test_correct_discounted_receipt_is_not_flagged():
    # regression: an after-validator that fell through on positive values
    # returned None, deleting the discount and flagging this as a 5.00 mismatch
    invoice = _invoice(
        grand_total=Decimal("105"), tax=Decimal("10"), discount=Decimal("5")
    )

    assert invoice.discount == Decimal("5")
    assert reconcile(invoice) == []


def test_wrong_grand_total_is_still_flagged_with_a_discount():
    # normalizing the sign must not make the check permissive
    invoice = _invoice(
        grand_total=Decimal("115"), tax=Decimal("10"), discount=Decimal("-5")
    )

    issues = reconcile(invoice)

    assert len(issues) == 1
    assert issues[0].category == "mismatch"
    assert "Grand total computation: 105" in issues[0].message
