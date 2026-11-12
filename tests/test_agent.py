from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from invoice_pipeline.workflow import nodes, runner
from invoice_pipeline.workflow.extractors import ExtractionResult
from invoice_pipeline.workflow.graph import build_graph
from invoice_pipeline.schema import Invoice, LineItem


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


def _unreconciled_invoice() -> Invoice:
    """Parses fine, but line items (100) don't add up to the grand total (500)."""
    return _invoice(grand_total=Decimal("500"))


class _FakeExtractor:
    """Stands in for a real extractor at the graph boundary.

    Returns a canned ExtractionResult and records how it was called, so tests
    can assert on how many (paid) calls were made and on the context fed back
    during the review retry.
    """

    def __init__(self, result: ExtractionResult | None = None):
        self.result = result
        self.calls = []

    async def extract_invoice(
        self, img_path: str, extra_context: str | None = None
    ) -> ExtractionResult:
        self.calls.append({"img_path": img_path, "extra_context": extra_context})
        return self.result


def _extracted(invoice: Invoice) -> ExtractionResult:
    return ExtractionResult(invoice=invoice, parse_error=None)


def _failed(parse_error: str) -> ExtractionResult:
    return ExtractionResult(invoice=None, parse_error=parse_error)


def _build(specialist_result=None, frontier_result=None):
    """Compile a graph wired to fake extractors."""
    specialist = _FakeExtractor(specialist_result)
    frontier = _FakeExtractor(frontier_result)
    return build_graph(specialist=specialist, frontier=frontier), specialist, frontier


def _initial_state(**overrides):
    # Mirrors run_extraction's input. The runner-level tests at the bottom of
    # this file go through run_extraction itself, so if the two ever drift
    # (e.g. a new state key the router reads but the runner never seeds),
    # those tests fail even when these graph-level ones don't.
    state = {
        "job_id": 1,
        "image": "fake.png",
        "invoice": None,
        "extracted_by": None,
        "frontier_attempted": False,
    }
    state.update(overrides)
    return state


# A runaway retry loop would otherwise spin for LangGraph's default limit
# (~10k steps, ~5k fake frontier calls) before failing. Cap it so a regression
# fails fast. The longest legitimate path is 5 node steps.
_LOOP_GUARD = {"recursion_limit": 12}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def no_real_ledger(monkeypatch):
    async def _find_duplicate(session, invoice_number, vendor_name):
        return None

    monkeypatch.setattr(nodes, "find_duplicate", _find_duplicate)

    written = []

    async def _write(
        session,
        job_id,
        invoice,
        needs_review=False,
        review_reason=None,
        extracted_by=None,
    ):
        written.append(
            {
                "job_id": job_id,
                "invoice": invoice,
                "needs_review": needs_review,
                "review_reason": review_reason,
                "extracted_by": extracted_by,
            }
        )
        return MagicMock(id=1)

    monkeypatch.setattr(nodes, "write_entry", _write)
    return written


# --- routing and provenance, one test per path -----------------------------
# Assertions target what outlives the run — the ledger row and how many
# frontier calls were paid for — rather than intermediate state.


@pytest.mark.anyio
async def test_specialist_success_reconciles_and_writes_ledger(no_real_ledger):
    graph, _, frontier = _build(specialist_result=_extracted(_invoice()))

    result = await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert result["reconciliation_issues"] == []
    assert frontier.calls == []
    assert result["frontier_attempted"] is False
    assert len(no_real_ledger) == 1
    # Every entry gets a human confirmation pass, even a clean reconciliation —
    # needs_review only flips to False once a verifier actually saves it.
    assert no_real_ledger[0]["needs_review"] is True
    assert no_real_ledger[0]["review_reason"] == []
    assert no_real_ledger[0]["job_id"] == 1
    assert no_real_ledger[0]["extracted_by"] == "specialist"


@pytest.mark.anyio
async def test_specialist_parse_failure_falls_back_to_frontier_and_succeeds(
    no_real_ledger,
):
    graph, _, frontier = _build(
        specialist_result=_failed("Pydantic Validation error: not valid json"),
        frontier_result=_extracted(_invoice()),
    )

    result = await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert result["reconciliation_issues"] == []
    assert len(frontier.calls) == 1
    assert len(no_real_ledger) == 1
    assert no_real_ledger[0]["needs_review"] is True
    assert no_real_ledger[0]["extracted_by"] == "frontier"


@pytest.mark.anyio
async def test_both_specialist_and_frontier_extraction_fail_skips_ledger_write(
    no_real_ledger,
):
    graph, _, frontier = _build(
        specialist_result=_failed("Pydantic Validation error: not valid json"),
        frontier_result=_failed("Frontier structured output error: frontier boom"),
    )

    result = await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert result["invoice"] is None
    # neither model produced anything, so nobody gets credited
    assert result["extracted_by"] is None
    # ...but the frontier call was still made (and paid for)
    assert len(frontier.calls) == 1
    assert result["frontier_attempted"] is True
    assert len(result["reconciliation_issues"]) == 1
    assert result["reconciliation_issues"][0].category == "unverifiable"
    assert (
        result["reconciliation_issues"][0].message
        == "Extraction failed entirely — Frontier structured output error: frontier boom"
    )
    # total failure routes straight to END — nothing to write, no ledger entry created
    assert no_real_ledger == []


@pytest.mark.anyio
async def test_specialist_reconciliation_failure_retries_via_frontier_and_resolves(
    no_real_ledger,
):
    graph, _, frontier = _build(
        specialist_result=_extracted(_unreconciled_invoice()),
        frontier_result=_extracted(_invoice()),
    )

    result = await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert result["reconciliation_issues"] == []
    assert len(frontier.calls) == 1
    assert len(no_real_ledger) == 1
    assert no_real_ledger[0]["needs_review"] is True
    assert no_real_ledger[0]["extracted_by"] == "frontier"


@pytest.mark.anyio
async def test_reconciliation_failure_persists_after_frontier_retry_goes_to_human_review(
    no_real_ledger,
):
    frontier_invoice = _unreconciled_invoice()
    graph, _, frontier = _build(
        specialist_result=_extracted(_unreconciled_invoice()),
        frontier_result=_extracted(frontier_invoice),
    )

    result = await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert len(result["reconciliation_issues"]) == 1
    # one retry only — still-unreconciled frontier output goes to a human
    assert len(frontier.calls) == 1
    assert len(no_real_ledger) == 1
    assert no_real_ledger[0]["needs_review"] is True
    # the retry *did* produce the invoice that was kept, so it's credited
    assert no_real_ledger[0]["invoice"] is frontier_invoice
    assert no_real_ledger[0]["extracted_by"] == "frontier"
    reasons = no_real_ledger[0]["review_reason"]
    assert any("Grand total computation" in r["message"] for r in reasons)


@pytest.mark.anyio
async def test_failed_review_retry_keeps_specialist_invoice_and_credits_specialist(
    no_real_ledger,
):
    """A failed review retry must neither discard the draft it was retrying
    nor take credit for it.

    Regression: the review node used to set the provenance unconditionally, so
    a specialist draft kept after a failed frontier retry was written to the
    ledger as extracted_by="frontier" — skewing per-model accuracy metrics and
    misleading the reviewer about what they're looking at.
    """
    specialist_invoice = _unreconciled_invoice()
    graph, _, frontier = _build(
        specialist_result=_extracted(specialist_invoice),
        frontier_result=_failed("Frontier structured output error: frontier boom"),
    )

    result = await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert result["invoice"] is specialist_invoice
    assert len(no_real_ledger) == 1
    assert no_real_ledger[0]["needs_review"] is True
    assert no_real_ledger[0]["invoice"] is specialist_invoice
    assert no_real_ledger[0]["extracted_by"] == "specialist"
    # the unresolved issue still reaches the reviewer
    reasons = no_real_ledger[0]["review_reason"]
    assert any("Grand total computation" in r["message"] for r in reasons)


@pytest.mark.anyio
async def test_failed_review_retry_calls_frontier_exactly_once(no_real_ledger):
    """The retry budget is spent when the frontier is *called*, not when it
    succeeds.

    Regression guard: if frontier_attempted were only set on success, a failed
    retry leaves the draft unreconciled with the budget apparently unspent, so
    the router sends it back to the frontier forever (observed: ~5,000 calls
    before LangGraph's recursion limit, x3 worker retries).
    """
    graph, _, frontier = _build(
        specialist_result=_extracted(_unreconciled_invoice()),
        frontier_result=_failed("Frontier structured output error: frontier boom"),
    )

    result = await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert len(frontier.calls) == 1
    assert result["frontier_attempted"] is True


@pytest.mark.anyio
async def test_frontier_fallback_with_issues_does_not_trigger_a_review_retry(
    no_real_ledger,
):
    """Either frontier node spends the budget: once the frontier produced the
    draft (after a specialist parse failure), a reconciliation issue goes to a
    human rather than back to the frontier."""
    frontier_invoice = _unreconciled_invoice()
    graph, _, frontier = _build(
        specialist_result=_failed("Pydantic Validation error: not valid json"),
        frontier_result=_extracted(frontier_invoice),
    )

    result = await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert len(frontier.calls) == 1
    # the fallback extraction is not a review retry — no issues were fed back
    assert frontier.calls[0]["extra_context"] in (None, "")
    assert len(result["reconciliation_issues"]) == 1
    assert len(no_real_ledger) == 1
    assert no_real_ledger[0]["invoice"] is frontier_invoice
    assert no_real_ledger[0]["extracted_by"] == "frontier"


@pytest.mark.anyio
async def test_duplicate_after_frontier_fallback_goes_to_human_review(
    monkeypatch, no_real_ledger
):
    graph, _, frontier = _build(
        specialist_result=_failed("Pydantic Validation error: not valid json"),
        frontier_result=_extracted(_invoice()),
    )

    async def _dup(session, invoice_number, vendor_name):
        return MagicMock(id=42)

    monkeypatch.setattr(nodes, "find_duplicate", _dup)

    result = await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert any(
        issue.category == "duplicate" for issue in result["reconciliation_issues"]
    )
    assert len(frontier.calls) == 1
    assert len(no_real_ledger) == 1
    assert no_real_ledger[0]["needs_review"] is True
    assert no_real_ledger[0]["extracted_by"] == "frontier"
    reasons = no_real_ledger[0]["review_reason"]
    assert any(r["category"] == "duplicate" for r in reasons)


@pytest.mark.anyio
async def test_ledger_write_returns_entry_id_in_state(no_real_ledger):
    graph, _, _ = _build(specialist_result=_extracted(_invoice()))

    result = await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert result["ledger_entry_id"] == 1


@pytest.mark.anyio
async def test_review_retry_feeds_reconciliation_issues_back_to_frontier(
    no_real_ledger,
):
    graph, _, frontier = _build(
        specialist_result=_extracted(_unreconciled_invoice()),
        frontier_result=_extracted(_invoice()),
    )

    await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert len(frontier.calls) == 1
    extra_context = frontier.calls[0]["extra_context"]
    assert extra_context is not None
    assert "Grand total computation" in extra_context


# --- through run_extraction: the runner's real initial state ----------------
# These drive the fake-wired graph via run_extraction, so they exercise the
# state the runner actually seeds instead of _initial_state(). Regression: the
# router reads frontier_attempted, and when the runner didn't seed it every
# specialist draft with an issue raised KeyError (-> worker retries -> error).


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("specialist_result", "frontier_result", "frontier_calls", "ledger_extracted_by"),
    [
        pytest.param(
            _extracted(_invoice()), None, 0, "specialist", id="specialist-clean"
        ),
        pytest.param(
            _extracted(_unreconciled_invoice()),
            _extracted(_invoice()),
            1,
            "frontier",
            id="specialist-issues-retry-resolves",
        ),
        pytest.param(
            _extracted(_unreconciled_invoice()),
            _failed("frontier boom"),
            1,
            "specialist",
            id="specialist-issues-retry-fails",
        ),
        pytest.param(
            _failed("not valid json"),
            _extracted(_invoice()),
            1,
            "frontier",
            id="specialist-unparseable-frontier-extracts",
        ),
        pytest.param(
            _failed("not valid json"),
            _extracted(_unreconciled_invoice()),
            1,
            "frontier",
            id="specialist-unparseable-frontier-has-issues",
        ),
        pytest.param(
            _failed("not valid json"),
            _failed("frontier boom"),
            1,
            None,
            id="both-fail",
        ),
    ],
)
async def test_run_extraction_routing_and_provenance(
    monkeypatch,
    no_real_ledger,
    specialist_result,
    frontier_result,
    frontier_calls,
    ledger_extracted_by,
):
    graph, _, frontier = _build(specialist_result, frontier_result)
    monkeypatch.setattr(runner, "graph", graph.with_config(_LOOP_GUARD))

    result = await runner.run_extraction(job_id=1, img_path="fake.png")

    assert len(frontier.calls) == frontier_calls
    if ledger_extracted_by is None:
        assert result.status == "extraction_failed"
        assert no_real_ledger == []
    else:
        assert result.status == "complete"
        assert len(no_real_ledger) == 1
        assert no_real_ledger[0]["extracted_by"] == ledger_extracted_by
