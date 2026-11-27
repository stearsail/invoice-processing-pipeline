from contextlib import asynccontextmanager
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx
import openai
import pytest

from invoice_pipeline.workflow import extractors, nodes, runner
from invoice_pipeline.workflow.extractors import (
    ExtractionResult,
    FrontierExtractor,
    SpecialistExtractor,
)
from invoice_pipeline.workflow.graph import build_graph
from invoice_pipeline.workflow.state import ExtractionRecord
from invoice_pipeline.schema import Invoice, LineItem
from invoice_pipeline.util.reconciliation import ReconciliationIssue


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
    # parses fine, but line items (100) don't add up to the grand total (500)
    return _invoice(grand_total=Decimal("500"))


class _FakeExtractor:
    # records calls so tests can count paid frontier calls and inspect retry context
    def __init__(self, result: ExtractionResult | None = None):
        self.result = result
        self.calls = []

    async def extract_invoice(
        self, img_path: str, extra_context: str | None = None
    ) -> ExtractionResult:
        self.calls.append({"img_path": img_path, "extra_context": extra_context})
        return self.result


def _extracted(invoice: Invoice, model: str = "fake-model") -> ExtractionResult:
    return ExtractionResult(
        invoice=invoice,
        parse_error=None,
        model_name=model,
        raw_output=invoice.model_dump_json(),
    )


def _failed(parse_error: str, model: str = "fake-model") -> ExtractionResult:
    return ExtractionResult(
        invoice=None, parse_error=parse_error, model_name=model, raw_output="garbage"
    )


def _build(specialist_result=None, frontier_result=None):
    specialist = _FakeExtractor(specialist_result)
    frontier = _FakeExtractor(frontier_result)
    return build_graph(specialist=specialist, frontier=frontier), specialist, frontier


def _initial_state(**overrides):
    # mirrors run_extraction's input; the run_extraction tests below catch drift
    state = {"job_id": 1, "image": "fake.png", "extractions": [], "selected": None}
    state.update(overrides)
    return state


# the longest legitimate path is 5 node steps; a runaway retry loop fails fast
_LOOP_GUARD = {"recursion_limit": 12}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def persisted(monkeypatch, duplicates):
    # a sentinel session: any operation that isn't faked fails loudly instead of hitting a real DB
    @asynccontextmanager
    async def _no_db():
        yield object()

    monkeypatch.setattr(nodes, "session_factory", _no_db)

    calls = []

    async def _write_extractions(session, job_id, extractions):
        calls.append({"job_id": job_id, "rows": extractions})

    monkeypatch.setattr(nodes, "write_extractions", _write_extractions)
    return calls


def _rows(persisted) -> list:
    assert len(persisted) == 1, "the graph must persist exactly once per run"
    return persisted[0]["rows"]


def _final(rows) -> list:
    return [r for r in rows if r.is_final]


def _issue_messages(row) -> list[str]:
    return [issue["message"] for issue in row.issues]


@pytest.fixture
def duplicates(monkeypatch):
    # set "approved" / "pending" to make the duplicate lookups match; calls are recorded
    state = {"approved": None, "pending": None, "pending_calls": []}

    async def _find_duplicate(session, invoice_number, vendor_name):
        return state["approved"]

    async def _find_pending_duplicate(
        session, invoice_number, vendor_name, exclude_job_id
    ):
        state["pending_calls"].append(exclude_job_id)
        return state["pending"]

    monkeypatch.setattr(nodes, "find_duplicate", _find_duplicate)
    monkeypatch.setattr(nodes, "find_pending_duplicate", _find_pending_duplicate)
    return state


# --- one test per path through the graph ------------------------------------


@pytest.mark.anyio
async def test_specialist_success_persists_one_final_row(persisted):
    graph, _, frontier = _build(specialist_result=_extracted(_invoice()))

    result = await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert frontier.calls == []
    assert result["selected"] == 0
    rows = _rows(persisted)
    assert [(r.extractor, r.trigger, r.is_final) for r in rows] == [
        ("specialist", "initial", True)
    ]
    assert rows[0].issues == []
    assert persisted[0]["job_id"] == 1


@pytest.mark.anyio
async def test_specialist_parse_failure_falls_back_to_frontier(persisted):
    graph, _, frontier = _build(
        specialist_result=_failed("invalid_output: not valid json"),
        frontier_result=_extracted(_invoice()),
    )

    result = await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert len(frontier.calls) == 1
    assert result["selected"] == 1
    rows = _rows(persisted)
    assert [(r.extractor, r.trigger, r.is_final) for r in rows] == [
        ("specialist", "initial", False),
        ("frontier", "parse_fallback", True),
    ]
    # the failed attempt is kept with its error and raw output, for metrics and training
    assert rows[0].invoice_data is None
    assert rows[0].parse_error == "invalid_output: not valid json"
    assert rows[0].raw_output == "garbage"


@pytest.mark.anyio
async def test_total_failure_still_persists_every_attempt_with_none_final(persisted):
    graph, _, frontier = _build(
        specialist_result=_failed("invalid_output: not valid json"),
        frontier_result=_failed("invalid_output: frontier boom"),
    )

    result = await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert len(frontier.calls) == 1
    assert result["selected"] is None
    rows = _rows(persisted)
    assert [(r.extractor, r.parse_error) for r in rows] == [
        ("specialist", "invalid_output: not valid json"),
        ("frontier", "invalid_output: frontier boom"),
    ]
    assert _final(rows) == []
    # no record parsed, so there is nothing to reconcile and no issue to invent
    assert all(r.issues == [] for r in rows)


@pytest.mark.anyio
async def test_mismatch_retries_via_frontier_and_the_retry_becomes_final(persisted):
    graph, _, frontier = _build(
        specialist_result=_extracted(_unreconciled_invoice()),
        frontier_result=_extracted(_invoice()),
    )

    result = await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert len(frontier.calls) == 1
    assert result["selected"] == 1
    rows = _rows(persisted)
    assert [(r.extractor, r.trigger, r.is_final) for r in rows] == [
        ("specialist", "initial", False),
        ("frontier", "reconcile_retry", True),
    ]
    # each record keeps the issues found on its own invoice
    assert any("Grand total computation" in m for m in _issue_messages(rows[0]))
    assert rows[1].issues == []


@pytest.mark.anyio
async def test_mismatch_surviving_the_retry_goes_to_review_without_a_second_retry(
    persisted,
):
    graph, _, frontier = _build(
        specialist_result=_extracted(_unreconciled_invoice()),
        frontier_result=_extracted(_unreconciled_invoice()),
    )

    await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert len(frontier.calls) == 1
    rows = _rows(persisted)
    assert _final(rows) == [rows[1]]
    assert any("Grand total computation" in m for m in _issue_messages(rows[1]))


@pytest.mark.anyio
async def test_failed_review_retry_keeps_the_specialist_invoice_final(persisted):
    # regression: a failed retry must neither discard the draft nor take credit for it
    specialist_invoice = _unreconciled_invoice()
    graph, _, frontier = _build(
        specialist_result=_extracted(specialist_invoice),
        frontier_result=_failed("invalid_output: frontier boom"),
    )

    result = await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert result["selected"] == 0
    assert result["extractions"][0].invoice is specialist_invoice
    rows = _rows(persisted)
    assert [(r.extractor, r.trigger, r.is_final) for r in rows] == [
        ("specialist", "initial", True),
        ("frontier", "reconcile_retry", False),
    ]
    assert rows[0].invoice_data == specialist_invoice.model_dump(mode="json")
    # the unresolved issue still reaches the reviewer, on the record they'll see
    assert any("Grand total computation" in m for m in _issue_messages(rows[0]))
    assert rows[1].parse_error == "invalid_output: frontier boom"


@pytest.mark.anyio
async def test_failed_review_retry_calls_frontier_exactly_once(persisted):
    # regression: the budget is spent when the frontier is called, not when it
    # succeeds; otherwise the still-selected specialist draft loops back forever
    graph, _, frontier = _build(
        specialist_result=_extracted(_unreconciled_invoice()),
        frontier_result=_failed("invalid_output: frontier boom"),
    )

    await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert len(frontier.calls) == 1


@pytest.mark.anyio
async def test_frontier_fallback_with_issues_does_not_trigger_a_review_retry(
    persisted,
):
    graph, _, frontier = _build(
        specialist_result=_failed("invalid_output: not valid json"),
        frontier_result=_extracted(_unreconciled_invoice()),
    )

    await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert len(frontier.calls) == 1
    # the fallback is a fresh read, not a review retry: no issues are fed back
    assert frontier.calls[0]["extra_context"] in (None, "")
    rows = _rows(persisted)
    assert _final(rows) == [rows[1]]
    assert len(rows[1].issues) == 1


@pytest.mark.anyio
async def test_review_retry_feeds_the_selected_records_issues_to_the_frontier(
    persisted,
):
    graph, _, frontier = _build(
        specialist_result=_extracted(_unreconciled_invoice()),
        frontier_result=_extracted(_invoice()),
    )

    await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    extra_context = frontier.calls[0]["extra_context"]
    assert "Grand total computation" in extra_context
    # plain messages, not the repr of ReconciliationIssue objects
    assert "ReconciliationIssue(" not in extra_context
    # appended to the base prompt, so it must not start glued to its last word
    assert extra_context.startswith("\n")


@pytest.mark.anyio
async def test_persisted_rows_carry_model_name_and_json_safe_payloads(persisted):
    graph, _, _ = _build(
        specialist_result=_extracted(_unreconciled_invoice(), model="qwen-ft"),
        frontier_result=_extracted(_invoice(), model="claude-test"),
    )

    await graph.ainvoke(_initial_state(job_id=7), config=_LOOP_GUARD)

    rows = _rows(persisted)
    assert [r.model for r in rows] == ["qwen-ft", "claude-test"]
    assert all(r.job_id == 7 for r in rows)
    # Decimal must be serialized for the JSON column
    assert rows[1].invoice_data["grand_total"] == "100"
    assert rows[0].issues[0]["category"] == "mismatch"
    # write_extractions numbers the run; the node must not guess it
    assert all(r.run is None for r in rows)


# --- issues that a re-read can't fix skip the frontier ----------------------


@pytest.mark.anyio
async def test_duplicate_specialist_extraction_skips_the_frontier(
    duplicates, persisted
):
    graph, _, frontier = _build(specialist_result=_extracted(_invoice()))

    duplicates["approved"] = MagicMock(id=42, job_id=7)

    await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert frontier.calls == []
    rows = _rows(persisted)
    assert [i["category"] for i in rows[0].issues] == ["duplicate"]
    # names the job, not the ledger entry id: the UI is job-centric
    assert "already in the ledger (job 7)" in rows[0].issues[0]["message"]


@pytest.mark.anyio
async def test_duplicate_after_frontier_fallback_is_attached_to_the_frontier_row(
    duplicates, persisted
):
    graph, _, frontier = _build(
        specialist_result=_failed("invalid_output: not valid json"),
        frontier_result=_extracted(_invoice()),
    )

    duplicates["approved"] = MagicMock(id=42, job_id=7)

    await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert len(frontier.calls) == 1
    rows = _rows(persisted)
    assert rows[0].issues == []
    assert [i["category"] for i in rows[1].issues] == ["duplicate"]


@pytest.mark.anyio
async def test_pending_duplicate_is_flagged_without_a_frontier_call(
    duplicates, persisted
):
    graph, _, frontier = _build(specialist_result=_extracted(_invoice()))
    duplicates["pending"] = 5

    await graph.ainvoke(_initial_state(job_id=9), config=_LOOP_GUARD)

    assert frontier.calls == []
    [issue] = _rows(persisted)[0].issues
    assert issue["category"] == "duplicate"
    assert "also waiting for review (job 5)" in issue["message"]
    # the job being validated is excluded, or a re-run would match itself
    assert duplicates["pending_calls"] == [9]


@pytest.mark.anyio
async def test_an_approved_duplicate_wins_over_a_pending_one(duplicates, persisted):
    graph, _, _ = _build(specialist_result=_extracted(_invoice()))
    duplicates["approved"] = MagicMock(id=42, job_id=7)
    duplicates["pending"] = 5

    await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    [issue] = _rows(persisted)[0].issues
    assert "already in the ledger (job 7)" in issue["message"]
    # the pending lookup is skipped once the ledger already matched
    assert duplicates["pending_calls"] == []


@pytest.mark.anyio
async def test_receipt_without_a_subtotal_skips_the_frontier(persisted):
    graph, _, frontier = _build(specialist_result=_extracted(_invoice(subtotal=None)))

    await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert frontier.calls == []
    assert [i["category"] for i in _rows(persisted)[0].issues] == ["unverifiable"]


@pytest.mark.anyio
async def test_mismatch_alongside_a_duplicate_still_retries_once(duplicates, persisted):
    graph, _, frontier = _build(
        specialist_result=_extracted(_unreconciled_invoice()),
        frontier_result=_extracted(_invoice()),
    )

    duplicates["approved"] = MagicMock(id=42, job_id=7)

    await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert len(frontier.calls) == 1
    assert _final(_rows(persisted))[0].extractor == "frontier"


# --- routers and nodes in isolation -----------------------------------------


def _record(extractor="specialist", invoice=None, issues=()) -> ExtractionRecord:
    return ExtractionRecord(
        extractor=extractor,
        model="m",
        trigger="initial",
        invoice=invoice,
        parse_error=None if invoice is not None else "invalid_output: x",
        raw_output=None,
        issues=list(issues),
    )


def _issue(category: str) -> ReconciliationIssue:
    return ReconciliationIssue(category=category, message=f"{category} issue")


def test_route_after_parsing_treats_index_zero_as_a_success():
    # 0 is falsy: a truthiness check here would pay for a frontier call on every good parse
    state = _initial_state(extractions=[_record(invoice=_invoice())], selected=0)

    assert nodes.route_after_parsing(state) == "success"


def test_route_after_parsing_falls_back_when_nothing_is_selected():
    state = _initial_state(extractions=[_record()], selected=None)

    assert nodes.route_after_parsing(state) == "needs_fallback"


@pytest.mark.parametrize(
    "categories, frontier_called, expected",
    [
        ([], False, "reconciled"),
        (["unverifiable"], False, "reconciled"),
        (["duplicate"], False, "reconciled"),
        (["unverifiable", "duplicate"], False, "reconciled"),
        (["mismatch"], False, "retry_via_frontier"),
        (["unverifiable", "mismatch"], False, "retry_via_frontier"),
        (["duplicate", "mismatch"], False, "retry_via_frontier"),
        (["mismatch"], True, "reconciled"),
    ],
)
def test_route_after_reconcile_retries_only_frontier_fixable_issues(
    categories, frontier_called, expected
):
    records = [_record(invoice=_invoice(), issues=[_issue(c) for c in categories])]
    if frontier_called:
        # a failed frontier call still spends the budget even though it isn't selected
        records.append(_record(extractor="frontier"))
    state = _initial_state(extractions=records, selected=0)

    assert nodes.route_after_reconcile(state) == expected


def test_route_after_reconcile_reports_total_failure_when_nothing_is_selected():
    state = _initial_state(extractions=[_record(), _record("frontier")], selected=None)

    assert nodes.route_after_reconcile(state) == "total_failure"


@pytest.mark.anyio
async def test_validate_reconcile_returns_no_update_when_nothing_is_selected():
    state = _initial_state(extractions=[_record()], selected=None)

    assert await nodes.validate_reconcile(state) == {}


@pytest.mark.anyio
async def test_validate_reconcile_does_not_mutate_the_incoming_state():
    original = _record(invoice=_unreconciled_invoice())
    extractions = [original]
    state = _initial_state(extractions=extractions, selected=0)

    update = await nodes.validate_reconcile(state)

    assert update["extractions"] is not extractions
    assert update["extractions"][0] is not original
    assert update["extractions"][0].issues
    assert extractions == [original]
    assert original.issues == []


@pytest.mark.anyio
async def test_failed_extractor_node_leaves_selected_out_of_its_update():
    # returning selected=None would erase an earlier, still-valid selection
    frontier = _FakeExtractor(_failed("invalid_output: boom"))
    state = _initial_state(
        extractions=[
            _record(invoice=_unreconciled_invoice(), issues=[_issue("mismatch")])
        ],
        selected=0,
    )

    update = await nodes.frontier_review_fallback(frontier, state)

    assert "selected" not in update
    assert len(update["extractions"]) == 2


# --- through run_extraction: the runner's real initial state ----------------


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("specialist_result", "frontier_result", "frontier_calls", "final_extractor"),
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
async def test_run_extraction_routing_and_final_extraction(
    monkeypatch,
    persisted,
    specialist_result,
    frontier_result,
    frontier_calls,
    final_extractor,
):
    graph, _, frontier = _build(specialist_result, frontier_result)
    monkeypatch.setattr(runner, "graph", graph.with_config(_LOOP_GUARD))

    result = await runner.run_extraction(job_id=1, img_path="fake.png")

    assert len(frontier.calls) == frontier_calls
    final = _final(_rows(persisted))
    if final_extractor is None:
        assert result.status == "extraction_failed"
        assert result.error == "not valid json; frontier boom"
        assert final == []
    else:
        assert result.status == "complete"
        assert [r.extractor for r in final] == [final_extractor]


# --- what the real extractors' error handling does to a document ------------

_REQ = httpx.Request("POST", "http://api.test/v1")


def _status_error(sdk, cls_name: str, code: int):
    cls = getattr(sdk, cls_name)
    return cls(cls_name, response=httpx.Response(code, request=_REQ), body=None)


def _real_specialist_raising(monkeypatch, exc) -> SpecialistExtractor:
    monkeypatch.setattr(extractors, "load_image_b64", lambda p: ("b64", "image/png"))
    specialist = SpecialistExtractor("http://vllm.test/v1", "key", "m", timeout=5)
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=exc)
    specialist._client = client
    return specialist


def _real_frontier_raising(monkeypatch, exc) -> FrontierExtractor:
    monkeypatch.setattr(extractors, "load_image_b64", lambda p: ("b64", "image/png"))
    frontier = FrontierExtractor("test-model", timeout=5)
    structured = MagicMock()
    structured.ainvoke = AsyncMock(side_effect=exc)
    client = MagicMock()
    client.with_structured_output.return_value = structured
    frontier._client = client
    return frontier


@pytest.mark.anyio
async def test_specialist_rejected_request_falls_back_to_the_frontier(
    monkeypatch, persisted
):
    specialist = _real_specialist_raising(
        monkeypatch, _status_error(openai, "BadRequestError", 400)
    )
    frontier = _FakeExtractor(_extracted(_invoice()))
    graph = build_graph(specialist=specialist, frontier=frontier)

    await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert len(frontier.calls) == 1
    rows = _rows(persisted)
    assert rows[0].parse_error.startswith("rejected_input")
    # no reply came back, so there is no raw output to keep
    assert rows[0].raw_output is None
    assert rows[0].model == "m"
    assert _final(rows)[0].extractor == "frontier"


@pytest.mark.anyio
async def test_wrong_specialist_key_fails_the_run_without_paying_for_the_frontier(
    monkeypatch, persisted
):
    specialist = _real_specialist_raising(
        monkeypatch, _status_error(openai, "AuthenticationError", 401)
    )
    frontier = _FakeExtractor(_extracted(_invoice()))
    graph = build_graph(specialist=specialist, frontier=frontier)

    with pytest.raises(openai.AuthenticationError):
        await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert frontier.calls == []
    assert persisted == []


@pytest.mark.anyio
async def test_wrong_frontier_key_during_review_retry_fails_the_run(
    monkeypatch, persisted
):
    # nothing is written: the job errors and the whole run is retried
    specialist = _FakeExtractor(_extracted(_unreconciled_invoice()))
    frontier = _real_frontier_raising(
        monkeypatch, _status_error(anthropic, "AuthenticationError", 401)
    )
    graph = build_graph(specialist=specialist, frontier=frontier)

    with pytest.raises(anthropic.AuthenticationError):
        await graph.ainvoke(_initial_state(), config=_LOOP_GUARD)

    assert persisted == []
