from unittest.mock import AsyncMock, MagicMock

import pytest

from invoice_pipeline.workflow import runner
from invoice_pipeline.util.reconciliation import ReconciliationIssue


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _fake_graph(final_state: dict) -> MagicMock:
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=final_state)
    return graph


@pytest.mark.anyio
async def test_run_extraction_reports_complete_with_ledger_entry_id(monkeypatch):
    graph = _fake_graph({"invoice": MagicMock(), "ledger_entry_id": 7})
    monkeypatch.setattr(runner, "graph", graph)

    result = await runner.run_extraction(job_id=1, img_path="fake.png")

    assert result.status == "complete"
    assert result.ledger_entry_id == 7
    assert result.error is None


@pytest.mark.anyio
async def test_run_extraction_seeds_state_from_job_and_path(monkeypatch):
    graph = _fake_graph({"invoice": MagicMock(), "ledger_entry_id": 1})
    monkeypatch.setattr(runner, "graph", graph)

    await runner.run_extraction(job_id=42, img_path="uploads/a1b2.png")

    seeded = graph.ainvoke.await_args.kwargs["input"]
    assert seeded["job_id"] == 42
    assert seeded["image"] == "uploads/a1b2.png"
    assert seeded["invoice"] is None


@pytest.mark.anyio
async def test_run_extraction_seeds_provenance_and_retry_budget(monkeypatch):
    # route_after_reconcile reads frontier_attempted on any draft with issues,
    # and LangGraph state only holds keys someone wrote — an unseeded key is a
    # KeyError mid-graph, not a default
    graph = _fake_graph({"invoice": MagicMock(), "ledger_entry_id": 1})
    monkeypatch.setattr(runner, "graph", graph)

    await runner.run_extraction(job_id=1, img_path="fake.png")

    seeded = graph.ainvoke.await_args.kwargs["input"]
    assert seeded["extracted_by"] is None
    assert seeded["frontier_attempted"] is False


@pytest.mark.anyio
async def test_run_extraction_reports_failure_with_joined_issue_messages(monkeypatch):
    graph = _fake_graph(
        {
            "invoice": None,
            "reconciliation_issues": [
                ReconciliationIssue(
                    category="unverifiable", message="Missing subtotal"
                ),
                ReconciliationIssue(category="unverifiable", message="No line items"),
            ],
        }
    )
    monkeypatch.setattr(runner, "graph", graph)

    result = await runner.run_extraction(job_id=1, img_path="fake.png")

    assert result.status == "extraction_failed"
    assert result.ledger_entry_id is None
    assert result.error == "Missing subtotal; No line items"


@pytest.mark.anyio
async def test_run_extraction_handles_missing_ledger_entry_id(monkeypatch):
    """An invoice that reached the ledger but produced no id must not raise."""
    graph = _fake_graph({"invoice": MagicMock()})
    monkeypatch.setattr(runner, "graph", graph)

    result = await runner.run_extraction(job_id=1, img_path="fake.png")

    assert result.status == "complete"
    assert result.ledger_entry_id is None
