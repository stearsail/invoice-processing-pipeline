from unittest.mock import AsyncMock, MagicMock

import pytest

from invoice_pipeline.workflow import runner
from invoice_pipeline.workflow.state import ExtractionRecord


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _fake_graph(final_state: dict) -> MagicMock:
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=final_state)
    return graph


def _record(invoice=None, parse_error=None) -> ExtractionRecord:
    return ExtractionRecord(
        extractor="specialist",
        model="m",
        trigger="initial",
        invoice=invoice,
        parse_error=parse_error,
        raw_output=None,
    )


@pytest.mark.anyio
async def test_run_extraction_reports_complete_when_an_invoice_is_selected(
    monkeypatch,
):
    graph = _fake_graph({"extractions": [_record(invoice=MagicMock())], "selected": 0})
    monkeypatch.setattr(runner, "graph", graph)

    result = await runner.run_extraction(job_id=1, img_path="fake.png")

    assert result.status == "complete"
    assert result.error is None


@pytest.mark.anyio
async def test_run_extraction_seeds_every_key_the_nodes_read(monkeypatch):
    # LangGraph state only holds keys someone wrote: an unseeded key is a KeyError mid-graph
    graph = _fake_graph({"extractions": [_record(invoice=MagicMock())], "selected": 0})
    monkeypatch.setattr(runner, "graph", graph)

    await runner.run_extraction(job_id=42, img_path="uploads/a1b2.png")

    assert graph.ainvoke.await_args.kwargs["input"] == {
        "job_id": 42,
        "image": "uploads/a1b2.png",
        "extractions": [],
        "selected": None,
    }


@pytest.mark.anyio
async def test_run_extraction_reports_failure_with_joined_parse_errors(monkeypatch):
    graph = _fake_graph(
        {
            "extractions": [
                _record(parse_error="invalid_output: not json"),
                _record(parse_error="rejected_input: 400"),
            ],
            "selected": None,
        }
    )
    monkeypatch.setattr(runner, "graph", graph)

    result = await runner.run_extraction(job_id=1, img_path="fake.png")

    assert result.status == "extraction_failed"
    assert result.error == "invalid_output: not json; rejected_input: 400"
