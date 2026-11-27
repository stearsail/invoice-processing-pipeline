import logging
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from invoice_pipeline.api.routers.ledger import (
    _to_response,
    delete_job_by_id,
    edit_job,
    get_full_ledger,
    get_job_by_id,
    get_reviewables,
    retry_job,
)
from invoice_pipeline.db.models import Extraction, Job, LedgerEntry
from invoice_pipeline.db.operations import create_job, update_job, write_extractions
from invoice_pipeline.queue.jobs import arq_job_id
from invoice_pipeline.schema import Invoice

_INVOICE_DATA = {"document_type": "receipt", "currency": "USD", "grand_total": "10.00"}


def _job(**overrides) -> Job:
    defaults = dict(
        id=1,
        status="complete",
        error=None,
        attempts=1,
        file_key="a1b2c3d4.png",
        created_at=datetime(2026, 7, 15, 12, 0, 0),
    )
    defaults.update(overrides)
    return Job(**defaults)


def _extraction_row(**overrides) -> Extraction:
    defaults = dict(
        id=4,
        job_id=1,
        run=1,
        extractor="specialist",
        model="qwen-ft",
        trigger="initial",
        invoice_data=_INVOICE_DATA,
        parse_error=None,
        issues=[{"category": "mismatch", "message": "Line items sum to 9.00"}],
        is_final=True,
        created_at=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Extraction(**defaults)


def _entry_row(**overrides) -> LedgerEntry:
    defaults = dict(
        id=9,
        job_id=1,
        source_extraction_id=4,
        invoice_number=None,
        vendor_name=None,
        issue_date=None,
        currency="USD",
        grand_total=Decimal("10.00"),
        invoice_data=_INVOICE_DATA,
        corrected=False,
        reviewed_at=datetime(2026, 7, 16, 9, 0, 0, tzinfo=UTC),
    )
    defaults.update(overrides)
    return LedgerEntry(**defaults)


# --- _to_response ------------------------------------------------------------


def test_to_response_for_a_job_with_nothing_attached():
    job = _job(status="extraction_failed", error="invalid_output: not json")

    response = _to_response(job, None, None)

    assert response.job_id == 1
    assert response.status == "extraction_failed"
    assert response.error == "invalid_output: not json"
    assert response.extraction is None
    assert response.extraction_error is None
    assert response.ledger_entry is None
    assert response.ledger_entry_error is None


def test_to_response_maps_the_extraction_the_reviewer_starts_from():
    response = _to_response(_job(), _extraction_row(), None)

    assert response.extraction.id == 4
    assert response.extraction.extractor == "specialist"
    assert response.extraction.model == "qwen-ft"
    assert response.extraction.trigger == "initial"
    assert response.extraction.invoice_data.grand_total == Decimal("10.00")
    assert response.extraction.issues[0].category == "mismatch"
    assert response.ledger_entry is None


def test_to_response_maps_an_approved_entry():
    response = _to_response(_job(), _extraction_row(), _entry_row(corrected=True))

    assert response.ledger_entry.id == 9
    assert response.ledger_entry.source_extraction_id == 4
    assert response.ledger_entry.corrected is True
    assert response.ledger_entry.reviewed_at == datetime(2026, 7, 16, 9, tzinfo=UTC)
    assert response.ledger_entry.invoice_data.document_type == "receipt"


def test_to_response_maps_an_entry_typed_in_for_a_total_failure():
    entry = _entry_row(source_extraction_id=None, corrected=None)

    response = _to_response(_job(status="extraction_failed"), None, entry)

    assert response.extraction is None
    assert response.ledger_entry.source_extraction_id is None
    assert response.ledger_entry.corrected is None


def test_to_response_isolates_a_corrupt_extraction_from_the_rest_of_the_row():
    extraction = _extraction_row(invoice_data={"this": "matches nothing"})

    response = _to_response(_job(), extraction, _entry_row())

    # one stale row must not 500 the whole list, nor blank out its other parts
    assert response.job_id == 1
    assert response.extraction is None
    assert "document_type" in response.extraction_error
    assert response.ledger_entry.id == 9
    assert response.ledger_entry_error is None


def test_to_response_isolates_a_corrupt_ledger_entry_from_the_rest_of_the_row():
    entry = _entry_row(invoice_data={"this": "matches nothing"})

    response = _to_response(_job(), _extraction_row(), entry)

    assert response.ledger_entry is None
    assert "document_type" in response.ledger_entry_error
    assert response.extraction.id == 4
    assert response.extraction_error is None


def test_to_response_accepts_a_failed_extraction_with_no_invoice():
    extraction = _extraction_row(
        invoice_data=None, parse_error="invalid_output: x", is_final=False, issues=[]
    )

    response = _to_response(_job(), extraction, None)

    assert response.extraction.invoice_data is None
    assert response.extraction.parse_error == "invalid_output: x"


# --- endpoints against a real session ---------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as s:
        yield s
    await engine.dispose()


def _invoice(**overrides) -> Invoice:
    defaults = {
        "document_type": "receipt",
        "currency": "USD",
        "grand_total": Decimal("10.00"),
    }
    defaults.update(overrides)
    return Invoice(**defaults)


async def _job_with_status(
    session, status: str, attempts: int = 1, error: str | None = None
) -> Job:
    job = await create_job(session, file_key=f"{status}.png")
    return await update_job(
        session,
        job_id=job.id,
        status_update=status,
        attempts_update=attempts,
        error_update=error,
    )


async def _extracted_job(session, invoice: Invoice) -> Job:
    # what the worker leaves behind: a complete job with one final extraction
    job = await _job_with_status(session, "complete")
    await write_extractions(
        session,
        job.id,
        [
            Extraction(
                job_id=job.id,
                extractor="specialist",
                model="qwen-ft",
                trigger="initial",
                invoice_data=invoice.model_dump(mode="json"),
                is_final=True,
            )
        ],
    )
    return job


async def _failed_job(session, attempts: int = 2, error: str = "boom") -> Job:
    job = await _job_with_status(
        session, "extraction_failed", attempts=attempts, error=error
    )
    await write_extractions(
        session,
        job.id,
        [
            Extraction(
                job_id=job.id,
                extractor="specialist",
                model="qwen-ft",
                trigger="initial",
                parse_error=error,
            )
        ],
    )
    return job


async def _entry_count(session, job_id: int) -> int:
    statement = (
        select(func.count())
        .select_from(LedgerEntry)
        .where(LedgerEntry.job_id == job_id)
    )
    return (await session.exec(statement)).one()


@pytest.mark.anyio
async def test_approving_an_extracted_job_links_the_entry_to_its_extraction(session):
    job = await _extracted_job(session, _invoice())

    response = await edit_job(job_id=job.id, invoice=_invoice(), session=session)

    # the response carries the entry just saved, not the pre-approval state
    assert response.ledger_entry is not None
    assert response.ledger_entry.source_extraction_id == response.extraction.id
    assert response.ledger_entry.corrected is False
    assert response.ledger_entry.reviewed_at is not None


@pytest.mark.anyio
async def test_approving_with_edits_marks_the_entry_corrected(session):
    job = await _extracted_job(session, _invoice())

    edited = _invoice(invoice_number="INV-42", grand_total=Decimal("99.99"))
    response = await edit_job(job_id=job.id, invoice=edited, session=session)

    assert response.ledger_entry.corrected is True
    assert response.ledger_entry.grand_total == Decimal("99.99")
    assert response.ledger_entry.invoice_number == "INV-42"
    # the model's output is kept untouched next to the human truth
    assert response.extraction.invoice_data.grand_total == Decimal("10.00")


@pytest.mark.anyio
async def test_the_same_amount_written_differently_is_not_a_correction(session):
    job = await _extracted_job(session, _invoice(grand_total=Decimal("100")))

    response = await edit_job(
        job_id=job.id, invoice=_invoice(grand_total=Decimal("100.00")), session=session
    )

    assert response.ledger_entry.corrected is False


@pytest.mark.anyio
async def test_a_discount_sign_flip_is_not_a_correction(session):
    # "-5.00" and "5.00" are the same discount once the schema normalizes it
    job = await _extracted_job(
        session, _invoice(grand_total=Decimal("95.00"), discount=Decimal("5.00"))
    )
    resubmitted = Invoice.model_validate(
        {
            "document_type": "receipt",
            "currency": "USD",
            "grand_total": "95.00",
            "discount": "-5.00",
        }
    )

    response = await edit_job(job_id=job.id, invoice=resubmitted, session=session)

    assert response.ledger_entry.corrected is False
    assert response.ledger_entry.invoice_data.discount == Decimal("5.00")


@pytest.mark.anyio
async def test_reapproving_updates_the_entry_in_place_and_compares_to_the_model(
    session,
):
    job = await _extracted_job(session, _invoice())
    first = await edit_job(
        job_id=job.id, invoice=_invoice(invoice_number="INV-1"), session=session
    )

    # back to what the model said: corrected is model-vs-human, not edit-vs-edit
    second = await edit_job(job_id=job.id, invoice=_invoice(), session=session)

    assert second.ledger_entry.id == first.ledger_entry.id
    assert first.ledger_entry.corrected is True
    assert second.ledger_entry.corrected is False
    assert second.ledger_entry.reviewed_at >= first.ledger_entry.reviewed_at
    assert await _entry_count(session, job.id) == 1


@pytest.mark.anyio
async def test_approving_a_total_failure_has_no_model_output_to_compare(session):
    job = await _failed_job(session)

    response = await edit_job(job_id=job.id, invoice=_invoice(), session=session)

    assert response.extraction is None
    assert response.ledger_entry.source_extraction_id is None
    assert response.ledger_entry.corrected is None


@pytest.mark.anyio
async def test_approving_leaves_the_pipelines_record_of_the_job_alone(session):
    # status, error and attempts describe what the pipeline did; approval is the entry
    job = await _failed_job(session, attempts=3, error="invalid_output: not json")

    response = await edit_job(job_id=job.id, invoice=_invoice(), session=session)

    assert response.status == "extraction_failed"
    assert response.error == "invalid_output: not json"
    assert response.attempts == 3


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["pending", "retrying", "error"])
async def test_approving_an_unfinished_job_is_rejected(session, status):
    # an approval saved before the extraction lands could never link to it
    job = await _job_with_status(session, status)

    with pytest.raises(HTTPException) as exc_info:
        await edit_job(job_id=job.id, invoice=_invoice(), session=session)

    assert exc_info.value.status_code == 409
    assert status in exc_info.value.detail
    assert await _entry_count(session, job.id) == 0


@pytest.mark.anyio
async def test_approving_an_unknown_job_is_a_404(session):
    with pytest.raises(HTTPException) as exc_info:
        await edit_job(job_id=999, invoice=_invoice(), session=session)

    assert exc_info.value.status_code == 404
    assert "999" in exc_info.value.detail


@pytest.mark.anyio
async def test_approval_moves_a_job_from_the_review_queue_to_the_ledger(session):
    job = await _extracted_job(session, _invoice())
    assert [r.job_id for r in await get_reviewables(session=session)] == [job.id]
    assert await get_full_ledger(session=session) == []

    await edit_job(job_id=job.id, invoice=_invoice(), session=session)

    assert await get_reviewables(session=session) == []
    [resolved] = await get_full_ledger(session=session)
    assert resolved.job_id == job.id
    assert resolved.ledger_entry is not None
    assert resolved.extraction is not None


@pytest.mark.anyio
async def test_review_queue_shows_total_failures_with_no_extraction(session):
    job = await _failed_job(session)

    [item] = await get_reviewables(session=session)

    assert item.job_id == job.id
    assert item.extraction is None
    assert item.ledger_entry is None


@pytest.mark.anyio
async def test_get_job_by_id_returns_the_job_with_everything_attached(session):
    job = await _extracted_job(session, _invoice())
    await edit_job(job_id=job.id, invoice=_invoice(), session=session)

    response = await get_job_by_id(job_id=job.id, session=session)

    assert response.job_id == job.id
    assert response.extraction is not None
    assert response.ledger_entry is not None


@pytest.mark.anyio
async def test_get_job_by_id_is_a_404_for_an_unknown_job(session):
    with pytest.raises(HTTPException) as exc_info:
        await get_job_by_id(job_id=999, session=session)

    assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_delete_removes_an_approved_job_and_echoes_what_was_deleted(session):
    job = await _extracted_job(session, _invoice())
    await edit_job(job_id=job.id, invoice=_invoice(), session=session)

    response = await delete_job_by_id(job_id=job.id, session=session)

    assert response.job_id == job.id
    assert response.ledger_entry is not None
    assert response.extraction is not None
    with pytest.raises(HTTPException) as exc_info:
        await get_job_by_id(job_id=job.id, session=session)
    assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_delete_is_a_404_for_an_unknown_job(session):
    with pytest.raises(HTTPException) as exc_info:
        await delete_job_by_id(job_id=999, session=session)

    assert exc_info.value.status_code == 404


# --- retry -------------------------------------------------------------------


async def _errored_job(session) -> Job:
    return await _job_with_status(session, "error", attempts=3, error="TimeoutError")


@pytest.mark.anyio
async def test_retry_job_enqueues_with_the_derived_arq_id(session):
    job = await _errored_job(session)
    pool = AsyncMock()

    response = await retry_job(job_id=job.id, session=session, pool=pool)

    assert response.status == "pending"
    assert response.error is None
    pool.enqueue_job.assert_awaited_once_with(
        "process_job", job.id, job.file_key, _job_id=arq_job_id(job.id)
    )


@pytest.mark.anyio
async def test_retry_job_of_a_job_arq_still_has_logs_and_returns_pending(
    session, caplog
):
    job = await _errored_job(session)
    pool = AsyncMock()
    pool.enqueue_job.return_value = None

    with caplog.at_level(logging.WARNING, logger="invoice_pipeline.api.routers.ledger"):
        response = await retry_job(job_id=job.id, session=session, pool=pool)

    assert response.status == "pending"
    assert arq_job_id(job.id) in caplog.text


@pytest.mark.anyio
async def test_retry_job_refuses_a_job_that_is_not_in_error(session):
    job = await _job_with_status(session, "complete")
    pool = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        await retry_job(job_id=job.id, session=session, pool=pool)

    assert exc_info.value.status_code == 409
    pool.enqueue_job.assert_not_awaited()
