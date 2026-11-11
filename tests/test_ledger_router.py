from datetime import datetime
from decimal import Decimal

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from invoice_pipeline.api.routers.ledger import _to_response, edit_job
from invoice_pipeline.db.models import Job, LedgerEntry
from invoice_pipeline.db.operations import create_job, update_job, write_entry
from invoice_pipeline.schema import Invoice


def _job(**overrides) -> Job:
    defaults = dict(
        id=1,
        status="complete",
        error=None,
        file_key="a1b2c3d4.png",
        created_at=datetime(2026, 7, 15, 12, 0, 0),
    )
    defaults.update(overrides)
    return Job(**defaults)


def _entry(**overrides) -> LedgerEntry:
    defaults = dict(
        id=1,
        invoice_number=None,
        vendor_name=None,
        issue_date=None,
        currency="USD",
        grand_total=Decimal("10.00"),
        invoice_data={
            "document_type": "receipt",
            "currency": "USD",
            "grand_total": "10.00",
        },
        created_at=datetime(2026, 7, 15, 12, 0, 0),
        needs_review=False,
        review_reason=None,
        job_id=1,
    )
    defaults.update(overrides)
    return LedgerEntry(**defaults)


def test_to_response_with_no_entry():
    job = _job(status="extraction_failed", error="extraction failed entirely")

    response = _to_response(job, None)

    assert response.job_id == 1
    assert response.status == "extraction_failed"
    assert response.error == "extraction failed entirely"
    assert response.ledger_entry is None
    assert response.ledger_entry_error is None


def test_to_response_with_valid_entry():
    job = _job()
    entry = _entry(
        needs_review=True,
        review_reason=[{"category": "unverifiable", "message": "Missing subtotal"}],
    )

    response = _to_response(job, entry)

    assert response.ledger_entry is not None
    assert response.ledger_entry.needs_review is True
    assert len(response.ledger_entry.review_reason) == 1
    assert response.ledger_entry.review_reason[0].category == "unverifiable"
    assert response.ledger_entry.review_reason[0].message == "Missing subtotal"
    assert response.ledger_entry.invoice_data.document_type == "receipt"
    assert response.ledger_entry_error is None


def test_to_response_degrades_gracefully_on_corrupt_invoice_data():
    job = _job()
    entry = _entry(invoice_data={"this": "does not match the Invoice schema at all"})

    response = _to_response(job, entry)

    # the whole row must still come back, not raise — job-level fields stay populated
    assert response.job_id == 1
    assert response.status == "complete"
    assert response.ledger_entry is None
    assert response.ledger_entry_error is not None
    assert "document_type" in response.ledger_entry_error


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


async def _failed_job(session, attempts: int = 2, error: str = "boom") -> Job:
    """A job whose extraction produced no invoice — so no ledger entry exists."""
    job = await create_job(session, file_key="a1b2c3d4.png")
    await update_job(
        session,
        job_id=job.id,
        status_update="extraction_failed",
        attempts_update=attempts,
        error_update=error,
    )
    return job


@pytest.mark.anyio
async def test_edit_job_on_failed_job_writes_a_confirmed_entry(session):
    job = await _failed_job(session)

    response = await edit_job(job_id=job.id, invoice=_invoice(), session=session)

    assert response.status == "complete"
    assert response.ledger_entry is not None
    assert response.ledger_entry.needs_review is False
    assert response.ledger_entry.review_reason is None
    # written from scratch by a human — no model attempt to attribute this to
    assert response.ledger_entry.extracted_by is None


@pytest.mark.anyio
async def test_edit_job_on_failed_job_preserves_the_worker_attempt_count(session):
    # attempts is owned by the worker (arq's job_try); a human edit must not reset it
    job = await _failed_job(session, attempts=3)

    response = await edit_job(job_id=job.id, invoice=_invoice(), session=session)

    assert response.attempts == 3


@pytest.mark.anyio
async def test_edit_job_on_failed_job_clears_the_stale_error(session):
    job = await _failed_job(session, error="Extraction failed entirely — timeout")

    response = await edit_job(job_id=job.id, invoice=_invoice(), session=session)

    assert response.error is None


@pytest.mark.anyio
async def test_edit_job_on_flagged_entry_corrects_it_and_clears_review(session):
    job = await create_job(session, file_key="b2c3d4e5.png")
    await update_job(
        session, job_id=job.id, status_update="complete", attempts_update=1
    )
    original = await write_entry(
        session,
        job.id,
        _invoice(),
        needs_review=True,
        review_reason=[{"category": "mismatch", "message": "Line items sum to 9.00"}],
    )

    corrected = _invoice(invoice_number="INV-42", grand_total=Decimal("99.99"))
    response = await edit_job(job_id=job.id, invoice=corrected, session=session)

    assert response.ledger_entry is not None
    # updated in place rather than appending a second entry for the same job
    assert response.ledger_entry.id == original.id
    assert response.ledger_entry.needs_review is False
    assert response.ledger_entry.review_reason is None
    assert response.ledger_entry.grand_total == Decimal("99.99")
    assert response.ledger_entry.invoice_data.invoice_number == "INV-42"
    assert response.ledger_entry.reviewed_with_changes is True


@pytest.mark.anyio
async def test_edit_job_on_flagged_entry_with_no_changes_marks_reviewed_without_changes(
    session,
):
    job = await create_job(session, file_key="c3d4e5f6.png")
    await update_job(
        session, job_id=job.id, status_update="complete", attempts_update=1
    )
    original_invoice = _invoice(invoice_number="INV-1")
    await write_entry(
        session,
        job.id,
        original_invoice,
        needs_review=True,
        review_reason=[{"category": "unverifiable", "message": "Missing subtotal"}],
    )

    # reviewer looks it over and confirms as-is, submitting the same data back
    response = await edit_job(job_id=job.id, invoice=original_invoice, session=session)

    assert response.ledger_entry.needs_review is False
    assert response.ledger_entry.reviewed_with_changes is False


@pytest.mark.anyio
async def test_edit_job_stores_a_reviewer_entered_negative_discount_as_positive(
    session,
):
    # FastAPI builds the request body with Invoice.model_validate, so a
    # reviewer typing "-5.00" goes through the same schema rule as the models
    job = await _failed_job(session)
    submitted = Invoice.model_validate(
        {
            "document_type": "receipt",
            "currency": "USD",
            "grand_total": "95.00",
            "discount": "-5.00",
        }
    )

    response = await edit_job(job_id=job.id, invoice=submitted, session=session)

    assert response.ledger_entry.invoice_data.discount == Decimal("5.00")


@pytest.mark.anyio
async def test_edit_job_does_not_count_a_discount_sign_flip_as_a_change(session):
    # "-5.00" and "5.00" are the same discount; confirming an entry while
    # typing the sign differently is a confirmation, not a correction
    job = await create_job(session, file_key="d4e5f6a7.png")
    await update_job(
        session, job_id=job.id, status_update="complete", attempts_update=1
    )
    await write_entry(
        session,
        job.id,
        _invoice(grand_total=Decimal("95.00"), discount=Decimal("5.00")),
        needs_review=True,
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

    assert response.ledger_entry.reviewed_with_changes is False


@pytest.mark.anyio
async def test_edit_job_raises_404_for_unknown_job(session):
    with pytest.raises(HTTPException) as exc_info:
        await edit_job(job_id=999, invoice=_invoice(), session=session)

    assert exc_info.value.status_code == 404
    assert "999" in exc_info.value.detail
