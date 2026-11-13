from datetime import date
from typing import get_args
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from invoice_pipeline.db.models import Job
from invoice_pipeline.db.operations import (
    PENDING_STATUSES,
    create_job,
    find_duplicate,
    mark_job_orphaned,
    query_errored_jobs,
    query_pending_jobs,
    query_resolved_ledger,
    query_job,
    query_reviewables,
    update_entry,
    update_job,
    write_entry,
)
from invoice_pipeline.schema import Invoice, Party


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


async def _job(session, file_key: str = "a1b2c3d4.png"):
    return await create_job(session, file_key=file_key)


@pytest.mark.anyio
async def test_write_invoice_persists_and_assigns_id(session):
    job = await _job(session)

    entry = await write_entry(session, job.id, _invoice())

    assert entry.id is not None
    assert entry.grand_total == Decimal("10.00")
    assert entry.currency == "USD"
    assert entry.job_id == job.id


@pytest.mark.anyio
async def test_write_invoice_stores_vendor_name_when_present(session):
    job = await _job(session)
    invoice = _invoice(vendor=Party(name="Acme Supplies SRL"))

    entry = await write_entry(session, job.id, invoice)

    assert entry.vendor_name == "Acme Supplies SRL"


@pytest.mark.anyio
async def test_write_invoice_vendor_name_is_none_when_vendor_missing(session):
    job = await _job(session)

    entry = await write_entry(session, job.id, _invoice())

    assert entry.vendor_name is None


@pytest.mark.anyio
async def test_write_invoice_stores_full_invoice_as_json(session):
    job = await _job(session)
    invoice = _invoice(invoice_number="INV-001", issue_date=date(2026, 7, 1))

    entry = await write_entry(session, job.id, invoice)
    restored = Invoice.model_validate(entry.invoice_data)

    assert restored.invoice_number == "INV-001"
    assert restored.issue_date == date(2026, 7, 1)
    assert restored.grand_total == Decimal("10.00")


@pytest.mark.anyio
async def test_write_entry_persists_extracted_by(session):
    job = await _job(session)

    entry = await write_entry(session, job.id, _invoice(), extracted_by="specialist")

    assert entry.extracted_by == "specialist"


@pytest.mark.anyio
async def test_write_entry_extracted_by_defaults_to_none(session):
    # a human-authored entry (edit_job on a job with no prior entry) has no
    # model attempt to attribute — extracted_by should stay unset, not guess.
    job = await _job(session)

    entry = await write_entry(session, job.id, _invoice())

    assert entry.extracted_by is None


@pytest.mark.anyio
async def test_update_entry_marks_reviewed_with_changes_true_when_invoice_differs(
    session,
):
    job = await _job(session)
    await write_entry(session, job.id, _invoice(invoice_number="INV-001"))

    updated = await update_entry(session, job.id, _invoice(invoice_number="INV-002"))

    assert updated.reviewed_with_changes is True


@pytest.mark.anyio
async def test_update_entry_marks_reviewed_with_changes_false_when_identical(
    session,
):
    job = await _job(session)
    original = _invoice(invoice_number="INV-001")
    await write_entry(session, job.id, original)

    updated = await update_entry(session, job.id, original)

    assert updated.reviewed_with_changes is False


@pytest.mark.anyio
async def test_multiple_writes_get_distinct_ids(session):
    job_one = await _job(session, "one.png")
    job_two = await _job(session, "two.png")

    entry_one = await write_entry(session, job_one.id, _invoice(invoice_number="A"))
    entry_two = await write_entry(session, job_two.id, _invoice(invoice_number="B"))

    assert entry_one.id != entry_two.id


@pytest.mark.anyio
async def test_find_duplicate_returns_none_when_nothing_matches(session):
    assert await find_duplicate(session, "INV-001", "Acme Supplies SRL") is None


@pytest.mark.anyio
async def test_find_duplicate_finds_same_invoice_number_and_vendor(session):
    job = await _job(session)
    invoice = _invoice(invoice_number="INV-001", vendor=Party(name="Acme Supplies SRL"))
    written = await write_entry(session, job.id, invoice)

    duplicate = await find_duplicate(session, "INV-001", "Acme Supplies SRL")

    assert duplicate is not None
    assert duplicate.id == written.id


@pytest.mark.anyio
async def test_find_duplicate_ignores_same_number_different_vendor(session):
    job = await _job(session)
    await write_entry(
        session,
        job.id,
        _invoice(invoice_number="INV-001", vendor=Party(name="Acme Supplies SRL")),
    )

    duplicate = await find_duplicate(session, "INV-001", "Beta Corp GmbH")

    assert duplicate is None


@pytest.mark.anyio
async def test_find_duplicate_returns_none_when_invoice_number_missing():
    assert await find_duplicate(object(), None, "Acme Supplies SRL") is None


@pytest.mark.anyio
async def test_find_duplicate_returns_none_when_vendor_name_missing():
    assert await find_duplicate(object(), "INV-001", None) is None


@pytest.mark.anyio
async def test_create_job_persists_and_assigns_id(session):
    job = await create_job(session, file_key="a1b2c3d4.png")

    assert job.id is not None
    assert job.status == "pending"
    assert job.file_key == "a1b2c3d4.png"
    assert job.error is None


@pytest.mark.anyio
async def test_multiple_jobs_get_distinct_ids(session):
    job_one = await create_job(session, file_key="a1b2c3d4.png")
    job_two = await create_job(session, file_key="e5f6g7h8.png")

    assert job_one.id != job_two.id


@pytest.mark.anyio
async def test_query_job_returns_matching_job(session):
    job = await _job(session)

    found = await query_job(session, job.id)

    assert found.id == job.id


@pytest.mark.anyio
async def test_query_job_returns_none_when_missing(session):
    assert await query_job(session, 999) is None


@pytest.mark.anyio
async def test_update_job_sets_status_and_error(session):
    job = await _job(session)

    updated = await update_job(
        session,
        job_id=job.id,
        status_update="error",
        attempts_update=3,
        error_update="boom",
    )

    assert updated.status == "error"
    assert updated.error == "boom"
    assert updated.attempts == 3


@pytest.mark.anyio
async def test_update_job_persists_across_reads(session):
    job = await _job(session)

    await update_job(
        session, job_id=job.id, status_update="complete", attempts_update=1
    )
    reread = await query_job(session, job.id)

    assert reread.status == "complete"


@pytest.mark.anyio
async def test_query_reviewables_excludes_clean_completed_jobs(session):
    job = await _job(session)
    await update_job(
        session, job_id=job.id, status_update="complete", attempts_update=1
    )
    await write_entry(session, job.id, _invoice(), needs_review=False)

    reviewables = await query_reviewables(session)

    assert reviewables == []


@pytest.mark.anyio
async def test_query_reviewables_includes_flagged_ledger_entries(session):
    job = await _job(session)
    await update_job(
        session, job_id=job.id, status_update="complete", attempts_update=1
    )
    await write_entry(
        session,
        job.id,
        _invoice(),
        needs_review=True,
        review_reason=[{"category": "duplicate", "message": "dup"}],
    )

    reviewables = await query_reviewables(session)

    assert len(reviewables) == 1
    returned_job, returned_entry = reviewables[0]
    assert returned_job.id == job.id
    assert returned_entry.review_reason == [{"category": "duplicate", "message": "dup"}]


@pytest.mark.anyio
async def test_query_reviewables_includes_extraction_failed_jobs_with_no_entry(session):
    job = await _job(session)
    await update_job(
        session,
        job_id=job.id,
        status_update="extraction_failed",
        attempts_update=1,
        error_update="no invoice",
    )

    reviewables = await query_reviewables(session)

    assert len(reviewables) == 1
    returned_job, returned_entry = reviewables[0]
    assert returned_job.id == job.id
    assert returned_entry is None


@pytest.mark.anyio
async def test_query_reviewables_excludes_errored_jobs(session):
    # "error" jobs (Arq retries exhausted) have no invoice data to review/edit —
    # they're surfaced separately via query_errored_jobs, not the review queue.
    job = await _job(session)
    await update_job(
        session,
        job_id=job.id,
        status_update="error",
        attempts_update=3,
        error_update="boom",
    )

    reviewables = await query_reviewables(session)

    assert reviewables == []


@pytest.mark.anyio
async def test_query_errored_jobs_returns_only_error_status(session):
    errored_job = await _job(session, "errored.png")
    await update_job(
        session,
        job_id=errored_job.id,
        status_update="error",
        attempts_update=3,
        error_update="boom",
    )
    other_job = await _job(session, "pending.png")

    errored = await query_errored_jobs(session)

    errored_ids = {job.id for job in errored}
    assert errored_ids == {errored_job.id}
    assert other_job.id not in errored_ids


@pytest.mark.anyio
async def test_query_resolved_ledger_excludes_pending_and_errored_jobs(session):
    clean_job = await _job(session, "clean.png")
    await update_job(
        session, job_id=clean_job.id, status_update="complete", attempts_update=1
    )
    await write_entry(session, clean_job.id, _invoice(), needs_review=False)

    failed_job = await _job(session, "failed.png")
    await update_job(
        session,
        job_id=failed_job.id,
        status_update="error",
        attempts_update=3,
        error_update="boom",
    )

    await _job(session, "pending.png")

    resolved = await query_resolved_ledger(session)

    assert len(resolved) == 1
    returned_job, returned_entry = resolved[0]
    assert returned_job.id == clean_job.id
    assert returned_entry.needs_review is False


_ALL_JOB_STATUSES = get_args(Job.model_fields["status"].annotation)
_TERMINAL_STATUSES = [s for s in _ALL_JOB_STATUSES if s not in PENDING_STATUSES]


async def _job_with_status(session, status: str, attempts: int = 1) -> Job:
    job = await create_job(session, file_key=f"{status}.png")
    return await update_job(
        session,
        job_id=job.id,
        status_update=status,
        attempts_update=attempts,
        error_update="previous error",
    )


@pytest.mark.anyio
@pytest.mark.parametrize("status", PENDING_STATUSES)
async def test_mark_job_orphaned_moves_pending_rows_to_error(session, status):
    job = await _job_with_status(session, status, attempts=2)

    reaped = await mark_job_orphaned(session, job.id, "orphaned: test")

    updated = await query_job(session, job.id)
    assert reaped is True
    assert updated.status == "error"
    assert updated.error == "orphaned: test"
    # the reaper didn't run an attempt
    assert updated.attempts == 2


@pytest.mark.anyio
@pytest.mark.parametrize("status", _TERMINAL_STATUSES)
async def test_mark_job_orphaned_never_overwrites_a_final_status(session, status):
    # race A: the worker wrote its final status before the reaper's UPDATE
    job = await _job_with_status(session, status)

    reaped = await mark_job_orphaned(session, job.id, "orphaned: test")

    updated = await query_job(session, job.id)
    assert reaped is False
    assert updated.status == status
    assert updated.error == "previous error"


@pytest.mark.anyio
async def test_mark_job_orphaned_returns_false_for_unknown_job(session):
    assert await mark_job_orphaned(session, 999, "orphaned: test") is False


@pytest.mark.anyio
@pytest.mark.parametrize("status", _ALL_JOB_STATUSES)
async def test_reaper_select_and_update_agree_on_which_statuses_are_pending(
    session, status
):
    # a row the reaper selects must be one it can update, and vice versa
    job = await _job_with_status(session, status)

    selected = job.id in {j.id for j in await query_pending_jobs(session)}
    updatable = await mark_job_orphaned(session, job.id, "orphaned: test")

    assert selected == updatable
