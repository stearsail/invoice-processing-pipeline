from datetime import UTC, date, datetime
from typing import get_args
from decimal import Decimal

import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from invoice_pipeline.db.models import Extraction, Job, LedgerEntry
from invoice_pipeline.db.operations import (
    PENDING_STATUSES,
    approve_job,
    create_job,
    delete_job_entry,
    find_duplicate,
    find_pending_duplicate,
    mark_job_orphaned,
    query_errored_jobs,
    query_pending_jobs,
    query_resolved_ledger,
    query_job,
    query_job_entry,
    query_reviewables,
    update_job,
    write_extractions,
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


async def _finished_job(
    session, status: str = "complete", final_invoice: Invoice | None = None
) -> Job:
    # what the worker leaves behind: a finished job plus its persisted extraction rows
    job = await _job(session, f"{status}-{final_invoice is not None}.png")
    await update_job(session, job_id=job.id, status_update=status, attempts_update=1)
    rows = [_extraction(job.id, parse_error="invalid_output: x")]
    if final_invoice is not None:
        rows.append(
            _extraction(
                job.id,
                is_final=True,
                extractor="frontier",
                invoice_data=final_invoice.model_dump(mode="json"),
            )
        )
    await write_extractions(session, job.id, rows)
    return job


async def _final_extraction_id(session, job_id: int) -> int:
    statement = select(Extraction.id).where(
        Extraction.job_id == job_id, Extraction.is_final
    )
    return (await session.exec(statement)).one()


@pytest.mark.anyio
async def test_approve_job_creates_the_entry_with_searchable_columns(session):
    job = await _finished_job(session, final_invoice=_invoice())
    invoice = _invoice(
        invoice_number="INV-7",
        vendor=Party(name="Acme Supplies SRL"),
        issue_date=date(2026, 3, 1),
        grand_total=Decimal("42.50"),
        currency="EUR",
    )

    entry = await approve_job(session, job.id, invoice)

    assert entry.id is not None
    assert entry.job_id == job.id
    assert entry.invoice_number == "INV-7"
    assert entry.vendor_name == "Acme Supplies SRL"
    assert entry.issue_date == date(2026, 3, 1)
    assert entry.currency == "EUR"
    assert entry.grand_total == Decimal("42.50")
    assert entry.invoice_data == invoice.model_dump(mode="json")
    assert entry.reviewed_at is not None


@pytest.mark.anyio
async def test_approve_job_vendor_name_is_none_when_vendor_missing(session):
    job = await _finished_job(session, final_invoice=_invoice())

    entry = await approve_job(session, job.id, _invoice())

    assert entry.vendor_name is None


@pytest.mark.anyio
async def test_approve_job_links_the_final_extraction_and_detects_no_change(session):
    job = await _finished_job(session, final_invoice=_invoice())

    entry = await approve_job(session, job.id, _invoice())

    # linked to the final row, not the failed specialist attempt before it
    assert entry.source_extraction_id == await _final_extraction_id(session, job.id)
    assert entry.corrected is False


@pytest.mark.anyio
async def test_approve_job_marks_corrected_when_the_reviewer_changed_a_value(session):
    job = await _finished_job(session, final_invoice=_invoice())

    entry = await approve_job(session, job.id, _invoice(invoice_number="INV-2"))

    assert entry.corrected is True


@pytest.mark.anyio
async def test_approve_job_compares_amounts_by_value_not_by_formatting(session):
    # as JSON these are "100" vs "100.00": a dict comparison would call it a correction
    job = await _finished_job(
        session, final_invoice=_invoice(grand_total=Decimal("100"))
    )

    entry = await approve_job(session, job.id, _invoice(grand_total=Decimal("100.00")))

    assert entry.corrected is False


@pytest.mark.anyio
async def test_approve_job_on_a_total_failure_has_no_source_to_compare(session):
    job = await _finished_job(session, status="extraction_failed")

    entry = await approve_job(session, job.id, _invoice())

    assert entry.source_extraction_id is None
    assert entry.corrected is None


@pytest.mark.anyio
async def test_reapproving_updates_the_same_entry(session):
    job = await _finished_job(session, final_invoice=_invoice())
    first = await approve_job(session, job.id, _invoice(invoice_number="INV-1"))
    first_id, first_reviewed_at = first.id, first.reviewed_at

    second = await approve_job(session, job.id, _invoice(invoice_number="INV-2"))

    assert second.id == first_id
    assert second.invoice_number == "INV-2"
    assert second.reviewed_at >= first_reviewed_at
    rows = (
        await session.exec(select(LedgerEntry).where(LedgerEntry.job_id == job.id))
    ).all()
    assert len(rows) == 1


@pytest.mark.anyio
async def test_approve_job_never_touches_the_job_row(session):
    job = await _finished_job(session, status="extraction_failed")

    await approve_job(session, job.id, _invoice())

    reread = await query_job(session, job.id)
    assert reread.status == "extraction_failed"


@pytest.mark.anyio
async def test_find_duplicate_returns_none_when_nothing_matches(session):
    assert await find_duplicate(session, "INV-001", "Acme Supplies SRL") is None


@pytest.mark.anyio
async def test_find_duplicate_finds_same_invoice_number_and_vendor(session):
    job = await _job(session)
    invoice = _invoice(invoice_number="INV-001", vendor=Party(name="Acme Supplies SRL"))
    written = await approve_job(session, job.id, invoice)

    duplicate = await find_duplicate(session, "INV-001", "Acme Supplies SRL")

    assert duplicate is not None
    assert duplicate.id == written.id


@pytest.mark.anyio
async def test_find_duplicate_ignores_same_number_different_vendor(session):
    job = await _job(session)
    await approve_job(
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


def _acme(**overrides) -> Invoice:
    return _invoice(
        invoice_number="INV-001", vendor=Party(name="Acme Supplies SRL"), **overrides
    )


@pytest.mark.anyio
async def test_find_pending_duplicate_finds_a_job_waiting_for_review(session):
    waiting = await _finished_job(session, final_invoice=_acme())

    found = await find_pending_duplicate(
        session, "INV-001", "Acme Supplies SRL", exclude_job_id=999
    )

    assert found == waiting.id


@pytest.mark.anyio
async def test_find_pending_duplicate_never_matches_the_job_itself(session):
    # on a re-run, the job's own earlier final extraction is already persisted
    job = await _finished_job(session, final_invoice=_acme())

    assert (
        await find_pending_duplicate(
            session, "INV-001", "Acme Supplies SRL", exclude_job_id=job.id
        )
        is None
    )


@pytest.mark.anyio
async def test_find_pending_duplicate_ignores_approved_jobs(session):
    # approved jobs belong to find_duplicate, which uses the reviewer's corrected values
    job = await _finished_job(session, final_invoice=_acme())
    await approve_job(session, job.id, _invoice(invoice_number="INV-999"))

    assert (
        await find_pending_duplicate(
            session, "INV-001", "Acme Supplies SRL", exclude_job_id=999
        )
        is None
    )


@pytest.mark.anyio
async def test_find_pending_duplicate_ignores_attempts_that_are_not_final(session):
    # _finished_job's failed specialist row is not final; a total failure has no final row
    await _finished_job(session, status="extraction_failed")
    job = await _job(session, "non-final.png")
    await write_extractions(
        session,
        job.id,
        [_extraction(job.id, invoice_data=_acme().model_dump(mode="json"))],
    )

    assert (
        await find_pending_duplicate(
            session, "INV-001", "Acme Supplies SRL", exclude_job_id=999
        )
        is None
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invoice_number, vendor_name",
    [("INV-001", "Beta Corp GmbH"), ("INV-002", "Acme Supplies SRL")],
    ids=["other-vendor", "other-number"],
)
async def test_find_pending_duplicate_needs_both_number_and_vendor_to_match(
    session, invoice_number, vendor_name
):
    await _finished_job(session, final_invoice=_acme())

    assert (
        await find_pending_duplicate(
            session, invoice_number, vendor_name, exclude_job_id=999
        )
        is None
    )


@pytest.mark.anyio
async def test_find_pending_duplicate_does_not_match_an_extraction_without_a_vendor(
    session,
):
    await _finished_job(session, final_invoice=_invoice(invoice_number="INV-001"))

    assert (
        await find_pending_duplicate(
            session, "INV-001", "Acme Supplies SRL", exclude_job_id=999
        )
        is None
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invoice_number, vendor_name", [(None, "Acme Supplies SRL"), ("INV-001", None)]
)
async def test_find_pending_duplicate_skips_the_query_without_both_keys(
    invoice_number, vendor_name
):
    assert (
        await find_pending_duplicate(
            object(), invoice_number, vendor_name, exclude_job_id=1
        )
        is None
    )


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
async def test_query_reviewables_returns_extracted_jobs_with_their_final_extraction(
    session,
):
    job = await _finished_job(session, final_invoice=_invoice())

    [(returned_job, extraction)] = await query_reviewables(session)

    assert returned_job.id == job.id
    assert extraction.is_final is True
    assert extraction.extractor == "frontier"


@pytest.mark.anyio
async def test_query_reviewables_keeps_total_failures_once_with_no_extraction(
    session,
):
    # is_final in the WHERE instead of the ON clause would drop this job entirely
    job = await _finished_job(session, status="extraction_failed")

    [(returned_job, extraction)] = await query_reviewables(session)

    assert returned_job.id == job.id
    assert extraction is None


@pytest.mark.anyio
async def test_query_reviewables_excludes_approved_jobs(session):
    approved = await _finished_job(session, final_invoice=_invoice())
    await approve_job(session, approved.id, _invoice())
    waiting = await _finished_job(session, status="extraction_failed")

    reviewables = await query_reviewables(session)

    assert [job.id for job, _ in reviewables] == [waiting.id]


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["pending", "retrying", "error"])
async def test_query_reviewables_excludes_unfinished_jobs(session, status):
    # "error" jobs have nothing to review; they're surfaced via query_errored_jobs
    await _job_with_status(session, status)

    assert await query_reviewables(session) == []


@pytest.mark.anyio
async def test_query_reviewables_is_ordered_oldest_first(session):
    first = await _finished_job(session, final_invoice=_invoice())
    second = await _finished_job(session, status="extraction_failed")

    reviewables = await query_reviewables(session)

    assert [job.id for job, _ in reviewables] == [first.id, second.id]


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
async def test_query_resolved_ledger_returns_only_approved_jobs(session):
    approved = await _finished_job(session, final_invoice=_invoice())
    entry = await approve_job(session, approved.id, _invoice())
    await _finished_job(session, final_invoice=_invoice())
    await _job_with_status(session, "error")

    [(job, extraction, returned_entry)] = await query_resolved_ledger(session)

    assert job.id == approved.id
    assert extraction.id == entry.source_extraction_id
    assert returned_entry.id == entry.id


@pytest.mark.anyio
async def test_query_resolved_ledger_includes_typed_in_total_failures(session):
    job = await _finished_job(session, status="extraction_failed")
    await approve_job(session, job.id, _invoice())

    [(returned_job, extraction, entry)] = await query_resolved_ledger(session)

    assert returned_job.id == job.id
    assert extraction is None
    assert entry is not None


@pytest.mark.anyio
async def test_query_job_entry_returns_none_for_an_unknown_job(session):
    assert await query_job_entry(session, 999) is None


@pytest.mark.anyio
async def test_query_job_entry_for_a_job_with_nothing_attached(session):
    job = await _job(session)

    assert await query_job_entry(session, job.id) == (job, None, None)


@pytest.mark.anyio
async def test_query_job_entry_returns_the_final_extraction_and_entry(session):
    job = await _finished_job(session, final_invoice=_invoice())
    entry = await approve_job(session, job.id, _invoice())

    returned_job, extraction, returned_entry = await query_job_entry(session, job.id)

    assert returned_job.id == job.id
    assert extraction.is_final is True
    assert returned_entry.id == entry.id


@pytest.mark.anyio
async def test_query_job_entry_does_not_lose_a_total_failure(session):
    job = await _finished_job(session, status="extraction_failed")

    returned_job, extraction, entry = await query_job_entry(session, job.id)

    assert returned_job.id == job.id
    assert extraction is None
    assert entry is None


@pytest.fixture
async def fk_session():
    # SQLite ignores foreign keys unless asked; Postgres always enforces them
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    event.listen(
        engine.sync_engine,
        "connect",
        lambda conn, _: conn.execute("PRAGMA foreign_keys=ON"),
    )
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


@pytest.mark.anyio
async def test_delete_job_entry_removes_everything_in_foreign_key_order(fk_session):
    job = await _finished_job(fk_session, final_invoice=_invoice())
    await approve_job(fk_session, job.id, _invoice())
    job_id = job.id

    deleted_job, extraction, entry = await delete_job_entry(fk_session, job_id)

    # echoes what was deleted
    assert deleted_job.id == job_id
    assert extraction is not None
    assert entry is not None
    assert await query_job(fk_session, job_id) is None
    assert await _extractions_of(fk_session, job_id) == []
    remaining = await fk_session.exec(
        select(LedgerEntry).where(LedgerEntry.job_id == job_id)
    )
    assert remaining.all() == []


@pytest.mark.anyio
async def test_delete_job_entry_leaves_other_jobs_alone(fk_session):
    doomed = await _finished_job(fk_session, final_invoice=_invoice())
    kept = await _finished_job(fk_session, status="extraction_failed")

    await delete_job_entry(fk_session, doomed.id)

    assert await query_job(fk_session, kept.id) is not None
    assert len(await _extractions_of(fk_session, kept.id)) == 1


@pytest.mark.anyio
async def test_delete_job_entry_returns_none_for_an_unknown_job(session):
    assert await delete_job_entry(session, 999) is None


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


@pytest.mark.anyio
async def test_a_job_can_have_only_one_ledger_entry(session):
    # approve_job upserts; this pins the database constraint behind it
    job = await create_job(session, file_key="once.png")
    session.add_all(
        [
            LedgerEntry(
                job_id=job.id,
                invoice_data={},
                currency="USD",
                grand_total=Decimal("1"),
                reviewed_at=datetime.now(UTC),
            )
            for _ in range(2)
        ]
    )

    with pytest.raises(IntegrityError):
        await session.commit()


def _extraction(job_id: int, is_final: bool = False, **overrides) -> Extraction:
    fields = {
        "job_id": job_id,
        "extractor": "specialist",
        "model": "m",
        "trigger": "initial",
        "is_final": is_final,
    }
    fields.update(overrides)
    return Extraction(**fields)


async def _extractions_of(session, job_id: int) -> list[Extraction]:
    statement = (
        select(Extraction).where(Extraction.job_id == job_id).order_by(Extraction.id)
    )
    return (await session.exec(statement)).all()


@pytest.mark.anyio
async def test_write_extractions_persists_every_row_with_run_one(session):
    job = await _job(session)

    await write_extractions(
        session,
        job.id,
        [
            _extraction(job.id, parse_error="invalid_output: x"),
            _extraction(job.id, is_final=True, extractor="frontier"),
        ],
    )

    rows = await _extractions_of(session, job.id)
    assert [(r.extractor, r.run, r.is_final) for r in rows] == [
        ("specialist", 1, False),
        ("frontier", 1, True),
    ]
    assert all(r.created_at is not None for r in rows)


@pytest.mark.anyio
async def test_write_extractions_rerun_moves_the_final_marker_and_keeps_history(
    session,
):
    job = await _job(session)
    await write_extractions(session, job.id, [_extraction(job.id, is_final=True)])

    # a second final row for the job would violate the partial unique index
    await write_extractions(
        session, job.id, [_extraction(job.id, is_final=True, extractor="frontier")]
    )

    rows = await _extractions_of(session, job.id)
    assert [(r.run, r.extractor, r.is_final) for r in rows] == [
        (1, "specialist", False),
        (2, "frontier", True),
    ]


@pytest.mark.anyio
async def test_write_extractions_total_failure_rerun_still_clears_the_old_final(
    session,
):
    # stale: the earlier run's invoice must not stay "final" next to a failed rerun
    job = await _job(session)
    await write_extractions(session, job.id, [_extraction(job.id, is_final=True)])

    await write_extractions(
        session, job.id, [_extraction(job.id, parse_error="invalid_output: x")]
    )

    rows = await _extractions_of(session, job.id)
    assert [r.is_final for r in rows] == [False, False]
    assert [r.run for r in rows] == [1, 2]


@pytest.mark.anyio
async def test_write_extractions_numbers_runs_per_job(session):
    job_a = await _job(session, "a.png")
    job_b = await _job(session, "b.png")
    await write_extractions(session, job_a.id, [_extraction(job_a.id)])
    await write_extractions(session, job_a.id, [_extraction(job_a.id)])

    await write_extractions(session, job_b.id, [_extraction(job_b.id, is_final=True)])

    assert [r.run for r in await _extractions_of(session, job_b.id)] == [1]
    # another job's write never touches this job's final marker
    await write_extractions(session, job_a.id, [_extraction(job_a.id, is_final=True)])
    assert [r.is_final for r in await _extractions_of(session, job_b.id)] == [True]


@pytest.mark.anyio
async def test_one_final_extraction_per_job_is_enforced_by_the_database(session):
    job = await _job(session)
    session.add_all([_extraction(job.id, is_final=True, run=1) for _ in range(2)])

    with pytest.raises(IntegrityError):
        await session.commit()


@pytest.mark.anyio
async def test_write_extractions_round_trips_json_payloads(session):
    job = await _job(session)
    invoice_data = _invoice(grand_total=Decimal("12.50")).model_dump(mode="json")
    issues = [{"category": "mismatch", "message": "off by 2"}]

    await write_extractions(
        session,
        job.id,
        [_extraction(job.id, is_final=True, invoice_data=invoice_data, issues=issues)],
    )

    session.expunge_all()
    [row] = await _extractions_of(session, job.id)
    assert row.invoice_data == invoice_data
    assert row.invoice_data["grand_total"] == "12.50"
    assert row.issues == issues
