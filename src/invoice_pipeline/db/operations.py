from datetime import UTC, datetime

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import and_, delete, func, select, update
from invoice_pipeline.db.models import Extraction, Job, LedgerEntry
from invoice_pipeline.schema import Invoice

PENDING_STATUSES = ("pending", "retrying")


async def create_job(session: AsyncSession, file_key: str) -> Job:
    job = Job(file_key=file_key)
    session.add(job)
    await session.commit()
    return job


async def update_job(
    session: AsyncSession,
    job_id: int,
    status_update: str,
    attempts_update: int,
    error_update: str | None = None,
) -> Job:
    job = await session.get(Job, job_id)
    job.status = status_update
    job.attempts = attempts_update
    job.error = error_update

    session.add(job)
    await session.commit()
    return job


async def query_job(session: AsyncSession, job_id: int) -> Job:
    job = await session.get(Job, job_id)
    return job


async def query_extractions(session: AsyncSession, job_id: int) -> list[Extraction]:
    statement = select(Extraction).where(Extraction.job_id == job_id)
    results = await session.exec(statement)
    rows = results.all()
    return list(rows)


async def write_extractions(
    session: AsyncSession, job_id: int, extractions: list[Extraction]
) -> None:
    # clear the old final before inserting, or autoflush trips the one-final-per-job index
    statement = (
        update(Extraction)
        .where(Extraction.job_id == job_id, Extraction.is_final)
        .values(is_final=False)
    )
    await session.exec(statement)
    last_run = (
        await session.exec(
            select(func.max(Extraction.run)).where(Extraction.job_id == job_id)
        )
    ).one()
    for extraction in extractions:
        extraction.run = (last_run or 0) + 1
    session.add_all(extractions)
    await session.commit()


async def approve_job(
    session: AsyncSession, job_id: int, invoice: Invoice
) -> LedgerEntry:
    final = (
        await session.exec(
            select(Extraction).where(Extraction.job_id == job_id, Extraction.is_final)
        )
    ).first()
    entry = (
        await session.exec(select(LedgerEntry).where(LedgerEntry.job_id == job_id))
    ).first()
    if entry is None:
        entry = LedgerEntry(job_id=job_id)
    entry.invoice_data = invoice.model_dump(mode="json")
    entry.invoice_number = invoice.invoice_number
    entry.vendor_name = invoice.vendor.name if invoice.vendor else None
    entry.issue_date = invoice.issue_date
    entry.currency = invoice.currency
    entry.grand_total = invoice.grand_total
    entry.source_extraction_id = final.id if final else None
    # compare parsed models, not JSON: "100" vs "100.00" is the same amount
    entry.corrected = (
        invoice != Invoice.model_validate(final.invoice_data) if final else None
    )
    entry.reviewed_at = datetime.now(UTC)
    session.add(entry)
    await session.commit()
    return entry


async def query_job_entry(
    session: AsyncSession, job_id: int
) -> tuple[Job, Extraction | None, LedgerEntry | None] | None:
    statement = (
        select(Job, Extraction, LedgerEntry)
        # is_final belongs in ON: in WHERE it would drop jobs with no final extraction
        .join(
            Extraction,
            onclause=and_(Extraction.job_id == Job.id, Extraction.is_final),
            isouter=True,
        )
        .join(LedgerEntry, onclause=LedgerEntry.job_id == Job.id, isouter=True)
        .where(Job.id == job_id)
    )
    results = await session.exec(statement)
    row = results.first()
    if row is None:
        return None
    job, extraction, entry = row
    return (job, extraction, entry)


async def query_reviewables(
    session: AsyncSession,
) -> list[tuple[Job, Extraction | None]]:
    statement = (
        select(Job, Extraction)
        .join(
            Extraction,
            onclause=and_(Extraction.job_id == Job.id, Extraction.is_final),
            isouter=True,
        )
        .join(LedgerEntry, onclause=LedgerEntry.job_id == Job.id, isouter=True)
        .where(
            Job.status.in_(("complete", "extraction_failed")),
            # anti-join: keep only jobs the outer join found no ledger entry for
            LedgerEntry.id.is_(None),
        )
        .order_by(Job.id)
    )
    results = await session.exec(statement)
    return [(job, extraction) for job, extraction in results]


async def delete_job_entry(
    session: AsyncSession, job_id: int
) -> tuple[Job, Extraction | None, LedgerEntry | None] | None:
    found = await query_job_entry(session, job_id)
    if found is None:
        return None
    # FK order: ledgerentry.source_extraction_id references extraction
    await session.exec(delete(LedgerEntry).where(LedgerEntry.job_id == job_id))
    await session.exec(delete(Extraction).where(Extraction.job_id == job_id))
    await session.exec(delete(Job).where(Job.id == job_id))
    await session.commit()
    return found


async def query_resolved_ledger(
    session: AsyncSession,
) -> list[tuple[Job, Extraction | None, LedgerEntry]]:
    statement = (
        select(Job, Extraction, LedgerEntry)
        .join(LedgerEntry, onclause=LedgerEntry.job_id == Job.id)
        .join(
            Extraction,
            onclause=and_(Extraction.job_id == Job.id, Extraction.is_final),
            isouter=True,
        )
        .order_by(Job.id)
    )
    results = await session.exec(statement)
    return [(job, extraction, entry) for job, extraction, entry in results]


async def query_pending_jobs(session: AsyncSession) -> list[Job]:
    statement = select(Job).where(Job.status.in_(PENDING_STATUSES))
    results = await session.exec(statement)
    return results.all()


async def mark_job_orphaned(session: AsyncSession, job_id: int, error: str) -> bool:
    # status check inside the UPDATE, so a worker finishing first can't be overwritten
    statement = (
        update(Job)
        .where(Job.id == job_id, Job.status.in_(PENDING_STATUSES))
        .values(status="error", error=error)
    )
    result = await session.exec(statement)
    await session.commit()
    return result.rowcount == 1


async def query_errored_jobs(session: AsyncSession) -> list[Job]:
    statement = select(Job).where(Job.status == "error")
    results = await session.exec(statement)
    return results.all()


async def find_duplicate(
    session: AsyncSession, invoice_number: str | None, vendor_name: str | None
) -> LedgerEntry | None:
    if invoice_number is None or vendor_name is None:
        return None

    statement = select(LedgerEntry).where(
        LedgerEntry.invoice_number == invoice_number,
        LedgerEntry.vendor_name == vendor_name,
    )
    result = await session.exec(statement)
    return result.first()


async def find_pending_duplicate(
    session: AsyncSession,
    invoice_number: str | None,
    vendor_name: str | None,
    exclude_job_id: int,
) -> int | None:
    if invoice_number is None or vendor_name is None:
        return None

    statement = (
        select(Extraction.job_id)
        .join(
            LedgerEntry, onclause=LedgerEntry.job_id == Extraction.job_id, isouter=True
        )
        .where(
            Extraction.is_final,
            # approved jobs are covered by find_duplicate, with the reviewer's corrected values
            LedgerEntry.id.is_(None),
            # a re-run must not match its own earlier extraction
            Extraction.job_id != exclude_job_id,
            Extraction.invoice_data["invoice_number"].as_string() == invoice_number,
            Extraction.invoice_data[("vendor", "name")].as_string() == vendor_name,
        )
        .order_by(Extraction.job_id)
    )
    result = await session.exec(statement)
    return result.first()
