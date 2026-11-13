from typing import Literal

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import or_, select, update
from invoice_pipeline.db.models import Job, LedgerEntry
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


async def write_entry(
    session: AsyncSession,
    job_id: int,
    invoice: Invoice,
    needs_review: bool = False,
    review_reason: list[dict] | None = None,
    extracted_by: Literal["specialist", "frontier"] | None = None,
) -> LedgerEntry:
    entry = LedgerEntry(
        invoice_number=invoice.invoice_number,
        vendor_name=invoice.vendor.name if invoice.vendor else None,
        issue_date=invoice.issue_date,
        currency=invoice.currency,
        grand_total=invoice.grand_total,
        invoice_data=invoice.model_dump(mode="json"),
        needs_review=needs_review,
        review_reason=review_reason,
        job_id=job_id,
        extracted_by=extracted_by,
    )
    session.add(entry)
    await session.commit()
    return entry


async def update_entry(
    session: AsyncSession, job_id: int, invoice: Invoice
) -> LedgerEntry:
    statement = select(LedgerEntry).where(LedgerEntry.job_id == job_id)
    result = await session.exec(statement)
    entry = result.first()
    new_data = invoice.model_dump(mode="json")
    entry.reviewed_with_changes = new_data != entry.invoice_data
    entry.invoice_number = invoice.invoice_number
    entry.vendor_name = invoice.vendor.name if invoice.vendor else None
    entry.issue_date = invoice.issue_date
    entry.currency = invoice.currency
    entry.grand_total = invoice.grand_total
    entry.invoice_data = new_data
    entry.needs_review = False
    entry.review_reason = None
    session.add(entry)
    await session.commit()
    return entry


async def query_job_entry(
    session: AsyncSession, job_id: int
) -> tuple[Job, LedgerEntry | None] | None:
    statement = (
        select(Job, LedgerEntry)
        .join(LedgerEntry, onclause=LedgerEntry.job_id == Job.id, isouter=True)
        .where(Job.id == job_id)
    )
    results = await session.exec(statement)
    row = results.first()
    if row is None:
        return None
    return (row[0], row[1])


async def delete_job_entry(
    session: AsyncSession, job_id: int
) -> tuple[Job, LedgerEntry | None] | None:
    statement = (
        select(Job, LedgerEntry)
        .join(LedgerEntry, onclause=LedgerEntry.job_id == Job.id, isouter=True)
        .where(Job.id == job_id)
    )
    result = await session.exec(statement)
    row = result.first()
    if row is None:
        return None
    job, entry = row
    if entry is not None:
        await session.delete(entry)
        await session.flush()
    await session.delete(job)
    await session.commit()
    return (job, entry)


async def query_reviewables(
    session: AsyncSession,
) -> list[tuple[Job, LedgerEntry | None]]:
    statement = (
        select(Job, LedgerEntry)
        .join(LedgerEntry, onclause=LedgerEntry.job_id == Job.id, isouter=True)
        .where(or_(Job.status == "extraction_failed", LedgerEntry.needs_review))
    )

    results = await session.exec(statement)
    reviewables = []
    for row in results:
        reviewables.append((row[0], row[1]))
    return reviewables


async def query_resolved_ledger(
    session: AsyncSession,
) -> list[tuple[Job, LedgerEntry | None]]:
    statement = (
        select(Job, LedgerEntry)
        .join(LedgerEntry, onclause=LedgerEntry.job_id == Job.id, isouter=False)
        .where(Job.status == "complete", LedgerEntry.needs_review == False)  # noqa: E712
        # RUFF SUGGESTS "not LedgerEntry.needs_review — this won't work since truthiness of a column object is always truthy"
    )
    results = await session.exec(statement)
    entries = []
    for row in results:
        entries.append((row[0], row[1]))
    return entries


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
