from arq import ArqRedis
from fastapi import APIRouter, HTTPException, Depends
import logging
from pydantic import ValidationError
from sqlmodel.ext.asyncio.session import AsyncSession
from invoice_pipeline.api.schema.responses import (
    ErroredJobResponse,
    LedgerEntryResponse,
    JobEntryPairResponse,
    PendingJobResponse,
)
from invoice_pipeline.db.engine import get_async_session
from invoice_pipeline.db.models import Job, LedgerEntry
from invoice_pipeline.db.operations import (
    delete_job_entry,
    query_errored_jobs,
    query_job,
    query_job_entry,
    query_pending_jobs,
    query_resolved_ledger,
    query_reviewables,
    update_entry,
    update_job,
    write_entry,
)
from invoice_pipeline.queue.jobs import arq_job_id
from invoice_pipeline.queue.pool import get_redis_pool
from invoice_pipeline.schema import Invoice


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ledger", tags=["ledger"])


def _to_response(job: Job, entry: LedgerEntry | None) -> JobEntryPairResponse:
    ledger_entry = None
    ledger_entry_error = None
    if entry is not None:
        try:
            ledger_entry = LedgerEntryResponse.model_validate(entry)
        except ValidationError as e:
            logger.error(
                "Failed to decode ledger entry %s for job %s: %s",
                entry.id,
                job.id,
                e,
            )
            ledger_entry_error = str(e)
    return JobEntryPairResponse(
        job_id=job.id,
        created_at=job.created_at,
        status=job.status,
        error=job.error,
        attempts=job.attempts,
        ledger_entry=ledger_entry,
        ledger_entry_error=ledger_entry_error,
    )


def _to_pending_response(job: Job) -> PendingJobResponse:
    return PendingJobResponse(
        job_id=job.id,
        created_at=job.created_at,
        status=job.status,
        error=job.error,
        attempts=job.attempts,
    )


def _to_errored_response(job: Job) -> ErroredJobResponse:
    return ErroredJobResponse(
        job_id=job.id,
        created_at=job.created_at,
        status=job.status,
        error=job.error,
        attempts=job.attempts,
    )


@router.get("/review", response_model=list[JobEntryPairResponse])
async def get_reviewables(
    session: AsyncSession = Depends(get_async_session),
) -> list[JobEntryPairResponse]:
    results = await query_reviewables(session)
    reviewables = []
    for job, entry in results:
        reviewables.append((_to_response(job, entry)))
    return reviewables


@router.get("/resolved", response_model=list[JobEntryPairResponse])
async def get_full_ledger(session: AsyncSession = Depends(get_async_session)):
    results = await query_resolved_ledger(session)
    entries = []
    for job, entry in results:
        entries.append((_to_response(job, entry)))
    return entries


@router.get("/pending", response_model=list[PendingJobResponse])
async def get_pending_jobs(
    session: AsyncSession = Depends(get_async_session),
) -> list[PendingJobResponse]:
    results = await query_pending_jobs(session)
    pending = []
    for job in results:
        pending.append(_to_pending_response(job))
    return pending


@router.get("/errored")
async def get_errored_jobs(
    session: AsyncSession = Depends(get_async_session),
) -> list[ErroredJobResponse]:
    results = await query_errored_jobs(session)
    responses = []
    for job in results:
        errored_job = _to_errored_response(job)
        responses.append(errored_job)
    return responses


@router.get("/{job_id}", response_model=JobEntryPairResponse)
async def get_job_by_id(
    job_id: int,
    session: AsyncSession = Depends(get_async_session),
) -> JobEntryPairResponse:
    result = await query_job_entry(session, job_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found",
        )
    job, entry = result
    return _to_response(job, entry)


@router.delete("/{job_id}", response_model=JobEntryPairResponse)
async def delete_job_by_id(
    job_id: int,
    session: AsyncSession = Depends(get_async_session),
) -> JobEntryPairResponse:
    result = await delete_job_entry(session, job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job, entry = result
    return _to_response(job, entry)


@router.put("/{job_id}/edit", response_model=JobEntryPairResponse)
async def edit_job(
    job_id: int, invoice: Invoice, session: AsyncSession = Depends(get_async_session)
) -> JobEntryPairResponse:
    result = await query_job_entry(session, job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job, entry = result
    if entry is None:
        entry = await write_entry(
            session=session,
            job_id=job_id,
            invoice=invoice,
            needs_review=False,
            review_reason=None,
        )
        job = await update_job(
            session=session,
            job_id=job_id,
            status_update="complete",
            attempts_update=job.attempts,
        )
    else:
        entry = await update_entry(session=session, job_id=job_id, invoice=invoice)
    return _to_response(job, entry)


@router.post("/{job_id}/retry", response_model=ErroredJobResponse)
async def retry_job(
    job_id: int,
    session: AsyncSession = Depends(get_async_session),
    pool: ArqRedis = Depends(get_redis_pool),
) -> ErroredJobResponse:
    job = await query_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status != "error":
        raise HTTPException(
            status_code=409, detail=f"Job {job_id} is not in an error state"
        )
    job = await update_job(
        session,
        job_id=job_id,
        status_update="pending",
        attempts_update=job.attempts,
        error_update=None,
    )
    arq_job = await pool.enqueue_job(
        "process_job", job.id, job.file_key, _job_id=arq_job_id(job.id)
    )
    if arq_job is None:
        # arq already has a live job for this row (e.g. the reaper marked it
        # error between commit and enqueue) — it will write the final status
        logger.warning(
            "Job %s (%s) is already queued or running in arq; retry not re-enqueued",
            job.id,
            arq_job_id(job.id),
        )
    return _to_errored_response(job)
