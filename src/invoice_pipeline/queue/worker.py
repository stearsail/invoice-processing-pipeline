import asyncio

from arq import Retry, cron
from arq.connections import RedisSettings
from arq.jobs import Job as ArqJob, JobStatus
from invoice_pipeline import config
import logging
from invoice_pipeline.workflow.runner import run_extraction
from invoice_pipeline.db.engine import session_factory
from invoice_pipeline.db.operations import (
    mark_job_orphaned,
    query_pending_jobs,
    update_job,
)
from invoice_pipeline.config import UPLOADS_DIR, JOB_TIMEOUT_S, EXTRACTION_DEADLINE_S
from invoice_pipeline.queue.jobs import arq_job_id

logger = logging.getLogger(__name__)

ORPHANED_ERROR = "orphaned: no queued, deferred or running worker job"


async def _set_status(
    job_id: int, status: str, attempts: int, error: str | None = None
):
    async with session_factory() as session:
        await update_job(
            session,
            job_id=job_id,
            status_update=status,
            attempts_update=attempts,
            error_update=error,
        )


def _describe_error(exc: BaseException) -> str:
    # str() alone loses information: str(TimeoutError()) is "", which would
    # store a timed-out job as an error with a blank message
    message = str(exc)
    name = type(exc).__name__
    return f"{name}: {message}" if message else name


async def _run_extraction_job(job_id: int, file_key: str) -> None:
    img_path = str(UPLOADS_DIR / file_key)
    return await run_extraction(job_id=job_id, img_path=img_path)


async def process_job(ctx, job_id, file_key):
    try:
        async with asyncio.timeout(EXTRACTION_DEADLINE_S):
            result = await _run_extraction_job(job_id, file_key)
    except Exception as e:
        if ctx["job_try"] < WorkerSettings.max_tries:
            async with session_factory() as session:
                await update_job(
                    session,
                    job_id=job_id,
                    status_update="retrying",
                    error_update=_describe_error(e),
                    attempts_update=ctx["job_try"],
                )
            raise Retry(defer=ctx["job_try"] * 20)
        else:
            logger.exception("Extraction job %s failed permanently", job_id)
            await _set_status(
                job_id,
                status="error",
                attempts=ctx["job_try"],
                error=_describe_error(e),
            )
            return
    await _set_status(
        job_id, status=result.status, attempts=ctx["job_try"], error=result.error
    )


async def reap_orphaned_jobs(ctx) -> int:
    reaped = 0
    async with session_factory() as session:
        for job in await query_pending_jobs(session):
            job_id, previous_status = job.id, job.status
            status = await ArqJob(arq_job_id(job_id), ctx["redis"]).status()
            if status == JobStatus.not_found:
                if await mark_job_orphaned(session, job_id, ORPHANED_ERROR):
                    reaped += 1
                    logger.warning(
                        "Reaped orphaned job %s (was %s)", job_id, previous_status
                    )
            elif status == JobStatus.complete:
                # unreachable with keep_result = 0
                logger.error(
                    "Job %s has a stored arq result; check keep_result", job.id
                )
    return reaped


class WorkerSettings:
    redis_settings = RedisSettings(host=config.REDIS_HOST, port=config.REDIS_PORT)
    functions = [process_job]
    cron_jobs = [
        cron(reap_orphaned_jobs, minute=set(range(0, 60, 5)), run_at_startup=True)
    ]
    max_tries = 3
    keep_result = 0
    job_timeout = JOB_TIMEOUT_S
