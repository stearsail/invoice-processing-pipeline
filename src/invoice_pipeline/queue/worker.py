import asyncio

from arq import Retry
from arq.connections import RedisSettings
from invoice_pipeline import config
import logging
from invoice_pipeline.workflow.runner import run_extraction
from invoice_pipeline.db.engine import session_factory
from invoice_pipeline.db.operations import update_job
from invoice_pipeline.config import UPLOADS_DIR, JOB_TIMEOUT_S, EXTRACTION_DEADLINE_S

logger = logging.getLogger(__name__)


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


class WorkerSettings:
    redis_settings = RedisSettings(host=config.REDIS_HOST, port=config.REDIS_PORT)
    functions = [process_job]
    max_tries = 3
    job_timeout = JOB_TIMEOUT_S
