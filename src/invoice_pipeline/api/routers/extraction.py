import logging
from arq import ArqRedis
from invoice_pipeline.queue.jobs import arq_job_id
from invoice_pipeline.queue.pool import get_redis_pool
import uuid
import aiofiles
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlmodel.ext.asyncio.session import AsyncSession
from invoice_pipeline.api.schema.responses import JobCreationResponse
from invoice_pipeline.db.engine import get_async_session
from invoice_pipeline.db.operations import create_job, query_job
from invoice_pipeline.config import UPLOADS_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/extraction", tags=["extraction"])

_ALLOWED_TYPES = {"image/jpeg": "jpeg", "image/png": "png"}


def _check_file_type(file: UploadFile) -> None:
    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"File type cannot be accessed: '{file.content_type}'. Please upload PNG, JPEG or PDF.",
        )


def _build_file_key(file: UploadFile) -> str:
    ext = _ALLOWED_TYPES[file.content_type]
    key = f"{uuid.uuid4()}.{ext}"
    return key


async def _save_upload(file: UploadFile, key: str) -> None:
    async with aiofiles.open(UPLOADS_DIR / key, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            await out.write(chunk)


@router.post("/upload_image")
async def upload(
    files: list[UploadFile],
    session: AsyncSession = Depends(get_async_session),
    pool: ArqRedis = Depends(get_redis_pool),
) -> list[JobCreationResponse]:
    created_jobs = []
    for file in files:
        _check_file_type(file)
        key = _build_file_key(file)
        await _save_upload(file, key)
        job = await create_job(session, key)
        created_jobs.append(job)
    for job in created_jobs:
        arq_job = await pool.enqueue_job(
            "process_job", job.id, job.file_key, _job_id=arq_job_id(job.id)
        )
        if arq_job is None:
            # a brand-new row should never collide; if it does, Redis still
            # holds a job from before the DB was reset (ids restarted)
            logger.warning(
                "Job %s (%s) already exists in arq on upload; not enqueued. "
                "Redis likely holds a stale job from before a DB reset",
                job.id,
                arq_job_id(job.id),
            )
    return [
        JobCreationResponse(job_id=job.id, status=job.status) for job in created_jobs
    ]


@router.get("/status/{job_id}")
async def get_extraction_job(
    job_id: int, session: AsyncSession = Depends(get_async_session)
):
    job = await query_job(session, job_id)
    return {"job": job}


@router.get("/image/{job_id}")
async def get_job_image(
    job_id: int, session: AsyncSession = Depends(get_async_session)
) -> FileResponse:
    job = await query_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    key = job.file_key
    file_path = UPLOADS_DIR / key
    if not Path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image does not exist")
    return FileResponse(file_path)
