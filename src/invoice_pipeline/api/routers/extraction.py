import asyncio
from collections import Counter
from dataclasses import dataclass
import hashlib
import logging
import os
from arq import ArqRedis
from PIL import Image, UnidentifiedImageError
from invoice_pipeline.queue.jobs import arq_job_id
from invoice_pipeline.queue.pool import get_redis_pool
import uuid
import aiofiles
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlmodel.ext.asyncio.session import AsyncSession
from invoice_pipeline.api.schema.responses import (
    JobCreation,
    RejectedFile,
    RejectReason,
    UploadResponse,
)
from invoice_pipeline.db.engine import get_async_session
from invoice_pipeline.db.operations import create_job, query_job
from invoice_pipeline.config import UPLOADS_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/extraction", tags=["extraction"])

_MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@dataclass
class UploadChecks:
    accepted: list[tuple[UploadFile, str]]
    rejected: list[RejectedFile]


async def _check_uploads(files: list[UploadFile]) -> UploadChecks:
    accepted_files = []
    rejected_files = []
    duplicates = await _find_batch_duplicates(files)
    for file in files:
        size_not_ok = _check_size(file)
        if size_not_ok is not None:
            rejection = RejectedFile(
                filename=file.filename,
                content_type=file.content_type,
                reason=size_not_ok,
            )
            rejected_files.append(rejection)
            continue
        reason, fmt = await asyncio.to_thread(_check_content, file)
        if reason is not None:
            rejection = RejectedFile(
                filename=file.filename, content_type=file.content_type, reason=reason
            )
            rejected_files.append(rejection)
            continue
        if file in duplicates:
            rejection = RejectedFile(
                filename=file.filename,
                content_type=file.content_type,
                reason="duplicate_in_batch",
            )
            rejected_files.append(rejection)
            continue
        accepted_files.append((file, fmt))
    if len(accepted_files) == 0:
        counts = Counter(r.reason for r in rejected_files)
        summary = ", ".join(f"{n} {reason}" for reason, n in counts.items())
        raise HTTPException(
            status_code=422,
            detail=f"Number of files rejected: {len(rejected_files)}\n{summary}.",
        )
    return UploadChecks(accepted=accepted_files, rejected=rejected_files)


def _check_size(file: UploadFile) -> RejectReason | None:
    size = file.size
    if size is None:
        size = file.file.seek(0, os.SEEK_END)
        file.file.seek(0)
    if size == 0:
        return "empty"
    if size > _MAX_UPLOAD_BYTES:
        return "too_large"
    return None


def _check_content(file: UploadFile) -> tuple[RejectReason | None, str | None]:
    # blocking (Pillow): call via asyncio.to_thread. Returns (reason, "png"/"jpeg")
    try:
        with Image.open(file.file, formats=["PNG", "JPEG"]) as img:
            fmt = img.format.lower()
            img.verify()
    except UnidentifiedImageError:
        if file.content_type in ("image/png", "image/jpeg"):
            return "type_mismatch", None
        return "unsupported_type", None
    except Exception:
        return "unreadable_image", None
    finally:
        file.file.seek(0)
    return None, fmt


async def _find_batch_duplicates(files: list[UploadFile]) -> list[UploadFile]:
    seen: set[str] = set()
    duplicates = []
    for file in files:
        digest = hashlib.sha256()
        while chunk := await file.read(1024 * 1024):
            digest.update(chunk)
        await file.seek(0)
        if digest.hexdigest() in seen:
            duplicates.append(file)
        seen.add(digest.hexdigest())
    return duplicates


def _build_file_key(fmt: str) -> str:
    key = f"{uuid.uuid4()}.{fmt}"
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
) -> UploadResponse:
    upload_checks = await _check_uploads(files)
    created_jobs = []
    for file, fmt in upload_checks.accepted:
        key = _build_file_key(fmt)
        await _save_upload(file, key)
        job = await create_job(session, key)
        created_jobs.append(job)
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
    return UploadResponse(
        jobs=[JobCreation(job_id=job.id, status=job.status) for job in created_jobs],
        rejected=upload_checks.rejected,
    )


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
