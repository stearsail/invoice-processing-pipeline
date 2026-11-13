import asyncio
import logging
from unittest.mock import AsyncMock

import pytest
from arq import Retry
from arq.jobs import JobStatus
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from invoice_pipeline.workflow.runner import ExtractionResult
from invoice_pipeline import config
from invoice_pipeline.api.routers.ledger import retry_job
from invoice_pipeline.db.operations import create_job, query_job, update_job
from invoice_pipeline.queue import worker
from invoice_pipeline.queue.jobs import arq_job_id


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def in_memory_session_factory(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(worker, "session_factory", factory)
    yield factory
    await engine.dispose()


async def _make_job(in_memory_session_factory, file_key: str = "a1b2c3d4.png"):
    async with in_memory_session_factory() as session:
        return await create_job(session, file_key=file_key)


@pytest.mark.anyio
async def test_run_extraction_job_resolves_file_key_against_uploads_dir(monkeypatch):
    fake_run = AsyncMock(return_value=ExtractionResult(status="complete"))
    monkeypatch.setattr(worker, "run_extraction", fake_run)

    await worker._run_extraction_job(1, "a1b2c3d4.png")

    img_path = fake_run.await_args.kwargs["img_path"]
    assert img_path.endswith("a1b2c3d4.png")
    assert str(worker.UPLOADS_DIR) in img_path


@pytest.mark.anyio
async def test_process_job_marks_complete_on_success(
    in_memory_session_factory, monkeypatch
):
    job = await _make_job(in_memory_session_factory)
    monkeypatch.setattr(
        worker,
        "run_extraction",
        AsyncMock(return_value=ExtractionResult(status="complete", ledger_entry_id=1)),
    )

    await worker.process_job({"job_try": 1}, job.id, job.file_key)

    async with in_memory_session_factory() as session:
        updated = await query_job(session, job.id)
    assert updated.status == "complete"
    assert updated.error is None
    assert updated.attempts == 1


@pytest.mark.anyio
async def test_process_job_marks_extraction_failed_when_no_invoice_extracted(
    in_memory_session_factory, monkeypatch
):
    job = await _make_job(in_memory_session_factory)
    monkeypatch.setattr(
        worker,
        "run_extraction",
        AsyncMock(
            return_value=ExtractionResult(
                status="extraction_failed",
                error="Extraction failed entirely — unknown: no invoice to reconcile",
            )
        ),
    )

    await worker.process_job({"job_try": 1}, job.id, job.file_key)

    async with in_memory_session_factory() as session:
        updated = await query_job(session, job.id)
    assert updated.status == "extraction_failed"
    assert "Extraction failed entirely" in updated.error


@pytest.mark.anyio
async def test_process_job_retries_when_tries_remain(
    in_memory_session_factory, monkeypatch
):
    job = await _make_job(in_memory_session_factory)
    monkeypatch.setattr(
        worker,
        "run_extraction",
        AsyncMock(side_effect=RuntimeError("vllm connection refused")),
    )

    with pytest.raises(Retry):
        await worker.process_job({"job_try": 1}, job.id, job.file_key)

    async with in_memory_session_factory() as session:
        updated = await query_job(session, job.id)
    assert updated.status == "retrying"
    assert updated.attempts == 1
    assert "vllm connection refused" in updated.error


@pytest.mark.anyio
async def test_process_job_marks_error_when_tries_exhausted(
    in_memory_session_factory, monkeypatch
):
    job = await _make_job(in_memory_session_factory)
    monkeypatch.setattr(
        worker,
        "run_extraction",
        AsyncMock(side_effect=RuntimeError("vllm connection refused")),
    )
    last_try = worker.WorkerSettings.max_tries

    await worker.process_job({"job_try": last_try}, job.id, job.file_key)

    async with in_memory_session_factory() as session:
        updated = await query_job(session, job.id)
    assert updated.status == "error"
    assert updated.attempts == last_try
    assert "vllm connection refused" in updated.error


def _fake_arq(monkeypatch, statuses: dict[str, JobStatus], on_status=None):
    asked = []

    class FakeArqJob:
        def __init__(self, job_id, redis):
            self.job_id = job_id

        async def status(self):
            asked.append(self.job_id)
            if on_status is not None:
                await on_status(self.job_id)
            return statuses.get(self.job_id, JobStatus.not_found)

    monkeypatch.setattr(worker, "ArqJob", FakeArqJob)
    return asked


async def _job_with_status(factory, status: str, attempts: int = 1):
    job = await _make_job(factory, file_key=f"{status}-{attempts}.png")
    async with factory() as session:
        return await update_job(
            session, job_id=job.id, status_update=status, attempts_update=attempts
        )


async def _status_of(factory, job_id: int) -> str:
    async with factory() as session:
        return (await query_job(session, job_id)).status


_CTX = {"redis": object()}


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["pending", "retrying"])
async def test_reaper_marks_orphaned_jobs_as_error(
    in_memory_session_factory, monkeypatch, status
):
    job = await _job_with_status(in_memory_session_factory, status, attempts=2)
    _fake_arq(monkeypatch, {})

    reaped = await worker.reap_orphaned_jobs(_CTX)

    async with in_memory_session_factory() as session:
        updated = await query_job(session, job.id)
    assert reaped == 1
    assert updated.status == "error"
    assert updated.error == worker.ORPHANED_ERROR
    assert updated.attempts == 2


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status, arq_status",
    [
        ("pending", JobStatus.queued),
        ("pending", JobStatus.in_progress),
        ("retrying", JobStatus.deferred),
        ("retrying", JobStatus.in_progress),
    ],
)
async def test_reaper_leaves_jobs_arq_still_has(
    in_memory_session_factory, monkeypatch, status, arq_status
):
    job = await _job_with_status(in_memory_session_factory, status)
    _fake_arq(monkeypatch, {arq_job_id(job.id): arq_status})

    reaped = await worker.reap_orphaned_jobs(_CTX)

    assert reaped == 0
    assert await _status_of(in_memory_session_factory, job.id) == status


@pytest.mark.anyio
async def test_reaper_looks_up_each_pending_job_by_its_derived_arq_id(
    in_memory_session_factory, monkeypatch
):
    pending = await _job_with_status(in_memory_session_factory, "pending")
    retrying = await _job_with_status(in_memory_session_factory, "retrying")
    for final in ["complete", "extraction_failed", "error"]:
        await _job_with_status(in_memory_session_factory, final)
    asked = _fake_arq(monkeypatch, {})

    await worker.reap_orphaned_jobs(_CTX)

    # finished jobs are never even looked up
    assert sorted(asked) == sorted([arq_job_id(pending.id), arq_job_id(retrying.id)])


@pytest.mark.anyio
async def test_reaper_does_not_overwrite_a_job_the_worker_finished_mid_sweep(
    in_memory_session_factory, monkeypatch
):
    job = await _job_with_status(in_memory_session_factory, "pending")

    async def worker_finishes_first(_arq_id):
        async with in_memory_session_factory() as session:
            await update_job(
                session, job_id=job.id, status_update="complete", attempts_update=1
            )

    _fake_arq(monkeypatch, {}, on_status=worker_finishes_first)

    reaped = await worker.reap_orphaned_jobs(_CTX)

    assert reaped == 0
    assert await _status_of(in_memory_session_factory, job.id) == "complete"


@pytest.mark.anyio
async def test_reaper_flags_but_does_not_reap_a_job_with_a_stored_arq_result(
    in_memory_session_factory, monkeypatch, caplog
):
    job = await _job_with_status(in_memory_session_factory, "pending")
    _fake_arq(monkeypatch, {arq_job_id(job.id): JobStatus.complete})

    with caplog.at_level(logging.ERROR, logger=worker.__name__):
        reaped = await worker.reap_orphaned_jobs(_CTX)

    assert reaped == 0
    assert await _status_of(in_memory_session_factory, job.id) == "pending"
    assert "keep_result" in caplog.text


@pytest.mark.anyio
async def test_reaper_sweeps_a_mix_and_counts_only_what_it_reaped(
    in_memory_session_factory, monkeypatch, caplog
):
    orphan_a = await _job_with_status(in_memory_session_factory, "pending")
    orphan_b = await _job_with_status(in_memory_session_factory, "retrying", 2)
    alive = await _job_with_status(in_memory_session_factory, "pending", 3)
    done = await _job_with_status(in_memory_session_factory, "complete")
    _fake_arq(monkeypatch, {arq_job_id(alive.id): JobStatus.queued})

    with caplog.at_level(logging.WARNING, logger=worker.__name__):
        reaped = await worker.reap_orphaned_jobs(_CTX)

    assert reaped == 2
    statuses = {
        j.id: await _status_of(in_memory_session_factory, j.id)
        for j in [orphan_a, orphan_b, alive, done]
    }
    assert statuses == {
        orphan_a.id: "error",
        orphan_b.id: "error",
        alive.id: "pending",
        done.id: "complete",
    }
    # the log names the status the row had, not the one just written
    assert f"Reaped orphaned job {orphan_b.id} (was retrying)" in caplog.text


@pytest.mark.anyio
async def test_reaped_job_can_be_retried_from_the_ledger(
    in_memory_session_factory, monkeypatch
):
    job = await _job_with_status(in_memory_session_factory, "pending")
    _fake_arq(monkeypatch, {})
    await worker.reap_orphaned_jobs(_CTX)
    pool = AsyncMock()

    async with in_memory_session_factory() as session:
        response = await retry_job(job_id=job.id, session=session, pool=pool)

    assert response.status == "pending"
    pool.enqueue_job.assert_awaited_once_with(
        "process_job", job.id, job.file_key, _job_id=arq_job_id(job.id)
    )


def test_worker_settings_schedule_the_reaper_at_startup():
    reapers = [
        c
        for c in worker.WorkerSettings.cron_jobs
        if c.coroutine is worker.reap_orphaned_jobs
    ]
    assert len(reapers) == 1
    assert reapers[0].run_at_startup is True


def test_worker_settings_keep_no_arq_results():
    # a stored result blocks re-enqueueing the same _job_id, silently breaking retry
    assert worker.WorkerSettings.keep_result == 0


def test_time_budget_is_ordered_so_the_inner_deadline_fires_first():
    assert (
        config.EXTRACTION_DEADLINE_S
        >= config.SPECIALIST_TIMEOUT_S + 2 * config.FRONTIER_TIMEOUT_S
    )
    assert config.JOB_TIMEOUT_S > config.EXTRACTION_DEADLINE_S
    assert worker.WorkerSettings.job_timeout == config.JOB_TIMEOUT_S


async def _hangs(**_kwargs):
    await asyncio.sleep(10)


@pytest.mark.anyio
async def test_process_job_deadline_marks_retrying_with_a_non_blank_error(
    in_memory_session_factory, monkeypatch
):
    job = await _make_job(in_memory_session_factory)
    monkeypatch.setattr(worker, "run_extraction", _hangs)
    monkeypatch.setattr(worker, "EXTRACTION_DEADLINE_S", 0.05)

    with pytest.raises(Retry):
        await worker.process_job({"job_try": 1}, job.id, job.file_key)

    async with in_memory_session_factory() as session:
        updated = await query_job(session, job.id)
    assert updated.status == "retrying"
    assert updated.error == "TimeoutError"


@pytest.mark.anyio
async def test_process_job_deadline_on_last_try_marks_error(
    in_memory_session_factory, monkeypatch
):
    job = await _make_job(in_memory_session_factory)
    monkeypatch.setattr(worker, "run_extraction", _hangs)
    monkeypatch.setattr(worker, "EXTRACTION_DEADLINE_S", 0.05)

    await worker.process_job(
        {"job_try": worker.WorkerSettings.max_tries}, job.id, job.file_key
    )

    async with in_memory_session_factory() as session:
        updated = await query_job(session, job.id)
    assert updated.status == "error"
    assert updated.error == "TimeoutError"


@pytest.mark.parametrize(
    "exc, expected",
    [
        (TimeoutError(), "TimeoutError"),
        (ValueError("bad input"), "ValueError: bad input"),
    ],
)
def test_describe_error_never_returns_a_blank_message(exc, expected):
    assert worker._describe_error(exc) == expected
