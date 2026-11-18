import io
import logging
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image
from starlette.datastructures import Headers
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from invoice_pipeline.api.routers import extraction
from invoice_pipeline.api.schema.responses import RejectedFile
from invoice_pipeline.db.models import Job
from invoice_pipeline.queue.jobs import arq_job_id


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def in_memory_session_factory(monkeypatch, tmp_path):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(extraction, "UPLOADS_DIR", tmp_path)
    yield factory
    await engine.dispose()


def _image(fmt: str = "PNG") -> bytes:
    # random noise: every call yields distinct bytes, and a PNG large enough to truncate
    buf = io.BytesIO()
    Image.effect_noise((64, 64), 64).convert("RGB").save(buf, format=fmt)
    return buf.getvalue()


def _upload_file(
    name: str,
    data: bytes | None = None,
    content_type: str = "image/png",
    size: int | None = None,
) -> UploadFile:
    payload = _image() if data is None else data
    # Starlette always sets size on real requests; mirror that so tests take the same path
    return UploadFile(
        file=io.BytesIO(payload),
        filename=name,
        size=len(payload) if size is None else size,
        headers=Headers({"content-type": content_type}),
    )


async def _upload(factory, files, pool=None):
    async with factory() as session:
        return await extraction.upload(
            files=files, session=session, pool=pool or AsyncMock()
        )


async def _job_rows(factory) -> list[Job]:
    async with factory() as session:
        return (await session.exec(select(Job))).all()


@pytest.mark.anyio
async def test_upload_creates_and_enqueues_a_job_per_valid_file(
    in_memory_session_factory,
):
    pool = AsyncMock()
    files = [
        _upload_file("a.png"),
        _upload_file("b.jpg", _image("JPEG"), "image/jpeg"),
        _upload_file("c.png"),
    ]

    response = await _upload(in_memory_session_factory, files, pool)

    assert [j.status for j in response.jobs] == ["pending"] * 3
    assert response.rejected == []
    enqueued = [call.args[1] for call in pool.enqueue_job.await_args_list]
    assert enqueued == [j.job_id for j in response.jobs]


@pytest.mark.anyio
async def test_upload_saves_the_whole_file_under_its_detected_format(
    in_memory_session_factory, tmp_path
):
    # labelled PNG, actually a JPEG: the bytes decide the extension
    jpeg = _image("JPEG")

    response = await _upload(
        in_memory_session_factory, [_upload_file("scan.png", jpeg, "image/png")]
    )

    [job] = await _job_rows(in_memory_session_factory)
    assert job.id == response.jobs[0].job_id
    assert job.file_key.endswith(".jpeg")
    # the checks read the file; the save must still get every byte
    assert (tmp_path / job.file_key).read_bytes() == jpeg


@pytest.mark.anyio
async def test_upload_accepts_a_real_png_with_a_generic_content_type(
    in_memory_session_factory,
):
    files = [_upload_file("scan", content_type="application/octet-stream")]

    response = await _upload(in_memory_session_factory, files)

    assert len(response.jobs) == 1
    [job] = await _job_rows(in_memory_session_factory)
    assert job.file_key.endswith(".png")


_PNG = _image()

_BAD_FILES = {
    "empty": ("empty.png", b"", "image/png", None),
    "too_large": ("huge.png", _PNG, "image/png", extraction._MAX_UPLOAD_BYTES + 1),
    "type_mismatch": ("evil.png", b"MZ" + b"\0" * 200, "image/png", None),
    "unsupported_type": (
        "doc.pdf",
        b"%PDF-1.7\n" + b"\0" * 200,
        "application/pdf",
        None,
    ),
    "unreadable_image": ("cut.png", _PNG[: len(_PNG) // 2], "image/png", None),
}


@pytest.mark.anyio
@pytest.mark.parametrize("reason", list(_BAD_FILES))
async def test_upload_rejects_a_bad_file_and_still_processes_the_rest(
    in_memory_session_factory, tmp_path, reason
):
    name, data, content_type, size = _BAD_FILES[reason]
    pool = AsyncMock()
    files = [_upload_file("good.png"), _upload_file(name, data, content_type, size)]

    response = await _upload(in_memory_session_factory, files, pool)

    assert len(response.jobs) == 1
    assert response.rejected == [
        RejectedFile(filename=name, content_type=content_type, reason=reason)
    ]
    # the rejected file is never saved, given a job row, or queued
    assert len(list(tmp_path.iterdir())) == 1
    assert len(await _job_rows(in_memory_session_factory)) == 1
    assert pool.enqueue_job.await_count == 1


@pytest.mark.anyio
async def test_upload_checks_size_before_content(in_memory_session_factory):
    # an empty PDF is reported as empty, not as an unsupported type
    files = [_upload_file("good.png"), _upload_file("x.pdf", b"", "application/pdf")]

    response = await _upload(in_memory_session_factory, files)

    assert [r.reason for r in response.rejected] == ["empty"]


@pytest.mark.anyio
async def test_upload_keeps_the_first_copy_of_a_duplicate(in_memory_session_factory):
    same = _image()
    files = [
        _upload_file("first.png", same),
        _upload_file("other.png"),
        _upload_file("copy.png", same),
    ]

    response = await _upload(in_memory_session_factory, files)

    assert len(response.jobs) == 2
    assert [(r.filename, r.reason) for r in response.rejected] == [
        ("copy.png", "duplicate_in_batch")
    ]


@pytest.mark.anyio
async def test_upload_with_every_file_rejected_is_422_with_grouped_reasons(
    in_memory_session_factory, tmp_path
):
    pool = AsyncMock()
    files = [
        _upload_file("a.pdf", b"%PDF-1.7\n" + b"\0" * 50, "application/pdf"),
        _upload_file("b.pdf", b"%PDF-1.7\n" + b"\1" * 50, "application/pdf"),
        _upload_file("c.png", b"", "image/png"),
    ]

    with pytest.raises(HTTPException) as exc_info:
        await _upload(in_memory_session_factory, files, pool)

    assert exc_info.value.status_code == 422
    assert "3" in exc_info.value.detail
    assert "2 unsupported_type" in exc_info.value.detail
    assert "1 empty" in exc_info.value.detail
    assert list(tmp_path.iterdir()) == []
    assert await _job_rows(in_memory_session_factory) == []
    pool.enqueue_job.assert_not_awaited()


@pytest.mark.anyio
async def test_upload_passes_the_derived_arq_id_as_arqs_job_id(
    in_memory_session_factory,
):
    pool = AsyncMock()

    response = await _upload(in_memory_session_factory, [_upload_file("a.png")], pool)

    [job] = response.jobs
    call = pool.enqueue_job.await_args
    # positional args go to process_job: the DB id must stay an int
    assert call.args[1] == job.job_id
    assert call.kwargs["_job_id"] == arq_job_id(job.job_id)


@pytest.mark.anyio
async def test_upload_logs_and_still_responds_when_arq_already_has_the_job(
    in_memory_session_factory, caplog
):
    pool = AsyncMock()
    pool.enqueue_job.return_value = None

    with caplog.at_level(logging.WARNING, logger=extraction.__name__):
        response = await _upload(
            in_memory_session_factory, [_upload_file("a.png")], pool
        )

    [job] = response.jobs
    assert job.status == "pending"
    assert arq_job_id(job.job_id) in caplog.text


@pytest.mark.parametrize(
    "data, expected", [(b"", "empty"), (b"x" * 10, None)], ids=["empty", "small"]
)
def test_check_size_measures_a_file_without_a_declared_size(data, expected):
    file = UploadFile(file=io.BytesIO(data), filename="f.png", size=None)

    assert extraction._check_size(file) == expected
    assert file.file.tell() == 0
