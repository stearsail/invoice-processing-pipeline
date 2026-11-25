import asyncio
import time
from contextlib import asynccontextmanager

import pytest
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import create_async_engine

from invoice_pipeline.db import engine as engine_module


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _FakeEngine:
    def __init__(self, on_connect):
        self._on_connect = on_connect

    @asynccontextmanager
    async def connect(self):
        await self._on_connect()
        yield


@pytest.mark.anyio
async def test_reachable_database(monkeypatch):
    sqlite = create_async_engine("sqlite+aiosqlite:///:memory:")
    monkeypatch.setattr(engine_module, "engine", sqlite)

    assert await engine_module.is_database_reachable(timeout=2) is True
    await sqlite.dispose()


@pytest.mark.anyio
async def test_refused_connection_is_unreachable(monkeypatch, caplog):
    # port 1 on localhost: nothing listens, connect is refused immediately
    dead = create_async_engine("postgresql+psycopg://u:p@127.0.0.1:1/x")
    monkeypatch.setattr(engine_module, "engine", dead)

    assert await engine_module.is_database_reachable(timeout=5) is False
    assert "OperationalError" in caplog.text
    await dead.dispose()


@pytest.mark.anyio
async def test_hanging_connect_hits_timeout(monkeypatch):
    async def hang():
        await asyncio.sleep(60)

    monkeypatch.setattr(engine_module, "engine", _FakeEngine(hang))

    start = time.monotonic()
    assert await engine_module.is_database_reachable(timeout=0.1) is False
    assert time.monotonic() - start < 1


@pytest.mark.anyio
async def test_non_outage_error_propagates(monkeypatch):
    async def bug():
        raise ProgrammingError("SELECT 1", {}, Exception("syntax error"))

    monkeypatch.setattr(engine_module, "engine", _FakeEngine(bug))

    with pytest.raises(ProgrammingError):
        await engine_module.is_database_reachable(timeout=2)
