import logging

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import BusyLoadingError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from invoice_pipeline.api.main import app
from invoice_pipeline.db.engine import get_async_session
from invoice_pipeline.queue.pool import get_redis_pool

_ORIGIN = "http://localhost:5173"


def _db_error(cls):
    return cls("SELECT 1", {}, Exception("could not connect"))


async def _no_session():
    yield None


@pytest.fixture
def client():
    # no `with`: lifespan (and its Redis connection) never runs
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def _raise_from(dependency, exc):
    async def failing():
        raise exc

    app.dependency_overrides[dependency] = failing


@pytest.mark.parametrize(
    "exc",
    [_db_error(OperationalError), PoolTimeoutError("QueuePool limit reached")],
    ids=["db_down", "pool_exhausted"],
)
def test_db_outage_returns_503_with_cors(client, exc):
    _raise_from(get_async_session, exc)

    r = client.get("/ledger/pending", headers={"Origin": _ORIGIN})

    assert r.status_code == 503
    assert r.json() == {"detail": "Database is unavailable"}
    assert r.headers["access-control-allow-origin"] == _ORIGIN


@pytest.mark.parametrize(
    "exc",
    [
        RedisConnectionError("Temporary failure in name resolution"),
        RedisTimeoutError("Timeout reading from socket"),
        BusyLoadingError("Redis is loading the dataset in memory"),
    ],
    ids=["redis_down", "redis_timeout", "redis_restarting_subclass"],
)
def test_queue_outage_returns_503_with_cors(client, exc):
    app.dependency_overrides[get_async_session] = _no_session
    _raise_from(get_redis_pool, exc)

    r = client.post("/ledger/1/retry", headers={"Origin": _ORIGIN})

    assert r.status_code == 503
    assert r.json() == {"detail": "Queue is unavailable"}
    assert r.headers["access-control-allow-origin"] == _ORIGIN


def test_outage_logs_one_warning_line_with_request_and_cause(client, caplog):
    _raise_from(get_async_session, _db_error(OperationalError))

    with caplog.at_level(logging.WARNING, logger="invoice_pipeline.api.main"):
        client.get("/ledger/pending")

    [record] = [r for r in caplog.records if r.name == "invoice_pipeline.api.main"]
    line = record.getMessage()
    assert record.levelno == logging.WARNING
    assert "\n" not in line
    assert "GET /ledger/pending" in line
    assert "could not connect" in line
    assert record.exc_info is None


def test_non_outage_db_error_is_not_reported_as_unavailable(client):
    # bugs and conflicts must stay a 500, never a "try again later"
    _raise_from(
        get_async_session, IntegrityError("INSERT", {}, Exception("duplicate key"))
    )

    r = client.get("/ledger/pending", headers={"Origin": _ORIGIN})

    assert r.status_code == 500
