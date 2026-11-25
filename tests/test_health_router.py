import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from invoice_pipeline.api.main import app
from invoice_pipeline.api.routers import health
from invoice_pipeline.queue.pool import get_redis_pool


@pytest.fixture
def client():
    # no `with`: lifespan (and its Redis connection) never runs
    app.dependency_overrides[get_redis_pool] = lambda: object()
    yield TestClient(app)
    app.dependency_overrides.clear()


def _checks(monkeypatch, *, db_ok: bool, redis_ok: bool, delay: float = 0):
    async def db(timeout):
        await asyncio.sleep(delay)
        return db_ok

    async def redis(timeout, pool):
        await asyncio.sleep(delay)
        return redis_ok

    monkeypatch.setattr(health, "is_database_reachable", db)
    monkeypatch.setattr(health, "is_redis_reachable", redis)


def test_all_up_returns_200(client, monkeypatch):
    _checks(monkeypatch, db_ok=True, redis_ok=True)

    r = client.get("/health")

    assert r.status_code == 200
    assert r.json() == {"database": "ok", "redis": "ok"}


@pytest.mark.parametrize(
    "db_ok, redis_ok, body",
    [
        (False, True, {"database": "unavailable", "redis": "ok"}),
        (True, False, {"database": "ok", "redis": "unavailable"}),
        (False, False, {"database": "unavailable", "redis": "unavailable"}),
    ],
    ids=["db_down", "redis_down", "both_down"],
)
def test_any_dependency_down_returns_503_with_per_dependency_body(
    client, monkeypatch, db_ok, redis_ok, body
):
    _checks(monkeypatch, db_ok=db_ok, redis_ok=redis_ok)

    r = client.get("/health")

    assert r.status_code == 503
    assert r.json() == body


def test_checks_run_concurrently(client, monkeypatch):
    # sequential would take 2 * 0.3s; compose's 3s healthcheck budget depends on this
    _checks(monkeypatch, db_ok=True, redis_ok=True, delay=0.3)

    start = time.monotonic()
    client.get("/health")

    assert time.monotonic() - start < 0.5


def test_health_check_budget_fits_compose_timeout():
    assert health.HEALTH_CHECK_TIMEOUT_S < 3
