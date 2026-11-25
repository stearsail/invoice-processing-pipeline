import asyncio

from arq import ArqRedis
from fastapi import APIRouter, Depends, Response
from invoice_pipeline.api.schema.responses import HealthResponse, HealthStatus
from invoice_pipeline.config import HEALTH_CHECK_TIMEOUT_S
from invoice_pipeline.db.engine import is_database_reachable
from invoice_pipeline.queue.pool import get_redis_pool, is_redis_reachable


router = APIRouter(tags=["health"])


def _check_health_status(ok: bool) -> HealthStatus:
    if ok:
        return "ok"
    return "unavailable"


@router.get("/health")
async def get_health(
    response: Response, pool: ArqRedis = Depends(get_redis_pool)
) -> HealthResponse:
    db_ok, redis_ok = await asyncio.gather(
        is_database_reachable(HEALTH_CHECK_TIMEOUT_S),
        is_redis_reachable(HEALTH_CHECK_TIMEOUT_S, pool=pool),
    )
    response.status_code = 200 if db_ok and redis_ok else 503
    return HealthResponse(
        database=_check_health_status(db_ok), redis=_check_health_status(redis_ok)
    )
