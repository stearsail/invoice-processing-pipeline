import asyncio
import logging
from arq import ArqRedis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from fastapi import Request

logger = logging.getLogger(__name__)


async def get_redis_pool(request: Request) -> ArqRedis:
    return request.app.state.redis


async def is_redis_reachable(timeout: float, pool: ArqRedis) -> bool:
    try:
        async with asyncio.timeout(timeout):
            await pool.ping()
        return True
    except (RedisConnectionError, RedisTimeoutError, TimeoutError) as e:
        logger.warning("Redis health check failed: %r", e)
        return False
