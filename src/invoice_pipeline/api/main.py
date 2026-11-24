import logging
from arq.connections import RedisSettings
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.exc import OperationalError
from redis.exceptions import TimeoutError as RedisTimeoutError
from redis.exceptions import ConnectionError as RedisConnectionError
from invoice_pipeline import config
from invoice_pipeline.api.routers.extraction import router as extraction_router
from invoice_pipeline.api.routers.ledger import router as ledger_router
from contextlib import asynccontextmanager
from invoice_pipeline.logging_config import setup_logging
from arq import create_pool

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis = await create_pool(
        RedisSettings(host=config.REDIS_HOST, port=config.REDIS_PORT)
    )
    app.state.redis = redis
    try:
        yield
    finally:
        await redis.close()


app = FastAPI(
    title="Invoice/Receipt OCR extraction pipeline", version="0.1.0", lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello!"}


_OUTAGE_MESSAGES = {
    OperationalError: "Database is unavailable",
    PoolTimeoutError: "Database is unavailable",
    RedisTimeoutError: "Queue is unavailable",
    RedisConnectionError: "Queue is unavailable",
}


async def backend_exception_handler(request: Request, exc: Exception):
    for cls, msg in _OUTAGE_MESSAGES.items():
        if isinstance(exc, cls):
            logger.warning(
                "%s: %s %s %s", msg, request.method, request.url.path, repr(exc)
            )
            return JSONResponse(
                content={"detail": msg},
                status_code=503,
            )


for cls in _OUTAGE_MESSAGES.keys():
    app.add_exception_handler(cls, backend_exception_handler)
app.include_router(extraction_router)
app.include_router(ledger_router)
