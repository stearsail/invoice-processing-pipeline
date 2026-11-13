import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)  # src/invoice_pipeline/ -> src/ -> repo root

# LOCAL STORAGE FOR PROJECT SCOPE, NO S3/R2
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY")
SPECIALIST_MODEL = os.environ.get("SPECIALIST_MODEL", "qwen3-vl-fullds-merged-r8")
FRONTIER_MODEL = os.environ.get("FRONTIER_MODEL", "claude-haiku-4-5-20251001")

# TIMEOUT VALUES FOR JOBS AND EXTRACTORS
SPECIALIST_TIMEOUT_S = 150  # S
FRONTIER_TIMEOUT_S = 60  # F * 2 FOR BOTH EXTRACTION AND REVIEW
EXTRACTION_DEADLINE_S = (
    SPECIALIST_TIMEOUT_S + 2 * FRONTIER_TIMEOUT_S + 30
)  # THIS IS S + 2F + 30(SLACK)
JOB_TIMEOUT_S = (
    EXTRACTION_DEADLINE_S + 30
)  # D+30(SLACK) FOR STATUS WRITE WHEN DEADLINE FIRES

REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = os.environ.get("REDIS_PORT")
