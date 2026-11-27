from datetime import date, datetime
from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, ConfigDict
from invoice_pipeline.schema import Invoice
from invoice_pipeline.util.reconciliation import ReconciliationIssue

HealthStatus = Literal["ok", "unavailable"]


class HealthResponse(BaseModel):
    database: HealthStatus
    redis: HealthStatus


class JobCreation(BaseModel):
    job_id: int
    status: str


RejectReason = Literal[
    "unsupported_type",
    "empty",
    "too_large",
    "type_mismatch",
    "unreadable_image",
    "duplicate_in_batch",
]


class RejectedFile(BaseModel):
    filename: str
    content_type: str | None
    reason: RejectReason


class UploadResponse(BaseModel):
    jobs: list[JobCreation]
    rejected: list[RejectedFile]


class LedgerEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    invoice_number: str | None
    vendor_name: str | None
    issue_date: date | None
    currency: str
    grand_total: Decimal
    invoice_data: Invoice
    source_extraction_id: int | None
    corrected: bool | None
    reviewed_at: datetime


class JobResponse(BaseModel):
    job_id: int
    created_at: datetime
    status: str
    error: str | None
    attempts: int


class ExtractionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    extractor: Literal["specialist", "frontier"]
    model: str
    trigger: Literal["initial", "parse_fallback", "reconcile_retry"]
    invoice_data: Invoice | None
    parse_error: str | None
    issues: list[ReconciliationIssue]
    created_at: datetime


class JobDetailResponse(JobResponse):
    ledger_entry: LedgerEntryResponse | None
    ledger_entry_error: str | None = None
    extraction: ExtractionResponse | None
    extraction_error: str | None = None
