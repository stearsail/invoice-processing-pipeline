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
    id: int | None
    invoice_number: str | None
    vendor_name: str | None
    issue_date: date | None
    currency: str
    grand_total: Decimal
    invoice_data: Invoice
    created_at: datetime
    needs_review: bool
    review_reason: list[ReconciliationIssue] | None
    extracted_by: Literal["specialist", "frontier"] | None
    reviewed_with_changes: bool | None


class JobResponse(BaseModel):
    job_id: int
    created_at: datetime
    status: str
    error: str | None
    attempts: int


class JobEntryPairResponse(JobResponse):
    ledger_entry: LedgerEntryResponse | None
    ledger_entry_error: str | None = None


class PendingJobResponse(JobResponse):
    pass


class ErroredJobResponse(JobResponse):
    pass
