from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Literal
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import (
    JSON,
    DateTime,
    Index,
    SQLModel,
    Field,
    String,
    UniqueConstraint,
    text,
)

JsonB = JSON().with_variant(JSONB(), "postgresql")


class LedgerEntry(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("job_id", name="uq_ledgerentry_job_id"),
        Index("ix_ledgerentry_vendor_invoice", "vendor_name", "invoice_number"),
    )

    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id", ondelete="CASCADE")
    source_extraction_id: int | None = Field(default=None, foreign_key="extraction.id")
    invoice_data: dict = Field(sa_type=JsonB)
    invoice_number: str | None = Field(default=None)
    vendor_name: str | None = Field(default=None)
    issue_date: date | None = Field(default=None)
    currency: str
    grand_total: Decimal
    corrected: bool | None = Field(default=None)
    reviewed_at: datetime = Field(sa_type=DateTime(timezone=True))


class Extraction(SQLModel, table=True):
    __table_args__ = (
        Index(
            "uq_extraction_one_final_per_job",
            "job_id",
            unique=True,
            postgresql_where=text("is_final"),
            sqlite_where=text("is_final"),
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id", index=True, ondelete="CASCADE")
    run: int
    extractor: Literal["specialist", "frontier"] = Field(sa_type=String)
    model: str
    invoice_data: dict | None = Field(default=None, sa_type=JsonB)
    parse_error: str | None = Field(default=None)
    raw_output: str | None = Field(default=None)
    issues: list[dict] = Field(default_factory=list, sa_type=JsonB)
    is_final: bool = Field(default=False)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=DateTime(timezone=True)
    )
    trigger: Literal["initial", "parse_fallback", "reconcile_retry"] = Field(
        sa_type=String
    )


class Job(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)
    attempts: int = Field(default=1)
    status: Literal["pending", "complete", "extraction_failed", "error", "retrying"] = (
        Field(default="pending", sa_type=String)
    )
    error: str | None = Field(default=None)
    file_key: str
