from dataclasses import dataclass, field
from typing import Literal, TypedDict
from invoice_pipeline.schema import Invoice
from invoice_pipeline.util.reconciliation import ReconciliationIssue


@dataclass
class ExtractionRecord:
    extractor: Literal["specialist", "frontier"]
    model: str
    trigger: Literal["initial", "parse_fallback", "reconcile_retry"]
    invoice: Invoice | None
    parse_error: str | None
    raw_output: str | None
    issues: list[ReconciliationIssue] = field(default_factory=list)


class State(TypedDict):
    job_id: int
    image: str
    extractions: list[ExtractionRecord]
    selected: int | None
