from dataclasses import replace
from invoice_pipeline.workflow.extractors import FrontierExtractor, SpecialistExtractor
from invoice_pipeline.workflow.state import ExtractionRecord, State
from invoice_pipeline.db.models import Extraction
from invoice_pipeline.db.operations import (
    find_duplicate,
    find_pending_duplicate,
    write_extractions,
)
from invoice_pipeline.util.reconciliation import ReconciliationIssue, reconcile
from invoice_pipeline.db.engine import session_factory

# only a re-read can fix these; duplicates and missing fields are facts of the document
FRONTIER_FIXABLE_CATEGORIES = frozenset({"mismatch"})


async def specialist_extract(specialist: SpecialistExtractor, state: State) -> dict:
    result = await specialist.extract_invoice(state["image"])
    record = ExtractionRecord(
        extractor="specialist",
        model=result.model_name,
        trigger="initial",
        invoice=result.invoice,
        parse_error=result.parse_error,
        raw_output=result.raw_output,
    )
    update = {"extractions": [*state["extractions"], record]}
    if result.invoice is not None:
        update["selected"] = len(state["extractions"])
    return update


def route_after_parsing(state: State) -> str:
    if state["selected"] is not None:
        return "success"
    return "needs_fallback"


async def frontier_extract(frontier: FrontierExtractor, state: State) -> dict:
    result = await frontier.extract_invoice(state["image"])
    record = ExtractionRecord(
        extractor="frontier",
        model=result.model_name,
        trigger="parse_fallback",
        invoice=result.invoice,
        parse_error=result.parse_error,
        raw_output=result.raw_output,
    )
    update = {"extractions": [*state["extractions"], record]}
    if result.invoice is not None:
        update["selected"] = len(state["extractions"])
    return update


async def validate_reconcile(state: State) -> dict:
    if state["selected"] is None:
        return {}
    index = state["selected"]
    extraction = state["extractions"][index]
    issues = reconcile(extraction.invoice)
    invoice_number = extraction.invoice.invoice_number
    vendor_name = extraction.invoice.vendor.name if extraction.invoice.vendor else None
    async with session_factory() as session:
        # an approved match is the stronger signal, so the ledger is checked first
        approved = await find_duplicate(session, invoice_number, vendor_name)
        pending_job_id = (
            None
            if approved is not None
            else await find_pending_duplicate(
                session, invoice_number, vendor_name, exclude_job_id=state["job_id"]
            )
        )
    if approved is not None:
        issues.append(
            ReconciliationIssue(
                category="duplicate",
                message=f"Duplicate invoice # — '{invoice_number}' is already in the ledger (job {approved.job_id})",
            )
        )
    elif pending_job_id is not None:
        issues.append(
            ReconciliationIssue(
                category="duplicate",
                message=f"Duplicate invoice # — '{invoice_number}' is also waiting for review (job {pending_job_id})",
            )
        )
    new_records = list(state["extractions"])
    new_records[index] = replace(extraction, issues=issues)
    return {"extractions": new_records}


async def frontier_review_fallback(frontier: FrontierExtractor, state: State) -> dict:
    selected_extraction = state["extractions"][state["selected"]]
    ctx = "; ".join(issue.message for issue in selected_extraction.issues)
    context = (
        f"\nValidation and reconciliation issues exist in the previous extraction "
        f"— take these into consideration: \n{ctx}"
    )
    result = await frontier.extract_invoice(state["image"], context)
    record = ExtractionRecord(
        extractor="frontier",
        model=result.model_name,
        trigger="reconcile_retry",
        invoice=result.invoice,
        parse_error=result.parse_error,
        raw_output=result.raw_output,
    )
    update = {"extractions": [*state["extractions"], record]}
    if result.invoice is not None:
        update["selected"] = len(state["extractions"])
    return update


def route_after_reconcile(state: State) -> str:
    if state["selected"] is None:
        return "total_failure"
    fixable = any(
        issue.category in FRONTIER_FIXABLE_CATEGORIES
        for issue in state["extractions"][state["selected"]].issues
    )
    if fixable and not any(r.extractor == "frontier" for r in state["extractions"]):
        return "retry_via_frontier"
    return "reconciled"


async def persist_extractions(state: State) -> dict:
    extractions = [
        Extraction(
            job_id=state["job_id"],
            extractor=record.extractor,
            model=record.model,
            trigger=record.trigger,
            invoice_data=record.invoice.model_dump(mode="json")
            if record.invoice is not None
            else None,
            parse_error=record.parse_error,
            raw_output=record.raw_output,
            issues=[issue.model_dump(mode="json") for issue in record.issues],
            is_final=index == state["selected"],
        )
        for index, record in enumerate(state["extractions"])
    ]
    async with session_factory() as session:
        await write_extractions(session, state["job_id"], extractions)
    return {}
