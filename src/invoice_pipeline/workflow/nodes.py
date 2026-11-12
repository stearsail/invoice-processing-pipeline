from invoice_pipeline.workflow.extractors import FrontierExtractor, SpecialistExtractor
from invoice_pipeline.workflow.state import State
from invoice_pipeline.db.operations import find_duplicate, write_entry
from invoice_pipeline.util.reconciliation import ReconciliationIssue, reconcile
from invoice_pipeline.db.engine import session_factory


async def specialist_extract(specialist: SpecialistExtractor, state: State) -> dict:
    result = await specialist.extract_invoice(state["image"])
    if result.invoice is None:
        extracted_by = None
    else:
        extracted_by = "specialist"
    return {
        "invoice": result.invoice,
        "parse_error": result.parse_error,
        "extracted_by": extracted_by,
    }


async def frontier_extract(
    frontier: FrontierExtractor, state: State, extra_context: str = ""
) -> dict:
    result = await frontier.extract_invoice(state["image"], extra_context)
    if result.invoice is None:
        extracted_by = None
    else:
        extracted_by = "frontier"
    return {
        "invoice": result.invoice,
        "parse_error": result.parse_error,
        "extracted_by": extracted_by,
        "frontier_attempted": True,
    }


async def validate_reconcile(state: State) -> dict:
    if state["invoice"] is None:
        return {
            "reconciliation_issues": [
                ReconciliationIssue(
                    category="unverifiable",
                    message=f"Extraction failed entirely — {state['parse_error']}",
                )
            ]
        }
    issues = reconcile(state["invoice"])
    vendor_name = state["invoice"].vendor.name if state["invoice"].vendor else None
    async with session_factory() as session:
        duplicate = await find_duplicate(
            session, state["invoice"].invoice_number, vendor_name
        )
    if duplicate is not None:
        issues.append(
            ReconciliationIssue(
                category="duplicate",
                message=f"Duplicate invoice # — '{state['invoice'].invoice_number}' already exists in ledger (Entry ID {duplicate.id})",
            )
        )
    return {"reconciliation_issues": issues}


async def frontier_review_fallback(frontier: FrontierExtractor, state: State) -> dict:
    context = (
        f"Validation and reconciliation issues exist in the previous extraction "
        f"— take these into consideration: {state['reconciliation_issues']}"
    )
    result = await frontier.extract_invoice(state["image"], context)
    update = {"parse_error": result.parse_error, "frontier_attempted": True}
    if result.invoice is not None:
        update["invoice"] = result.invoice
        update["extracted_by"] = "frontier"
    return update


def route_after_parsing(state: State) -> str:
    if state["parse_error"] is None:
        return "success"
    return "needs_fallback"


def route_after_reconcile(state: State) -> str:
    if state["invoice"] is None:
        return "total_failure"
    if state["reconciliation_issues"] and not state["frontier_attempted"]:
        return "retry_via_frontier"
    return "reconciled"


async def ledger_write(state: State) -> dict:
    # every extraction needs human verification — reconciliation issues are informational
    # needs_review flips to false when a verifier saves the entry (update_entry)
    review_reason = [issue.model_dump() for issue in state["reconciliation_issues"]]
    async with session_factory() as session:
        entry = await write_entry(
            session,
            state["job_id"],
            state["invoice"],
            needs_review=True,
            review_reason=review_reason,
            extracted_by=state["extracted_by"],
        )
    return {"ledger_entry_id": entry.id}
