from dataclasses import dataclass
from invoice_pipeline.workflow import graph
from typing_extensions import Literal


@dataclass
class ExtractionResult:
    status: Literal["complete", "extraction_failed"]
    ledger_entry_id: int | None = None
    error: str | None = None


async def run_extraction(job_id: int, img_path: str) -> ExtractionResult:
    final_state = await graph.ainvoke(
        input={
            "job_id": job_id,
            "image": img_path,
            "invoice": None,
            "extracted_by": None,
            "frontier_attempted": False,
        }
    )
    if final_state["invoice"] is None:
        return ExtractionResult(
            status="extraction_failed",
            error="; ".join(
                issue.message for issue in final_state["reconciliation_issues"]
            ),
        )
    return ExtractionResult(
        status="complete",
        ledger_entry_id=final_state.get("ledger_entry_id"),
        error=None,
    )
