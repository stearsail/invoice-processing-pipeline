from dataclasses import dataclass
from typing import Literal

from invoice_pipeline.workflow import graph


@dataclass
class ExtractionResult:
    status: Literal["complete", "extraction_failed"]
    error: str | None = None


async def run_extraction(job_id: int, img_path: str) -> ExtractionResult:
    final_state = await graph.ainvoke(
        input={
            "job_id": job_id,
            "image": img_path,
            "extractions": [],
            "selected": None,
        }
    )
    if final_state["selected"] is None:
        # nothing parsed, so every record carries its own parse_error
        return ExtractionResult(
            status="extraction_failed",
            error="; ".join(r.parse_error for r in final_state["extractions"]),
        )
    return ExtractionResult(status="complete")
