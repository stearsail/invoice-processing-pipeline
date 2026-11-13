from functools import partial
from invoice_pipeline.workflow.extractors import FrontierExtractor, SpecialistExtractor
from langgraph.graph import END, START, StateGraph
from invoice_pipeline.workflow.nodes import (
    frontier_extract,
    frontier_review_fallback,
    ledger_write,
    route_after_parsing,
    route_after_reconcile,
    specialist_extract,
    validate_reconcile,
)
from invoice_pipeline.workflow.state import Context, State
from invoice_pipeline.config import (
    FRONTIER_MODEL,
    SPECIALIST_MODEL,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    FRONTIER_TIMEOUT_S,
    SPECIALIST_TIMEOUT_S,
)


def build_graph(specialist: SpecialistExtractor, frontier: FrontierExtractor):
    builder = StateGraph(state_schema=State, context_schema=Context)
    # partial is used because add_node stores the function and calls later as node(state), so object hadnded must be a CALLABLE that takes one argument
    # partial supplies one argument now (at node construction) and the rest later (the specialist arg for extraction) — hence partial aplication
    builder.add_node("specialist_extract", partial(specialist_extract, specialist))
    builder.add_node("frontier_extract_fallback", partial(frontier_extract, frontier))
    builder.add_node("validate_reconcile", validate_reconcile)
    builder.add_node(
        "frontier_review_fallback", partial(frontier_review_fallback, frontier)
    )
    builder.add_node("ledger_write", ledger_write)

    builder.add_edge(START, "specialist_extract")
    builder.add_conditional_edges(
        "specialist_extract",
        route_after_parsing,
        {
            "success": "validate_reconcile",
            "needs_fallback": "frontier_extract_fallback",
        },
    )
    builder.add_edge("frontier_extract_fallback", "validate_reconcile")
    builder.add_conditional_edges(
        "validate_reconcile",
        route_after_reconcile,
        {
            "retry_via_frontier": "frontier_review_fallback",
            "reconciled": "ledger_write",
            "total_failure": END,
        },
    )
    builder.add_edge("frontier_review_fallback", "validate_reconcile")
    builder.add_edge("ledger_write", END)
    return builder.compile()


graph = build_graph(
    specialist=SpecialistExtractor(
        VLLM_BASE_URL, VLLM_API_KEY, SPECIALIST_MODEL, timeout=SPECIALIST_TIMEOUT_S
    ),
    frontier=FrontierExtractor(
        FRONTIER_MODEL, timeout=FRONTIER_TIMEOUT_S, temperature=0.1
    ),
)
