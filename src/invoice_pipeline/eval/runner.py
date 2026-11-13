import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from invoice_pipeline import config
from invoice_pipeline.eval.metrics import accuracy
from invoice_pipeline.eval.report import (
    EvaluationAccumulator,
    filtered_scores,
    overall_scores,
    score_invoice,
    summarize,
)
from invoice_pipeline.schema import Invoice
from invoice_pipeline.workflow.extractors import (
    ExtractionResult,
    FrontierExtractor,
    SpecialistExtractor,
)

GOLDEN_SET_PATH = config.PROJECT_ROOT / "data" / "golden" / "test.jsonl"
IMAGE_ROOT = config.PROJECT_ROOT / "data" / "processed"

# The benchmark has no job deadline and no arq around it to retry, so it keeps
# the SDKs' own defaults (600s, and 2 retries for the specialist) rather than
# the production budget in config — published numbers were produced this way.
_EVAL_TIMEOUT_S = 600.0
_EVAL_SPECIALIST_MAX_RETRIES = 2

# rajistics/sroie (the HF source behind data/processed/SROIE) only ever
# carries company/date/address/total per example — 0% coverage for every
# other field across all 624 training rows, vs. the golden set's own SROIE
# docs having e.g. invoice_number 88% of the time. The model was never
# taught these fields for SROIE-shaped input. This restricts a diagnostic
# score to the fields it actually got trained on, to separate "the model is
# bad at this" from "the training data never covered this" — it does NOT
# replace the full, honest per-source score above it.
_TRAINED_FIELDS_BY_SOURCE: dict[str, tuple[str, ...]] = {
    "sroie": ("vendor", "vendor.name", "vendor.address", "issue_date"),
}


class Predictor(Protocol):
    async def extract_invoice(self, img_path: str) -> ExtractionResult: ...


@dataclass
class GoldRecord:
    source: str
    gold: Invoice
    image_path: Path


@dataclass
class EvaluationRun:
    overall: EvaluationAccumulator = field(default_factory=EvaluationAccumulator)
    by_source: dict[str, EvaluationAccumulator] = field(default_factory=dict)
    total: int = 0
    parse_failures: int = 0
    latencies_seconds: list[float] = field(default_factory=list)


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[GoldRecord]:
    records = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            records.append(
                GoldRecord(
                    source=row["source"],
                    gold=Invoice.model_validate(row["invoice"]),
                    image_path=IMAGE_ROOT / row["image_path"],
                )
            )
    return records


def schema_conformance_rate(run: EvaluationRun) -> float:
    return accuracy(run.total - run.parse_failures, run.total)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round(fraction * (len(ordered) - 1))
    return ordered[index]


async def _score_record(
    predictor: Predictor,
    record: GoldRecord,
    run: EvaluationRun,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        start = time.perf_counter()
        result = await predictor.extract_invoice(str(record.image_path))
        run.latencies_seconds.append(time.perf_counter() - start)

    run.total += 1
    if result.invoice is None:
        run.parse_failures += 1
        return

    score_invoice(run.overall, result.invoice, record.gold)
    by_source = run.by_source.setdefault(record.source, EvaluationAccumulator())
    score_invoice(by_source, result.invoice, record.gold)


async def evaluate(
    predictor: Predictor, golden_set: list[GoldRecord], concurrency: int = 5
) -> EvaluationRun:
    run = EvaluationRun()
    semaphore = asyncio.Semaphore(concurrency)
    await asyncio.gather(
        *(_score_record(predictor, record, run, semaphore) for record in golden_set)
    )
    return run


def _print_field_breakdown(acc: EvaluationAccumulator) -> None:
    for report in summarize(acc):
        if report.metric == "accuracy":
            print(
                f"  {report.name:<24} accuracy={report.accuracy:.3f} (n={report.support})"
            )
        else:
            print(
                f"  {report.name:<24} precision={report.precision:.3f} "
                f"recall={report.recall:.3f} f1={report.f1:.3f} (n={report.support})"
            )


def print_report(run: EvaluationRun) -> None:
    print(f"Documents evaluated: {run.total}")
    print(f"Schema-conformance rate: {schema_conformance_rate(run):.1%}")
    if run.latencies_seconds:
        print(f"Latency p50: {percentile(run.latencies_seconds, 0.5):.2f}s")
        print(f"Latency p95: {percentile(run.latencies_seconds, 0.95):.2f}s")

    scored_documents = run.total - run.parse_failures
    if scored_documents == 0:
        print(
            "\nNo documents were successfully parsed — no field-level metrics to report."
        )
        return

    scores = overall_scores(run.overall)
    print(f"Overall macro-F1: {scores['macro_f1']:.3f}")
    print(f"Overall micro-F1: {scores['micro_f1']:.3f}")

    print("\nPer-field breakdown:")
    _print_field_breakdown(run.overall)

    for source, acc in sorted(run.by_source.items()):
        source_scores = overall_scores(acc)
        print(
            f"\n[{source}] macro-F1={source_scores['macro_f1']:.3f} "
            f"micro-F1={source_scores['micro_f1']:.3f}"
        )
        _print_field_breakdown(acc)

        trained_fields = _TRAINED_FIELDS_BY_SOURCE.get(source)
        if trained_fields:
            trained_scores = filtered_scores(acc, trained_fields)
            print(
                f"  [{source}] trained-fields-only ({', '.join(trained_fields)}): "
                f"macro-F1={trained_scores['macro_f1']:.3f} "
                f"micro-F1={trained_scores['micro_f1']:.3f}"
            )


def _build_predictor(
    name: str,
    model: str | None = None,
    seed: int | None = None,
    base_url: str | None = None,
) -> Predictor:
    if name == "specialist":
        return SpecialistExtractor(
            base_url=base_url or config.VLLM_BASE_URL,
            api_key=config.VLLM_API_KEY,
            model_name=model or config.SPECIALIST_MODEL,
            timeout=_EVAL_TIMEOUT_S,
            max_retries=_EVAL_SPECIALIST_MAX_RETRIES,
            seed=seed,
        )
    if name == "frontier":
        return FrontierExtractor(
            model_name=model or config.FRONTIER_MODEL, timeout=_EVAL_TIMEOUT_S
        )
    raise ValueError(f"Unknown extractor: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the invoice extraction benchmark against the golden set."
    )
    parser.add_argument(
        "--extractor", choices=["specialist", "frontier"], required=True
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model name (defaults to config.FRONTIER_MODEL / "
        "config.SPECIALIST_MODEL depending on --extractor)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible sampling. Only affects --extractor specialist "
        "(vLLM/OpenAI-compatible) — Anthropic's API has no seed parameter, so this "
        "is ignored for --extractor frontier.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override the vLLM base URL (defaults to config.VLLM_BASE_URL). "
        "Only relevant for --extractor specialist — e.g. pointing at a separate "
        "deployment serving a different checkpoint for comparison.",
    )
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--golden-set", type=Path, default=GOLDEN_SET_PATH)
    args = parser.parse_args()

    if args.extractor == "frontier":
        print(
            "Note: --seed has no effect for --extractor frontier "
            "(Anthropic's API has no seed parameter)."
        )

    predictor = _build_predictor(
        args.extractor, model=args.model, seed=args.seed, base_url=args.base_url
    )
    golden_set = load_golden_set(args.golden_set)
    run = asyncio.run(evaluate(predictor, golden_set, concurrency=args.concurrency))
    print_report(run)


if __name__ == "__main__":
    main()
