import json
from decimal import Decimal
from pathlib import Path

import pytest

from invoice_pipeline import config
from invoice_pipeline.eval.runner import (
    EvaluationRun,
    GoldRecord,
    _build_predictor,
    evaluate,
    load_golden_set,
    percentile,
    print_report,
    schema_conformance_rate,
)
from invoice_pipeline.schema import Invoice
from invoice_pipeline.workflow.extractors import (
    ExtractionResult,
    FrontierExtractor,
    SpecialistExtractor,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _invoice(**overrides) -> Invoice:
    defaults = dict(
        document_type="invoice",
        currency="USD",
        grand_total=Decimal("100"),
    )
    defaults.update(overrides)
    return Invoice(**defaults)


class _FakePredictor:
    def __init__(self, responses: dict[str, ExtractionResult]):
        self._responses = responses
        self.calls: list[str] = []

    async def extract_invoice(self, img_path: str) -> ExtractionResult:
        self.calls.append(img_path)
        return self._responses[img_path]


@pytest.mark.anyio
async def test_evaluate_scores_successful_extractions():
    gold = _invoice()
    record = GoldRecord(source="fatura", gold=gold, image_path=Path("a.png"))
    predictor = _FakePredictor(
        {
            "a.png": ExtractionResult(
                invoice=gold, parse_error=None, model_name="m", raw_output=None
            )
        }
    )

    run = await evaluate(predictor, [record])

    assert run.total == 1
    assert run.parse_failures == 0
    assert run.overall.accuracy_fields["document_type"].correct == 1
    assert "fatura" in run.by_source
    assert run.by_source["fatura"].accuracy_fields["document_type"].correct == 1


@pytest.mark.anyio
async def test_evaluate_counts_parse_failures_without_scoring_fields():
    record = GoldRecord(source="sroie", gold=_invoice(), image_path=Path("b.png"))
    predictor = _FakePredictor(
        {
            "b.png": ExtractionResult(
                invoice=None,
                parse_error="malformed: not json",
                model_name="m",
                raw_output=None,
            )
        }
    )

    run = await evaluate(predictor, [record])

    assert run.total == 1
    assert run.parse_failures == 1
    assert run.overall.accuracy_fields == {}
    assert run.by_source == {}


@pytest.mark.anyio
async def test_evaluate_records_one_latency_sample_per_document():
    gold = _invoice()
    records = [
        GoldRecord(source="fatura", gold=gold, image_path=Path(f"{i}.png"))
        for i in range(3)
    ]
    predictor = _FakePredictor(
        {
            f"{i}.png": ExtractionResult(
                invoice=gold, parse_error=None, model_name="m", raw_output=None
            )
            for i in range(3)
        }
    )

    run = await evaluate(predictor, records)

    assert len(run.latencies_seconds) == 3


def test_schema_conformance_rate():
    run = EvaluationRun(total=10, parse_failures=3)
    assert schema_conformance_rate(run) == pytest.approx(0.7)


def test_schema_conformance_rate_no_documents_is_vacuously_perfect():
    assert schema_conformance_rate(EvaluationRun()) == 1.0


def test_percentile_p50_of_sorted_values():
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0


def test_percentile_p0_is_minimum():
    assert percentile([3.0, 1.0, 2.0], 0.0) == 1.0


def test_percentile_p100_is_maximum():
    assert percentile([3.0, 1.0, 2.0], 1.0) == 3.0


def test_print_report_flags_total_parse_failure_instead_of_fake_perfect_score(capsys):
    run = EvaluationRun(total=5, parse_failures=5)

    print_report(run)

    output = capsys.readouterr().out
    assert "No documents were successfully parsed" in output
    assert "macro-F1" not in output


@pytest.mark.anyio
async def test_print_report_includes_per_source_field_breakdown(capsys):
    fatura_invoice = _invoice(document_type="invoice")
    sroie_invoice = _invoice(document_type="receipt")
    records = [
        GoldRecord(source="fatura", gold=fatura_invoice, image_path=Path("a.png")),
        GoldRecord(source="sroie", gold=sroie_invoice, image_path=Path("b.png")),
    ]
    predictor = _FakePredictor(
        {
            "a.png": ExtractionResult(
                invoice=fatura_invoice,
                parse_error=None,
                model_name="m",
                raw_output=None,
            ),
            "b.png": ExtractionResult(
                invoice=sroie_invoice, parse_error=None, model_name="m", raw_output=None
            ),
        }
    )

    run = await evaluate(predictor, records)
    print_report(run)

    output = capsys.readouterr().out
    assert "[fatura] macro-F1=" in output
    assert "[sroie] macro-F1=" in output
    fatura_section = output.split("[fatura]")[1].split("[sroie]")[0]
    assert "document_type" in fatura_section


@pytest.mark.anyio
async def test_print_report_adds_trained_fields_diagnostic_for_sroie_only(capsys):
    fatura_invoice = _invoice(document_type="invoice")
    sroie_invoice = _invoice(document_type="receipt")
    records = [
        GoldRecord(source="fatura", gold=fatura_invoice, image_path=Path("a.png")),
        GoldRecord(source="sroie", gold=sroie_invoice, image_path=Path("b.png")),
    ]
    predictor = _FakePredictor(
        {
            "a.png": ExtractionResult(
                invoice=fatura_invoice,
                parse_error=None,
                model_name="m",
                raw_output=None,
            ),
            "b.png": ExtractionResult(
                invoice=sroie_invoice, parse_error=None, model_name="m", raw_output=None
            ),
        }
    )

    run = await evaluate(predictor, records)
    print_report(run)

    output = capsys.readouterr().out
    sroie_section = output.split("[sroie]", 1)[1]
    assert "trained-fields-only" in sroie_section

    fatura_section = output.split("[fatura]")[1].split("[sroie]")[0]
    assert "trained-fields-only" not in fatura_section


def test_build_predictor_frontier_defaults_to_config_model(monkeypatch):
    monkeypatch.setattr(config, "FRONTIER_MODEL", "claude-haiku-4-5-20251001")
    predictor = _build_predictor("frontier")
    assert isinstance(predictor, FrontierExtractor)
    assert predictor._client.model == "claude-haiku-4-5-20251001"


def test_build_predictor_frontier_uses_override_model(monkeypatch):
    monkeypatch.setattr(config, "FRONTIER_MODEL", "claude-haiku-4-5-20251001")
    predictor = _build_predictor("frontier", model="claude-opus-5")
    assert predictor._client.model == "claude-opus-5"


def test_build_predictor_specialist_uses_override_model(monkeypatch):
    monkeypatch.setattr(config, "VLLM_BASE_URL", "http://fake")
    monkeypatch.setattr(config, "VLLM_API_KEY", "fake-key")
    predictor = _build_predictor("specialist", model="custom-checkpoint")
    assert isinstance(predictor, SpecialistExtractor)
    assert predictor._model_name == "custom-checkpoint"


def test_build_predictor_specialist_passes_seed_through(monkeypatch):
    monkeypatch.setattr(config, "VLLM_BASE_URL", "http://fake")
    monkeypatch.setattr(config, "VLLM_API_KEY", "fake-key")
    predictor = _build_predictor("specialist", seed=42)
    assert predictor._seed == 42


def test_build_predictor_specialist_defaults_to_no_seed(monkeypatch):
    monkeypatch.setattr(config, "VLLM_BASE_URL", "http://fake")
    monkeypatch.setattr(config, "VLLM_API_KEY", "fake-key")
    predictor = _build_predictor("specialist")
    assert predictor._seed is None


def test_build_predictor_specialist_uses_override_base_url(monkeypatch):
    monkeypatch.setattr(config, "VLLM_BASE_URL", "http://fake-default")
    monkeypatch.setattr(config, "VLLM_API_KEY", "fake-key")
    predictor = _build_predictor("specialist", base_url="http://fake-override")
    assert str(predictor._client.base_url) == "http://fake-override"


def test_build_predictor_specialist_defaults_to_config_base_url(monkeypatch):
    monkeypatch.setattr(config, "VLLM_BASE_URL", "http://fake-default")
    monkeypatch.setattr(config, "VLLM_API_KEY", "fake-key")
    predictor = _build_predictor("specialist")
    assert str(predictor._client.base_url) == "http://fake-default"


def test_build_predictor_unknown_extractor_raises():
    with pytest.raises(ValueError, match="Unknown extractor"):
        _build_predictor("gpt4-mini")


def test_load_golden_set_parses_jsonl(tmp_path):
    golden_path = tmp_path / "test.jsonl"
    row = {
        "source": "fatura",
        "split": "train",
        "index": 1,
        "invoice": json.loads(_invoice().model_dump_json()),
        "image_path": "FATURA/images/train/1.png",
    }
    golden_path.write_text(json.dumps(row) + "\n")

    records = load_golden_set(golden_path)

    assert len(records) == 1
    assert records[0].source == "fatura"
    assert records[0].gold.currency == "USD"
    assert str(records[0].image_path).endswith("FATURA/images/train/1.png")
