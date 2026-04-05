from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.core.models import CanonicalDocument, Record
from parhaf_clinbench.scoring.pseudo import compute_pseudo_metrics
from parhaf_clinbench.scoring.response import compute_response_metrics


def test_pseudo_official_f1_perfect_match() -> None:
    doc_pred = CanonicalDocument(
        document_id="d1",
        task=TaskId.PSEUDO,
        speciality=None,
        records=[Record(label="FIRST_NAME", text="Jean", start=0, end=4, attributes={})],
    )
    doc_gold = CanonicalDocument(
        document_id="d1",
        task=TaskId.PSEUDO,
        speciality=None,
        records=[Record(label="FIRST_NAME", text="Jean", start=0, end=4, attributes={})],
    )

    result = compute_pseudo_metrics(
        predictions=[doc_pred],
        references=[doc_gold],
        robustness={
            "json_valid_rate": 1.0,
            "schema_conformity_rate": 1.0,
            "empty_output_rate": 0.0,
            "latency_mean_ms": 0.0,
            "latency_median_ms": 0.0,
            "latency_p95_ms": 0.0,
            "input_tokens_mean": 0.0,
            "output_tokens_mean": 0.0,
            "throughput_tokens_per_second": 0.0,
        },
    )
    assert result.metrics.official.f1 == 1.0


def test_response_secondary_label_metric_is_document_level() -> None:
    pred = CanonicalDocument(
        document_id="doc-1",
        task=TaskId.RESPONSE,
        speciality=None,
        records=[],
    )
    gold = CanonicalDocument(
        document_id="doc-1",
        task=TaskId.RESPONSE,
        speciality=None,
        records=[
            # Label documentaire conservé même sans justification span.
            Record(label="NonApplicable", text="", start=None, end=None, attributes={}),
        ],
    )
    result = compute_response_metrics(
        predictions=[pred],
        references=[gold],
        robustness={
            "raw_json_valid_rate": 1.0,
            "repair_applied_rate": 0.0,
            "schema_conformity_rate": 1.0,
            "empty_output_rate": 0.0,
            "latency_mean_ms": 0.0,
            "latency_median_ms": 0.0,
            "latency_p95_ms": 0.0,
            "input_tokens_mean": 0.0,
            "output_tokens_mean": 0.0,
            "throughput_tokens_per_second": 0.0,
        },
    )

    # Officiel sur (text,label): gold vide en texte normalisé, donc pas de match.
    assert result.metrics.official.f1 == 0.0
    # Secondaire documentaire: classe manquée.
    assert result.metrics.secondary["micro_f1_label"].f1 == 0.0
