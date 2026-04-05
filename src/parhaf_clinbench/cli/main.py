"""Command-line entrypoint for the `parhaf-clinbench` package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from parhaf_clinbench.core.enums import RuntimeName, TaskId
from parhaf_clinbench.core.settings import get_settings
from parhaf_clinbench.data.contracts_audit import audit_suite_contracts
from parhaf_clinbench.data.prefetch import prefetch_hf_dataset
from parhaf_clinbench.orchestration.experiment_plan import load_model, load_suite, load_task
from parhaf_clinbench.orchestration.runner import run_campaign, score_from_jsonl
from parhaf_clinbench.runtimes.prefetch import prefetch_hf_model


def _dataset_prefetch_configs_for_task(task: TaskId) -> list[str] | None:
    """Return dataset config names to prefetch for a given task."""

    if task == TaskId.SCENARIO:
        return None
    return ["document_metadata", "spans"]


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the root CLI argument parser."""

    settings = get_settings()
    parser = argparse.ArgumentParser(prog="parhaf-clinbench")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Lancer une campagne benchmark")
    run_parser.add_argument("--suite", default=settings.parhaf_suite)
    run_parser.add_argument(
        "--task",
        choices=["pseudo", "infectio", "response", "scenario", "all"],
        default="all",
    )
    run_parser.add_argument(
        "--track",
        choices=["zeroshot", "fewshot", "all"],
        default="all",
    )
    run_parser.add_argument("--model", default="all")
    run_parser.add_argument("--output-dir", default=settings.parhaf_output_dir)

    smoke_parser = sub.add_parser("smoke", help="Exécuter la suite smoke")
    smoke_parser.add_argument("--suite", default="configs/suites/v1_smoke.yaml")
    smoke_parser.add_argument("--output-dir", default="results/smoke")

    score_parser = sub.add_parser("score", help="Calculer un score offline")
    score_parser.add_argument("--predictions", required=True)
    score_parser.add_argument("--gold", required=True)
    score_parser.add_argument(
        "--task",
        choices=["pseudo", "infectio", "response", "scenario"],
        required=True,
    )

    report_parser = sub.add_parser("report", help="Afficher le chemin du rapport d'un run")
    report_parser.add_argument("--run-dir", required=True)

    prefetch_parser = sub.add_parser("prefetch", help="Précharger un modèle HF vers cache local")
    group = prefetch_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Model ID configuré dans configs/models/*.yaml")
    group.add_argument("--hf-id", help="Identifiant HF direct (ex: Qwen/Qwen2.5-7B-Instruct)")
    prefetch_parser.add_argument("--revision", default="main")
    prefetch_parser.add_argument("--cache-root", default=str(settings.model_cache_root))
    prefetch_parser.add_argument("--output-json", default="")
    prefetch_parser.add_argument("--print-path-only", action="store_true")

    prefetch_suite_parser = sub.add_parser(
        "prefetch-suite",
        help="Précharger tous les modèles/datasets d'une suite",
    )
    prefetch_suite_parser.add_argument("--suite", default=settings.parhaf_suite)
    prefetch_suite_parser.add_argument("--model-cache-root", default=str(settings.model_cache_root))
    prefetch_suite_parser.add_argument("--dataset-cache-root", default=str(settings.dataset_cache_root))
    prefetch_suite_parser.add_argument("--output-json", default="")

    audit_contracts_parser = sub.add_parser(
        "audit-contracts",
        help="Auditer les contrats datasets (labels/attributs/champs) sur HF",
    )
    audit_contracts_parser.add_argument("--suite", default=settings.parhaf_suite)
    audit_contracts_parser.add_argument(
        "--dataset-cache-root",
        default=str(settings.dataset_cache_root),
    )
    audit_contracts_parser.add_argument("--output-json", default="")
    audit_contracts_parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Retourne 0 même si des écarts de contrat sont détectés.",
    )

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    """Handle the `run` command."""

    run_dirs = run_campaign(
        suite_path=Path(args.suite),
        task_selection=args.task,
        track_selection=args.track,
        model_selection=args.model,
        output_dir=Path(args.output_dir),
    )
    for path in run_dirs:
        print(path)
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    """Handle the `smoke` command."""

    run_dirs = run_campaign(
        suite_path=Path(args.suite),
        task_selection="all",
        track_selection="all",
        model_selection="all",
        output_dir=Path(args.output_dir),
    )
    for path in run_dirs:
        print(path)
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    """Handle the `score` command."""

    result = score_from_jsonl(
        task=TaskId(args.task),
        predictions_path=Path(args.predictions),
        gold_path=Path(args.gold),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Handle the `report` command."""

    run_dir = Path(args.run_dir)
    report_path = run_dir / "report.md"
    if not report_path.exists():
        raise FileNotFoundError(f"Rapport introuvable: {report_path}")
    print(report_path)
    return 0


def _cmd_prefetch(args: argparse.Namespace) -> int:
    """Handle the `prefetch` command."""

    settings = get_settings()
    if args.model:
        model_cfg = load_model(args.model)
        hf_id = model_cfg.hf_id
        revision = model_cfg.revision
    else:
        hf_id = str(args.hf_id)
        revision = str(args.revision)

    result = prefetch_hf_model(
        hf_id=hf_id,
        revision=revision,
        cache_root=Path(args.cache_root),
        hf_token=settings.hf_token,
    )
    payload = result.model_dump(mode="json")
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.print_path_only:
        print(payload["local_path"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_prefetch_suite(args: argparse.Namespace) -> int:
    """Handle the `prefetch-suite` command."""

    settings = get_settings()
    suite = load_suite(Path(args.suite))
    payload: dict[str, Any] = {
        "suite_id": suite.suite_id,
        "models": [],
        "datasets": [],
    }

    for model_id in suite.models:
        runtime_name = suite.runtime_overrides.get(model_id, suite.runtime_default)
        if runtime_name not in {RuntimeName.VLLM, RuntimeName.GLINER}:
            continue
        model_cfg = load_model(model_id)
        prefetched = prefetch_hf_model(
            hf_id=model_cfg.hf_id,
            revision=model_cfg.revision,
            cache_root=Path(args.model_cache_root),
            hf_token=settings.hf_token,
        )
        payload["models"].append(
            {
                "model_id": model_id,
                "runtime": runtime_name.value,
                **prefetched.model_dump(mode="json"),
            }
        )

    for task in suite.tasks:
        task_cfg = load_task(task)
        prefetched_dataset = prefetch_hf_dataset(
            dataset_name=task_cfg.dataset,
            revision=task_cfg.dataset_revision,
            cache_root=Path(args.dataset_cache_root),
            hf_token=settings.hf_token,
            configs=_dataset_prefetch_configs_for_task(task),
        )
        payload["datasets"].append(
            {
                "task": task.value,
                **prefetched_dataset.model_dump(mode="json"),
            }
        )

    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _cmd_audit_contracts(args: argparse.Namespace) -> int:
    """Handle the `audit-contracts` command."""

    settings = get_settings()
    report = audit_suite_contracts(
        suite_path=Path(args.suite),
        dataset_cache_root=Path(args.dataset_cache_root),
        hf_token=settings.hf_token,
    )
    rendered = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report.all_ok or args.allow_mismatch:
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run CLI command dispatch.

    Args:
        argv: Optional CLI argument list. When `None`, `sys.argv[1:]` is used.

    Returns:
        Process exit code.

    Examples:
        >>> main(["score", "--predictions", "preds.jsonl", "--gold", "gold.jsonl", "--task", "pseudo"])
        0
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "smoke":
        return _cmd_smoke(args)
    if args.command == "score":
        return _cmd_score(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "prefetch":
        return _cmd_prefetch(args)
    if args.command == "prefetch-suite":
        return _cmd_prefetch_suite(args)
    if args.command == "audit-contracts":
        return _cmd_audit_contracts(args)
    parser.error("Commande non supportée")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
