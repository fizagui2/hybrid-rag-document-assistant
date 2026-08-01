"""Chunking strategy comparison: run the same golden eval suite across all
three chunking strategies (fixed, structure, semantic) against fresh
indexes built from the eval corpus, and report which strategy wins on
which metric. This is the concrete, data-driven answer to "which chunking
strategy should this project actually use," not a guess.
"""
from pathlib import Path
from typing import Optional

from ..ingestion.chunker import chunk_document
from ..ingestion.loaders import load_document
from ..retrieval.index import HybridIndex
from .metrics import run_eval_suite, summarize_results

STRATEGIES = ["fixed", "structure", "semantic"]
METRICS = ["correctness", "faithfulness", "retrieval_relevance", "citation_accuracy"]


def build_index_for_strategy(corpus_dir: Path, chroma_dir: Path, bm25_path: Path, strategy: str) -> HybridIndex:
    """A fresh, isolated index (caller supplies chroma_dir/bm25_path so
    three strategies never share or clobber each other's index files).
    """
    index = HybridIndex(chroma_dir=chroma_dir, bm25_path=bm25_path)
    for path in sorted(corpus_dir.glob("*.md")):
        doc = load_document(path)
        chunks = chunk_document(doc, strategy=strategy)
        index.add(chunks)
    return index


def compare_strategies(
    corpus_dir: Path,
    cases: list[dict],
    work_dir: Path,
    strategies: list[str] = STRATEGIES,
) -> dict:
    """Returns {strategy: summarize_results()} for each strategy, run
    against the same golden dataset cases.
    """
    report = {}
    for strategy in strategies:
        index = build_index_for_strategy(
            corpus_dir, work_dir / strategy / "chroma", work_dir / strategy / "bm25.pkl", strategy
        )
        results = run_eval_suite(index, cases)
        report[strategy] = summarize_results(results)
    return report


def _fmt(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "N/A"


def _best_strategy(values: dict[str, Optional[float]]) -> str:
    """max() alone would silently pick whichever strategy happens to come
    first on an exact tie (e.g. every strategy scoring a perfect 1.00 on an
    easy question) — that's a fabricated "winner," not a real finding, so
    ties are reported explicitly instead.
    """
    present = {strategy: value for strategy, value in values.items() if value is not None}
    if not present:
        return "—"
    best_value = max(present.values())
    winners = [strategy for strategy, value in present.items() if value == best_value]
    return winners[0] if len(winners) == 1 else "tie (" + ", ".join(winners) + ")"


def _metric_table(report: dict, section: str, category: Optional[str] = None) -> list[str]:
    strategies = list(report)
    lines = ["| Metric | " + " | ".join(s.capitalize() for s in strategies) + " | Winner |"]
    lines.append("|---|" + "---|" * (len(strategies) + 1))
    for metric in METRICS:
        scope = report_scope(report, section, category)
        values = {s: scope[s].get(metric) for s in strategies}
        row = " | ".join(_fmt(values[s]) for s in strategies)
        lines.append(f"| {metric} | {row} | {_best_strategy(values)} |")
    return lines


def report_scope(report: dict, section: str, category: Optional[str]) -> dict:
    if section == "overall":
        return {s: report[s]["overall"] for s in report}
    return {s: report[s]["by_category"][category] for s in report}


def build_comparison_report(report: dict) -> str:
    lines = ["# Chunking Strategy Comparison", ""]
    counts = {s: report[s]["overall"]["count"] for s in report}
    lines.append(f"Cases evaluated per strategy: {counts}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.extend(_metric_table(report, "overall"))
    lines.append("")

    categories = sorted(next(iter(report.values()))["by_category"].keys())
    for category in categories:
        lines.append(f"## {category}")
        lines.append("")
        lines.extend(_metric_table(report, "by_category", category))
        lines.append("")

    return "\n".join(lines)
