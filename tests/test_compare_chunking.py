from pathlib import Path

from src.eval.compare_chunking import (
    STRATEGIES,
    _best_strategy,
    _fmt,
    build_comparison_report,
    build_index_for_strategy,
    compare_strategies,
)
from tests.conftest import requires_groq

CORPUS_DIR = Path("eval/corpus")


# --- _fmt / _best_strategy: pure logic ---

def test_fmt_formats_float():
    assert _fmt(0.6667) == "0.67"


def test_fmt_handles_none():
    assert _fmt(None) == "N/A"


def test_best_strategy_picks_highest():
    assert _best_strategy({"fixed": 0.5, "structure": 0.9, "semantic": 0.7}) == "structure"


def test_best_strategy_ignores_none():
    assert _best_strategy({"fixed": None, "structure": 0.4, "semantic": None}) == "structure"


def test_best_strategy_all_none():
    assert _best_strategy({"fixed": None, "structure": None}) == "—"


def test_best_strategy_reports_ties_honestly_not_a_fabricated_winner():
    # every strategy scoring identically (e.g. a trivially easy question)
    # must not silently credit whichever one happens to come first
    result = _best_strategy({"fixed": 1.0, "structure": 1.0, "semantic": 1.0})
    assert result == "tie (fixed, structure, semantic)"


# --- build_comparison_report: pure formatting logic ---

def test_build_comparison_report_includes_all_strategies_and_categories():
    fake_summary = {
        "count": 2,
        "correctness": 0.8,
        "faithfulness": 0.9,
        "retrieval_relevance": 1.0,
        "citation_accuracy": 0.7,
    }
    report = {
        "fixed": {"overall": fake_summary, "by_category": {"straightforward": fake_summary}},
        "structure": {"overall": fake_summary, "by_category": {"straightforward": fake_summary}},
    }

    markdown = build_comparison_report(report)

    assert "Fixed" in markdown
    assert "Structure" in markdown
    assert "straightforward" in markdown
    assert "correctness" in markdown


# --- build_index_for_strategy: real chunking + indexing, no LLM needed ---

def test_build_index_for_strategy_produces_chunks(tmp_path):
    index = build_index_for_strategy(CORPUS_DIR, tmp_path / "chroma", tmp_path / "bm25.pkl", "structure")
    assert index.count() > 0


def test_different_strategies_produce_different_chunk_counts(tmp_path):
    # not asserting a specific relationship (chunk_size/corpus dependent) —
    # just confirming the strategy parameter genuinely changes behavior,
    # since a bug that silently ignored it would still "pass" a vaguer test
    counts = {}
    for strategy in STRATEGIES:
        index = build_index_for_strategy(
            CORPUS_DIR, tmp_path / strategy / "chroma", tmp_path / strategy / "bm25.pkl", strategy
        )
        counts[strategy] = index.count()
    assert len(set(counts.values())) > 1, f"expected chunk counts to differ across strategies, got {counts}"


# --- compare_strategies: real end-to-end, minimal case count to limit LLM calls ---

@requires_groq
def test_compare_strategies_runs_all_strategies(tmp_path):
    cases = [
        {
            "id": "t1",
            "question": "How many days per week can employees work remotely?",
            "expected_answer": "Up to 3 days per week.",
            "category": "straightforward",
            "source_documents": ["hr_policies.md"],
        }
    ]

    report = compare_strategies(CORPUS_DIR, cases, tmp_path)

    assert set(report.keys()) == set(STRATEGIES)
    for strategy in STRATEGIES:
        assert report[strategy]["overall"]["count"] == 1
