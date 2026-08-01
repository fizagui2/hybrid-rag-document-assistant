"""Run the golden eval suite across all three chunking strategies and save
a comparison report to eval/results/.

Usage:
    venv\\Scripts\\python.exe scripts\\run_chunking_comparison.py
    venv\\Scripts\\python.exe scripts\\run_chunking_comparison.py --limit 9
    venv\\Scripts\\python.exe scripts\\run_chunking_comparison.py --ids sf-01,mh-01,mh-04,na-01,amb-03

--limit N restricts to the first N golden dataset cases in file order —
fast, but the dataset is grouped by category, so a small --limit only
exercises whichever category comes first (all "straightforward" for small
N), which isn't a meaningful comparison on its own.

--ids is a comma-separated list of specific case ids, for a deliberately
category-diverse quick run instead of just "the first N."

The full suite (51 cases x 3 strategies) makes several hundred LLM calls
and can take a long time; run it deliberately, not by accident.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.compare_chunking import build_comparison_report, compare_strategies  # noqa: E402

CORPUS_DIR = Path("eval/corpus")
GOLDEN_DATASET = Path("eval/golden_dataset.json")
RESULTS_DIR = Path("eval/results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N golden dataset cases")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated list of specific case ids to run")
    args = parser.parse_args()

    cases = json.loads(GOLDEN_DATASET.read_text(encoding="utf-8"))
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]
        missing = wanted - {c["id"] for c in cases}
        if missing:
            raise SystemExit(f"Unknown case id(s): {sorted(missing)}")
    elif args.limit:
        cases = cases[: args.limit]

    print(f"Running {len(cases)} case(s) across each of 3 chunking strategies "
          f"({len(cases) * 3} total eval runs)...")

    # tempfile.mkdtemp() (not TemporaryDirectory()'s context manager) on
    # purpose: Chroma's PersistentClient keeps SQLite/hnswlib file handles
    # open on Windows, which blocks deletion — a TemporaryDirectory's
    # automatic __exit__ cleanup would crash mid-teardown and take the
    # freshly computed (real, LLM-call-expensive) results down with it,
    # since the crash happens before anything gets saved. Results are
    # written to disk FIRST here; cleanup is attempted afterward, best
    # effort, and can never destroy already-saved work.
    tmp = tempfile.mkdtemp()
    try:
        report = compare_strategies(CORPUS_DIR, cases, Path(tmp))

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "chunking_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        markdown = build_comparison_report(report)
        (RESULTS_DIR / "chunking_comparison.md").write_text(markdown, encoding="utf-8")

        print()
        print(markdown)
        print(f"\nSaved to {RESULTS_DIR}/chunking_comparison.md and .json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
