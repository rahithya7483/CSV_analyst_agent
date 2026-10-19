"""Measure the agent's accuracy against pandas ground truth.

    python -m evaluation.run_eval                 # agent only
    python -m evaluation.run_eval --baseline      # also run the no-tools baseline (sees 20 rows)
    python -m evaluation.run_eval --baseline --baseline-rows 300   # baseline sees the whole sales file
    python -m evaluation.run_eval --show-answers  # print every answer for manual review

Needs a real Groq key (GROQ_API in .env); mock mode is not meaningful here.
"""
import argparse
import json
import sys
from pathlib import Path

from evaluation.baseline import baseline_answer
from evaluation.questions import QUESTIONS
from evaluation.runner import run_questions, summarize
from src.agent import CsvAnalystAgent
from src.config import DATA_DIR, MOCK_LLM, MODEL
from src.data_loader import DatasetStore, load_samples
from src.llm_client import get_client

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def print_report(title, rows, show_answers):
    print(f"\n=== {title} ===")
    for r in rows:
        mark = "PASS" if r["correct"] else "FAIL"
        print(f"[{mark}] ({r['dataset']}) {r['question']}")
        if show_answers or not r["correct"]:
            print(f"        expected: {r['expected']}\n        answer:   {r['answer'][:300]!r}")
    summary = summarize(rows)
    o = summary["overall"]
    print(f"\n{title}: {o['correct']}/{o['questions']} correct = {o['accuracy']}% | "
          f"avg tool calls {o['avg_tool_calls']} | avg {o['avg_seconds']}s | errors {o['errors']}")
    for name, s in summary["by_dataset"].items():
        print(f"   {name}: {s['correct']}/{s['questions']} = {s['accuracy']}%")
    return summary


def save(report):
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "latest.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", action="store_true", help="also run the no-tools baseline")
    parser.add_argument("--baseline-rows", type=int, default=20,
                        help="how many rows of each CSV the baseline gets to see (default 20; use 300 to show it all)")
    parser.add_argument("--show-answers", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between questions (rate limits)")
    args = parser.parse_args()

    if MOCK_LLM:
        sys.exit("MOCK_LLM is true. Set MOCK_LLM=false and add GROQ_API to .env to run the evaluation.")

    store = DatasetStore()
    load_samples(store, DATA_DIR)
    datasets = {name: store.get(name) for name in store.names()}
    client = get_client()

    def agent_ask(name, df, question):
        result = CsvAnalystAgent(df, client).ask(question)  # fresh agent per question: no memory carry-over
        return result.answer, {"tool_calls": len(result.tool_calls), "steps": result.steps, "error": result.error}

    report = {"model": MODEL, "questions": len(QUESTIONS)}
    agent_rows = run_questions(datasets, QUESTIONS, agent_ask, delay=args.delay)
    report["agent"] = {"summary": print_report("Agent with tools", agent_rows, args.show_answers), "rows": agent_rows}
    print(f"\nSaved agent results to {save(report)}")  # saved now, so a baseline failure cannot lose them

    if args.baseline:
        def baseline_ask(name, df, question):
            return baseline_answer(client, df, question, sample_rows=args.baseline_rows), {"tool_calls": 0, "steps": 1}
        try:
            base_rows = run_questions(datasets, QUESTIONS, baseline_ask, delay=args.delay)
        except Exception as exc:
            print(f"\nBaseline stopped early: {str(exc)[:300]}")
            print("If this is a 413 / tokens-per-minute error, lower --baseline-rows: the Groq free tier "
                  "allows only about 8,000 tokens per minute for this model.")
            return
        title = f"Baseline (no tools, first {args.baseline_rows} rows)"
        report["baseline_rows_shown"] = args.baseline_rows
        report["baseline"] = {"summary": print_report(title, base_rows, args.show_answers), "rows": base_rows}
        print(f"\nSaved agent + baseline results to {save(report)}")


if __name__ == "__main__":
    main()