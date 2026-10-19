"""Run questions through an `ask_fn` and score them against pandas ground truth."""
import time
from collections import defaultdict

from evaluation.checks import matches


def run_questions(datasets, questions, ask_fn, delay: float = 0.0, sleep_fn=time.sleep):
    """ask_fn(dataset_name, df, question_text) -> (answer_text, {"tool_calls": int, "steps": int, "error": str|None})"""
    rows = []
    for q in questions:
        df = datasets[q.dataset]
        expected = q.expected(df)
        expected = list(expected) if q.kind == "list" else (float(expected) if q.kind == "number" else str(expected))
        start = time.perf_counter()
        answer, meta = ask_fn(q.dataset, df, q.text)
        rows.append({
            "dataset": q.dataset, "question": q.text, "expected": expected, "answer": answer,
            "correct": bool(matches(answer, expected, q.kind)),
            "tool_calls": meta.get("tool_calls", 0), "steps": meta.get("steps", 1),
            "error": meta.get("error"), "seconds": round(time.perf_counter() - start, 2),
        })
        if delay:
            sleep_fn(delay)
    return rows


def summarize(rows):
    def stats(subset):
        n = len(subset)
        return {
            "questions": n,
            "correct": sum(r["correct"] for r in subset),
            "accuracy": round(100 * sum(r["correct"] for r in subset) / n, 1) if n else 0.0,
            "avg_tool_calls": round(sum(r["tool_calls"] for r in subset) / n, 2) if n else 0.0,
            "avg_seconds": round(sum(r["seconds"] for r in subset) / n, 2) if n else 0.0,
            "errors": sum(1 for r in subset if r["error"]),
        }

    by_dataset = defaultdict(list)
    for r in rows:
        by_dataset[r["dataset"]].append(r)
    return {"overall": stats(rows), "by_dataset": {k: stats(v) for k, v in by_dataset.items()}}
