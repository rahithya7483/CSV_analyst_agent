"""The CSV analyst agent: a bounded tool-calling loop over one dataset."""
import json
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pandas as pd

from src.config import MAX_RETRIES, MAX_STEPS, MAX_TOOL_CHARS, MODEL
from src.data_loader import column_overview
from src.llm_client import chat_with_retry
from src.tools import TOOL_SCHEMAS, CsvTools, run_tool

SYSTEM_PROMPT = """You are a careful data analyst answering questions about ONE tabular dataset by calling tools.

Dataset: {rows} rows. Columns: {columns}.

Rules:
- Every number in your answer must come from a tool result. You may combine tool results with simple arithmetic (a difference, ratio or percentage); show the numbers you used.
- Never estimate, and never add up or average rows yourself: use get_stats or group_stats for totals and averages.
- For comparisons across groups (per department, per region, ...) call group_stats.
- If a tool returns an error, read it and call the tool again with corrected arguments.
- If the question cannot be answered from these columns, say so plainly instead of guessing.
- Do not add currency symbols or units that are not present in the data (a % sign for a percentage is fine).
- Keep answers short: state the key numbers and any filters you applied."""

MAX_HISTORY_MESSAGES = 10


@dataclass
class ToolTrace:
    name: str
    arguments: Any
    result: Any


@dataclass
class AgentResult:
    answer: str
    tool_calls: List[ToolTrace] = field(default_factory=list)
    steps: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "tool_calls": [{"name": t.name, "arguments": t.arguments, "result": t.result} for t in self.tool_calls],
            "steps": self.steps,
            "error": self.error,
        }


def _parse(text: str):
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text


class CsvAnalystAgent:
    def __init__(self, df: pd.DataFrame, client, model: str = MODEL, max_steps: int = MAX_STEPS,
                 max_retries: int = MAX_RETRIES, sleep_fn=time.sleep):
        self.df = df
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.sleep_fn = sleep_fn
        self.tools = CsvTools(df)
        self.history: list = []  # earlier user/assistant turns, so follow-up questions work

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(rows=len(self.df), columns=column_overview(self.df))

    def _remember(self, question: str, answer: str) -> None:
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        self.history = self.history[-MAX_HISTORY_MESSAGES:]

    def ask(self, question: str) -> AgentResult:
        messages = [{"role": "system", "content": self._system_prompt()}, *self.history,
                    {"role": "user", "content": question}]
        trace: List[ToolTrace] = []

        for step in range(1, self.max_steps + 1):
            try:
                response = chat_with_retry(
                    self.client, max_retries=self.max_retries, sleep_fn=self.sleep_fn,
                    model=self.model, messages=messages, tools=TOOL_SCHEMAS,
                    tool_choice="auto", temperature=0)
            except Exception as exc:
                return AgentResult("Sorry, the language model request failed. Please try again.",
                                   trace, step, error=f"llm_error: {exc}")

            message = response.choices[0].message
            calls = getattr(message, "tool_calls", None)
            if not calls:
                answer = (message.content or "").strip() or "The model returned an empty answer."
                self._remember(question, answer)
                return AgentResult(answer, trace, step)

            messages.append({
                "role": "assistant", "content": message.content or "",
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name,
                                             "arguments": c.function.arguments or "{}"}} for c in calls],
            })
            for call in calls:
                result = run_tool(self.tools, call.function.name, call.function.arguments)
                trace.append(ToolTrace(call.function.name, _parse(call.function.arguments), _parse(result)))
                content = result if len(result) <= MAX_TOOL_CHARS else result[:MAX_TOOL_CHARS] + "...(truncated)"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

        return AgentResult(
            f"I could not finish within {self.max_steps} steps. Try a more specific question.",
            trace, self.max_steps, error="step_limit")