"""Deterministic analysis tools that the LLM can call.

The model never does arithmetic itself: every number in an answer comes from one
of these functions. Each tool returns a JSON-serialisable dict. Problems come back
as {"error": ...} so the model can read the message and retry with better arguments.
"""
import json
import math
import operator

import numpy as np
import pandas as pd

from src.config import MAX_ROWS_RETURNED

OPERATORS = ("==", "!=", ">", ">=", "<", "<=", "contains")
AGGREGATIONS = ("mean", "sum", "min", "max", "count", "median")
MAX_GROUPS = 50
_COMPARE = {
    "==": operator.eq, "!=": operator.ne, ">": operator.gt,
    ">=": operator.ge, "<": operator.lt, "<=": operator.le,
}


class ToolError(Exception):
    """A problem with the arguments the model supplied."""


def _to_py(x):
    """Convert numpy/pandas values into plain JSON-safe Python (NaN -> None)."""
    if isinstance(x, dict):
        return {str(k): _to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else round(f, 2)
    if isinstance(x, str) or x is None:
        return x
    if isinstance(x, pd.Timestamp):
        return str(x)
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return str(x)


def _clamp(value, low, high, default):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


class CsvTools:
    """The tool set, bound to one DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ---- helpers -------------------------------------------------------
    def _col(self, name) -> str:
        if not isinstance(name, str):
            raise ToolError("Column name must be a string.")
        if name in self.df.columns:
            return name
        lowered = {str(c).strip().lower(): c for c in self.df.columns}
        if name.strip().lower() in lowered:
            return lowered[name.strip().lower()]
        raise ToolError(f"Column '{name}' not found. Available columns: {list(self.df.columns)}")

    def _numeric(self, name) -> str:
        col = self._col(name)
        if not pd.api.types.is_numeric_dtype(self.df[col]):
            numeric = [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]
            raise ToolError(f"Column '{col}' is not numeric (it holds text). Numeric columns: {numeric}")
        return col

    def _apply_filters(self, filters) -> pd.DataFrame:
        df = self.df
        if not filters:
            return df
        if not isinstance(filters, list):
            raise ToolError("filters must be a list of {column, operator, value} objects.")
        for f in filters:
            if not isinstance(f, dict) or not {"column", "operator", "value"} <= f.keys():
                raise ToolError("Each filter needs 'column', 'operator' and 'value'.")
            col, op, val = self._col(f["column"]), f["operator"], f["value"]
            if op not in OPERATORS:
                raise ToolError(f"Unsupported operator '{op}'. Use one of {list(OPERATORS)}.")
            series = df[col]
            if op == "contains":
                mask = series.astype(str).str.contains(str(val), case=False, regex=False, na=False)
            elif pd.api.types.is_numeric_dtype(series):
                try:
                    number = float(val)
                except (TypeError, ValueError):
                    raise ToolError(f"Column '{col}' is numeric but '{val}' is not a number.")
                mask = _COMPARE[op](series, number)
            else:
                if op not in ("==", "!="):
                    raise ToolError(f"Operator '{op}' needs a numeric column; '{col}' holds text.")
                left = series.astype(str).str.strip().str.lower()
                mask = _COMPARE[op](left, str(val).strip().lower())
            df = df[mask]
        return df

    # ---- tools ---------------------------------------------------------
    def describe_dataset(self):
        columns = []
        for c in self.df.columns:
            s = self.df[c]
            columns.append({
                "name": c,
                "type": "numeric" if pd.api.types.is_numeric_dtype(s) else "text",
                "missing": int(s.isna().sum()),
                "examples": list(s.dropna().unique()[:3]),
            })
        return {"rows": len(self.df), "columns": columns}

    def get_stats(self, column, filters=None):
        col = self._numeric(column)
        s = self._apply_filters(filters)[col].dropna()
        if s.empty:
            return {"column": col, "count": 0, "note": "No matching non-null values."}
        return {
            "column": col, "filters_applied": filters or [], "count": int(s.count()),
            "mean": s.mean(), "median": s.median(), "min": s.min(),
            "max": s.max(), "sum": s.sum(), "std": s.std(),
        }

    def filter_rows(self, filters=None, limit=10):
        df = self._apply_filters(filters)
        limit = _clamp(limit, 1, MAX_ROWS_RETURNED, 10)
        return {"count": len(df), "rows": df.head(limit).to_dict("records"), "truncated": len(df) > limit}

    def group_stats(self, group_by, column, agg="mean", filters=None):
        gcol = self._col(group_by)
        if agg not in AGGREGATIONS:
            raise ToolError(f"Unsupported agg '{agg}'. Use one of {list(AGGREGATIONS)}.")
        vcol = self._col(column) if agg == "count" else self._numeric(column)
        df = self._apply_filters(filters)
        grouped = df.groupby(gcol)[vcol]
        values = grouped.agg(agg).sort_values(ascending=False)
        sizes = grouped.size()
        groups = [{"group": k, "value": v, "rows": sizes[k]} for k, v in values.items()]
        return {"group_by": gcol, "column": vcol, "agg": agg, "groups": groups[:MAX_GROUPS]}

    def top_n(self, column, n=5, ascending=False, filters=None):
        col = self._numeric(column)
        if isinstance(ascending, str):
            ascending = ascending.strip().lower() == "true"
        df = self._apply_filters(filters)
        n = _clamp(n, 1, MAX_ROWS_RETURNED, 5)
        rows = df.sort_values(col, ascending=bool(ascending), kind="stable").head(n)
        return {"sorted_by": col, "order": "ascending" if ascending else "descending",
                "rows": rows.to_dict("records")}

    def value_counts(self, column, top=10):
        col = self._col(column)
        counts = self.df[col].value_counts(dropna=False)
        top = _clamp(top, 1, MAX_ROWS_RETURNED, 10)
        return {"column": col, "distinct_values": int(len(counts)),
                "counts": [{"value": k, "count": v} for k, v in counts.head(top).items()]}


TOOL_NAMES = ("describe_dataset", "get_stats", "filter_rows", "group_stats", "top_n", "value_counts")


def run_tool(tools: CsvTools, name: str, raw_args) -> str:
    """Execute one tool call safely. Always returns a JSON string, never raises."""
    try:
        if isinstance(raw_args, str):
            raw_args = json.loads(raw_args.strip() or "{}")
        args = dict(raw_args or {})
        if name not in TOOL_NAMES:
            raise ToolError(f"Unknown tool '{name}'. Available tools: {list(TOOL_NAMES)}")
        result = getattr(tools, name)(**args)
    except json.JSONDecodeError:
        result = {"error": "Tool arguments were not valid JSON."}
    except ToolError as exc:
        result = {"error": str(exc)}
    except TypeError as exc:
        result = {"error": f"Invalid arguments for {name}: {exc}"}
    except Exception as exc:  # keep the agent loop alive on any unexpected pandas error
        result = {"error": f"{type(exc).__name__}: {exc}"}
    return json.dumps(_to_py(result))


# ---- JSON schemas sent to the model ------------------------------------
_FILTERS = {
    "type": "array",
    "description": "Optional conditions, all must hold (AND).",
    "items": {
        "type": "object",
        "properties": {
            "column": {"type": "string"},
            "operator": {"type": "string", "enum": list(OPERATORS)},
            "value": {"type": "string", "description": "Value to compare with; numbers as text, e.g. '50000'."},
        },
        "required": ["column", "operator", "value"],
    },
}


def _tool(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOL_SCHEMAS = [
    _tool("describe_dataset",
          "Row count plus every column's type, missing count and example values.", {}, []),
    _tool("get_stats",
          "Count, mean, median, min, max, sum and std of ONE numeric column, optionally on a filtered subset.",
          {"column": {"type": "string"}, "filters": _FILTERS}, ["column"]),
    _tool("filter_rows",
          "Count the rows matching the filters and return a preview of them.",
          {"filters": _FILTERS, "limit": {"type": "integer", "description": "Max rows to preview (default 10)."}}, []),
    _tool("group_stats",
          "Aggregate a column per group, e.g. average salary per department. Use this for any comparison "
          "across groups.",
          {"group_by": {"type": "string", "description": "Column to group by."},
           "column": {"type": "string", "description": "Column to aggregate (numeric unless agg is 'count')."},
           "agg": {"type": "string", "enum": list(AGGREGATIONS)},
           "filters": _FILTERS}, ["group_by", "column", "agg"]),
    _tool("top_n",
          "Rows with the highest (or lowest) values of a numeric column.",
          {"column": {"type": "string"}, "n": {"type": "integer"},
           "ascending": {"type": "boolean", "description": "true for the lowest values."},
           "filters": _FILTERS}, ["column"]),
    _tool("value_counts",
          "How often each distinct value of a column occurs (most frequent first).",
          {"column": {"type": "string"}, "top": {"type": "integer"}}, ["column"]),
]
