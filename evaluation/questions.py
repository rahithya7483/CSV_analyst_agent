"""Evaluation questions. Ground truth is computed with pandas at run time, never typed by hand."""
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class Question:
    dataset: str
    text: str
    expected: Callable[[pd.DataFrame], Any]
    kind: str  # "number" | "text" | "list"


def _top(series: pd.Series, largest: bool = True):
    """Index label of the max/min; raises if two groups tie, so ground truth is never ambiguous."""
    ordered = series.sort_values(ascending=not largest)
    if len(ordered) > 1 and ordered.iloc[0] == ordered.iloc[1]:
        raise ValueError("Ambiguous ground truth: tie for first place")
    return ordered.index[0]


def _first_n(df: pd.DataFrame, by: str, label: str, n: int, largest: bool):
    ordered = df.sort_values(by, ascending=not largest)
    if ordered[by].iloc[n - 1] == ordered[by].iloc[n]:
        raise ValueError("Ambiguous ground truth: tie at the cut-off")
    return list(ordered[label].head(n))


EMP, SALES = "employees", "sales"

QUESTIONS = [
    # ---- employees (15 rows) ----
    Question(EMP, "What is the average salary?", lambda d: d.salary.mean(), "number"),
    Question(EMP, "What is the highest salary?", lambda d: d.salary.max(), "number"),
    Question(EMP, "Who earns the highest salary?", lambda d: d.loc[d.salary.idxmax(), "name"], "text"),
    Question(EMP, "How many employees work in Engineering?", lambda d: (d.department == "Engineering").sum(), "number"),
    Question(EMP, "How many employees are based in Hyderabad?", lambda d: (d.city == "Hyderabad").sum(), "number"),
    Question(EMP, "What is the average salary in the Engineering department?",
             lambda d: d[d.department == "Engineering"].salary.mean(), "number"),
    Question(EMP, "Which department has the highest average salary?",
             lambda d: _top(d.groupby("department").salary.mean()), "text"),
    Question(EMP, "How much higher is the average Engineering salary than the average Marketing salary?",
             lambda d: d[d.department == "Engineering"].salary.mean() - d[d.department == "Marketing"].salary.mean(),
             "number"),
    Question(EMP, "What is the total salary paid to the HR department?",
             lambda d: d[d.department == "HR"].salary.sum(), "number"),
    Question(EMP, "How many employees have more than 5 years of experience?",
             lambda d: (d.experience_years > 5).sum(), "number"),
    Question(EMP, "What is the average salary of Engineering employees based in Hyderabad?",
             lambda d: d[(d.department == "Engineering") & (d.city == "Hyderabad")].salary.mean(), "number"),
    Question(EMP, "Which department has the most employees?",
             lambda d: _top(d.department.value_counts()), "text"),
    Question(EMP, "Who are the three lowest-paid employees?",
             lambda d: _first_n(d, "salary", "name", 3, largest=False), "list"),
    Question(EMP, "What is the average experience in years of HR employees?",
             lambda d: d[d.department == "HR"].experience_years.mean(), "number"),
    Question(EMP, "What percentage of employees work in Marketing?",
             lambda d: (d.department == "Marketing").mean() * 100, "number"),
    # ---- sales (240 rows, synthetic) ----
    Question(SALES, "What is the total revenue?", lambda d: d.revenue.sum(), "number"),
    Question(SALES, "What is the average number of units per order?", lambda d: d.units.mean(), "number"),
    Question(SALES, "Which region has the highest total revenue?",
             lambda d: _top(d.groupby("region").revenue.sum()), "text"),
    Question(SALES, "What is the total revenue for the West region?",
             lambda d: d[d.region == "West"].revenue.sum(), "number"),
    Question(SALES, "How many orders came from the Corporate segment?",
             lambda d: (d.segment == "Corporate").sum(), "number"),
    Question(SALES, "Which product category has the highest average revenue per order?",
             lambda d: _top(d.groupby("category").revenue.mean()), "text"),
    Question(SALES, "What is the average unit price of Laptop orders?",
             lambda d: d[d.category == "Laptop"].unit_price.mean(), "number"),
    Question(SALES, "Which three orders have the highest revenue? Give their order IDs.",
             lambda d: _first_n(d, "revenue", "order_id", 3, largest=True), "list"),
    Question(SALES, "Which month has the highest total revenue?",
             lambda d: _top(d.groupby("month").revenue.sum()), "text"),
    Question(SALES, "How many orders had more than 20 units?", lambda d: (d.units > 20).sum(), "number"),
    Question(SALES, "How much more total revenue did the North region make than the South region?",
             lambda d: d[d.region == "North"].revenue.sum() - d[d.region == "South"].revenue.sum(), "number"),
    Question(SALES, "What is the total revenue from Phone orders in the Retail segment?",
             lambda d: d[(d.category == "Phone") & (d.segment == "Retail")].revenue.sum(), "number"),
]
