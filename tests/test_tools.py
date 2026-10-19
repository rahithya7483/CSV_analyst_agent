import json
import unittest
from pathlib import Path

import pandas as pd

from src.tools import CsvTools, run_tool

DATA = Path(__file__).resolve().parent.parent / "data"


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA / "employees.csv")
        cls.tools = CsvTools(cls.df)

    def call(self, name, **args):
        return json.loads(run_tool(self.tools, name, args))

    def test_describe_dataset(self):
        out = self.call("describe_dataset")
        self.assertEqual(out["rows"], 15)
        self.assertEqual([c["name"] for c in out["columns"]],
                         ["name", "department", "salary", "experience_years", "city"])

    def test_get_stats_matches_pandas(self):
        out = self.call("get_stats", column="salary")
        self.assertAlmostEqual(out["mean"], round(self.df.salary.mean(), 2))
        self.assertEqual(out["max"], self.df.salary.max())
        self.assertEqual(out["count"], 15)

    def test_get_stats_with_filter(self):
        out = self.call("get_stats", column="salary",
                        filters=[{"column": "department", "operator": "==", "value": "Engineering"}])
        self.assertAlmostEqual(out["mean"], 81666.67)

    def test_group_stats_fixes_department_comparison(self):
        """The v1 notebook had no group-by tool and hallucinated 78,500 vs 55,000 for this question."""
        out = self.call("group_stats", group_by="department", column="salary", agg="mean")
        means = {g["group"]: g["value"] for g in out["groups"]}
        self.assertAlmostEqual(means["Engineering"], 81666.67)
        self.assertAlmostEqual(means["Marketing"], 60000.0)
        self.assertAlmostEqual(means["HR"], 51250.0)
        self.assertEqual(out["groups"][0]["group"], "Engineering")  # sorted, highest first

    def test_group_stats_count_and_filters(self):
        out = self.call("group_stats", group_by="city", column="name", agg="count",
                        filters=[{"column": "department", "operator": "==", "value": "HR"}])
        counts = {g["group"]: g["value"] for g in out["groups"]}
        self.assertEqual(sum(counts.values()), 4)

    def test_filter_rows_numeric_and_multiple(self):
        out = self.call("filter_rows", filters=[{"column": "salary", "operator": ">", "value": "80000"}])
        self.assertEqual(out["count"], int((self.df.salary > 80000).sum()))
        out = self.call("filter_rows", filters=[
            {"column": "department", "operator": "==", "value": "Engineering"},
            {"column": "city", "operator": "==", "value": "Hyderabad"}])
        self.assertEqual(out["count"], 4)

    def test_case_insensitive_names_and_values(self):
        out = self.call("filter_rows", filters=[{"column": "Department", "operator": "==", "value": "engineering"}])
        self.assertEqual(out["count"], 6)

    def test_contains_and_not_equal(self):
        self.assertEqual(self.call("filter_rows", filters=[
            {"column": "name", "operator": "contains", "value": "AR"}])["count"],
            int(self.df.name.str.contains("ar", case=False).sum()))
        self.assertEqual(self.call("filter_rows", filters=[
            {"column": "department", "operator": "!=", "value": "HR"}])["count"], 11)

    def test_top_n(self):
        out = self.call("top_n", column="salary", n=3)
        self.assertEqual([r["name"] for r in out["rows"]], ["Ravi", "Vikram", "Arjun"])
        out = self.call("top_n", column="salary", n=2, ascending=True)
        self.assertEqual([r["name"] for r in out["rows"]], ["Sneha", "Kavya"])

    def test_value_counts(self):
        out = self.call("value_counts", column="department")
        self.assertEqual(out["counts"][0], {"value": "Engineering", "count": 6})
        self.assertEqual(out["distinct_values"], 3)

    def test_errors_are_returned_not_raised(self):
        out = self.call("get_stats", column="salry")
        self.assertIn("not found", out["error"])
        self.assertIn("salary", out["error"])                 # lists available columns
        self.assertIn("not numeric", self.call("get_stats", column="city")["error"])
        self.assertIn("Unsupported operator", self.call(
            "filter_rows", filters=[{"column": "salary", "operator": "~", "value": "1"}])["error"])
        self.assertIn("not a number", self.call(
            "filter_rows", filters=[{"column": "salary", "operator": ">", "value": "lots"}])["error"])
        self.assertIn("Unsupported agg", self.call(
            "group_stats", group_by="city", column="salary", agg="mode")["error"])
        self.assertIn("Unknown tool", self.call("delete_everything")["error"])
        self.assertIn("Invalid arguments", self.call("get_stats", colum="salary")["error"])

    def test_bad_json_and_empty_arguments(self):
        self.assertIn("not valid JSON", json.loads(run_tool(self.tools, "get_stats", "{oops"))["error"])
        self.assertEqual(json.loads(run_tool(self.tools, "describe_dataset", ""))["rows"], 15)

    def test_output_is_json_safe_with_missing_values(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]})
        out = json.loads(run_tool(CsvTools(df), "filter_rows", {}))
        self.assertIsNone(out["rows"][1]["a"])
        self.assertIsNone(out["rows"][2]["b"])


if __name__ == "__main__":
    unittest.main()
