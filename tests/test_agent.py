import unittest
from pathlib import Path

import pandas as pd

from src.agent import CsvAnalystAgent
from src.llm_client import MockClient, chat_with_retry, is_retryable
from tests.fakes import FakeClient, RateLimitError

DF = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "employees.csv")
NO_SLEEP = lambda seconds: None


def make_agent(client, **kw):
    return CsvAnalystAgent(DF, client, sleep_fn=NO_SLEEP, **kw)


class AgentTests(unittest.TestCase):
    def test_tool_call_then_answer(self):
        client = FakeClient([("tool", "group_stats", {"group_by": "department", "column": "salary", "agg": "mean"}),
                             ("text", "Engineering earns more (81,667 vs 60,000).")])
        result = make_agent(client).ask("Engineering or Marketing?")
        self.assertIsNone(result.error)
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.tool_calls[0].name, "group_stats")
        self.assertAlmostEqual(result.tool_calls[0].result["groups"][0]["value"], 81666.67)
        self.assertIn("81,667", result.answer)

    def test_tool_error_is_fed_back_and_model_recovers(self):
        client = FakeClient([("tool", "get_stats", {"column": "salry"}),
                             ("tool", "get_stats", {"column": "salary"}),
                             ("text", "The average salary is 66,333.")])
        result = make_agent(client).ask("Average salary?")
        self.assertIn("error", result.tool_calls[0].result)
        self.assertNotIn("error", result.tool_calls[1].result)
        self.assertEqual(result.steps, 3)

    def test_step_limit(self):
        client = FakeClient([], loop_tool=("tool", "describe_dataset", {}))
        result = make_agent(client, max_steps=3).ask("Loop forever")
        self.assertEqual(result.error, "step_limit")
        self.assertEqual(client.calls, 3)
        self.assertEqual(len(result.tool_calls), 3)

    def test_retries_transient_errors_then_succeeds(self):
        client = FakeClient([RateLimitError("slow down"), RateLimitError("slow down"), ("text", "Done.")])
        result = make_agent(client).ask("Anything")
        self.assertIsNone(result.error)
        self.assertEqual(result.answer, "Done.")
        self.assertEqual(client.calls, 3)

    def test_gives_up_after_max_retries(self):
        client = FakeClient([RateLimitError("x")] * 10)
        result = make_agent(client, max_retries=2).ask("Anything")
        self.assertTrue(result.error.startswith("llm_error"))
        self.assertEqual(client.calls, 3)  # first try + 2 retries

    def test_non_retryable_error_fails_fast(self):
        client = FakeClient([ValueError("bad request")])
        result = make_agent(client).ask("Anything")
        self.assertTrue(result.error.startswith("llm_error"))
        self.assertEqual(client.calls, 1)

    def test_follow_up_questions_see_history(self):
        client = FakeClient([("text", "It is 15."), ("text", "Six.")])
        agent = make_agent(client)
        agent.ask("How many employees?")
        agent.ask("And in Engineering?")
        roles = [m["role"] for m in client.last_messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertIn("How many employees?", client.last_messages[1]["content"])

    def test_system_prompt_lists_columns(self):
        prompt = make_agent(FakeClient([]))._system_prompt()
        self.assertIn("15 rows", prompt)
        self.assertIn("salary (numeric)", prompt)

    def test_mock_client_runs_end_to_end(self):
        result = make_agent(MockClient()).ask("Hello")
        self.assertIsNone(result.error)
        self.assertEqual(result.tool_calls[0].name, "describe_dataset")
        self.assertIn("mock mode", result.answer)


class RetryHelperTests(unittest.TestCase):
    def test_is_retryable(self):
        self.assertTrue(is_retryable(RateLimitError("x")))
        self.assertTrue(is_retryable(Exception("Error code: 400 - tool_use_failed")))
        self.assertFalse(is_retryable(ValueError("nope")))

    def test_backoff_delays_double(self):
        delays = []
        client = FakeClient([RateLimitError("x"), RateLimitError("x"), RateLimitError("x"), ("text", "ok")])
        chat_with_retry(client, max_retries=3, backoff=1.0, sleep_fn=delays.append, messages=[])
        self.assertEqual(delays, [1.0, 2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
