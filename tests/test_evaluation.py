import unittest
from pathlib import Path

from evaluation.checks import extract_numbers, matches, number_match
from evaluation.questions import QUESTIONS
from evaluation.runner import run_questions, summarize
from src.data_loader import DatasetStore, load_samples

DATA = Path(__file__).resolve().parent.parent / "data"


class CheckTests(unittest.TestCase):
    def test_extract_numbers_handles_commas_and_decimals(self):
        self.assertEqual(extract_numbers("Average is 81,666.67 (6 people)"), [81666.67, 6.0])

    def test_number_match_tolerance(self):
        self.assertTrue(number_match("about 81,667", 81666.67))
        self.assertFalse(number_match("about 78,500", 81666.67))  # the v1 notebook's wrong answer

    def test_unicode_thousands_separator(self):
        """Seen in a real run: 'Average salary: 66<U+202F>333.33.' was wrongly marked as a failure."""
        self.assertEqual(extract_numbers("Average salary: 66\u202f333.33."), [66333.33])
        self.assertTrue(number_match("Engineering 81\u202f666.67; difference 21\u202f666.67.", 21666.67))
        self.assertEqual(extract_numbers("5\u202fyears of experience"), [5.0])

    def test_non_breaking_hyphen_in_ids(self):
        self.assertTrue(matches("1. ORD\u20111185  2. ORD\u20111157", ["ORD-1185", "ORD-1157"], "list"))

    def test_text_and_list_match(self):
        self.assertTrue(matches("Ravi earns the most.", "Ravi", "text"))
        self.assertFalse(matches("Vikram earns the most.", "Ravi", "text"))
        self.assertTrue(matches("Sneha, Kavya and Karthik", ["Sneha", "Kavya", "Karthik"], "list"))
        self.assertFalse(matches("Sneha and Kavya", ["Sneha", "Kavya", "Karthik"], "list"))


class QuestionSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        store = DatasetStore()
        load_samples(store, DATA)
        cls.datasets = {n: store.get(n) for n in store.names()}

    def test_ground_truth_is_computable_and_unambiguous(self):
        """Every expected() runs without a tie error, so no question has two right answers."""
        for q in QUESTIONS:
            with self.subTest(q.text):
                q.expected(self.datasets[q.dataset])

    def test_oracle_scores_100_and_wrong_answers_score_0(self):
        def oracle(name, df, text):
            q = next(x for x in QUESTIONS if x.text == text)
            value = q.expected(df)
            return (", ".join(map(str, value)) if q.kind == "list" else str(value)), {"tool_calls": 1}

        rows = run_questions(self.datasets, QUESTIONS, oracle)
        self.assertEqual(summarize(rows)["overall"]["accuracy"], 100.0)
        rows = run_questions(self.datasets, QUESTIONS, lambda n, d, t: ("I don't know", {}))
        self.assertEqual(summarize(rows)["overall"]["accuracy"], 0.0)

    def test_question_counts(self):
        by = summarize(run_questions(self.datasets, QUESTIONS, lambda n, d, t: ("", {})))["by_dataset"]
        self.assertEqual(by["employees"]["questions"], 15)
        self.assertEqual(by["sales"]["questions"], 12)


if __name__ == "__main__":
    unittest.main()