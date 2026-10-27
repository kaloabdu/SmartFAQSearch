
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from src.main import search_faqs


class FakeModel:
    """Simulate an embedding model without loading the real model."""

    def encode(self, texts):
        return np.array([[0.1, 0.2]])


class FakeCollection:
    """Simulate ChromaDB search results."""

    def __init__(self, questions, answers, categories, distances):
        self.questions = questions
        self.answers = answers
        self.categories = categories
        self.distances = distances

    def count(self):
        return len(self.questions)

    def query(self, query_embeddings, n_results, include):
        return {
            "documents": [self.questions[:n_results]],
            "metadatas": [
                [
                    {
                        "answer": self.answers[i],
                        "category": self.categories[i],
                    }
                    for i in range(n_results)
                ]
            ],
            "distances": [self.distances[:n_results]],
        }


class SearchTests(unittest.TestCase):

    def setUp(self):
        self.model = FakeModel()

    def run_search(self, collection, query="test question"):
        output = StringIO()

        with redirect_stdout(output):
            search_faqs(query, collection, self.model)

        return output.getvalue()

    def test_relevant_question_returns_answer(self):
        collection = FakeCollection(
            ["How can I reset my password?"],
            ["Use the university portal."],
            ["Account"],
            [0.2162],
        )

        output = self.run_search(collection)

        self.assertIn("How can I reset my password?", output)
        self.assertIn("Use the university portal.", output)

    def test_unrelated_question_is_rejected(self):
        collection = FakeCollection(
            ["Where can I find my grades?"],
            ["Check the student portal."],
            ["Grades"],
            [0.9414],
        )

        output = self.run_search(
            collection,
            "What is the best restaurant in Addis Ababa?",
        )

        self.assertIn(
            "couldn't find a sufficiently relevant FAQ",
            output,
        )
        self.assertNotIn("Check the student portal.", output)

    
    def test_clear_question_shows_only_best_answer(self):
        collection = FakeCollection(
            [
            "How can I reset my password?",
            "How do I contact student support?",
            "Where can I find my exam schedule?",
            ],
            [
            "Use password recovery.",
            "Contact IT support.",
            "Check the academic portal.",
            ],
            ["Account", "Account", "Exams"],
            [0.2162, 0.5954, 0.7543],
        )

        output = self.run_search(collection)

        self.assertIn("Use password recovery.", output)
        self.assertNotIn("Contact IT support.", output)
        self.assertNotIn("Check the academic portal.", output)
        self.assertNotIn("more than one interpretation", output)
    
    def test_ambiguous_question_shows_closest_answer_and_alternatives(self):
        collection = FakeCollection(
            [
                "When are examination results published?",
                "Where can I find my exam schedule?",
                "Where can I find my grades?",
            ],
            [
                "Check official announcements.",
                "Check the academic portal.",
                "Check the student portal.",
            ],
            ["Grades", "Exams", "Grades"],
            [0.3125, 0.3617, 0.5070],
        )

        output = self.run_search(
            collection,
            "How do I see my exam results?",
        )

        self.assertIn("more than one interpretation", output)
        self.assertIn("Check official announcements.", output)
        self.assertIn("Did you mean one of these questions?", output)
        self.assertIn("Where can I find my exam schedule?", output)
        self.assertIn("Where can I find my grades?", output)

    
    def test_empty_collection_is_handled(self):
        collection = FakeCollection([], [], [], [])

        output = self.run_search(collection)

        self.assertIn("no FAQs are available", output)


if __name__ == "__main__":
    unittest.main()
