import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribution_steering.dataset import (
    FactExample,
    answer_is_correct,
    build_messages,
    build_prompt,
    normalize_text,
)


class DatasetTests(unittest.TestCase):
    def test_normalize_text_removes_punctuation(self) -> None:
        self.assertEqual(normalize_text("George Orwell."), "george orwell")

    def test_answer_is_correct_matches_aliases(self) -> None:
        example = FactExample(
            id="orwell",
            question="Who wrote 1984?",
            answer="George Orwell",
            aliases=("Orwell",),
        )
        self.assertTrue(answer_is_correct("It was written by Orwell.", example))

    def test_build_prompt_uses_misleading_context(self) -> None:
        example = FactExample(
            id="capital",
            question="What is the capital of France?",
            answer="Paris",
            misleading_context="France's capital city is Lyon.",
        )
        prompt = build_prompt(example, "misleading")
        self.assertIn("France's capital city is Lyon.", prompt)

    def test_build_messages_splits_system_and_user_content(self) -> None:
        example = FactExample(
            id="capital",
            question="What is the capital of France?",
            answer="Paris",
            misleading_context="France's capital city is Lyon.",
        )
        messages = build_messages(example, "misleading")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("France's capital city is Lyon.", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
