import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribution_steering.modeling import render_prompt_for_model


class _FakeChatTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        del tokenize
        rendered = " | ".join(f"{item['role']}={item['content']}" for item in messages)
        if add_generation_prompt:
            rendered += " | assistant="
        return rendered


class ModelingTests(unittest.TestCase):
    def test_render_prompt_for_model_uses_chat_template_when_available(self) -> None:
        tokenizer = _FakeChatTokenizer()
        rendered = render_prompt_for_model(
            tokenizer,
            prompt="fallback prompt",
            messages=[
                {"role": "system", "content": "system text"},
                {"role": "user", "content": "user text"},
            ],
        )
        self.assertIn("system=system text", rendered)
        self.assertIn("assistant=", rendered)

    def test_render_prompt_for_model_falls_back_to_plain_prompt(self) -> None:
        class PlainTokenizer:
            chat_template = None

        rendered = render_prompt_for_model(PlainTokenizer(), prompt="fallback prompt", messages=None)
        self.assertEqual(rendered, "fallback prompt")


if __name__ == "__main__":
    unittest.main()
