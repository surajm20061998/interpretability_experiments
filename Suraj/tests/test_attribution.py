import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribution_steering.attribution import build_attribution_graph


class _TinyTokenizer:
    def convert_ids_to_tokens(self, token_ids):
        return [f"tok_{token_id}" for token_id in token_ids]

    def decode(self, token_ids, skip_special_tokens: bool = True):
        del skip_special_tokens
        return " ".join(f"tok_{token_id}" for token_id in token_ids)


class AttributionTests(unittest.TestCase):
    def test_build_attribution_graph_handles_missing_attentions(self) -> None:
        hidden_states = (
            torch.randn(1, 3, 4, requires_grad=True),
            torch.randn(1, 3, 4, requires_grad=True),
            torch.randn(1, 3, 4, requires_grad=True),
        )
        for hidden_state in hidden_states:
            hidden_state.retain_grad()
            hidden_state.grad = torch.ones_like(hidden_state)

        graph = build_attribution_graph(
            hidden_states=hidden_states,
            attentions=None,
            input_ids=torch.tensor([[1, 2, 3]]),
            tokenizer=_TinyTokenizer(),
            target_token_id=3,
            top_k_nodes=3,
            top_k_edges=3,
        )

        self.assertEqual(len(graph["layer_scores"]), 2)
        self.assertEqual(len(graph["top_edges"]), 3)


if __name__ == "__main__":
    unittest.main()
