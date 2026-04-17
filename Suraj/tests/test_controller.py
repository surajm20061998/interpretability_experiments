import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribution_steering.controller import train_hallucination_controller


class ControllerTests(unittest.TestCase):
    def test_train_hallucination_controller_returns_usable_state(self) -> None:
        truthful = torch.randn(6, 4, 8) * 0.1 + 1.0
        hallucinated = torch.randn(6, 4, 8) * 0.1 - 1.0
        activations = torch.cat([truthful, hallucinated], dim=0)
        labels = torch.tensor([True] * 6 + [False] * 6)
        layer_scores = torch.cat(
            [
                torch.full((6, 4), 0.2),
                torch.full((6, 4), 1.0),
            ],
            dim=0,
        )

        state = train_hallucination_controller(
            activations=activations,
            labels=labels,
            layer_scores=layer_scores,
            top_k_layers=2,
            hidden_dim=16,
            dropout=0.0,
            epochs=40,
            learning_rate=1e-2,
            val_fraction=0.25,
            random_seed=0,
        )

        self.assertEqual(len(state.steering_state.selected_layers), 2)
        self.assertGreaterEqual(state.threshold, 0.1)
        self.assertLessEqual(state.threshold, 0.9)
        self.assertIn("final_train_metrics", state.training_summary)


if __name__ == "__main__":
    unittest.main()
