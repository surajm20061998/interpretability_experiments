import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribution_steering.steering import fit_steering_state


class SteeringTests(unittest.TestCase):
    def test_fit_steering_state_selects_layers(self) -> None:
        activations = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0], [0.5, 0.0]],
                [[0.9, 0.1], [0.8, 0.2], [0.4, 0.1]],
                [[-1.0, 0.0], [-0.8, 0.2], [-0.5, 0.0]],
                [[-0.9, -0.1], [-0.7, 0.1], [-0.6, 0.0]],
            ]
        )
        labels = torch.tensor([True, True, False, False])
        layer_scores = torch.tensor(
            [
                [0.8, 0.7, 0.2],
                [0.7, 0.6, 0.2],
                [0.1, 0.2, 0.5],
                [0.2, 0.3, 0.6],
            ]
        )

        state = fit_steering_state(activations, labels, layer_scores, top_k_layers=2)
        self.assertEqual(len(state.selected_layers), 2)
        self.assertEqual(state.thresholds.shape[0], activations.shape[1])


if __name__ == "__main__":
    unittest.main()
