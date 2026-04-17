from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from .modeling import find_decoder_layers


@dataclass
class SteeringState:
    truthful_centroids: torch.Tensor
    hallucinated_centroids: torch.Tensor
    directions: torch.Tensor
    thresholds: torch.Tensor
    layer_weights: torch.Tensor
    selected_layers: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "truthful_centroids": self.truthful_centroids,
            "hallucinated_centroids": self.hallucinated_centroids,
            "directions": self.directions,
            "thresholds": self.thresholds,
            "layer_weights": self.layer_weights,
            "selected_layers": self.selected_layers,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SteeringState":
        return cls(
            truthful_centroids=payload["truthful_centroids"],
            hallucinated_centroids=payload["hallucinated_centroids"],
            directions=payload["directions"],
            thresholds=payload["thresholds"],
            layer_weights=payload["layer_weights"],
            selected_layers=list(payload["selected_layers"]),
        )

    def save(self, path: str | Path) -> None:
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "SteeringState":
        return cls.from_dict(torch.load(path, map_location="cpu"))


def hallucination_score(hidden_state: torch.Tensor, state: SteeringState, layer_index: int) -> torch.Tensor:
    truthful = state.truthful_centroids[layer_index].unsqueeze(0)
    hallucinated = state.hallucinated_centroids[layer_index].unsqueeze(0)
    return _centroid_score(hidden_state, truthful, hallucinated)


def _centroid_score(
    hidden_state: torch.Tensor,
    truthful_centroid: torch.Tensor,
    hallucinated_centroid: torch.Tensor,
) -> torch.Tensor:
    hallucinated_score = F.cosine_similarity(hidden_state, hallucinated_centroid, dim=-1)
    truthful_score = F.cosine_similarity(hidden_state, truthful_centroid, dim=-1)
    return hallucinated_score - truthful_score


def fit_steering_state(
    activations: torch.Tensor,
    labels: torch.Tensor,
    layer_scores: torch.Tensor,
    top_k_layers: int = 6,
) -> SteeringState:
    if activations.ndim != 3:
        raise ValueError("Expected activations with shape [num_examples, num_layers, hidden_size].")
    if labels.ndim != 1:
        raise ValueError("Expected labels with shape [num_examples].")
    if activations.shape[0] != labels.shape[0]:
        raise ValueError("Activations and labels must have matching example counts.")
    if layer_scores.ndim != 2:
        raise ValueError("Expected layer_scores with shape [num_examples, num_layers].")
    if layer_scores.shape[0] != activations.shape[0] or layer_scores.shape[1] != activations.shape[1]:
        raise ValueError("layer_scores must align with the example and layer dimensions of activations.")

    truthful_mask = labels.bool()
    hallucinated_mask = ~truthful_mask
    truthful_count = int(truthful_mask.sum().item())
    hallucinated_count = int(hallucinated_mask.sum().item())
    if truthful_count == 0 or hallucinated_count == 0:
        raise ValueError(
            "Need at least one truthful and one hallucinated example to fit a steering state."
        )

    truthful_centroids = activations[truthful_mask].mean(dim=0)
    hallucinated_centroids = activations[hallucinated_mask].mean(dim=0)
    raw_directions = truthful_centroids - hallucinated_centroids
    directions = F.normalize(raw_directions, dim=-1)

    truthful_scores = []
    hallucinated_scores = []
    for layer_index in range(activations.shape[1]):
        truthful_scores.append(
            _centroid_score(
                activations[truthful_mask, layer_index, :],
                truthful_centroids[layer_index].unsqueeze(0),
                hallucinated_centroids[layer_index].unsqueeze(0),
            )
        )
        hallucinated_scores.append(
            _centroid_score(
                activations[hallucinated_mask, layer_index, :],
                truthful_centroids[layer_index].unsqueeze(0),
                hallucinated_centroids[layer_index].unsqueeze(0),
            )
        )

    thresholds = torch.stack(
        [(truth.mean() + hall.mean()) / 2.0 for truth, hall in zip(truthful_scores, hallucinated_scores)]
    )
    score_gap = torch.abs(layer_scores[truthful_mask].mean(dim=0) - layer_scores[hallucinated_mask].mean(dim=0))
    if torch.all(score_gap == 0):
        score_gap = torch.ones_like(score_gap)
    layer_weights = score_gap / score_gap.max()
    top_k = min(top_k_layers, activations.shape[1])
    selected_layers = torch.topk(layer_weights, k=top_k).indices.tolist()

    return SteeringState(
        truthful_centroids=truthful_centroids.detach().cpu(),
        hallucinated_centroids=hallucinated_centroids.detach().cpu(),
        directions=directions.detach().cpu(),
        thresholds=thresholds.detach().cpu(),
        layer_weights=layer_weights.detach().cpu(),
        selected_layers=selected_layers,
    )


class ActivationSteerer:
    def __init__(
        self,
        model: PreTrainedModel,
        state: SteeringState,
        steering_scale: float = 2.0,
    ) -> None:
        self.model = model
        self.state = state
        self.steering_scale = steering_scale
        self._handles: list[Any] = []

    def __enter__(self) -> "ActivationSteerer":
        decoder_layers = find_decoder_layers(self.model)
        for layer_index in self.state.selected_layers:
            layer = decoder_layers[layer_index]
            self._handles.append(layer.register_forward_hook(self._make_hook(layer_index)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden_states: torch.Tensor
            remainder: tuple[Any, ...] = ()
            if isinstance(output, tuple):
                hidden_states = output[0]
                remainder = output[1:]
            else:
                hidden_states = output

            if hidden_states.ndim != 3 or hidden_states.shape[1] == 0:
                return output

            state_device = hidden_states.device
            state_dtype = hidden_states.dtype
            current_state = hidden_states[:, -1, :]
            truthful_centroid = self.state.truthful_centroids[layer_index].to(
                device=state_device, dtype=state_dtype
            )
            hallucinated_centroid = self.state.hallucinated_centroids[layer_index].to(
                device=state_device, dtype=state_dtype
            )
            score = _centroid_score(
                current_state,
                truthful_centroid.unsqueeze(0),
                hallucinated_centroid.unsqueeze(0),
            )
            threshold = self.state.thresholds[layer_index].to(device=state_device, dtype=state_dtype)
            mask = (score > threshold).to(state_dtype).unsqueeze(-1)
            delta = (
                self.steering_scale
                * self.state.layer_weights[layer_index].to(device=state_device, dtype=state_dtype)
                * self.state.directions[layer_index].to(device=state_device, dtype=state_dtype)
            )

            steered_hidden = hidden_states.clone()
            steered_hidden[:, -1, :] = hidden_states[:, -1, :] + mask * delta
            if isinstance(output, tuple):
                return (steered_hidden, *remainder)
            return steered_hidden

        return hook
