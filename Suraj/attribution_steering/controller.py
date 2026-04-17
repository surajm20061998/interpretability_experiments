from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .modeling import find_decoder_layers
from .steering import SteeringState, fit_steering_state


class HallucinationSignalNet(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        embedding_dim = max(16, min(64, hidden_dim // 4))
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_proj = nn.Linear(hidden_size, hidden_dim)
        self.layer_embedding = nn.Embedding(num_layers, embedding_dim)
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_state: torch.Tensor, layer_positions: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(hidden_state.float())
        projected = self.input_proj(normalized)
        layer_features = self.layer_embedding(layer_positions.long())
        logits = self.classifier(torch.cat([projected, layer_features], dim=-1))
        return logits.squeeze(-1)


@dataclass
class NeuralControllerState:
    steering_state: SteeringState
    controller_state_dict: dict[str, torch.Tensor]
    hidden_size: int
    hidden_dim: int
    dropout: float
    threshold: float
    training_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steering_state": self.steering_state.to_dict(),
            "controller_state_dict": {
                key: value.detach().cpu() for key, value in self.controller_state_dict.items()
            },
            "hidden_size": self.hidden_size,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "threshold": self.threshold,
            "training_summary": self.training_summary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NeuralControllerState":
        return cls(
            steering_state=SteeringState.from_dict(payload["steering_state"]),
            controller_state_dict=payload["controller_state_dict"],
            hidden_size=int(payload["hidden_size"]),
            hidden_dim=int(payload["hidden_dim"]),
            dropout=float(payload["dropout"]),
            threshold=float(payload["threshold"]),
            training_summary=dict(payload.get("training_summary", {})),
        )

    def save(self, path: str | Path) -> None:
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralControllerState":
        return cls.from_dict(torch.load(path, map_location="cpu"))

    def build_model(self, device: torch.device | str | None = None) -> HallucinationSignalNet:
        model = HallucinationSignalNet(
            hidden_size=self.hidden_size,
            num_layers=len(self.steering_state.selected_layers),
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        model.load_state_dict(self.controller_state_dict)
        if device is not None:
            model.to(device)
        model.eval()
        return model


class NeuralActivationSteerer:
    def __init__(
        self,
        model: PreTrainedModel,
        state: NeuralControllerState,
        steering_scale: float = 2.0,
    ) -> None:
        self.model = model
        self.state = state
        self.steering_scale = steering_scale
        self._handles: list[Any] = []
        self._controller: HallucinationSignalNet | None = None
        self._layer_to_position = {
            layer_index: position for position, layer_index in enumerate(self.state.steering_state.selected_layers)
        }

    def __enter__(self) -> "NeuralActivationSteerer":
        device = next(self.model.parameters()).device
        self._controller = self.state.build_model(device=device)
        decoder_layers = find_decoder_layers(self.model)
        for layer_index in self.state.steering_state.selected_layers:
            layer = decoder_layers[layer_index]
            self._handles.append(layer.register_forward_hook(self._make_hook(layer_index)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._controller = None

    def _make_hook(self, layer_index: int):
        layer_position = self._layer_to_position[layer_index]

        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            if self._controller is None:
                return output

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
            layer_positions = torch.full(
                (current_state.shape[0],),
                layer_position,
                dtype=torch.long,
                device=state_device,
            )
            with torch.no_grad():
                hallucination_logits = self._controller(current_state.float(), layer_positions)
                hallucination_probabilities = torch.sigmoid(hallucination_logits)

            threshold = self.state.threshold
            normalizer = max(1.0 - threshold, 1e-6)
            gate = torch.clamp((hallucination_probabilities - threshold) / normalizer, min=0.0, max=1.0)
            gate = gate.to(dtype=state_dtype).unsqueeze(-1)
            direction = self.state.steering_state.directions[layer_index].to(
                device=state_device,
                dtype=state_dtype,
            )
            layer_weight = self.state.steering_state.layer_weights[layer_index].to(
                device=state_device,
                dtype=state_dtype,
            )
            delta = self.steering_scale * layer_weight * gate * direction

            steered_hidden = hidden_states.clone()
            steered_hidden[:, -1, :] = hidden_states[:, -1, :] + delta
            if isinstance(output, tuple):
                return (steered_hidden, *remainder)
            return steered_hidden

        return hook


def _example_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    labels_bool = labels.bool()
    true_positive = int((predictions & labels_bool).sum().item())
    true_negative = int((~predictions & ~labels_bool).sum().item())
    false_positive = int((predictions & ~labels_bool).sum().item())
    false_negative = int((~predictions & labels_bool).sum().item())
    total = max(labels.numel(), 1)

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    accuracy = (true_positive + true_negative) / total
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _select_threshold(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    if probabilities.numel() == 0:
        return 0.5
    best_threshold = 0.5
    best_metrics = None
    for threshold in torch.linspace(0.1, 0.9, steps=17):
        metrics = _example_metrics(probabilities, labels, float(threshold.item()))
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold.item())
            best_metrics = metrics
    return best_threshold


def _stratified_split_indices(
    hallucination_labels: torch.Tensor,
    val_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    positive = torch.nonzero(hallucination_labels, as_tuple=False).flatten()
    negative = torch.nonzero(~hallucination_labels, as_tuple=False).flatten()

    positive = positive[torch.randperm(len(positive), generator=generator)] if len(positive) else positive
    negative = negative[torch.randperm(len(negative), generator=generator)] if len(negative) else negative

    def _split(bucket: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(bucket) <= 1 or val_fraction <= 0:
            return bucket, bucket.new_empty((0,), dtype=torch.long)
        val_size = max(1, int(round(len(bucket) * val_fraction)))
        if val_size >= len(bucket):
            val_size = len(bucket) - 1
        train_size = len(bucket) - val_size
        return bucket[:train_size], bucket[train_size:]

    positive_train, positive_val = _split(positive)
    negative_train, negative_val = _split(negative)
    train_indices = torch.cat([positive_train, negative_train])
    val_indices = torch.cat([positive_val, negative_val])
    if len(train_indices):
        train_indices = train_indices[torch.randperm(len(train_indices), generator=generator)]
    if len(val_indices):
        val_indices = val_indices[torch.randperm(len(val_indices), generator=generator)]
    return train_indices, val_indices


def _build_layer_samples(
    activations: torch.Tensor,
    hallucination_labels: torch.Tensor,
    selected_layers: list[int],
    example_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if example_indices.numel() == 0:
        hidden_size = activations.shape[-1]
        return (
            torch.empty((0, hidden_size), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    layer_slice = activations[example_indices][:, selected_layers, :].float()
    num_examples, num_layers, hidden_size = layer_slice.shape
    hidden_states = layer_slice.reshape(num_examples * num_layers, hidden_size)
    layer_positions = (
        torch.arange(num_layers, dtype=torch.long).unsqueeze(0).expand(num_examples, num_layers).reshape(-1)
    )
    labels = (
        hallucination_labels[example_indices]
        .unsqueeze(1)
        .expand(num_examples, num_layers)
        .reshape(-1)
        .float()
    )
    return hidden_states, layer_positions, labels


def train_hallucination_controller(
    activations: torch.Tensor,
    labels: torch.Tensor,
    layer_scores: torch.Tensor,
    top_k_layers: int = 6,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.25,
    random_seed: int = 0,
) -> NeuralControllerState:
    activations = activations.float().cpu()
    labels = labels.bool().cpu()
    layer_scores = layer_scores.float().cpu()
    steering_state = fit_steering_state(
        activations=activations,
        labels=labels,
        layer_scores=layer_scores,
        top_k_layers=top_k_layers,
    )

    hallucination_labels = (~labels).cpu()
    train_indices, val_indices = _stratified_split_indices(
        hallucination_labels=hallucination_labels,
        val_fraction=val_fraction,
        seed=random_seed,
    )
    if train_indices.numel() == 0:
        train_indices = torch.arange(activations.shape[0])

    train_hidden, train_layer_positions, train_targets = _build_layer_samples(
        activations=activations,
        hallucination_labels=hallucination_labels,
        selected_layers=steering_state.selected_layers,
        example_indices=train_indices,
    )
    val_hidden, val_layer_positions, val_targets = _build_layer_samples(
        activations=activations,
        hallucination_labels=hallucination_labels,
        selected_layers=steering_state.selected_layers,
        example_indices=val_indices,
    )

    model = HallucinationSignalNet(
        hidden_size=activations.shape[-1],
        num_layers=len(steering_state.selected_layers),
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    positive_count = max(int(train_targets.sum().item()), 1)
    negative_count = max(int(train_targets.numel() - positive_count), 1)
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state_dict = deepcopy(model.state_dict())
    best_epoch = 0
    best_threshold = 0.5
    best_metric = -1.0
    best_train_metrics: dict[str, float] = {}
    best_val_metrics: dict[str, float] = {}
    has_validation = val_hidden.numel() > 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(train_hidden, train_layer_positions)
        loss = criterion(train_logits, train_targets)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_probabilities = torch.sigmoid(model(train_hidden, train_layer_positions))
            train_threshold = 0.5
            train_metrics = _example_metrics(train_probabilities, train_targets, train_threshold)

            if has_validation:
                val_probabilities = torch.sigmoid(model(val_hidden, val_layer_positions))
                threshold = _select_threshold(val_probabilities, val_targets)
                val_metrics = _example_metrics(val_probabilities, val_targets, threshold)
                score = val_metrics["f1"]
            else:
                threshold = _select_threshold(train_probabilities, train_targets)
                val_metrics = {}
                score = _example_metrics(train_probabilities, train_targets, threshold)["f1"]

        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_threshold = threshold
            best_state_dict = deepcopy(model.state_dict())
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics

    model.load_state_dict(best_state_dict)
    model.eval()
    with torch.no_grad():
        final_train_probabilities = torch.sigmoid(model(train_hidden, train_layer_positions))
        final_train_metrics = _example_metrics(final_train_probabilities, train_targets, best_threshold)
        if has_validation:
            final_val_probabilities = torch.sigmoid(model(val_hidden, val_layer_positions))
            final_val_metrics = _example_metrics(final_val_probabilities, val_targets, best_threshold)
        else:
            final_val_metrics = {}

    summary = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "train_examples": int(train_indices.numel()),
        "val_examples": int(val_indices.numel()),
        "train_layer_samples": int(train_hidden.shape[0]),
        "val_layer_samples": int(val_hidden.shape[0]),
        "selected_layers": steering_state.selected_layers,
        "best_epoch": best_epoch,
        "threshold": float(best_threshold),
        "best_selection_metric": float(best_metric),
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
    }
    return NeuralControllerState(
        steering_state=steering_state,
        controller_state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        hidden_size=int(activations.shape[-1]),
        hidden_dim=hidden_dim,
        dropout=dropout,
        threshold=float(best_threshold),
        training_summary=summary,
    )
