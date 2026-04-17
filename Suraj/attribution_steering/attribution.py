from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


def retain_hidden_state_grads(hidden_states: tuple[torch.Tensor, ...]) -> None:
    for hidden_state in hidden_states:
        hidden_state.retain_grad()


def _node_scores(hidden_states: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    scores: list[torch.Tensor] = []
    for hidden_state in hidden_states[1:]:
        if hidden_state.grad is None:
            raise RuntimeError("Missing gradients on hidden states. Call backward() first.")
        grad_times_act = hidden_state.grad * hidden_state
        scores.append(grad_times_act.norm(dim=-1)[0].detach().cpu())
    return scores


def build_attribution_graph(
    hidden_states: tuple[torch.Tensor, ...],
    attentions: tuple[torch.Tensor, ...],
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    target_token_id: int,
    top_k_nodes: int = 20,
    top_k_edges: int = 40,
) -> dict[str, Any]:
    node_scores = _node_scores(hidden_states)
    token_ids = input_ids[0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    flat_nodes: list[dict[str, Any]] = []
    for layer_index, layer_scores in enumerate(node_scores):
        for token_index, score in enumerate(layer_scores.tolist()):
            flat_nodes.append(
                {
                    "layer": layer_index,
                    "token_index": token_index,
                    "token": tokens[token_index],
                    "score": float(score),
                }
            )
    top_nodes = sorted(flat_nodes, key=lambda item: item["score"], reverse=True)[:top_k_nodes]

    flat_edges: list[dict[str, Any]] = []
    for layer_index, attention in enumerate(attentions):
        layer_scores = node_scores[layer_index]
        if attention is None:
            attention_map = torch.full(
                (layer_scores.shape[0], layer_scores.shape[0]),
                1.0 / max(layer_scores.shape[0], 1),
                dtype=torch.float32,
            )
        else:
            attention_map = attention[0].mean(dim=0).detach().cpu()
        for dst_index in range(attention_map.shape[0]):
            dst_score = layer_scores[dst_index].item()
            for src_index in range(attention_map.shape[1]):
                src_score = layer_scores[src_index].item()
                score = attention_map[dst_index, src_index].item() * (src_score + dst_score) / 2.0
                flat_edges.append(
                    {
                        "layer": layer_index,
                        "src_index": src_index,
                        "dst_index": dst_index,
                        "src_token": tokens[src_index],
                        "dst_token": tokens[dst_index],
                        "score": float(score),
                    }
                )
    top_edges = sorted(flat_edges, key=lambda item: item["score"], reverse=True)[:top_k_edges]
    layer_scores = [float(layer.mean().item()) for layer in node_scores]

    return {
        "target_token_id": int(target_token_id),
        "target_token": tokenizer.decode([target_token_id]).strip(),
        "layer_scores": layer_scores,
        "top_nodes": top_nodes,
        "top_edges": top_edges,
        "tokens": tokens,
    }
