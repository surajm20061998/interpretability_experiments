from __future__ import annotations

import json
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .attribution import build_attribution_graph, retain_hidden_state_grads
from .dataset import answer_is_correct, build_messages, build_prompt, load_dataset, parse_conditions
from .modeling import decode_new_tokens, load_model_and_tokenizer, render_prompt_for_model
from .steering import ActivationSteerer, SteeringState, fit_steering_state


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _move_encoding(encoded: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in encoded.items()}


def _select_next_token(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1, keepdim=True)


def _prepare_prompts(tokenizer: Any, example: Any, condition: str) -> tuple[str, str, list[dict[str, str]]]:
    prompt = build_prompt(example, condition)
    messages = build_messages(example, condition)
    model_prompt = render_prompt_for_model(tokenizer, prompt, messages=messages)
    return prompt, model_prompt, messages


def run_prompt_trace(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    encoded = _move_encoding(tokenizer(prompt, return_tensors="pt"), str(device))

    model.zero_grad(set_to_none=True)
    outputs = model(
        **encoded,
        output_hidden_states=True,
        output_attentions=True,
        use_cache=False,
    )
    retain_hidden_state_grads(outputs.hidden_states)
    next_token_logits = outputs.logits[:, -1, :]
    next_token = _select_next_token(next_token_logits)
    selected_logit = next_token_logits.gather(-1, next_token).sum()
    selected_logit.backward()

    graph = build_attribution_graph(
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        input_ids=encoded["input_ids"],
        tokenizer=tokenizer,
        target_token_id=int(next_token.item()),
    )
    layer_activations = torch.stack(
        [hidden_state[0, -1, :].detach().cpu() for hidden_state in outputs.hidden_states[1:]],
        dim=0,
    )
    model.zero_grad(set_to_none=True)
    return {
        "graph": graph,
        "layer_activations": layer_activations,
        "layer_scores": torch.tensor(graph["layer_scores"], dtype=torch.float32),
    }


def generate_answer(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 8,
    steerer: ActivationSteerer | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    encoded = _move_encoding(tokenizer(prompt, return_tensors="pt"), str(device))
    prompt_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    eos_token_id = tokenizer.eos_token_id
    generated = torch.empty((1, 0), dtype=prompt_ids.dtype, device=prompt_ids.device)

    steering_context = steerer if steerer is not None else nullcontext()
    with steering_context:
        with torch.no_grad():
            outputs = model(**encoded, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = _select_next_token(outputs.logits[:, -1, :])
            generated = torch.cat([generated, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device),
                    ],
                    dim=-1,
                )

            for _ in range(max_new_tokens - 1):
                if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                    break
                step_outputs = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = step_outputs.past_key_values
                next_token = _select_next_token(step_outputs.logits[:, -1, :])
                generated = torch.cat([generated, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (attention_mask.shape[0], 1),
                                dtype=attention_mask.dtype,
                                device=attention_mask.device,
                            ),
                        ],
                        dim=-1,
                    )

    full_ids = torch.cat([prompt_ids, generated], dim=-1)
    text = decode_new_tokens(tokenizer, prompt_ids, full_ids)
    return {
        "text": text,
        "token_ids": generated[0].detach().cpu().tolist(),
    }


def _accuracy_by_condition(records: list[dict[str, Any]], key: str = "is_correct") -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        grouped[record["condition"]].append(bool(record[key]))
    return {
        condition: sum(values) / len(values)
        for condition, values in sorted(grouped.items())
        if values
    }


def _summarize_collection(records: list[dict[str, Any]]) -> dict[str, Any]:
    correctness = Counter(record["is_correct"] for record in records)
    return {
        "num_records": len(records),
        "num_correct": int(correctness.get(True, 0)),
        "num_hallucinated": int(correctness.get(False, 0)),
        "accuracy_by_condition": _accuracy_by_condition(records),
    }


def collect_dataset(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    device: str | None = None,
    max_new_tokens: int = 8,
    max_examples: int | None = None,
    conditions: str | list[str] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    examples = load_dataset(dataset_path, limit=max_examples)
    requested_conditions = parse_conditions(conditions)

    records: list[dict[str, Any]] = []
    activation_records: list[dict[str, Any]] = []
    for example in tqdm(examples, desc="collecting"):
        for condition in requested_conditions:
            prompt, model_prompt, messages = _prepare_prompts(tokenizer, example, condition)
            trace = run_prompt_trace(model, tokenizer, model_prompt)
            generation = generate_answer(
                model,
                tokenizer,
                model_prompt,
                max_new_tokens=max_new_tokens,
            )
            is_correct = answer_is_correct(generation["text"], example)

            record = {
                "example_id": example.id,
                "condition": condition,
                "question": example.question,
                "answer": example.answer,
                "aliases": list(example.aliases),
                "prompt": prompt,
                "model_prompt": model_prompt,
                "messages": messages,
                "generated_text": generation["text"],
                "generated_token_ids": generation["token_ids"],
                "is_correct": is_correct,
                "graph": trace["graph"],
            }
            records.append(record)
            activation_records.append(
                {
                    "example_id": example.id,
                    "condition": condition,
                    "is_correct": bool(is_correct),
                    "activations": trace["layer_activations"],
                    "layer_scores": trace["layer_scores"],
                }
            )

    summary = _summarize_collection(records)
    _write_jsonl(output_path / "records.jsonl", records)
    _write_json(output_path / "summary.json", summary)
    torch.save(
        {
            "model_name": model_name,
            "dataset_path": dataset_path,
            "records": activation_records,
        },
        output_path / "activations.pt",
    )
    return summary


def _aggregate_graph_features(
    records: list[dict[str, Any]],
    is_correct: bool,
    feature_name: str,
    top_n: int = 15,
) -> list[dict[str, Any]]:
    scores: dict[tuple[Any, ...], float] = defaultdict(float)
    for record in records:
        if bool(record["is_correct"]) != is_correct:
            continue
        for item in record["graph"][feature_name]:
            if feature_name == "top_nodes":
                key = (item["layer"], item["token"])
                scores[key] += float(item["score"])
            else:
                key = (item["layer"], item["src_token"], item["dst_token"])
                scores[key] += float(item["score"])

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_n]
    results: list[dict[str, Any]] = []
    for key, value in ranked:
        if feature_name == "top_nodes":
            layer, token = key
            results.append({"layer": layer, "token": token, "score": value})
        else:
            layer, src_token, dst_token = key
            results.append(
                {
                    "layer": layer,
                    "src_token": src_token,
                    "dst_token": dst_token,
                    "score": value,
                }
            )
    return results


def analyze_collection(
    input_dir: str,
    output_dir: str,
    fit_conditions: str | list[str] | None = "misleading",
    top_k_layers: int = 6,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    activation_bundle = torch.load(input_path / "activations.pt", map_location="cpu")
    raw_records = _read_jsonl(input_path / "records.jsonl")
    selected_conditions = set(parse_conditions(fit_conditions))

    fit_records = [
        record for record in activation_bundle["records"] if record["condition"] in selected_conditions
    ]
    if not fit_records:
        raise ValueError("No activation records matched the requested fit conditions.")

    activations = torch.stack([record["activations"] for record in fit_records], dim=0)
    labels = torch.tensor([record["is_correct"] for record in fit_records], dtype=torch.bool)
    layer_scores = torch.stack([record["layer_scores"] for record in fit_records], dim=0)
    state = fit_steering_state(
        activations=activations,
        labels=labels,
        layer_scores=layer_scores,
        top_k_layers=top_k_layers,
    )
    state.save(output_path / "steering_state.pt")

    truthful_mask = labels.bool()
    hallucinated_mask = ~truthful_mask
    truthful_layer_scores = layer_scores[truthful_mask].mean(dim=0)
    hallucinated_layer_scores = layer_scores[hallucinated_mask].mean(dim=0)

    summary = {
        "model_name": activation_bundle["model_name"],
        "fit_conditions": sorted(selected_conditions),
        "num_fit_examples": len(fit_records),
        "num_truthful": int(truthful_mask.sum().item()),
        "num_hallucinated": int(hallucinated_mask.sum().item()),
        "selected_layers": state.selected_layers,
        "layer_weights": state.layer_weights.tolist(),
        "truthful_layer_scores": truthful_layer_scores.tolist(),
        "hallucinated_layer_scores": hallucinated_layer_scores.tolist(),
        "graph_contrast": {
            "truthful_top_nodes": _aggregate_graph_features(raw_records, True, "top_nodes"),
            "hallucinated_top_nodes": _aggregate_graph_features(raw_records, False, "top_nodes"),
            "truthful_top_edges": _aggregate_graph_features(raw_records, True, "top_edges"),
            "hallucinated_top_edges": _aggregate_graph_features(raw_records, False, "top_edges"),
        },
    }
    _write_json(output_path / "analysis.json", summary)
    return summary


def evaluate_steering(
    model_name: str,
    dataset_path: str,
    steering_state_path: str,
    output_dir: str,
    device: str | None = None,
    max_new_tokens: int = 8,
    max_examples: int | None = None,
    conditions: str | list[str] | None = None,
    steering_scale: float = 2.0,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    state = SteeringState.load(steering_state_path)
    examples = load_dataset(dataset_path, limit=max_examples)
    requested_conditions = parse_conditions(conditions)

    records: list[dict[str, Any]] = []
    for example in tqdm(examples, desc="steering"):
        for condition in requested_conditions:
            prompt, model_prompt, _messages = _prepare_prompts(tokenizer, example, condition)
            baseline = generate_answer(model, tokenizer, model_prompt, max_new_tokens=max_new_tokens)
            steered = generate_answer(
                model,
                tokenizer,
                model_prompt,
                max_new_tokens=max_new_tokens,
                steerer=ActivationSteerer(model, state, steering_scale=steering_scale),
            )

            baseline_correct = answer_is_correct(baseline["text"], example)
            steered_correct = answer_is_correct(steered["text"], example)
            records.append(
                {
                    "example_id": example.id,
                    "condition": condition,
                    "question": example.question,
                    "answer": example.answer,
                    "prompt": prompt,
                    "model_prompt": model_prompt,
                    "baseline_text": baseline["text"],
                    "baseline_correct": baseline_correct,
                    "steered_text": steered["text"],
                    "steered_correct": steered_correct,
                }
            )

    summary = {
        "num_records": len(records),
        "baseline_accuracy_by_condition": _accuracy_by_condition(records, key="baseline_correct"),
        "steered_accuracy_by_condition": _accuracy_by_condition(records, key="steered_correct"),
    }
    _write_jsonl(output_path / "steering_records.jsonl", records)
    _write_json(output_path / "steering_summary.json", summary)
    return summary
