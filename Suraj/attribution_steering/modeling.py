from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def resolve_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    model_name: str,
    device: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    resolved_device = resolve_device(device)
    torch_dtype = torch.float16 if resolved_device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
    model.to(resolved_device)
    model.eval()
    return model, tokenizer


def find_decoder_layers(model: PreTrainedModel) -> Sequence[torch.nn.Module]:
    candidates = [
        ("model.layers", getattr(getattr(model, "model", None), "layers", None)),
        ("transformer.h", getattr(getattr(model, "transformer", None), "h", None)),
        ("gpt_neox.layers", getattr(getattr(model, "gpt_neox", None), "layers", None)),
        ("backbone.layers", getattr(getattr(model, "backbone", None), "layers", None)),
    ]
    for _, layers in candidates:
        if layers is not None:
            return layers
    raise ValueError(
        "Could not find decoder layers for this model architecture. "
        "Add the layer collection path in find_decoder_layers()."
    )


def decode_new_tokens(
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,
    full_ids: torch.Tensor,
) -> str:
    prompt_length = prompt_ids.shape[-1]
    new_ids = full_ids[0, prompt_length:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def render_prompt_for_model(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    messages: list[Mapping[str, str]] | None = None,
) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if messages and chat_template and callable(apply_chat_template):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt
