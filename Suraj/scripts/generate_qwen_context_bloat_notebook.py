from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "qwen_context_bloat_experiments.ipynb"


def _lines(text: str) -> list[str]:
    return [line + "\n" for line in dedent(text).strip("\n").split("\n")]


def markdown_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(text),
    }


def code_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


cells = [
    markdown_cell(
        """
        # Qwen Context-Bloat Experiments

        This notebook implements the context-bloat study from scratch for Qwen checkpoints.

        It is structured to measure three concrete long-context failure modes:

        - lost-in-the-middle / position bias
        - length-only degradation as irrelevant context grows
        - attention sinks, where early useless tokens soak up attention

        The code is heavily commented so you can see what is happening, when it is happening, and why each metric is collected.
        """
    ),
    markdown_cell(
        """
        ## Before You Run

        The repo includes a reproducible conda environment and activation hooks for local caches.

        From the project root:

        ```bash
        bash scripts/setup_qwen_context_bloat_env.sh
        conda activate .conda/envs/qwen-context-bloat
        python scripts/generate_qwen_context_bloat_notebook.py
        jupyter lab notebooks/qwen_context_bloat_experiments.ipynb
        ```

        Notes:

        - The activation hook keeps Hugging Face downloads and experiment artifacts inside this repo.
        - The default notebook config is intentionally modest so the first run is debuggable.
        - Once the pipeline works, scale `num_examples`, `position_lengths`, and `bloat_lengths` upward.
        """
    ),
    code_cell(
        """
        from __future__ import annotations

        import json
        import math
        import os
        import random
        import time
        from collections import defaultdict
        from dataclasses import asdict, dataclass
        from pathlib import Path
        from typing import Any

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import torch
        import torch.nn.functional as F
        from tqdm import tqdm
        from transformers import AutoModelForCausalLM, AutoTokenizer


        def find_project_root(start: Path | None = None) -> Path:
            start = (start or Path.cwd()).resolve()
            for candidate in (start, *start.parents):
                if (candidate / "environment.yml").exists() and (candidate / "scripts").exists():
                    return candidate
            return start


        PROJECT_ROOT = find_project_root()
        ARTIFACT_ROOT = Path(os.getenv("QCB_ARTIFACT_DIR", PROJECT_ROOT / "artifacts" / "context_bloat"))
        TABLE_DIR = ARTIFACT_ROOT / "tables"
        FIGURE_DIR = ARTIFACT_ROOT / "figures"
        NOTE_DIR = ARTIFACT_ROOT / "notes"

        for directory in (ARTIFACT_ROOT, TABLE_DIR, FIGURE_DIR, NOTE_DIR):
            directory.mkdir(parents=True, exist_ok=True)

        pd.set_option("display.max_columns", 200)
        pd.set_option("display.max_colwidth", 120)
        sns.set_theme(style="whitegrid", context="notebook")

        print(f"Project root: {PROJECT_ROOT}")
        print(f"Artifact root: {ARTIFACT_ROOT}")
        print(f"Python: {os.sys.version.split()[0]}")
        print(f"PyTorch: {torch.__version__}")
        print(f"Transformers: {__import__('transformers').__version__}")
        """
    ),
    markdown_cell(
        """
        ## Configuration

        The notebook defaults to a smoke-test scale so you can verify the pipeline before running expensive sweeps.

        Recommended progression:

        1. Run the defaults as a correctness and memory check.
        2. Increase `num_examples`.
        3. Increase the prompt-length sweeps.
        4. Only then move to 16K+ prompts or larger Qwen checkpoints.
        """
    ),
    code_cell(
        """
        @dataclass
        class ExperimentConfig:
            model_name: str = os.getenv("QCB_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
            seed: int = 7
            num_examples: int = 8
            max_new_tokens: int = 12
            position_lengths: tuple[int, ...] = (2048, 4096)
            position_ratios: tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90)
            fixed_bloat_position: float = 0.50
            bloat_lengths: tuple[int, ...] = (1024, 2048, 4096, 8192)
            competition_length: int = 4096
            sink_total_length: int = 4096
            sink_prefix_lengths: tuple[int, ...] = (0, 64, 256, 512)
            sink_answer_position: float = 0.70
            filler_types: tuple[str, ...] = ("natural", "boilerplate", "code", "table")
            attention_probe_examples: int = 3
            attention_max_prompt_tokens: int = 4096
            sink_ratio_tokens: int = 32
            device: str | None = None


        CONFIG = ExperimentConfig()
        pd.Series(asdict(CONFIG), dtype="object")
        """
    ),
    code_cell(
        """
        def seed_everything(seed: int) -> None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)


        def resolve_device(requested: str | None = None) -> str:
            if requested:
                return requested
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"


        def pick_torch_dtype(device: str) -> torch.dtype:
            # We keep non-CUDA runs in float32 because interpretability probes are
            # easier to reason about when we are not also debugging dtype issues.
            if device == "cuda":
                return torch.float16
            return torch.float32


        def extract_answer_candidate(text: str) -> str:
            stripped = text.strip()
            if not stripped:
                return ""

            first_nonempty_line = ""
            for line in stripped.splitlines():
                candidate = line.strip()
                if candidate:
                    first_nonempty_line = candidate
                    break

            if not first_nonempty_line:
                return ""

            if ":" in first_nonempty_line:
                prefix, suffix = first_nonempty_line.split(":", 1)
                if prefix.strip().upper() in {"ANSWER", "SECURITY ANSWER"}:
                    first_nonempty_line = suffix.strip()

            return first_nonempty_line.strip().strip("`\"'.,;:!?()[]{}")


        def normalize_text(text: str) -> str:
            normalized = extract_answer_candidate(text)
            return " ".join(normalized.strip().split()).upper()


        seed_everything(CONFIG.seed)
        """
    ),
    code_cell(
        """
        FILLER_BANKS: dict[str, list[str]] = {
            "natural": [
                "The archive memo described a routine maintenance window and noted that all labels had been reviewed twice.\\n",
                "Several employees discussed cafeteria seating, shipment timing, and a delayed badge printer that was unrelated to security.\\n",
                "A facilities report mentioned dim hallway lights, a squeaky cart wheel, and a pending request for new storage bins.\\n",
                "The operations summary listed travel reimbursements, printer toner levels, and a rescheduled onboarding session.\\n",
                "A logistics note described cardboard box counts, loading dock congestion, and weather-related delivery timing.\\n",
            ],
            "boilerplate": [
                "Standard disclaimer: this paragraph is placeholder documentation and should not be treated as the answer.\\n",
                "Template footer: repeated administrative language appears here for formatting consistency only.\\n",
                "Policy reminder: archival headers, page breaks, and routing notes are not evidence-bearing fields.\\n",
                "Separator text: repeated boilerplate is included to inflate context length without adding useful facts.\\n",
            ],
            "code": [
                "def reconcile_records(batch):\\n    return [row for row in batch if row.get('status') != 'archived']\\n",
                "for idx, item in enumerate(queue):\\n    if item['flag']:\\n        results.append((idx, item['name']))\\n",
                "SELECT project_id, created_at, status FROM nightly_jobs WHERE status != 'failed' ORDER BY created_at DESC;\\n",
                "if cache_key in registry:\\n    metadata = registry[cache_key]\\nelse:\\n    metadata = {'status': 'missing'}\\n",
            ],
            "table": [
                "| Field | Value | Notes |\\n| Queue | East | Routine |\\n| Batch | 14 | Closed |\\n",
                "| Region | Desk | Count |\\n| North | A2 | 18 |\\n| South | C7 | 11 |\\n",
                "| Team | Shift | Floor |\\n| Orion | PM | 3 |\\n| Delta | AM | 1 |\\n",
                "| Category | State | Priority |\\n| Paperwork | open | low |\\n| Badges | closed | low |\\n",
            ],
        }

        ADJECTIVES = [
            "opal",
            "amber",
            "cobalt",
            "ivory",
            "scarlet",
            "sable",
            "silver",
            "golden",
            "teal",
            "crimson",
            "azure",
            "jade",
        ]
        NOUNS = [
            "river",
            "harbor",
            "anchor",
            "signal",
            "meadow",
            "comet",
            "falcon",
            "lantern",
            "vertex",
            "summit",
            "breeze",
            "cedar",
        ]
        PROJECT_WORDS = [
            "mercury",
            "delta",
            "atlas",
            "solstice",
            "ember",
            "matrix",
            "vector",
            "aurora",
            "pioneer",
            "quartz",
        ]


        def make_secret(index: int, salt: int) -> str:
            adjective = ADJECTIVES[(index + salt) % len(ADJECTIVES)].upper()
            noun = NOUNS[(index * 3 + salt) % len(NOUNS)].upper()
            number = 10000 + index + salt * 17
            return f"{adjective}-{noun}-{number}"


        def build_example_bank(num_examples: int, seed: int) -> list[dict[str, Any]]:
            examples: list[dict[str, Any]] = []
            for index in range(num_examples):
                rng = random.Random(seed + index * 997)
                project_left = PROJECT_WORDS[(index + 1) % len(PROJECT_WORDS)].upper()
                project_right = PROJECT_WORDS[(index * 2 + 3) % len(PROJECT_WORDS)].upper()
                answer = make_secret(index=index, salt=1)
                front_decoy = make_secret(index=index, salt=5)
                middle_decoy = make_secret(index=index, salt=9)
                back_decoy = make_secret(index=index, salt=13)
                examples.append(
                    {
                        "example_id": f"example_{index:04d}",
                        "employee_id": f"{42000 + index}",
                        "project_code": f"{project_left}-{project_right}-{rng.randint(100, 999)}",
                        "answer": answer,
                        "front_decoy": front_decoy,
                        "middle_decoy": middle_decoy,
                        "back_decoy": back_decoy,
                    }
                )
            return examples
        """
    ),
    code_cell(
        """
        def encode_text(tokenizer: Any, text: str) -> list[int]:
            return tokenizer(text, add_special_tokens=False)["input_ids"]


        def decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
            return tokenizer.decode(token_ids, skip_special_tokens=False)


        def build_filler_ids(tokenizer: Any, target_tokens: int, filler_type: str, seed: int) -> list[int]:
            if target_tokens <= 0:
                return []
            rng = random.Random(seed)
            bank = [encode_text(tokenizer, piece) for piece in FILLER_BANKS[filler_type]]
            token_ids: list[int] = []
            while len(token_ids) < target_tokens:
                token_ids.extend(list(rng.choice(bank)))
            return token_ids[:target_tokens]


        def build_sink_prefix_ids(tokenizer: Any, target_tokens: int) -> list[int]:
            if target_tokens <= 0:
                return []
            unit_ids = encode_text(tokenizer, "### HEADER ### HEADER ### HEADER ###\\n")
            token_ids: list[int] = []
            while len(token_ids) < target_tokens:
                token_ids.extend(unit_ids)
            return token_ids[:target_tokens]


        def build_single_needle_prompt(
            tokenizer: Any,
            example: dict[str, Any],
            total_tokens: int,
            answer_position: float,
            filler_type: str,
            seed: int,
            sink_prefix_tokens: int = 0,
        ) -> dict[str, Any]:
            doc_preamble_ids = encode_text(
                tokenizer,
                "Answer using only the document. Return only the security answer.\\n\\nDocument:\\n",
            )
            record_header_ids = encode_text(tokenizer, "Important record:\\n")
            employee_ids = encode_text(tokenizer, f"Employee ID: {example['employee_id']}\\n")
            project_ids = encode_text(tokenizer, f"Project codename: {example['project_code']}\\n")
            answer_prefix_ids = encode_text(tokenizer, "Security answer: ")
            answer_ids = encode_text(tokenizer, example["answer"])
            line_break_ids = encode_text(tokenizer, "\\n")
            suffix_ids = encode_text(tokenizer, "\\nQuestion: What is the security answer?\\nAnswer:")
            sink_ids = build_sink_prefix_ids(tokenizer, sink_prefix_tokens)

            fixed_tokens = (
                len(sink_ids)
                + len(doc_preamble_ids)
                + len(record_header_ids)
                + len(employee_ids)
                + len(project_ids)
                + len(answer_prefix_ids)
                + len(answer_ids)
                + len(line_break_ids)
                + len(suffix_ids)
            )
            filler_budget = total_tokens - fixed_tokens
            if filler_budget < 0:
                raise ValueError(
                    f"Prompt length {total_tokens} is too small for the fixed template ({fixed_tokens} tokens)."
                )

            desired_answer_start = int(round(total_tokens * answer_position))
            fixed_before_answer = (
                len(sink_ids)
                + len(doc_preamble_ids)
                + len(record_header_ids)
                + len(employee_ids)
                + len(project_ids)
                + len(answer_prefix_ids)
            )
            pre_target = max(0, min(filler_budget, desired_answer_start - fixed_before_answer))
            post_target = filler_budget - pre_target

            pre_ids = build_filler_ids(tokenizer, pre_target, filler_type, seed=seed + 1)
            post_ids = build_filler_ids(tokenizer, post_target, filler_type, seed=seed + 2)

            prompt_ids = (
                sink_ids
                + doc_preamble_ids
                + pre_ids
                + record_header_ids
                + employee_ids
                + project_ids
                + answer_prefix_ids
                + answer_ids
                + line_break_ids
                + post_ids
                + suffix_ids
            )
            answer_start = (
                len(sink_ids)
                + len(doc_preamble_ids)
                + len(pre_ids)
                + len(record_header_ids)
                + len(employee_ids)
                + len(project_ids)
                + len(answer_prefix_ids)
            )
            answer_end = answer_start + len(answer_ids)

            return {
                "prompt_ids": prompt_ids,
                "prompt_text": decode_ids(tokenizer, prompt_ids),
                "answer_text": example["answer"],
                "answer_ids": answer_ids,
                "relevant_span": (answer_start, answer_end),
                "sink_span": (0, min(CONFIG.sink_ratio_tokens, len(prompt_ids))),
                "observed_answer_position": answer_start / max(1, len(prompt_ids)),
            }


        def build_competition_prompt(
            tokenizer: Any,
            example: dict[str, Any],
            total_tokens: int,
            correct_location: str,
            filler_type: str,
            seed: int,
        ) -> dict[str, Any]:
            prompt_intro_ids = encode_text(
                tokenizer,
                "Three records appear below. Exactly one record is verified.\\n"
                "Use the verified record only and return only its security answer.\\n\\n"
                "Document:\\n",
            )
            suffix_ids = encode_text(
                tokenizer,
                "\\nQuestion: Which security answer belongs to the verified record?\\nAnswer:",
            )

            answer_by_location = {
                "front": example["front_decoy"],
                "middle": example["middle_decoy"],
                "back": example["back_decoy"],
            }
            answer_by_location[correct_location] = example["answer"]

            block_specs = [
                ("Front record", "front"),
                ("Middle record", "middle"),
                ("Back record", "back"),
            ]

            encoded_blocks: list[tuple[str, list[int], list[int], list[int]]] = []
            fixed_tokens = len(prompt_intro_ids) + len(suffix_ids)
            for label, location in block_specs:
                verified = "verified" if location == correct_location else "decoy"
                block_prefix_ids = encode_text(
                    tokenizer,
                    f"{label}:\\n"
                    f"Employee ID: {example['employee_id']}\\n"
                    f"Project codename: {example['project_code']}\\n"
                    f"Verification status: {verified}\\n"
                    "Security answer: ",
                )
                answer_text = answer_by_location[location]
                answer_ids = encode_text(tokenizer, answer_text)
                block_suffix_ids = encode_text(tokenizer, "\\n")
                encoded_blocks.append((location, block_prefix_ids, answer_ids, block_suffix_ids))
                fixed_tokens += len(block_prefix_ids) + len(answer_ids) + len(block_suffix_ids)

            filler_budget = total_tokens - fixed_tokens
            if filler_budget < 0:
                raise ValueError(
                    f"Competition prompt length {total_tokens} is too small for the fixed template ({fixed_tokens} tokens)."
                )

            filler_targets = [
                int(round(filler_budget * 0.08)),
                int(round(filler_budget * 0.34)),
                int(round(filler_budget * 0.34)),
            ]
            filler_targets.append(filler_budget - sum(filler_targets))
            filler_segments = [
                build_filler_ids(tokenizer, target, filler_type, seed + 10 + index)
                for index, target in enumerate(filler_targets)
            ]

            prompt_ids = list(prompt_intro_ids)
            candidate_spans: dict[str, tuple[int, int]] = {}
            prompt_ids.extend(filler_segments[0])
            for block_index, (location, block_prefix_ids, answer_ids, block_suffix_ids) in enumerate(encoded_blocks):
                answer_start = len(prompt_ids) + len(block_prefix_ids)
                answer_end = answer_start + len(answer_ids)
                candidate_spans[location] = (answer_start, answer_end)
                prompt_ids.extend(block_prefix_ids)
                prompt_ids.extend(answer_ids)
                prompt_ids.extend(block_suffix_ids)
                prompt_ids.extend(filler_segments[block_index + 1])
            prompt_ids.extend(suffix_ids)

            correct_answer_ids = encode_text(tokenizer, example["answer"])
            relevant_span = candidate_spans[correct_location]
            return {
                "prompt_ids": prompt_ids,
                "prompt_text": decode_ids(tokenizer, prompt_ids),
                "answer_text": example["answer"],
                "answer_ids": correct_answer_ids,
                "relevant_span": relevant_span,
                "sink_span": (0, min(CONFIG.sink_ratio_tokens, len(prompt_ids))),
                "observed_answer_position": relevant_span[0] / max(1, len(prompt_ids)),
                "candidate_answers": answer_by_location,
                "candidate_spans": candidate_spans,
            }
        """
    ),
    markdown_cell(
        """
        ## Model Loading

        We force eager attention when possible because interpretability diagnostics need attention tensors.

        Important practical note:

        - collecting `output_attentions=True` is quadratic in sequence length
        - that is why the notebook probes attention on only a small subset of cases
        - exact-match, log-prob, and drift metrics can still be run much more broadly
        """
    ),
    code_cell(
        """
        def find_decoder_layers(model: torch.nn.Module) -> Any:
            candidates = [
                getattr(getattr(model, "model", None), "layers", None),
                getattr(getattr(model, "transformer", None), "h", None),
                getattr(getattr(model, "gpt_neox", None), "layers", None),
                getattr(getattr(model, "backbone", None), "layers", None),
            ]
            for layers in candidates:
                if layers is not None:
                    return layers
            raise ValueError("Could not locate the decoder layers for this architecture.")


        def load_model_and_tokenizer(model_name: str, device: str | None = None):
            resolved_device = resolve_device(device)
            torch_dtype = pick_torch_dtype(resolved_device)

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_kwargs = {
                "torch_dtype": torch_dtype,
            }
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    attn_implementation="eager",
                    **model_kwargs,
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            set_attn_implementation = getattr(model, "set_attn_implementation", None)
            if callable(set_attn_implementation):
                try:
                    set_attn_implementation("eager")
                except (TypeError, ValueError):
                    pass

            model.to(resolved_device)
            model.eval()

            if getattr(model.generation_config, "pad_token_id", None) is None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id

            return model, tokenizer, resolved_device


        model, tokenizer, DEVICE = load_model_and_tokenizer(CONFIG.model_name, device=CONFIG.device)
        total_params = sum(parameter.numel() for parameter in model.parameters())
        print(f"Loaded {CONFIG.model_name} on {DEVICE}")
        print(f"Parameters: {total_params / 1e6:.1f}M")
        print(f"Decoder layers: {len(find_decoder_layers(model))}")
        """
    ),
    code_cell(
        """
        def tensors_from_ids(token_ids: list[int], device: str) -> tuple[torch.Tensor, torch.Tensor]:
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            return input_ids, attention_mask


        def capture_prompt_states(
            model: torch.nn.Module,
            token_ids: list[int],
            device: str,
            collect_attentions: bool = False,
        ) -> tuple[Any, torch.Tensor]:
            input_ids, attention_mask = tensors_from_ids(token_ids, device=device)
            layers = find_decoder_layers(model)
            captured_states: list[torch.Tensor] = []
            hooks = []

            def _capture_last_token(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                hidden_states = output[0] if isinstance(output, tuple) else output
                captured_states.append(hidden_states[:, -1, :].detach().float().cpu())

            for layer in layers:
                hooks.append(layer.register_forward_hook(_capture_last_token))

            try:
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=collect_attentions,
                        use_cache=False,
                    )
            finally:
                for hook in hooks:
                    hook.remove()

            last_token_states = torch.cat(captured_states, dim=0)
            return outputs, last_token_states


        def compute_answer_logprob(
            model: torch.nn.Module,
            prompt_ids: list[int],
            answer_ids: list[int],
            device: str,
        ) -> dict[str, float]:
            full_ids = prompt_ids + answer_ids
            input_ids, attention_mask = tensors_from_ids(full_ids, device=device)
            prompt_len = len(prompt_ids)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            logits = outputs.logits[0]
            answer_logits = logits[prompt_len - 1 : prompt_len - 1 + len(answer_ids), :]
            targets = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)
            logprobs = F.log_softmax(answer_logits, dim=-1)
            token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            first_step_logits = answer_logits[0]
            first_step_probs = F.softmax(first_step_logits, dim=-1)
            first_step_entropy = float(
                (-(first_step_probs * F.log_softmax(first_step_logits, dim=-1)).sum()).detach().cpu()
            )

            return {
                "sequence_logprob": float(token_logprobs.sum().detach().cpu()),
                "mean_token_logprob": float(token_logprobs.mean().detach().cpu()),
                "first_token_entropy": first_step_entropy,
            }


        def greedy_generate_ids(
            model: torch.nn.Module,
            tokenizer: Any,
            prompt_ids: list[int],
            max_new_tokens: int,
            device: str,
        ) -> list[int]:
            input_ids, attention_mask = tensors_from_ids(prompt_ids, device=device)
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            return generated[0, len(prompt_ids) :].detach().cpu().tolist()


        def summarize_attention(
            attentions: Any,
            relevant_span: tuple[int, int],
            sink_span: tuple[int, int],
        ) -> pd.DataFrame:
            if attentions is None:
                return pd.DataFrame()

            rows: list[dict[str, float]] = []
            for layer_index, layer_attn in enumerate(attentions):
                layer_tensor = layer_attn[0].detach().float().cpu()
                query_index = layer_tensor.shape[-2] - 1
                query_slice = layer_tensor[:, query_index, :]

                relevant_mass = query_slice[:, relevant_span[0] : relevant_span[1]].sum(dim=-1).numpy()
                sink_mass = query_slice[:, sink_span[0] : sink_span[1]].sum(dim=-1).numpy()
                total_mass = query_slice.sum(dim=-1).numpy()

                for head_index in range(query_slice.shape[0]):
                    rel = float(relevant_mass[head_index])
                    sink = float(sink_mass[head_index])
                    total = float(total_mass[head_index])
                    rows.append(
                        {
                            "layer": layer_index,
                            "head": head_index,
                            "relevant_mass": rel,
                            "sink_mass": sink,
                            "relevant_ratio": rel / max(total, 1e-8),
                            "sink_over_relevant": sink / max(rel, 1e-8),
                        }
                    )
            return pd.DataFrame(rows)


        def evaluate_prompt_case(
            model: torch.nn.Module,
            tokenizer: Any,
            prompt_package: dict[str, Any],
            device: str,
            experiment: str,
            baseline_states: torch.Tensor | None = None,
            collect_attentions: bool = False,
            metadata: dict[str, Any] | None = None,
        ) -> tuple[dict[str, Any], torch.Tensor, pd.DataFrame, pd.DataFrame]:
            metadata = dict(metadata or {})
            prompt_ids = prompt_package["prompt_ids"]
            answer_ids = prompt_package["answer_ids"]

            outputs, state_stack = capture_prompt_states(
                model=model,
                token_ids=prompt_ids,
                device=device,
                collect_attentions=collect_attentions,
            )
            logprob_stats = compute_answer_logprob(model, prompt_ids, answer_ids, device=device)
            generated_ids = greedy_generate_ids(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                max_new_tokens=CONFIG.max_new_tokens,
                device=device,
            )

            attention_details = summarize_attention(
                outputs.attentions if collect_attentions else None,
                relevant_span=prompt_package["relevant_span"],
                sink_span=prompt_package["sink_span"],
            )

            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generated_answer_text = extract_answer_candidate(generated_text)
            token_exact_match = generated_ids[: len(answer_ids)] == answer_ids
            normalized_text_exact_match = (
                normalize_text(generated_answer_text) == normalize_text(prompt_package["answer_text"])
            )

            row = {
                "experiment": experiment,
                "prompt_tokens": len(prompt_ids),
                "observed_answer_position": prompt_package["observed_answer_position"],
                "answer_text": prompt_package["answer_text"],
                "generated_text": generated_text,
                "generated_answer_text": generated_answer_text,
                "token_exact_match": token_exact_match,
                "normalized_text_exact_match": normalized_text_exact_match,
                # Keep an `exact_match` alias so downstream cells stay simple.
                "exact_match": normalized_text_exact_match,
                "sequence_logprob": logprob_stats["sequence_logprob"],
                "mean_token_logprob": logprob_stats["mean_token_logprob"],
                "first_token_entropy": logprob_stats["first_token_entropy"],
                "mean_relevant_attention": float(attention_details["relevant_ratio"].mean())
                if not attention_details.empty
                else np.nan,
                "mean_sink_over_relevant": float(attention_details["sink_over_relevant"].mean())
                if not attention_details.empty
                else np.nan,
                **metadata,
            }

            drift_details = pd.DataFrame()
            if baseline_states is not None:
                cosine_values = F.cosine_similarity(
                    state_stack.float(),
                    baseline_states.float(),
                    dim=-1,
                ).detach().cpu().numpy()
                drift_details = pd.DataFrame(
                    {
                        "layer": np.arange(len(cosine_values), dtype=int),
                        "cosine_similarity": cosine_values,
                        **metadata,
                    }
                )
                row["mean_layer_cosine"] = float(np.mean(cosine_values))
                row["late_layer_cosine"] = float(np.mean(cosine_values[max(0, len(cosine_values) * 3 // 4) :]))
            else:
                row["mean_layer_cosine"] = np.nan
                row["late_layer_cosine"] = np.nan

            return row, state_stack, attention_details, drift_details


        def score_candidate_answers(
            model: torch.nn.Module,
            tokenizer: Any,
            prompt_ids: list[int],
            candidate_answers: dict[str, str],
            device: str,
        ) -> dict[str, dict[str, float]]:
            scores: dict[str, dict[str, float]] = {}
            for label, answer_text in candidate_answers.items():
                answer_ids = encode_text(tokenizer, answer_text)
                scores[label] = compute_answer_logprob(
                    model=model,
                    prompt_ids=prompt_ids,
                    answer_ids=answer_ids,
                    device=device,
                )
            return scores
        """
    ),
    markdown_cell(
        """
        ## Synthetic Examples

        Each example contains:

        - a unique employee identifier
        - a unique project code
        - a unique correct answer string
        - decoy answers for the competition experiment

        Keeping these synthetic and programmatic gives us precise control over prompt length and answer placement.
        """
    ),
    code_cell(
        """
        examples = build_example_bank(CONFIG.num_examples, CONFIG.seed)
        pd.DataFrame(examples).head()
        """
    ),
    code_cell(
        """
        # These placeholders let you run experiments one section at a time without
        # editing later cells that expect the result DataFrames to exist.
        position_results = pd.DataFrame()
        position_attention = pd.DataFrame()
        position_drift = pd.DataFrame()

        bloat_results = pd.DataFrame()
        bloat_attention = pd.DataFrame()
        bloat_drift = pd.DataFrame()

        competition_results = pd.DataFrame()
        competition_attention = pd.DataFrame()

        sink_results = pd.DataFrame()
        sink_attention = pd.DataFrame()
        sink_drift = pd.DataFrame()
        """
    ),
    code_cell(
        """
        def run_position_sweep(
            model: torch.nn.Module,
            tokenizer: Any,
            examples: list[dict[str, Any]],
            config: ExperimentConfig,
            device: str,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            results: list[dict[str, Any]] = []
            attention_frames: list[pd.DataFrame] = []
            drift_frames: list[pd.DataFrame] = []
            baseline_states: dict[tuple[str, float, str], torch.Tensor] = {}

            sorted_lengths = tuple(sorted(config.position_lengths))
            total_cases = (
                len(config.filler_types)
                * len(config.position_ratios)
                * len(sorted_lengths)
                * len(examples)
            )
            progress = tqdm(total=total_cases, desc="position sweep")

            for filler_type in config.filler_types:
                for position_ratio in config.position_ratios:
                    for total_tokens in sorted_lengths:
                        for example_index, example in enumerate(examples):
                            prompt_package = build_single_needle_prompt(
                                tokenizer=tokenizer,
                                example=example,
                                total_tokens=total_tokens,
                                answer_position=position_ratio,
                                filler_type=filler_type,
                                seed=config.seed + example_index * 10_000 + total_tokens,
                            )
                            baseline_key = (filler_type, position_ratio, example["example_id"])
                            is_probe = (
                                example_index < config.attention_probe_examples
                                and total_tokens <= config.attention_max_prompt_tokens
                            )
                            row, states, attention_detail, drift_detail = evaluate_prompt_case(
                                model=model,
                                tokenizer=tokenizer,
                                prompt_package=prompt_package,
                                device=device,
                                experiment="position_sweep",
                                baseline_states=baseline_states.get(baseline_key),
                                collect_attentions=is_probe,
                                metadata={
                                    "example_id": example["example_id"],
                                    "filler_type": filler_type,
                                    "position_ratio": position_ratio,
                                    "total_tokens": total_tokens,
                                },
                            )
                            if total_tokens == sorted_lengths[0]:
                                baseline_states[baseline_key] = states
                            if not attention_detail.empty:
                                attention_frames.append(attention_detail.assign(
                                    example_id=example["example_id"],
                                    filler_type=filler_type,
                                    position_ratio=position_ratio,
                                    total_tokens=total_tokens,
                                    experiment="position_sweep",
                                ))
                            if not drift_detail.empty:
                                drift_frames.append(drift_detail.assign(
                                    experiment="position_sweep",
                                    total_tokens=total_tokens,
                                    position_ratio=position_ratio,
                                    filler_type=filler_type,
                                    example_id=example["example_id"],
                                ))
                            results.append(row)
                            progress.update(1)

            progress.close()
            return (
                pd.DataFrame(results),
                pd.concat(attention_frames, ignore_index=True) if attention_frames else pd.DataFrame(),
                pd.concat(drift_frames, ignore_index=True) if drift_frames else pd.DataFrame(),
            )


        def run_bloat_sweep(
            model: torch.nn.Module,
            tokenizer: Any,
            examples: list[dict[str, Any]],
            config: ExperimentConfig,
            device: str,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            results: list[dict[str, Any]] = []
            attention_frames: list[pd.DataFrame] = []
            drift_frames: list[pd.DataFrame] = []
            baseline_states: dict[tuple[str, str], torch.Tensor] = {}

            sorted_lengths = tuple(sorted(config.bloat_lengths))
            total_cases = len(config.filler_types) * len(sorted_lengths) * len(examples)
            progress = tqdm(total=total_cases, desc="bloat sweep")

            for filler_type in config.filler_types:
                for total_tokens in sorted_lengths:
                    for example_index, example in enumerate(examples):
                        prompt_package = build_single_needle_prompt(
                            tokenizer=tokenizer,
                            example=example,
                            total_tokens=total_tokens,
                            answer_position=config.fixed_bloat_position,
                            filler_type=filler_type,
                            seed=config.seed + example_index * 20_000 + total_tokens,
                        )
                        baseline_key = (filler_type, example["example_id"])
                        is_probe = (
                            example_index < config.attention_probe_examples
                            and total_tokens <= config.attention_max_prompt_tokens
                        )
                        row, states, attention_detail, drift_detail = evaluate_prompt_case(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_package=prompt_package,
                            device=device,
                            experiment="bloat_sweep",
                            baseline_states=baseline_states.get(baseline_key),
                            collect_attentions=is_probe,
                            metadata={
                                "example_id": example["example_id"],
                                "filler_type": filler_type,
                                "position_ratio": config.fixed_bloat_position,
                                "total_tokens": total_tokens,
                            },
                        )
                        if total_tokens == sorted_lengths[0]:
                            baseline_states[baseline_key] = states
                        if not attention_detail.empty:
                            attention_frames.append(attention_detail.assign(
                                example_id=example["example_id"],
                                filler_type=filler_type,
                                total_tokens=total_tokens,
                                experiment="bloat_sweep",
                            ))
                        if not drift_detail.empty:
                            drift_frames.append(drift_detail.assign(
                                experiment="bloat_sweep",
                                total_tokens=total_tokens,
                                filler_type=filler_type,
                                example_id=example["example_id"],
                            ))
                        results.append(row)
                        progress.update(1)

            progress.close()
            return (
                pd.DataFrame(results),
                pd.concat(attention_frames, ignore_index=True) if attention_frames else pd.DataFrame(),
                pd.concat(drift_frames, ignore_index=True) if drift_frames else pd.DataFrame(),
            )


        def run_competition_sweep(
            model: torch.nn.Module,
            tokenizer: Any,
            examples: list[dict[str, Any]],
            config: ExperimentConfig,
            device: str,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            results: list[dict[str, Any]] = []
            attention_frames: list[pd.DataFrame] = []
            locations = ("front", "middle", "back")
            total_cases = len(config.filler_types) * len(locations) * len(examples)
            progress = tqdm(total=total_cases, desc="competition sweep")

            for filler_type in config.filler_types:
                for correct_location in locations:
                    for example_index, example in enumerate(examples):
                        prompt_package = build_competition_prompt(
                            tokenizer=tokenizer,
                            example=example,
                            total_tokens=config.competition_length,
                            correct_location=correct_location,
                            filler_type=filler_type,
                            seed=config.seed + example_index * 30_000,
                        )
                        is_probe = (
                            example_index < config.attention_probe_examples
                            and config.competition_length <= config.attention_max_prompt_tokens
                        )
                        row, _states, attention_detail, _drift_detail = evaluate_prompt_case(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_package=prompt_package,
                            device=device,
                            experiment="competition_sweep",
                            baseline_states=None,
                            collect_attentions=is_probe,
                            metadata={
                                "example_id": example["example_id"],
                                "filler_type": filler_type,
                                "correct_location": correct_location,
                                "total_tokens": config.competition_length,
                            },
                        )
                        scores = score_candidate_answers(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_ids=prompt_package["prompt_ids"],
                            candidate_answers=prompt_package["candidate_answers"],
                            device=device,
                        )
                        predicted_location = max(
                            scores,
                            key=lambda label: scores[label]["mean_token_logprob"],
                        )
                        sorted_locations = sorted(
                            scores,
                            key=lambda label: scores[label]["mean_token_logprob"],
                            reverse=True,
                        )
                        row.update(
                            {
                                "predicted_location": predicted_location,
                                "candidate_margin": (
                                    scores[sorted_locations[0]]["mean_token_logprob"]
                                    - scores[sorted_locations[1]]["mean_token_logprob"]
                                ),
                                "front_mean_token_logprob": scores["front"]["mean_token_logprob"],
                                "middle_mean_token_logprob": scores["middle"]["mean_token_logprob"],
                                "back_mean_token_logprob": scores["back"]["mean_token_logprob"],
                                "front_sequence_logprob": scores["front"]["sequence_logprob"],
                                "middle_sequence_logprob": scores["middle"]["sequence_logprob"],
                                "back_sequence_logprob": scores["back"]["sequence_logprob"],
                                "candidate_exact_match": predicted_location == correct_location,
                            }
                        )
                        if not attention_detail.empty:
                            attention_frames.append(attention_detail.assign(
                                example_id=example["example_id"],
                                filler_type=filler_type,
                                correct_location=correct_location,
                                total_tokens=config.competition_length,
                                experiment="competition_sweep",
                            ))
                        results.append(row)
                        progress.update(1)

            progress.close()
            return (
                pd.DataFrame(results),
                pd.concat(attention_frames, ignore_index=True) if attention_frames else pd.DataFrame(),
            )


        def run_sink_stress(
            model: torch.nn.Module,
            tokenizer: Any,
            examples: list[dict[str, Any]],
            config: ExperimentConfig,
            device: str,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            results: list[dict[str, Any]] = []
            attention_frames: list[pd.DataFrame] = []
            drift_frames: list[pd.DataFrame] = []
            baseline_states: dict[tuple[str, str], torch.Tensor] = {}

            total_cases = len(config.filler_types) * len(config.sink_prefix_lengths) * len(examples)
            progress = tqdm(total=total_cases, desc="sink stress")

            for filler_type in config.filler_types:
                for sink_prefix_tokens in config.sink_prefix_lengths:
                    for example_index, example in enumerate(examples):
                        prompt_package = build_single_needle_prompt(
                            tokenizer=tokenizer,
                            example=example,
                            total_tokens=config.sink_total_length,
                            answer_position=config.sink_answer_position,
                            filler_type=filler_type,
                            seed=config.seed + example_index * 40_000 + sink_prefix_tokens,
                            sink_prefix_tokens=sink_prefix_tokens,
                        )
                        baseline_key = (filler_type, example["example_id"])
                        is_probe = (
                            example_index < config.attention_probe_examples
                            and config.sink_total_length <= config.attention_max_prompt_tokens
                        )
                        row, states, attention_detail, drift_detail = evaluate_prompt_case(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_package=prompt_package,
                            device=device,
                            experiment="sink_stress",
                            baseline_states=baseline_states.get(baseline_key),
                            collect_attentions=is_probe,
                            metadata={
                                "example_id": example["example_id"],
                                "filler_type": filler_type,
                                "sink_prefix_tokens": sink_prefix_tokens,
                                "total_tokens": config.sink_total_length,
                            },
                        )
                        if sink_prefix_tokens == min(config.sink_prefix_lengths):
                            baseline_states[baseline_key] = states
                        if not attention_detail.empty:
                            attention_frames.append(attention_detail.assign(
                                example_id=example["example_id"],
                                filler_type=filler_type,
                                sink_prefix_tokens=sink_prefix_tokens,
                                total_tokens=config.sink_total_length,
                                experiment="sink_stress",
                            ))
                        if not drift_detail.empty:
                            drift_frames.append(drift_detail.assign(
                                experiment="sink_stress",
                                sink_prefix_tokens=sink_prefix_tokens,
                                total_tokens=config.sink_total_length,
                                filler_type=filler_type,
                                example_id=example["example_id"],
                            ))
                        results.append(row)
                        progress.update(1)

            progress.close()
            return (
                pd.DataFrame(results),
                pd.concat(attention_frames, ignore_index=True) if attention_frames else pd.DataFrame(),
                pd.concat(drift_frames, ignore_index=True) if drift_frames else pd.DataFrame(),
            )
        """
    ),
    markdown_cell(
        """
        ## Experiment 1: Position Sweep

        The task stays the same and the answer stays the same.
        Only the answer-bearing span moves through the prompt.

        If accuracy drops in the middle while the edges remain stronger, that is classic lost-in-the-middle behavior.
        """
    ),
    code_cell(
        """
        position_results, position_attention, position_drift = run_position_sweep(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            config=CONFIG,
            device=DEVICE,
        )
        position_results.head()
        """
    ),
    markdown_cell(
        """
        ## Experiment 2: Bloat-Only Sweep

        The answer-bearing span stays at the same relative position.
        Only the amount of irrelevant context changes.

        This isolates length sensitivity from position sensitivity.
        """
    ),
    code_cell(
        """
        bloat_results, bloat_attention, bloat_drift = run_bloat_sweep(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            config=CONFIG,
            device=DEVICE,
        )
        bloat_results.head()
        """
    ),
    markdown_cell(
        """
        ## Experiment 3: Primacy vs Recency Competition

        Three candidate records appear in one prompt:

        - one near the front
        - one near the middle
        - one near the back

        Only one record is marked verified.
        We rotate which location is correct and compare candidate-answer log-probs.
        """
    ),
    code_cell(
        """
        competition_results, competition_attention = run_competition_sweep(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            config=CONFIG,
            device=DEVICE,
        )
        competition_results.head()
        """
    ),
    markdown_cell(
        """
        ## Experiment 4: Sink-Token Stress Test

        We keep the total prompt length fixed and only increase the meaningless repeated prefix at the front.

        If sink attention rises and answer retrieval falls, that is direct evidence of sink-like behavior rather than generic task difficulty.
        """
    ),
    code_cell(
        """
        sink_results, sink_attention, sink_drift = run_sink_stress(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            config=CONFIG,
            device=DEVICE,
        )
        sink_results.head()
        """
    ),
    markdown_cell(
        """
        ## Persist Results

        Saving intermediate tables is helpful because long-context experiments are slow and you usually do not want to rerun everything after a plotting tweak.
        """
    ),
    code_cell(
        """
        def save_dataframe(frame: pd.DataFrame, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            print(f"Saved {path}")


        if not position_results.empty:
            save_dataframe(position_results, TABLE_DIR / "position_results.csv")
        if not position_attention.empty:
            save_dataframe(position_attention, TABLE_DIR / "position_attention.csv")
        if not position_drift.empty:
            save_dataframe(position_drift, TABLE_DIR / "position_drift.csv")

        if not bloat_results.empty:
            save_dataframe(bloat_results, TABLE_DIR / "bloat_results.csv")
        if not bloat_attention.empty:
            save_dataframe(bloat_attention, TABLE_DIR / "bloat_attention.csv")
        if not bloat_drift.empty:
            save_dataframe(bloat_drift, TABLE_DIR / "bloat_drift.csv")

        if not competition_results.empty:
            save_dataframe(competition_results, TABLE_DIR / "competition_results.csv")
        if not competition_attention.empty:
            save_dataframe(competition_attention, TABLE_DIR / "competition_attention.csv")

        if not sink_results.empty:
            save_dataframe(sink_results, TABLE_DIR / "sink_results.csv")
        if not sink_attention.empty:
            save_dataframe(sink_attention, TABLE_DIR / "sink_attention.csv")
        if not sink_drift.empty:
            save_dataframe(sink_drift, TABLE_DIR / "sink_drift.csv")
        """
    ),
    markdown_cell(
        """
        ## Plotting Helpers

        These plots are designed to surface the failure signatures we care about:

        - accuracy vs answer position
        - accuracy vs context length
        - sink ratio vs sink-prefix size
        - layerwise cosine drift vs baseline
        """
    ),
    code_cell(
        """
        def finalize_figure(fig: plt.Figure, output_name: str) -> Path:
            fig.tight_layout()
            output_path = FIGURE_DIR / output_name
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"Saved {output_path}")
            return output_path


        if not position_results.empty:
            summary = (
                position_results.groupby(["total_tokens", "position_ratio"], as_index=False)
                .agg(accuracy=("normalized_text_exact_match", "mean"))
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.lineplot(data=summary, x="position_ratio", y="accuracy", hue="total_tokens", marker="o", ax=ax)
            ax.set_title("Position Sweep Accuracy")
            ax.set_ylabel("Normalized exact-match accuracy")
            ax.set_xlabel("Target answer position")
            finalize_figure(fig, "position_accuracy.png")
            plt.show()


        if not bloat_results.empty:
            summary = (
                bloat_results.groupby(["total_tokens", "filler_type"], as_index=False)
                .agg(accuracy=("normalized_text_exact_match", "mean"))
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.lineplot(data=summary, x="total_tokens", y="accuracy", hue="filler_type", marker="o", ax=ax)
            ax.set_title("Bloat Sweep Accuracy")
            ax.set_ylabel("Normalized exact-match accuracy")
            ax.set_xlabel("Total prompt tokens")
            finalize_figure(fig, "bloat_accuracy.png")
            plt.show()


        if not sink_results.empty:
            summary = (
                sink_results.groupby(["sink_prefix_tokens", "filler_type"], as_index=False)
                .agg(accuracy=("normalized_text_exact_match", "mean"))
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.lineplot(data=summary, x="sink_prefix_tokens", y="accuracy", hue="filler_type", marker="o", ax=ax)
            ax.set_title("Sink Stress Accuracy")
            ax.set_ylabel("Normalized exact-match accuracy")
            ax.set_xlabel("Sink prefix tokens")
            finalize_figure(fig, "sink_accuracy.png")
            plt.show()


        if not position_drift.empty:
            summary = (
                position_drift.groupby(["layer", "total_tokens"], as_index=False)
                .agg(cosine_similarity=("cosine_similarity", "mean"))
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=summary, x="layer", y="cosine_similarity", hue="total_tokens", ax=ax)
            ax.set_title("Representation Drift by Layer (Position Sweep)")
            ax.set_ylabel("Cosine similarity to shortest-context baseline")
            ax.set_xlabel("Layer")
            finalize_figure(fig, "position_drift_layers.png")
            plt.show()


        if not sink_attention.empty:
            summary = (
                sink_attention.groupby(["sink_prefix_tokens"], as_index=False)
                .agg(
                    relevant_ratio=("relevant_ratio", "mean"),
                    sink_over_relevant=("sink_over_relevant", "mean"),
                )
            )
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            sns.lineplot(data=summary, x="sink_prefix_tokens", y="relevant_ratio", marker="o", ax=axes[0])
            axes[0].set_title("Relevant Attention vs Sink Size")
            axes[0].set_ylabel("Mean relevant attention ratio")
            axes[0].set_xlabel("Sink prefix tokens")
            sns.lineplot(data=summary, x="sink_prefix_tokens", y="sink_over_relevant", marker="o", ax=axes[1])
            axes[1].set_title("Sink Ratio vs Sink Size")
            axes[1].set_ylabel("Mean sink / relevant attention")
            axes[1].set_xlabel("Sink prefix tokens")
            finalize_figure(fig, "sink_attention_summary.png")
            plt.show()
        """
    ),
    markdown_cell(
        """
        ## Interpreting the Outputs

        Strong evidence of context bloat usually looks like this:

        - exact match is healthy at short lengths and weakens as irrelevant context grows
        - the middle positions perform worse than the edges
        - relevant-span attention falls while sink attention rises
        - the final query-token representation drifts away from the short-context baseline

        If you want to push this further, the next clean extensions are:

        - larger Qwen checkpoints
        - longer prompt sweeps such as 16K and 32K
        - candidate-head masking or layer ablations
        - richer filler distributions such as real code files or semi-structured logs
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Qwen Context Bloat (Python 3.11)",
            "language": "python",
            "name": "qwen-context-bloat",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
