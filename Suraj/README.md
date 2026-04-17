# Attribution Steering Experiments

This repo now includes a lightweight experiment harness for studying hallucinations with attribution-style graphs and activation steering.

The workflow is:

1. Run a causal language model on factual QA prompts under `clean` and `misleading` retrieved-note conditions.
2. Label each answer as correct or hallucinated against a known gold answer.
3. Cache last-token residual activations plus an attention-informed attribution graph for the first generated token.
4. Fit per-layer steering vectors from the contrast between truthful and hallucinated traces.
5. Re-run the same prompts with steering enabled only when the live activation looks more hallucination-like than truthful.

## What "attribution graph" means here

This implementation is intentionally lightweight and reproducible:

- Node importance uses `||grad * activation||` on each layer/token hidden state for the first answer token.
- Edge importance uses average attention weight within a layer, scaled by the node importance of the connected tokens.

That is useful for contrastive experiments, but it is **not** a full transformer circuit reconstruction in the Anthropic sense. It is best thought of as an attention-informed attribution proxy.

## Repo Layout

- `attribution_steering/`: core package
- `datasets/factual_conflict.jsonl`: small factual conflict benchmark
- `tests/`: basic unit tests
- `trial.ipynb`: original notebook draft

## Quick Start

From the `Suraj/` directory, install dependencies in your environment, then run:

```bash
pip install -r requirements.txt
python3 -m attribution_steering.cli collect \
  --model gpt2 \
  --dataset datasets/factual_conflict.jsonl \
  --output-dir runs/gpt2_collect \
  --max-examples 12
```

For Qwen Instruct, the closest official checkpoint name is `Qwen/Qwen2.5-3B-Instruct` rather than `Qwen 2.5B`. The code now automatically uses the tokenizer chat template when the model provides one, which is important for instruct-tuned models.

```bash
python3 -m attribution_steering.cli collect \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset datasets/factual_conflict.jsonl \
  --output-dir runs/qwen25_3b_collect \
  --max-examples 12
```

Fit a steering state from misleading-condition traces:

```bash
python3 -m attribution_steering.cli analyze \
  --input-dir runs/qwen25_3b_collect \
  --output-dir runs/qwen25_3b_analysis \
  --fit-conditions misleading \
  --top-k-layers 4
```

Evaluate baseline versus steered generations:

```bash
python3 -m attribution_steering.cli steer \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset datasets/factual_conflict.jsonl \
  --steering-state runs/qwen25_3b_analysis/steering_state.pt \
  --output-dir runs/qwen25_3b_steer \
  --max-examples 12 \
  --steering-scale 2.0
```

## Output Artifacts

`collect` writes:

- `records.jsonl`: prompts, generations, correctness labels, and top attribution graph features
- `activations.pt`: cached layer activations and layer attribution scores
- `summary.json`: basic accuracy summary

`analyze` writes:

- `analysis.json`: layer-level contrasts and repeated graph motifs
- `steering_state.pt`: centroids, thresholds, layer weights, and steering directions

`steer` writes:

- `steering_records.jsonl`: baseline vs steered answers
- `steering_summary.json`: per-condition accuracy comparison

## Notes

- The included dataset is a starter benchmark, not a publishable hallucination benchmark.
- Small base models may fail on both clean and misleading conditions. That is still useful because the pipeline will show where the traces diverge.
- On Apple Silicon, you can add `--device mps`. On NVIDIA GPUs, CUDA will be auto-detected.
- If you want stronger experiments, the next step is to replace the dataset with a larger factual benchmark and swap the current graph proxy for a richer attribution method.
