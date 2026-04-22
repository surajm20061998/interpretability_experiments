# Attention Dilution Re-Scope Plan

This document re-scopes the project around the proposal in [Project_Proposal_LLM_Reasoners (1).pdf](/Users/surajmishra/Downloads/Project_Proposal_LLM_Reasoners%20%281%29.pdf) and maps the work onto the current repo.

## Proposal Summary

The proposal defines three phases:

1. Phase 1: Find refusal-related heads and a refusal direction on direct harmful prompts.
2. Phase 2: Add benign context bloat before the harmful request and test whether refusal-related attribution weakens as context grows.
3. Phase 3: Inject the refusal vector back into the model during long-context jailbreaks and test whether refusal is restored.

## Validity Check

The core research question is valid, but two parts of the wording are currently stronger than the literature supports.

### What is well-supported

- Long-context jailbreaks are real.
  - Anthropic's "Many-shot jailbreaking" (April 2, 2024) shows that increasing the number of demonstrations in a long prompt can increase harmful responses.
- Refusal behavior is mechanistically localized enough to be steerable.
  - Arditi et al. show that refusal is mediated by a one-dimensional direction in the residual stream.
- Safety-relevant attention heads are plausible objects of study.
  - Zhou et al. (ICLR 2025) identify safety-critical attention heads whose ablation compromises safety.
- Long-context degradation and position sensitivity are real.
  - Liu et al. show performance often drops when relevant information is buried in the middle of long contexts.

### What should be weakened

- "Attention mass allocated to Guardrail Heads" should remain a hypothesis, not a stated mechanism.
  - Existing work supports refusal directions and safety-relevant heads, but does not fully prove that context bloat works specifically by starving those heads of attention mass.
- "Integrated Gradients isolates the exact heads responsible" is too strong if used alone.
  - IG is a strong attribution tool for ranking candidates, but causal claims should be checked with ablation, activation patching, or path patching.

### Recommended revised hypothesis

Safety fine-tuning induces refusal-related structure in the residual stream, likely mediated in part by a small subset of safety-relevant attention heads. We hypothesize that long-context jailbreak prompts reduce the influence of these components on the first generated token, and that this reduction can be observed as a drop in refusal-direction strength and in attribution to safety-relevant heads. If so, injecting the refusal direction back into the relevant layer should partially restore refusal behavior.

## What We Can Reuse From This Repo

### Directly reusable

- `attribution_steering/modeling.py`
  - model loading
  - device resolution
  - decoder layer discovery
- `attribution_steering/experiment.py`
  - prompt tracing pattern
  - activation capture pattern
  - generation loop
- `attribution_steering/steering.py`
  - inference-time activation steering hooks

### Needs to change for the new scope

- The current dataset and labeling are for hallucination vs truthful QA, not refusal vs compliance.
- The current attribution graph is `grad * activation` plus attention-weight heuristics.
  - That is useful as a fast exploratory proxy.
  - It does not satisfy the proposal's Integrated Gradients requirement.
- The current notebook is focused on context-bloat retrieval, not safety refusal.

## Recommended Phase 1 MVP

### Goal

Locate a small set of refusal-relevant heads and layers on direct harmful prompts using a small aligned chat model.

### Recommended model choices

Use one of these:

1. `Qwen/Qwen2.5-1.5B-Instruct`
2. `meta-llama/Llama-3.2-1B-Instruct`
3. If access is available and refusal behavior is too weak, use a stronger but still manageable aligned chat model around 3B parameters.

Do not use Llama Guard or ShieldGemma as the main model for the mechanistic analysis.
They are safety classifiers, not general chat models that naturally refuse and then continue generation.

### Phase 1 dataset

Use a small curated harmful-prompt evaluation set:

- 20 to 50 harmful prompts across categories
  - violence
  - fraud
  - illicit behavior
  - self-harm related requests
- 20 to 50 benign control prompts with similar surface form

Do not hardcode unsafe content in the repo if avoidable.
Prefer benchmark IDs, placeholders, or local private data loading.

### Phase 1 metric

For each prompt, record:

- generated first token
- refusal classification
- refusal score
- compliance score

Recommended refusal score:

- logit difference between a refusal token set and a compliance token set on the first generated token

Example anchor sets:

- refusal anchors: `["I", "I'm", "Sorry", "cannot", "can't"]`
- compliance anchors: `["Sure", "Here", "First", "To"]`

This should be calibrated per model after a small pilot run.

### Phase 1 attribution workflow

1. Run the direct harmful prompt.
2. Target the refusal score or the first refusal token logit.
3. Compute Integrated Gradients on:
   - token embeddings
   - per-layer residual stream at the final prompt position
   - optionally per-head outputs if hooks are available
4. Rank candidate heads and layers by attribution.
5. Verify the top candidates with causal tests:
   - zero ablation
   - activation patching between refused and non-refused runs

### Phase 1 deliverables by tonight

- a refusal-eval JSONL
- a notebook or script that runs direct harmful prompts
- a table of top refusal-relevant layers/heads
- at least one causal validation plot showing that ablating top candidates reduces refusal score

## Recommended Phase 2 MVP

### Goal

Show that adding benign context bloat weakens the refusal mechanism found in Phase 1.

### Prompt construction

Build paired prompts for the same harmful request:

- direct harmful prompt
- harmful prompt preceded by benign roleplay or many-shot bloat

Sweep context lengths:

- 0
- 512
- 1024
- 2048
- 4096
- 8192

If the model supports more and the hardware allows it, extend upward after the first successful run.

### Phase 2 metrics

For every length:

- refusal rate
- refusal score
- first-token entropy
- IG attribution mass on Phase 1 safety heads
- attribution mass on early prompt tokens
- cosine drift of the final prompt-token hidden state relative to the no-bloat baseline

### Phase 2 proof target

The strongest Phase 2 result is a plot showing:

- refusal score decreases with context length
- attribution mass on Phase 1 safety heads decreases with context length
- early-token or benign-bloat attribution rises
- the model crosses from refusal to compliance beyond a prompt-length threshold

### Phase 2 deliverables by tonight

- a long-context prompt generator
- a run script for context sweeps
- summary CSVs
- at least three plots:
  - refusal rate vs context length
  - refusal-head attribution vs context length
  - representation drift vs context length

## Phase 3 Stretch Plan

### Goal

Use activation steering to restore refusal in long-context jailbreak settings.

### Minimal version

Compute a refusal vector using one of these:

1. difference between mean residual activations for direct harmful refused prompts and long-context complied prompts
2. difference between refused and complied runs at the target layer identified in Phase 1

Then inject:

`h_l <- h_l + alpha * v_refusal`

at one or a few target layers during inference.

### Phase 3 success criterion

On long-context jailbreak prompts:

- refusal score increases after steering
- refusal rate improves after steering
- benign prompts are minimally harmed

### What counts as enough for a first Phase 3 result

- one model
- one target layer or a small layer set
- one steering vector
- one sweep over `alpha`
- before vs after refusal plots

## Concrete Implementation Plan For Today

### Step 1: Freeze scope

Stop extending the retrieval notebook.
Keep it as a side experiment.

### Step 2: Build a new safety-focused path

Create a new package or module group:

- `safety_dilution/dataset.py`
- `safety_dilution/attribution.py`
- `safety_dilution/experiment.py`
- `safety_dilution/cli.py`
- `notebooks/safety_attention_dilution.ipynb`

### Step 3: Implement refusal labeling first

Before IG, make sure you can reliably score:

- refused
- complied
- ambiguous

If this is noisy, the rest of the pipeline will be unstable.

### Step 4: Implement IG on the first-token refusal score

Prioritize:

- residual-stream IG by layer
- per-token attribution

Then add per-head attribution once the layer-level result is stable.

### Step 5: Add causal checks

For top-ranked heads/layers:

- ablate
- patch

Do not rely on attribution-only claims for the final report.

### Step 6: Add the context-length sweep

Once direct harmful prompts are behaving correctly:

- generate bloat
- sweep lengths
- log refusal score and head attribution

### Step 7: Only then attempt steering

Do not begin Phase 3 until:

- refusal scoring works
- at least one safety-relevant layer or head is replicated across prompts
- the context sweep shows a measurable drop

## Fastest Path To A Strong Result

If time gets tight, the best reduced-scope path is:

1. One model: `Qwen/Qwen2.5-1.5B-Instruct`
2. 20 harmful prompts
3. 20 benign prompts
4. Layer-level IG instead of full head-level IG
5. One context sweep: `0 -> 512 -> 1024 -> 2048 -> 4096`
6. One steering intervention at the best layer

That is enough for a defensible Phase 1 and Phase 2, and a credible Phase 3 pilot.

## Internet Sources Used To Validate The Scope

- Proposal PDF: [Project_Proposal_LLM_Reasoners (1).pdf](/Users/surajmishra/Downloads/Project_Proposal_LLM_Reasoners%20%281%29.pdf)
- Anthropic, "Many-shot jailbreaking" (April 2, 2024): https://www.anthropic.com/research/many-shot-jailbreaking
- Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction" (NeurIPS 2024): https://openreview.net/forum?id=pH3XAQME6c
- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (July 6, 2023): https://huggingface.co/papers/2307.03172
- Turner et al., "Activation Addition: Steering Language Models Without Optimization" (August 20, 2023): https://huggingface.co/papers/2308.10248
- Zhou et al., "On the Role of Attention Heads in Large Language Model Safety" (ICLR 2025): https://openreview.net/forum?id=h0Ak8A5yqw
- Sundararajan et al., "Axiomatic Attribution for Deep Networks" (March 4, 2017): https://huggingface.co/papers/1703.01365
- Captum Integrated Gradients docs: https://captum.ai/docs/extension/integrated_gradients
- Qwen2.5-1.5B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

