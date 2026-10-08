# steering-awareness

A GPU-free reproduction of Section 4.3, "Intervention awareness: can models detect activation steering?", of
[Can LLMs Introspect? A Reality Check](https://arxiv.org/abs/2605.26242) (Singh, Linzen, Ravfogel).
All generation and steering run on the hosted [Neuronpedia API](https://www.neuronpedia.org/api-doc).

The model is told that a researcher may inject "thoughts" into its activations and is asked what it detects.
Each trial is one of three conditions:

- **Control**: the task prompt alone.
- **Hidden intervention**: the task prompt, with a concept direction added to the residual stream.
- **Gaslight**: a block of text urging the model to talk about the concept, followed by the task prompt. No steering.

In the 2-way task the model answers "injected" or "not injected"; in the 3-way task it also has a "prompt manipulation" option.
The paper's point is that models which pass the 2-way task also call gaslight trials injections, and fail to separate the two in the 3-way task.

## Setup

```bash
uv sync
export NEURONPEDIA_API_KEY=...  # from https://www.neuronpedia.org/account
```

## Pipeline

Run per model (`model=llama8b` or `model=llama70b`):

```bash
uv run src/find_features.py model=llama8b                            # SAE latent per concept; prints a table to review
uv run src/sweep_alpha.py model=llama8b                              # 2-way pilot over model.alphas; picks the best alpha
uv run src/collect_trials.py model=llama8b collect_trials.task=2way  # main trials at that alpha
uv run src/collect_trials.py model=llama8b collect_trials.task=3way
uv run src/analyze_trials.py model=llama8b analyze_trials.task=2way  # heatmaps and comparison with the paper
uv run src/analyze_trials.py model=llama8b analyze_trials.task=3way
```

- To replace a concept's latent, add `concept: index` to `overrides` in `conf/model/<model>.yaml` (pick from the printed candidates) and rerun `src/find_features.py`.
- Steering endpoints allow 240 calls per hour per IP, so a model takes about 6 hours. Requests wait out the limit.
- To resume an interrupted run, rerun the same command with its timestamp appended, e.g. `timestamp=20261004-213517`.

Each run writes to `results/{script}/{experiment_name}/{timestamp}/` with its resolved `config.yaml`; `latest` links to the most recently started run.
Trial logs (`trials.jsonl`) are not committed.

## Differences from the paper

| | Paper | Here |
|---|---|---|
| Models | Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct | `llama3.1-8b-it`, `llama3.3-70b-it` |
| Concept direction | Mean-difference residual vector per concept | Decoder direction of the SAE latent whose explanation best matches the concept word |
| Layer (input of block) | 8B: 4 (per Appendix R; Appendix J lists 2 for 2-way); 70B: 2 | 8B: 4 (`3-resid-post-aa`); 70B: 51 (`50-resid-post-gf`) |
| Strength α (on a unit vector) | Grid search per model and task (8B: 8 for 2-way, 4 for 3-way; 70B: 2) | 2-way pilot over four values, 1 trial per concept; the best is used for both tasks |
| Steered positions | Tokens of "Trial 1: What do you detect?" | Every non-special prompt token and every generated token |
| Sampling | T=0.7 steered, T=0.8 otherwise, top-p 0.95, 512 tokens | T=0.7 everywhere, top-p 0.9 (the checkpoint default, which the API does not expose), 64 tokens |
| Trials per concept | 250 hidden intervention, 500 gaslight, 500 control | 20 hidden intervention, 10 gaslight, 10 control (2-way adds 2 + 2 from the pilot) |
| Answer parsing | First integer in the response ("loose") | Same, but 0 and signed numbers are invalid (the official code wraps them to an option) |

With 10-20 trials per concept, part of the ± across concepts is sampling noise, so compare the means with the paper.

The trial prompts and the gaslight text in `prompts/` are copied from the
[official implementation](https://github.com/shashwat1002/introspection_reality_check).
