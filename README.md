# Evals

## Deepresearch Bench

Eval questions and answers taken from the RACE evals from [Deepresearch Bench](https://github.com/Ayanami0730/deep_research_bench)

```bibtex
@article{du2025deepresearch,
  author    = {Mingxuan Du and Benfeng Xu and Chiwei Zhu and Xiaorui Wang and Zhendong Mao},
  title     = {DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents},
  journal   = {arXiv preprint},
  year      = {2025},
}
```

## Running the Weave-Instrumented Evaluation

- CLI: `uv run python evaluation/eval.py --target data/model_outputs.jsonl --api-key $OPENAI_API_KEY`
- Programmatic: call `evaluate_agent(dr.run, config)` from `evaluation.eval` to weave-trace an agent directly (trials, project names, and custom attributes come from `EvalConfig`).
- Every evaluation writes JSONL + summary files locally and publishes traces to the Weave project configured via `EvalConfig`.

## Tests

- Run the test suite with `uv run pytest`.
- The end-to-end weave test requires a live W&B project; set `WANDB_ENTITY` and `WANDB_PROJECT` before running to enable it (it is skipped otherwise).
