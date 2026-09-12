# Unified FIRM pipeline

Read the repository [README](../README.md) for the corrected setup and supported exploratory commands, and [research audit](../RESEARCH_AUDIT.md) for implementation gaps and the paper's experiment plan.

**The complete FIRM pipeline is not research-ready.** The legacy causal tracer used layer-position scores; causal training saved an adapter without optimization; the four-variant evaluator did not apply the joint intervention. These paths now reject execution rather than emit misleading successful results.

Useful entry points:

- `preflight.py`: dependency, cache/storage, config, and model-access checks before weights are downloaded.
- `run_unified_pipeline.py`: single-variant exploratory evaluation, configured by model YAML and `configs/datasets.yaml`.
- `train_fairsteer.py`: static contrastive DSV artifacts for a configured model. Built-in pairs are toy data; dynamic FairSteer is not reproduced by this command.
- `model_variant_loader.py`: checks explicit artifacts, model identity, steering direction, layer, and dimensions. Sycophancy checkpoints require `sycophancy.model_path`.
- `firm_pipeline.py` and `run_integrated_pipeline.py`: incomplete FIRM research entry points; currently fail explicitly.

From the repository root, run:

```bash
python unified_pipeline/preflight.py \
  --model-config unified_pipeline/configs/models/qwen2.5-1.5b-instruct.yaml
python -m pytest tests/test_research_reliability.py -q
```

Use the root README's tiny dataset configuration for model smoke tests. Saved historical results, “complete” filenames, and a `publication` preset are not evidence of a valid experiment. The audit preserves old artifacts for traceability.
