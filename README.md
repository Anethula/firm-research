# FIRM: Fairness Interventions at Runtime and Model-training

FIRM is a research prototype for combining selective training with inference-time bias steering. **This checkout does not yet implement a validated end-to-end version of the attached paper.** Successful installation or a `publication` preset does not establish research validity.

The September 2026 audit found both runtime failures and scientific correctness problems. Read [RESEARCH_AUDIT.md](RESEARCH_AUDIT.md) for the evidence, changes, remaining gaps, and experiment plan. Existing run folders have been preserved, but must not be treated as validated results merely because their filenames say `COMPLETE` or `robust`.

## Current status

| Component | Status |
|---|---|
| Environment preflight | Checks imports, config identity, writable storage, free space, model config/tokenizer access |
| Single-model evaluation | Available for exploratory debugging; benchmark protocols still need validation |
| Static contrastive DSV | Available for exploratory debugging; not a reproduction of dynamic FairSteer |
| External sycophancy pipeline | Requires an external implementation; its head-artifact importer is unfinished |
| FIRM causal localization and training | Incomplete; execution stops instead of saving heuristic circuits/untrained adapters as results |
| Four-variant / multi-seed FIRM sweep | Blocked by the incomplete FIRM implementation |
| Layer-alignment and checkpoint-drift conclusions | Require actual behavioral/checkpoint experiments; placeholder entry points now reject execution |

The four conditions for the paper are **baseline, bias-targeted training only, steering only, and training plus steering**. A sycophancy-trained model can be an additional comparator; it is not automatically the paper's training-only condition.

## Set up the EC2 environment

Use a fresh Python 3.11 environment. The old Python 3.9 instructions and conflicting PyTorch 2.1/2.8 installs were incorrect.

```bash
# Run from the repository root on EC2.
conda create -n firm-research python=3.11 -y
conda activate firm-research

# Inspect disk capacity and the GPU before downloading weights.
df -h
df -i
nvidia-smi

python setup.py
python -m pip check
```

`setup.py` uses the active interpreter and installs `requirements.txt` once; a failed install exits nonzero. Python 3.10–3.12 is the supported setup range, with 3.11 recommended. The requirements retain PyTorch 2.8.0, Transformers 4.45.2, and PEFT 0.13.2; spaCy is bounded and Hub is constrained below 1.0 for Transformers 4.x. This dependency set has **not** been installed and GPU-tested on your EC2 machine during this audit.

Choose a PyTorch build appropriate to the host driver using the [official version instructions](https://pytorch.org/get-started/previous-versions/). VRAM needs depend on model, sequence length, precision, batch size, optimizer, and checkpointing; a fixed 16 GB promise is not reliable.

### Cache and output storage

`No space left on device` is a disk/cache problem, distinct from CUDA out-of-memory. Model shards, download staging, Xet cache, adapters, datasets, and temporary files all need room.

If the default filesystem is full, place the caches on an **existing mounted volume you can write to**. Substitute its actual path below; creating a directory alone does not add disk capacity.

```bash
export FIRM_STORAGE=/path/to/your/mounted-volume/firm
mkdir -p "$FIRM_STORAGE/huggingface" "$FIRM_STORAGE/tmp"
export HF_HOME="$FIRM_STORAGE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TMPDIR="$FIRM_STORAGE/tmp"
```

Set these before Python starts, and log in after selecting `HF_HOME` so the intended token cache is used. Check the filesystem holding the checkout too, because default run outputs are stored there. Do not delete old research outputs to free space without backing them up. See [Hugging Face cache documentation](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

### Model access and datasets

```bash
hf auth login
hf auth whoami
# Older installed Hub releases may instead expose: huggingface-cli login

git submodule update --init --recursive
bash pull_datasets.sh
```

For Gemma and Llama, accept the model's license/request access on its Hugging Face page using the account associated with the token. Authentication alone does not grant gated-repository access. Qwen's smaller configured model is a useful first environment check. Do not put tokens in source code, logs, or chat. [Hugging Face authentication instructions](https://huggingface.co/docs/huggingface_hub/guides/cli).

The `sycophancy-interpretability` folder in the audited checkout is empty and is not declared in `.gitmodules`. Updating the dataset submodules does not install it. Recover the collaborators' exact external checkout/commit and local modifications; this integration also expects Gemma-specific data/configuration files. Simply copying an arbitrary upstream revision does not validate that interface.

## Validate before a long run

From the repository root:

```bash
python unified_pipeline/preflight.py \
  --model-config unified_pipeline/configs/models/qwen2.5-1.5b-instruct.yaml
```

The preflight downloads config/tokenizer files, not model weights. Its default minimum is 20 GiB free on each model-cache/output filesystem and 1 GiB on the temporary filesystem; these are early checks, not a guarantee that a complete run fits. Use `--min-free-gb` to set the requirement for your planned run, `--offline` for local checks, and `--require-sycophancy` to check the external scripts. An `environment_ok` result does not change `firm_research_ready: false`.

## Exploratory smoke runs

Start with a tiny sample configuration before an expensive evaluation. These commands test execution, **not publication-ready measurements**.

```bash
cd unified_pipeline

# Prepare a small, strict dataset configuration.
python - <<'PY'
import yaml
from pathlib import Path
config = yaml.safe_load(Path('configs/datasets.yaml').read_text())
config['evaluation_suites']['quick_evaluation']['datasets'] = ['CrowsPairs', 'WinoBias']
config['integration']['skip_failed_datasets'] = False
for name in ['CrowsPairs', 'WinoBias']:
    config['dataset_configs'][name].update(sample_size=10, seed=100, enabled=True)
Path('smoke_datasets.yaml').write_text(yaml.safe_dump(config))
PY

python run_unified_pipeline.py \
  --model-config configs/models/qwen2.5-1.5b-instruct.yaml \
  --dataset-config smoke_datasets.yaml \
  --suite quick_evaluation
```

The printed output directory contains `evaluation/baseline/evaluation_results.json` and `reports/`. The legacy directory name `baseline` is also used for variant evaluations: record the actual model config with every run. There is no maintained `latest/` symlink.

To compute a static DSV, still from `unified_pipeline`:

```bash
python train_fairsteer.py \
  --config configs/models/qwen2.5-1.5b-instruct.yaml \
  --output-dir steering_vectors_smoke
```

Without `--pairs`, this uses a small built-in toy set and labels the artifact accordingly. For curated inputs, use `--pairs /absolute/path/train_pairs.json`, containing a JSON list of `[biased_text, neutral_text]` pairs. This script computes all supplied pairs. It selects a layer by vector norm within a configured range; that is not a measured optimal debiasing layer.

Create a copy of the model YAML with `model_variant: fairsteer` and an explicit artifact path under `fairsteer.vectors_path`, then run `run_unified_pipeline.py` with that YAML and the same dataset config. Keep the same seed and sample IDs for paired comparisons. The loader rejects old artifacts without model identity and vector-direction metadata; regenerate them rather than guessing the sign. The root `fairsteer_debiasing.py --config ...` command delegates to this same CLI.

The previous README's `run_unified_pipeline.py --model-variant`, `--steering-vectors`, and `--output-dir` flags were not implemented. Set variant/artifact options in YAML. Evaluation suites live in `configs/datasets.yaml` unless a different dataset config is explicitly passed.

## Before paper experiments

Follow the acceptance criteria in [RESEARCH_AUDIT.md](RESEARCH_AUDIT.md). In particular:

1. Implement causal patching and actual selective training on audited, disjoint development data.
2. Validate benchmark scoring, preserve raw outcomes, and verify each saved/reloaded variant changes the intended computation.
3. Run the four conditions with held-out layer/strength selection, then alignment ablations and distinct post-training checkpoints.
4. Analyze each benchmark separately with paired outcomes and training-seed uncertainty. Do not average accuracy, effect sizes, and bias percentages into one “mean bias” number.

`quick`, `standard`, and `publication` are historical seed-count presets (2×2, 4×4, 6×6), not statistical power guarantees. More evaluation seeds do not create more independently trained models. The four-variant entry points now stop while FIRM remains incomplete.

## Local regression tests

```bash
python -m pip install pytest
python -m pytest tests/test_research_reliability.py -q
```

These tests use CPU toy models and mocks for failure paths. They do not reproduce the paper's effect sizes or validate large-model performance. Keep a package lock/freeze, git commit, dataset hashes, split manifest, seed, model revision, and exact artifact paths with every future research run.
