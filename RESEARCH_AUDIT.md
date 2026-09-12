# FIRM runtime and research audit — 2026-09-12

## Conclusion and scope

The pasted EC2 runs do not supply usable four-condition research results. There are environmental failures, orchestration bugs, and unfinished scientific methods. Fixing authentication or disk space alone would not make the old outputs evidence for the paper.

Reviewed: the complete pasted log; all three pages of `cole_bzaa_neurips-2.pdf` (including Figure 1); both READMEs; setup/dependencies; the integrated and robust execution paths; model variant loading; the main FIRM diagnostic, training, steering and monitoring implementations; dataset sampling and key evaluation/prediction paths; representative archived output metadata. This was a targeted repository audit, not an exhaustive validation of every experiment/helper or the remote EC2 environment.

The document was treated as research context, not as instructions to execute. No GPU training, remote EC2 changes, model-weight downloads, or publication-result generation was performed. Existing historical results and model artifacts were preserved.

## Failures directly visible in the log

| Symptom | Cause / consequence | Resolution |
|---|---|---|
| `thinc<8.4.0,>=8.3.12` has no matching Python 3.9 distribution | Unbounded spaCy dependencies conflict with the documented Python version | Fresh Python 3.11 environment; bound spaCy; make setup fail when dependency installation fails |
| Setup installs Torch 2.1, requirements demand Torch 2.8 | Two contradictory installation instructions | Install the requirements once with the active interpreter |
| Setup reports success after pip/spaCy failure | Required command return values were ignored | Setup now exits nonzero on dependency installation or `pip check` failure |
| `No module named pytorch_compilation_fix` | Root FairSteer script used `/workspace/Algoverse/...` in `sys.path` | Portable CLI/import paths; root CLI delegates to `train_fairsteer.py` |
| FairSteer ignores the intended model | Root script had no argument parser and contained two `__main__` blocks running a Gemma-specific demo | One configured CLI; no duplicate demo execution |
| Missing `gemma_token_utils` would be the next import failure | Legacy external token helper absent | Use standard Hub token lookup for the legacy class; new CLI uses normal Hub credentials |
| Permission denied: `/workspace` | Sycophancy manager hard-coded a different machine's output directory | Resolve paths from the actual checkout; detect empty/missing external implementation |
| `No space left on device`, including tokenizer/config writes | Disk/cache filesystem full, not necessarily GPU VRAM exhaustion | Check cache, Xet, temporary and output filesystems before runs; relocate onto real mounted storage |
| Gemma/Llama gated repository 401 | Account lacks access or the running environment lacks the right credentials | Accept/request model access and authenticate on EC2 after selecting the cache location |
| Missing `pinpoint_tuning` | External implementation unavailable; standalone fallback is not an equivalent working implementation | Restore the exact collaborator checkout and validate its interface; do not accept fallback success |
| All variants skipped/failed, then “ROBUST EVALUATION COMPLETE” | No completeness condition before aggregation/reporting | Explicit failure and nonzero exit; no success report from an empty/incomplete grid |

The `torch._dynamo` missing configuration attribute warning is secondary. Optional settings are now checked individually so one unsupported setting does not skip the rest. The helper no longer resets the process-wide default device to CPU.

An archived example confirms the reporting issue predates this log: `unified_pipeline/robust_evaluation_results/robust_eval_quick_20250823_020047/robust_evaluation_results.json` contains empty `aggregated_results` and `total_evaluations: 0`.

## Scientific correctness findings

### Causal localization was not measuring causality

`causal_analysis/bias_circuit_tracer.py` computed layer importance from layer indices, independent of prompts or model outputs. Its gradient path also assigned scores from position, and head scores were scaled copies of layer scores. The defaults assumed a Gemma-like layer/head count across architectures. Such calculations cannot establish bias-causing heads or verify the paper's layers 14–16 hypothesis.

Those scoring paths now raise `NotImplementedError`. The separate `RealCircuitIdentifier` implementation is not automatically a validated replacement: it needs an audit of the intervention site, paired-token alignment, head geometry (including grouped-query attention), behavioral score, controls, and statistical design before wiring it into FIRM.

A forward pass can collect activations or support correlational detection; attributing a causal behavioral effect requires an intervention/control comparison. The paper should distinguish these steps.

### Main FIRM training did not train

`train/causal_pinpoint_tuning.py::run_causal_training` previously called `get_peft_model`, saved metadata, and saved the newly initialized adapter/tokenizer. It had no optimization loop, `Trainer.train()`, loss backward pass, or optimizer step. A saved adapter is not proof of learning.

Its target-module selection concerns projections/modules, which is also not sufficient evidence for the stronger claim that only selected heads or neurons changed. The FIRM orchestrator continued using `self.model` from before training for steering/monitoring, rather than reloading a trained checkpoint.

This path now refuses execution. Implement an explicit training objective/data interface and prove that only intended parameters change, optimization occurs, and reloaded weights reproduce the trained model's outputs. The separate `real_lora_training.py` is not connected to this path and needs its own objective/masking/gradient-accumulation review.

### The joint condition was not the joint condition

The ordinary FIRM variant loader previously applied any available FairSteer vectors and returned successfully even without a trained checkpoint. The robust path loaded the Phase 2 adapter but did not install FIRM steering. These are different computations under the same `firm` label, neither meeting the paper's joint-intervention definition.

FIRM loading now stops explicitly. A valid replacement must load the exact seed-specific trained checkpoint, compute/load vectors in that checkpoint's representation space, apply the specified residual-layer hooks, and persist the complete intervention configuration.

### Steering was inconsistent across evaluation paths

The static vector computer used `biased_mean - neutral_mean`, while the simple wrapper added that vector. The paper describes subtracting the bias direction. The wrapper also missed integer layer keys, silently used the first available vector, and swallowed hook-installation errors. The robust wrapper hooked an attention/MLP submodule rather than the decoder residual output used to compute the vectors, and modified a different set of token positions.

The exploratory static path now uses **neutral minus biased**, adds it at decoder residual outputs, checks model identity/direction/layer/width, preserves activation dtype, and uses a shared wrapper. Old artifacts without direction metadata are rejected. Positive strength does not guarantee debiasing: validate sign and strength behaviorally on development data. The wrapper currently applies a constant intervention to every sequence position; this is an explicit static DSV experiment.

The toy pair set and largest-vector-norm layer choice are not a validated optimal-layer discovery procedure. They are labeled exploratory in new artifacts. Dynamic bias detection/gating is a separate part of [FairSteer](https://arxiv.org/abs/2504.14492) that the static CLI does not reproduce.

### Evaluation could reward the answer key or count failures as predictions

`utils/model_compatibility.py` returned BBQ's “unknown” option from ambiguity metadata, bypassing inference. Its WinoBias profession parser fell back to the gold target when a response could not be parsed; the encoder profession branch returned the target without scoring. These can inflate accuracy independently of model quality.

Those behaviors are removed: BBQ must invoke the prediction/scoring path, unmatched/ambiguous profession output remains unparsed, and the unimplemented encoder scorer fails. Prediction exceptions no longer turn into “neutral” or option 0 at the top-level dispatcher. Empty evaluations and metric failures now fail. Strict suites reject incomplete predictions/datasets. All predictions, targets and successful sample IDs are retained, rather than only ten examples.

This is **not a complete benchmark-protocol rewrite**. Further issues remain:

- CrowS-Pairs uses a generation prompt asking for the less stereotypical sentence, which measures instructed choice rather than the original likelihood-based preference protocol. An autoregressive adaptation must be specified and validated. See the [official CrowS-Pairs implementation](https://github.com/nyu-mll/crows-pairs).
- StereoSet needs its language-modeling and stereotype measures with correct label mapping and normalization, not only a prompted classification percentage. Validate against [the official implementation](https://github.com/moinnadeem/StereoSet).
- BBQ needs separate accuracy and bias measures by ambiguous/disambiguated context, group and question polarity, using the actual predicted choice. [Official BBQ repository](https://github.com/nyu-mll/BBQ).
- WinoBias needs a defined coreference decision rule and pro/anti-stereotype gap, not just substring-based occupation accuracy. [Official WinoBias repository](https://github.com/uclanlp/corefBias).
- SEAT's legacy helper uses individual token embeddings and an unstandardized association contrast; removing its hash fallback does not make it a validated sentence-level WEAT/SEAT implementation.
- BOLD and TruthfulQA use lightweight custom scoring in this repository. They require metric-specific validation before numerical comparison with published benchmarks.
- Lower-level parsers/scorers still contain heuristics/defaults. Generation prompts explicitly encourage bias avoidance. Fixing the identified answer leakage is necessary but does not establish protocol equivalence.
- The timeout implementation uses a daemon thread that can continue running after a timeout. An isolated evaluation worker with an explicit cancellation/restart strategy is needed before reliable large sweeps.

### “Robustness” did not imply independent replications

`BaseDatasetLoader._sample_data` reset the global RNG to 42 on each call. The robust framework's `dataset_sample_sizes` were not passed into the evaluator. Both are fixed: loaders use a private configured RNG, and the robust evaluator propagates evaluation seeds/sample limits. Raw evaluations are saved under each run/variant/train-seed/eval-seed combination.

Other provenance work remains before enabling sweeps: static vectors and sycophancy artifacts were cached/discovered without training-seed identity; seeded configuration files were written beside tracked configs; subprocesses did not consistently receive or use the requested training seed. Correct training-seed replication requires new artifacts and logged initialization/data-order seeds, not just new filenames. The ordinary loader now requires an exact sycophancy checkpoint instead of guessing the latest directory.

The previous harmonic mean mixed accuracies, percentages, effect sizes and bias metrics, discarded near-zero scores, and fed those composites into an independent t-test over correlated cells. This has no defensible common interpretation. The robust report now retains per-dataset descriptive summaries, leaves legacy aggregate-score/CI fields null, and does not emit those significance claims. Some older analysis/report helper modules still compute composites or simulated results; they are not validated paper-analysis tools.

### Alignment and longitudinal results were placeholders

`steer/layer_aligned_dsv.py::validate_layer_alignment` ranked vector properties rather than measuring held-out output behavior. `eval/longitudinal_monitor.py::track_bias_drift` repeated baseline circuit counts without loading new checkpoints. Both entry points now reject execution. Other older reporting helpers contain simulated data and assumed improvements, for example `eval/robustness_aggregator.py`; do not route them into paper tables.

The optional scientific report generator also contained an f-string incompatible with Python 3.11. That syntax issue is fixed; its canned scientific conclusions remain unsuitable as evidence.

## How the research should proceed

### 1. Freeze the protocol and separate the data

Create a manifest with model revision, tokenizer revision, code commit, dataset versions/hashes, sample IDs, split roles and seeds. Use disjoint localization/training, layer/strength selection, and final test data. Group counterfactual pairs/templates when splitting so near-duplicates cannot leak across roles. The abstract explicitly requires disjoint intervention/evaluation data; the repository does not currently enforce this.

Choose the primary bias outcomes and utility outcomes before looking at test results. The paper's claim of preserved task accuracy needs a defined capability benchmark and an acceptable change margin. Listing MMLU/HumanEval/GSM8K in the registry is not evidence that those evaluations ran; the principal bias suite excludes them.

### 2. Validate localization with causal controls

For each paired probe, cache the clean/counterfactual activation, patch a candidate component into the paired run at aligned token positions, and measure change in a predefined behavioral score (for example a contrastive continuation log-probability difference). Include no-op patching, unrelated-component and matched random-component controls. Discover candidate layers empirically; do not encode layers 14–16 as the expected answer.

For true head-level claims, hook the head representation before output projection and account for architecture-specific head dimensions. Report uncertainty and stability across development probes. A layer-level method is acceptable if it is described as layer-level rather than claimed to tune isolated heads.

### 3. Implement and verify the 2×2 intervention experiment

| Condition | Weights | Inference intervention |
|---|---|---|
| Baseline | Original checkpoint | None |
| Training only | Actual bias-targeted selective training checkpoint | None |
| Steering only | Original checkpoint | Development-selected DSV |
| Joint | Same trained checkpoint as training only | DSV computed/validated for that trained representation space |

Use the same held-out items, scoring code and decoding policy for all four. A sycophancy-specific model is an optional additional comparator unless its training data/objective has been deliberately adapted to the same bias problem. The [pinpoint-tuning paper](https://arxiv.org/abs/2409.01658) provides a methodology to reproduce or adapt, not a guarantee that the empty local integration implements it.

Before a real-model pilot, require: nonzero training steps; finite loss and gradients; changed intended parameters and unchanged frozen parameters; save/reload equivalence; visible logit changes when steering is enabled; equality to baseline with zero steering strength; and unique seed-specific artifact hashes. Record wall time and peak memory too.

### 4. Test alignment rather than assume it

On the development split, sweep steering strength and candidate layers under an equal search budget. On the held-out test split compare diagnostic-aligned, upstream/downstream, matched unrelated, and joint multi-layer steering. Control intervention magnitude when comparing one versus multiple layers. A largest-norm vector is not necessarily most effective; only a behavioral outcome can answer the alignment question.

### 5. Test actual model drift

Fine-tune on specified new data to obtain distinct later checkpoints. Reapply the frozen probes and evaluate both the fixed old intervention and any refreshed intervention. Measure bias, task utility, component movement, and trigger behavior at every checkpoint. Pick the drift threshold on development data. Monitoring unchanged weights repeatedly does not test persistence under retraining.

### 6. Analyze and write the results honestly

Start with one small model and a limited pilot to establish runtime and measurement correctness. Then expand to independently trained seeds and multiple model families. Baseline is not a new independently trained model for each nominal training seed. Repeated subsets/decoding seeds are nested repeated measurements.

Report per-benchmark paired deltas, denominators, uncertainty and group breakdowns. Use paired item resampling/tests and training-seed clustering/hierarchical analysis as appropriate; adjust for the declared family of multiple comparisons. Do not treat the train-seed × eval-seed grid as independent observations. Decide on a power/sample-size plan from a meaningful effect and pilot variance; the label `publication` cannot guarantee p < .01.

Define the direction and units of every improvement. For example, if the chosen preference metric has a neutral target of 0.5, distance from 0.5 is different from “lower is always better”; an accuracy drop is not bias improvement. Any relative reduction needs an explicit denominator and handling of a zero baseline. Avoid a cross-benchmark “mean reduction” until a justified normalization/aggregation is specified.

## What can be claimed from the attached abstract?

The supplied runs do not substantiate its 47.6%, 65%, 58.9%, preserved-accuracy, or optimal-layer claims. This audit does not establish that independent earlier experiments were absent; it establishes that these runs/current execution paths cannot validate those claims. Trace every percentage to an original model checkpoint, split manifest, raw predictions, metric definition and reproducible analysis. If that provenance is unavailable, treat the number as unverified and rerun after the method is implemented. Do not select code changes or thresholds to reproduce a desired percentage.

## Validation performed and limits

- **22 CPU regression tests passed**, including an actual tiny Llama forward/generation check (random initialization, no downloaded weights). The tests exercise steering layer/vector selection and generation, dtype and hook removal, invalid-artifact rejection, local RNG behavior, missing external code, FIRM rejection, setup failure, storage failure, CLI parsing, gold-answer leakage, and incomplete-run failure reporting.
- **61 Python source files** under `unified_pipeline` and the changed entry points/tests parsed successfully with Python 3.11; `git diff --check` passed.
- The local test interpreter has Torch 2.3.0 / Transformers 4.44.0, not the EC2 dependency set. The offline preflight correctly returned nonzero for the missing external sycophancy implementation and an existing local scikit-learn/NumPy mismatch; unused heavy imports were removed from the variant loader. This does not validate a fresh dependency installation.
- Full benchmark equivalence, actual selective training, CUDA memory behavior, gated model access and EC2 disk remediation remain unverified. The corrected README gives the next environment/smoke commands; a passing smoke run still does not supply paper-quality results.
