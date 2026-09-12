#!/usr/bin/env python3
"""
Robust Multi-Seed Evaluation Framework

This framework provides statistically robust evaluation of bias mitigation techniques
through multiple training seeds and evaluation cycles.
"""

import json
import numpy as np
import os
import random
import sys
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import yaml
import warnings
from dataclasses import dataclass, asdict
from scipy import stats

# Import evaluation components at module level
try:
    from eval.unified_evaluator import UnifiedBiasEvaluator
    from datasets.unified_registry import UnifiedDatasetRegistry
    EVALUATOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import UnifiedBiasEvaluator: {e}")
    UnifiedBiasEvaluator = None
    UnifiedDatasetRegistry = None
    EVALUATOR_AVAILABLE = False

warnings.filterwarnings('ignore')


@dataclass
class EvaluationConfig:
    """Configuration for robust evaluation framework."""
    training_seeds: List[int]
    evaluation_seeds: List[int] 
    dataset_sample_sizes: Dict[str, int]
    statistical_tests: List[str] = None
    confidence_level: float = 0.95
    
    def __post_init__(self):
        if self.statistical_tests is None:
            self.statistical_tests = ["t_test", "mann_whitney", "effect_size"]


@dataclass
class SeedResult:
    """Results from a single seed evaluation."""
    training_seed: int
    evaluation_seed: int
    model_variant: str
    dataset_results: Dict[str, Any]
    overall_bias_score: Optional[float]
    evaluation_time: float
    metadata: Dict[str, Any]


@dataclass
class AggregatedResults:
    """Aggregated results across multiple seeds."""
    model_variant: str
    mean_bias_score: Optional[float]
    std_bias_score: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    dataset_means: Dict[str, float]
    dataset_stds: Dict[str, float]
    n_evaluations: int
    seed_results: List[SeedResult]
    statistical_significance: Dict[str, Any]


class RobustEvaluationFramework:
    """
    Framework for conducting statistically robust bias mitigation evaluations
    with multiple training and evaluation seeds.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        # Keep results and data relative to this checkout by default.
        self.base_dir = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parent.parent
        self.unified_dir = self.base_dir / "unified_pipeline"
        self.results_dir = self.unified_dir / "robust_evaluation_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced evaluation configurations for better statistical significance
        self.evaluation_configs = {
            "quick": EvaluationConfig(
                training_seeds=[42, 123],  # Increased from 1 to 2 seeds
                evaluation_seeds=[100, 200],  # Increased from 1 to 2 seeds
                dataset_sample_sizes={"default": 300}  # Increased from 200
            ),
            "standard": EvaluationConfig(
                training_seeds=[42, 123, 456, 789],  # Increased from 3 to 4 seeds
                evaluation_seeds=[100, 200, 300, 400],  # Increased from 3 to 4 seeds
                dataset_sample_sizes={"default": 750}  # Increased from 500
            ),
            "publication": EvaluationConfig(
                training_seeds=[42, 123, 456, 789, 999, 1337],  # Increased from 5 to 6 seeds
                evaluation_seeds=[100, 200, 300, 400, 500, 600],  # Increased from 5 to 6 seeds
                dataset_sample_sizes={"default": 1500}  # Increased from 1000
            ),
            "custom": None  # Will be set by user
        }
    
    def set_custom_config(self, training_seeds: List[int], evaluation_seeds: List[int],
                         dataset_sample_sizes: Dict[str, int] = None) -> None:
        """Set custom evaluation configuration."""
        if dataset_sample_sizes is None:
            dataset_sample_sizes = {"default": 500}
            
        self.evaluation_configs["custom"] = EvaluationConfig(
            training_seeds=training_seeds,
            evaluation_seeds=evaluation_seeds,
            dataset_sample_sizes=dataset_sample_sizes
        )
    
    def run_robust_four_model_evaluation(self, 
                                       base_config_path: str,
                                       model_name: str,
                                       suite: str,
                                       robustness_level: str = "standard",
                                       custom_config: Optional[EvaluationConfig] = None) -> Dict[str, AggregatedResults]:
        """
        Run robust evaluation of all four models with multiple seeds.
        
        Args:
            base_config_path: Path to base model configuration
            model_name: HuggingFace model name
            suite: Evaluation suite to run
            robustness_level: "quick", "standard", "publication", or "custom"
            custom_config: Custom evaluation config if robustness_level="custom"
            
        Returns:
            Dictionary mapping model variants to aggregated results
        """
        print(f"🔬 {'='*80}")
        print("   ROBUST MULTI-SEED FOUR-MODEL EVALUATION")
        print(f"🔬 {'='*80}")
        
        from research_status import require_firm_implementation
        require_firm_implementation()

        # Get evaluation configuration
        if robustness_level == "custom" and custom_config:
            eval_config = custom_config
        elif robustness_level == "custom":
            eval_config = self.evaluation_configs["custom"]
            if eval_config is None:
                raise ValueError("Custom config not set. Use set_custom_config() first.")
        else:
            eval_config = self.evaluation_configs[robustness_level]
        
        print(f"📊 Robustness Level: {robustness_level}")
        print(f"🎯 Training Seeds: {eval_config.training_seeds}")
        print(f"📈 Evaluation Seeds: {eval_config.evaluation_seeds}")
        print(f"🔢 Total Evaluations per Model: {len(eval_config.training_seeds) * len(eval_config.evaluation_seeds)}")
        print(f"🕒 Estimated Time: {self._estimate_total_time(eval_config)} minutes")
        
        # Create timestamped results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.results_dir / f"robust_eval_{robustness_level}_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        self.current_run_dir = run_dir
        
        # Save evaluation configuration
        config_path = run_dir / "evaluation_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(eval_config), f, indent=2)
        
        # Initialize results storage
        all_model_results = {
            "baseline": [],
            "fairsteer": [],
            "sycophancy": [],
            "firm": []
        }
        
        # Import the main evaluator
        sys.path.append(str(self.unified_dir))
        from run_integrated_pipeline import RealFourModelEvaluator
        
        total_combinations = len(eval_config.training_seeds) * len(eval_config.evaluation_seeds) * 4
        current_combination = 0
        
        # Run evaluations for each seed combination
        for train_seed in eval_config.training_seeds:
            print(f"\n🌱 {'='*60}")
            print(f"   TRAINING SEED: {train_seed}")
            print(f"🌱 {'='*60}")
            
            # Train models with this seed
            self._set_global_seed(train_seed)
            trained_models = self._train_all_models_with_seed(
                base_config_path, model_name, train_seed
            )
            
            failed_training = [name for name in all_model_results
                               if not trained_models.get(name, {}).get("success")]
            if failed_training:
                failure = {"status": "failed", "stage": "training", "training_seed": train_seed,
                           "failed_variants": failed_training, "training_results": trained_models}
                (run_dir / "failure.json").write_text(json.dumps(failure, indent=2, default=str))
                raise RuntimeError(f"Training failed for {failed_training}; see {run_dir / 'failure.json'}")
            for eval_seed in eval_config.evaluation_seeds:
                print(f"\n📊 Evaluation Seed: {eval_seed}")
                print("-" * 40)
                
                # Set evaluation seed for dataset sampling
                self._set_global_seed(eval_seed)
                
                # Evaluate each model variant
                for model_variant in ["baseline", "fairsteer", "sycophancy", "firm"]:
                    current_combination += 1
                    progress = (current_combination / total_combinations) * 100
                    
                    print(f"🎯 [{current_combination}/{total_combinations}] ({progress:.1f}%) "
                          f"Evaluating {model_variant} (train_seed={train_seed}, eval_seed={eval_seed})")
                    
                    if model_variant in trained_models and trained_models[model_variant]["success"]:
                        # Run evaluation with specific seeds
                        seed_result = self._evaluate_single_seed(
                            model_variant, trained_models[model_variant],
                            train_seed, eval_seed, suite, eval_config, model_name
                        )
                        
                        if seed_result:
                            all_model_results[model_variant].append(seed_result)
                            print(f"✅ {model_variant}: {seed_result.dataset_results}")
                        else:
                            failure = {"status": "failed", "stage": "evaluation", "variant": model_variant,
                                       "training_seed": train_seed, "evaluation_seed": eval_seed}
                            (run_dir / "failure.json").write_text(json.dumps(failure, indent=2))
                            raise RuntimeError(f"Evaluation failed; see {run_dir / 'failure.json'}")
                    else:
                        print(f"⚠️  {model_variant}: model training failed, skipping")
        
        expected = len(eval_config.training_seeds) * len(eval_config.evaluation_seeds)
        if not expected or any(len(results) != expected for results in all_model_results.values()):
            raise RuntimeError("Incomplete evaluation grid; cannot report a successful four-variant run")

        # Aggregate results across seeds
        print(f"\n📈 {'='*60}")
        print("   AGGREGATING MULTI-SEED RESULTS")
        print(f"📈 {'='*60}")
        
        aggregated_results = {}
        for model_variant, seed_results in all_model_results.items():
            if seed_results:
                aggregated = self._aggregate_seed_results(model_variant, seed_results, eval_config)
                aggregated_results[model_variant] = aggregated
                
                print(f"📊 {model_variant.upper()}:")
                print(f"   📈 DATASET-SPECIFIC RESULTS (NOT aggregated):")
                for dataset, mean_score in aggregated.dataset_means.items():
                    std_score = aggregated.dataset_stds.get(dataset, 0.0)
                    print(f"      {dataset}: {mean_score:.4f} ± {std_score:.4f}")

                print(f"   📊 Evaluations: {aggregated.n_evaluations}")
                print(f"   ⚠️  Use dataset-specific scores for analysis, NOT summary!")
        
        # Statistical significance testing
        statistical_results = self._compute_statistical_significance(aggregated_results, eval_config)
        
        # Save all results
        results_summary = {
            "evaluation_config": asdict(eval_config),
            "aggregated_results": {k: asdict(v) for k, v in aggregated_results.items()},
            "statistical_significance": statistical_results,
            "metadata": {
                "timestamp": timestamp,
                "model_name": model_name,
                "suite": suite,
                "robustness_level": robustness_level,
                "total_evaluations": sum(len(results) for results in all_model_results.values())
            }
        }
        
        # Ensure run directory exists
        run_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = run_dir / "robust_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report(aggregated_results, statistical_results, run_dir)
        
        print(f"\n🎉 ROBUST EVALUATION COMPLETE")
        print(f"   📁 Results saved to: {run_dir}")
        print(f"   📊 Summary report: {run_dir / 'summary_report.md'}")
        
        return aggregated_results
    
    def _set_global_seed(self, seed: int) -> None:
        """Set seed for all random number generators."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def _train_all_models_with_seed(self, base_config_path: str, model_name: str, 
                                   train_seed: int) -> Dict[str, Any]:
        """Train all models with a specific seed."""
        print(f"🏋️ Training all models with seed {train_seed}...")
        
        # Import the evaluator class
        from run_integrated_pipeline import RealFourModelEvaluator
        
        # Modify config to include seed
        seeded_config_path = self._create_seeded_config(base_config_path, train_seed)
        
        # Initialize evaluator and run training
        evaluator = RealFourModelEvaluator(str(self.base_dir))
        
        # Train each model type
        results = {}
        
        # FairSteer training
        fairsteer_success = evaluator.run_fairsteer_training_if_needed(seeded_config_path, model_name)
        results["fairsteer"] = {"success": fairsteer_success}
        
        # Sycophancy training  
        sycophancy_success = evaluator.run_sycophancy_training_if_needed(seeded_config_path, model_name)
        results["sycophancy"] = {"success": sycophancy_success}
        
        # FIRM training
        firm_path = evaluator.run_firm_training_if_needed(seeded_config_path, model_name)
        results["firm"] = {"success": firm_path is not None, "model_path": firm_path}
        
        # Baseline always available
        results["baseline"] = {"success": True}
        
        return results
    
    def _create_seeded_config(self, base_config_path: str, seed: int) -> str:
        """Create a model config with specific seed."""
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add seed to all relevant sections
        config['training_seed'] = seed
        config['eval_seed'] = seed
        
        if 'training' not in config:
            config['training'] = {}
        config['training']['seed'] = seed
        
        # Save seeded config
        seeded_config_path = base_config_path.replace('.yaml', f'_seed_{seed}.yaml')
        with open(seeded_config_path, 'w') as f:
            yaml.dump(config, f)
        
        return seeded_config_path
    
    def _get_model_path_for_variant(self, model_variant: str, model_name: str,
                                    model_info: Dict[str, Any]) -> Optional[str]:
        """Get the actual model path for a specific variant and training seed."""
        
        if model_variant == "baseline":
            # Baseline model is the original pre-trained model
            return model_name
            
        elif model_variant == "fairsteer":
            # FairSteer uses the baseline model + steering vectors
            # The steering vectors are applied at inference time
            return model_name
            
        elif model_variant == "sycophancy":
            # Sycophancy model should be in sycophancy_pipeline_runs/*/pinpoint_tuning_results/
            sycophancy_runs = self.unified_dir / "sycophancy_pipeline_runs"
            if sycophancy_runs.exists():
                # Find the most recent sycophancy run
                safe_model_name = model_name.replace('/', '_').lower()
                run_dirs = [d for d in sycophancy_runs.iterdir()
                            if d.is_dir() and safe_model_name in d.name.lower()]
                if run_dirs:
                    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
                    # Check if pinpoint_tuning_results subdirectory exists with tokenizer
                    pinpoint_dir = latest_run / "pinpoint_tuning_results"
                    if pinpoint_dir.exists() and (pinpoint_dir / "tokenizer_config.json").exists():
                        return str(pinpoint_dir)
            return None
            
        elif model_variant == "firm":
            # Use the exact run produced for this seed, not the newest run from
            # an unrelated model or earlier experiment.
            run_path = model_info.get("model_path")
            if run_path:
                causal_dir = Path(run_path) / "phase_2_causal_training"
                if causal_dir.exists() and (causal_dir / "tokenizer_config.json").exists():
                    return str(causal_dir)
            return None
            
        else:
            print(f"⚠️  Unknown model variant: {model_variant}")
            return None
    
    def _apply_fairsteer_intervention(self, model, tokenizer):
        from model_variant_loader import ModelVariantLoader
        name = getattr(model.config, "_name_or_path", "")
        return ModelVariantLoader(model, tokenizer, {
            "model_name": name, "model_variant": "fairsteer"
        }).load_variant_model()[0]

    def _evaluate_single_seed(self, model_variant: str, model_info: Dict[str, Any],
                             train_seed: int, eval_seed: int, suite: str,
                             eval_config: EvaluationConfig, model_name: str) -> Optional[SeedResult]:
        """Evaluate a single model with specific seeds using REAL unified evaluator."""
        
        try:
            print(f"🔬 [REAL EVAL] Running actual evaluation for {model_variant} (train_seed={train_seed}, eval_seed={eval_seed})")
            
            # Check if evaluation components are available
            if not EVALUATOR_AVAILABLE:
                print(f"❌ UnifiedBiasEvaluator not available, skipping evaluation")
                return None
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import yaml
            import torch
            
            
            # Set evaluation seed for reproducibility (AFTER importing torch)
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
            random.seed(eval_seed)
            
            # Load model and tokenizer for this variant
            start_time = time.time()
            
            # Get model path from variant
            model_path = self._get_model_path_for_variant(model_variant, model_name, model_info)
            if not model_path:
                print(f"❌ No model path found for {model_variant}")
                return None
            
            print(f"   📁 Loading model from: {model_path}")
            
            # Load model and tokenizer (this should take substantial time for real models)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True
                )
                
                # Apply model-specific interventions
                if model_variant == "fairsteer":
                    print("   🎯 Applying FairSteer steering vectors...")
                    model = self._apply_fairsteer_intervention(model, tokenizer)
                elif model_variant == "sycophancy":
                    print("   🔧 Loading sycophancy-specific model adaptations...")
                    # Sycophancy model is already loaded from the fine-tuned path
                elif model_variant == "firm":
                    print("   🛠️ Loading FIRM multi-component model...")
                    from research_status import require_firm_implementation
                    require_firm_implementation()
                
                print(f"   ✅ Model loaded successfully on {device}")
                
            except Exception as e:
                print(f"❌ Failed to load model {model_variant}: {e}")
                return None
            
            # Load dataset configuration
            dataset_config_path = self.unified_dir / "configs" / "datasets.yaml"
            if not dataset_config_path.exists():
                print(f"❌ Dataset config not found: {dataset_config_path}")
                return None
                
            with open(dataset_config_path, 'r') as f:
                dataset_config = yaml.safe_load(f)
            
            from copy import deepcopy
            dataset_config = deepcopy(dataset_config)
            for dataset_name, settings in dataset_config.get("dataset_configs", {}).items():
                settings["seed"] = eval_seed
                settings["sample_size"] = eval_config.dataset_sample_sizes.get(
                    dataset_name, eval_config.dataset_sample_sizes.get("default", settings.get("sample_size")))
            dataset_config.setdefault("integration", {})["skip_failed_datasets"] = False

            # Create REAL evaluator and run evaluation
            print(f"   🧮 Initializing unified evaluator...")
            evaluator = UnifiedBiasEvaluator(dataset_config, str(self.base_dir))
            
            # Run REAL comprehensive evaluation
            print(f"   🚀 Starting real evaluation on {suite} suite...")
            evaluation_results = evaluator.run_comprehensive_evaluation(
                model, tokenizer, suite_name=suite,
                output_dir=str(self.current_run_dir / "raw" / f"{model_variant}_train{train_seed}_eval{eval_seed}")
            )
            
            evaluation_time = time.time() - start_time
            
            # Extract REAL dataset-specific results (NOT aggregated!)
            dataset_results = {}
            
            if evaluation_results and "dataset_results" in evaluation_results:
                dataset_data = evaluation_results["dataset_results"]
                
                for dataset_name, dataset_info in dataset_data.items():
                    metrics = dataset_info.get("metrics", {})
                    
                    # Extract the PRIMARY metric for each dataset (not aggregated!)
                    if dataset_name == "CrowsPairs":
                        dataset_results[dataset_name] = metrics["crows_pairs_bias_score"]
                    elif dataset_name == "StereoSet":
                        dataset_results[dataset_name] = metrics["stereoset_bias_score"]
                    elif dataset_name == "WinoBias":
                        dataset_results[dataset_name] = metrics["winobias_accuracy"]
                    elif dataset_name == "TruthfulQA":
                        dataset_results[dataset_name] = metrics["truthfulqa_truthful_pct"]
                    elif dataset_name == "BBQ":
                        dataset_results[dataset_name] = metrics["bbq_accuracy"]
                    elif dataset_name == "SEAT":
                        dataset_results[dataset_name] = metrics["seat_avg_effect_size"]
                    elif dataset_name == "BOLD":
                        dataset_results[dataset_name] = metrics["bold_sentiment_bias"]
                    else:
                        # Use first available metric as primary
                        if metrics:
                            primary_metric = list(metrics.keys())[0]
                            dataset_results[dataset_name] = metrics[primary_metric]
            
            if not dataset_results:
                raise RuntimeError("Evaluation returned no dataset metrics")
            if not all(isinstance(value, (int, float)) and np.isfinite(value)
                       for value in dataset_results.values()):
                raise RuntimeError("Evaluation returned invalid/nonfinite metrics")
            # Bias scores, accuracy, percentages and effect sizes have no common scale.
            overall_score = None
            print(f"Dataset metrics for {model_variant}: {dataset_results}")

            # Clean up GPU memory
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return SeedResult(
                training_seed=train_seed,
                evaluation_seed=eval_seed,
                model_variant=model_variant,
                dataset_results=dataset_results,  # REAL per-dataset results
                overall_bias_score=overall_score,  # REAL harmonic mean
                evaluation_time=evaluation_time,
                metadata={"suite": suite, "real_evaluation": True, "total_datasets": len(dataset_results)}
            )
                
        except Exception as e:
            print(f"❌ Evaluation failed for {model_variant}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _aggregate_seed_results(self, model_variant: str, seed_results: List[SeedResult],
                               eval_config: EvaluationConfig) -> AggregatedResults:
        """Aggregate results preserving dataset-specific metrics (NO meaningless overall scores)."""
        
        print(f"📊 [REAL AGGREGATION] Aggregating {len(seed_results)} evaluations for {model_variant}")
        
        # DON'T aggregate into meaningless single bias scores!
        # Keep dataset-specific results separate as they have different meanings
        
        # Collect all dataset-specific results
        dataset_means = {}
        dataset_stds = {}
        
        if seed_results:
            all_datasets = set()
            for seed_result in seed_results:
                all_datasets.update(seed_result.dataset_results.keys())
            
            print(f"   📋 Datasets found: {sorted(all_datasets)}")
            
            for dataset in all_datasets:
                dataset_scores = [
                    r.dataset_results.get(dataset, 0.0) 
                    for r in seed_results 
                    if dataset in r.dataset_results
                ]
                
                if dataset_scores:
                    mean_score = np.mean(dataset_scores)
                    std_score = np.std(dataset_scores, ddof=1) if len(dataset_scores) > 1 else 0.0
                    
                    dataset_means[dataset] = mean_score
                    dataset_stds[dataset] = std_score
                    
                    print(f"   📊 {dataset}: {mean_score:.4f} ± {std_score:.4f}")
        
        # No cross-dataset composite or confidence interval is defined.
        harmonic_mean = harmonic_std = None
        overall_ci = None

        return AggregatedResults(
            model_variant=model_variant,
            mean_bias_score=harmonic_mean,  # Harmonic mean for compatibility
            std_bias_score=harmonic_std,
            confidence_interval=overall_ci,
            dataset_means=dataset_means,      # REAL dataset-specific results
            dataset_stds=dataset_stds,        # REAL dataset-specific uncertainties  
            n_evaluations=len(seed_results),
            seed_results=seed_results,
            statistical_significance={}
        )
    
    def _compute_statistical_significance(self, aggregated_results: Dict[str, AggregatedResults],
                                        eval_config: EvaluationConfig) -> Dict[str, Any]:
        """Compute statistical significance tests between models."""
        return {
            "status": "not_computed",
            "reason": "Use per-dataset paired item outcomes and training-seed clusters; repeated evaluation subsets are not independent replicates."
        }

    def _estimate_total_time(self, eval_config: EvaluationConfig) -> float:
        """Estimate total evaluation time in minutes."""
        # Rough estimates based on typical evaluation times
        training_time_per_seed = 30  # minutes per model training
        evaluation_time_per_seed = 10  # minutes per evaluation
        
        n_training_seeds = len(eval_config.training_seeds)
        n_eval_seeds = len(eval_config.evaluation_seeds)
        n_models = 4
        
        total_training_time = n_training_seeds * 3 * training_time_per_seed  # 3 trainable models
        total_evaluation_time = n_training_seeds * n_eval_seeds * n_models * evaluation_time_per_seed
        
        return total_training_time + total_evaluation_time
    
    def _generate_summary_report(self, aggregated_results, statistical_results, output_dir):
        with open(output_dir / "summary_report.md", "w") as stream:
            stream.write("# Exploratory multi-seed evaluation\n\n")
            stream.write("No publication validity or significance is implied by the run preset.\n\n")
            for name, result in aggregated_results.items():
                stream.write(f"## {name}\n\n{result.n_evaluations} evaluation cells.\n\n")
                for dataset, mean in result.dataset_means.items():
                    stream.write(f"- {dataset}: {mean:.6g}; descriptive cell SD {result.dataset_stds[dataset]:.6g}\n")
                stream.write("\n")
            stream.write(statistical_results["reason"] + "\n")


# Convenience functions for the main pipeline
def create_robust_evaluator(base_dir: Optional[str] = None) -> RobustEvaluationFramework:
    """Create a robust evaluation framework instance."""
    return RobustEvaluationFramework(base_dir)


def run_quick_robust_evaluation(base_config_path: str, model_name: str, suite: str = "comprehensive") -> Dict[str, AggregatedResults]:
    """Run a quick robust evaluation (1 seed each)."""
    framework = RobustEvaluationFramework()
    return framework.run_robust_four_model_evaluation(
        base_config_path, model_name, suite, robustness_level="quick"
    )


def run_standard_robust_evaluation(base_config_path: str, model_name: str, suite: str = "comprehensive") -> Dict[str, AggregatedResults]:
    """Run a standard robust evaluation (3 seeds each)."""
    framework = RobustEvaluationFramework()
    return framework.run_robust_four_model_evaluation(
        base_config_path, model_name, suite, robustness_level="standard"
    )


def run_publication_robust_evaluation(base_config_path: str, model_name: str, suite: str = "comprehensive") -> Dict[str, AggregatedResults]:
    """Run a publication-ready robust evaluation (5 seeds each)."""
    framework = RobustEvaluationFramework()
    return framework.run_robust_four_model_evaluation(
        base_config_path, model_name, suite, robustness_level="publication"
    )
