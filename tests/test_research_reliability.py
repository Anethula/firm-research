"""CPU-only regression tests; no model downloads or research results are generated."""
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "unified_pipeline"))

from datasets.base_loader import BaseDatasetLoader
from model_variant_loader import ModelVariantLoader
from preflight import check_storage
from steer.simple_fairsteer_wrapper import SimpleFairSteerWrapper
from utils.model_compatibility import ModelCompatibilityHandler


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.config = SimpleNamespace(hidden_size=3)

    def forward(self, states):
        for layer in self.model.layers:
            states = layer(states)
        return states

    def generate(self, states):
        return self(states)


@pytest.mark.parametrize("key", [1, "1"])
def test_steering_selects_exact_layer_vector_and_applies_in_generation(key):
    model = ToyModel()
    wrapper = SimpleFairSteerWrapper(model, {0: [99., 99., 99.], key: [1., 2., 3.]}, 1, 0.5)
    states = torch.zeros(1, 2, 3, dtype=torch.float64)
    expected = torch.tensor([[[0.5, 1., 1.5], [0.5, 1., 1.5]]], dtype=states.dtype)
    assert torch.equal(wrapper(states), expected)
    assert torch.equal(wrapper.generate(states), expected)
    assert wrapper(states).dtype == states.dtype
    wrapper._remove_hook()
    assert torch.equal(model(states), states)


@pytest.mark.parametrize("vectors,layer", [({0: [1., 2., 3.]}, 1), ({2: [1., 2., 3.]}, 2), ({0: [1.]}, 0)])
def test_invalid_steering_cannot_silently_evaluate_baseline(vectors, layer):
    with pytest.raises(ValueError):
        SimpleFairSteerWrapper(ToyModel(), vectors, layer)


def test_sampling_honors_seed_without_resetting_global_rng():
    loader = SimpleNamespace(config={"seed": 100})
    values = list(range(100))
    state = random.getstate()
    first = BaseDatasetLoader._sample_data(loader, values, 12)
    assert random.getstate() == state
    assert first == BaseDatasetLoader._sample_data(loader, values, 12)
    loader.config["seed"] = 200
    assert first != BaseDatasetLoader._sample_data(loader, values, 12)


def test_firm_cannot_be_relabelled_baseline():
    loader = ModelVariantLoader(ToyModel(), None, {"model_name": "test/model", "model_variant": "firm"})
    with pytest.raises(NotImplementedError, match="optimizer"):
        loader.load_variant_model()


def test_sycophancy_requires_exact_checkpoint(tmp_path):
    loader = ModelVariantLoader(ToyModel(), None, {"model_name": "test/model"})
    with pytest.raises(ValueError, match="exact"):
        loader._find_sycophancy_path()
    (tmp_path / "adapter_config.json").write_text("{}")
    loader.config["sycophancy"] = {"model_path": str(tmp_path)}
    assert loader._find_sycophancy_path() == str(tmp_path)


def test_empty_external_repo_rejected_before_workspace_write(tmp_path):
    from train.sycophancy_pipeline import SycophancyPipelineManager
    with pytest.raises(FileNotFoundError, match="External"):
        SycophancyPipelineManager(str(tmp_path))


def test_bbq_ambiguity_metadata_does_not_supply_the_answer():
    handler = object.__new__(ModelCompatibilityHandler)
    handler.supports_generation = lambda: False
    handler._score_multiple_choice_options = Mock(return_value=0)
    sample = {"text": "Question", "metadata": {"choices": ["person A", "Unknown", "person B"], "context_condition": "ambig"}}
    assert handler._evaluate_bbq_question(sample) == 0
    handler._score_multiple_choice_options.assert_called_once()


def test_prediction_failure_is_not_neutral_or_first_choice():
    handler = object.__new__(ModelCompatibilityHandler)
    handler.model_type = "test"
    handler._evaluate_bbq_question = Mock(side_effect=RuntimeError("inference failed"))
    with pytest.raises(RuntimeError, match="inference failed"):
        handler.evaluate_bias_sample({}, "bbq_qa")


@pytest.mark.parametrize("response", ["I cannot tell", "nurse or doctor", 50])
def test_coreference_parser_never_fills_in_gold_answer(response):
    handler = object.__new__(ModelCompatibilityHandler)
    assert handler._parse_profession_response(response, "doctor", ["nurse", "doctor"]) == "__unparsed__"


def test_setup_stops_after_failed_dependency_install(monkeypatch):
    spec = importlib.util.spec_from_file_location("firm_setup", ROOT / "setup.py")
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    monkeypatch.setattr(setup, "check_python_version", lambda: True)
    monkeypatch.setattr(setup, "check_cuda", lambda: False)
    monkeypatch.setattr(setup, "setup_environment", lambda: None)
    install = Mock(return_value=False)
    monkeypatch.setattr(setup, "run_command", install)
    with pytest.raises(SystemExit) as exc:
        setup.main()
    assert exc.value.code == 1
    assert install.call_count == 1
    assert sys.executable in install.call_args.args[0]


def test_disk_preflight_checks_actual_target(tmp_path, monkeypatch):
    import preflight
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match="GiB free"):
        check_storage(tmp_path, 1)


def test_legacy_fairsteer_cli_accepts_config_and_exits_on_bad_args():
    result = subprocess.run([sys.executable, str(ROOT / "fairsteer_debiasing.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "static contrastive" in result.stdout


def test_robust_training_failure_saves_failure_and_never_claims_complete(tmp_path, monkeypatch):
    import research_status
    from robust_evaluation_framework import RobustEvaluationFramework, EvaluationConfig
    # Exercise the runner's failure handling independently of the explicit FIRM gate.
    monkeypatch.setattr(research_status, "require_firm_implementation", lambda: None)
    framework = RobustEvaluationFramework(str(tmp_path))
    monkeypatch.setattr(framework, "_train_all_models_with_seed", lambda *args: {
        name: {"success": name == "baseline"} for name in ("baseline", "fairsteer", "sycophancy", "firm")})
    with pytest.raises(RuntimeError, match="Training failed"):
        framework.run_robust_four_model_evaluation("unused", "test/model", "quick_evaluation", "custom",
                                                   EvaluationConfig([42], [100], {"default": 2}))
    failures = list(tmp_path.rglob("failure.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())["status"] == "failed"
    assert not list(tmp_path.rglob("robust_evaluation_results.json"))


def test_static_dsv_points_from_biased_to_neutral():
    from steer.model_agnostic_fairsteer import ModelAgnosticFairSteer
    computer = object.__new__(ModelAgnosticFairSteer)
    computer.model_name = "toy"
    computer.config = SimpleNamespace(get_optimal_layers=lambda: [0])
    computer._get_layer_activations = lambda text, layer: torch.tensor([[1., 2., 3.]]) if text == "biased" else torch.tensor([[4., 6., 8.]])
    result = computer.compute_steering_vectors([("biased", "neutral")])
    assert torch.equal(result[0], torch.tensor([[3., 4., 5.]]))


def test_steering_changes_real_tiny_transformer_logits_and_zero_strength_is_noop():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=2, num_attention_heads=2,
                                        num_key_value_heads=2, pad_token_id=0)).eval()
    tokens = torch.tensor([[1, 5, 8]])
    with torch.no_grad():
        baseline = model(tokens).logits.clone()
        wrapper = SimpleFairSteerWrapper(model, {0: torch.arange(16).float() / 100}, 0, 0.0)
        assert torch.equal(wrapper(tokens).logits, baseline)
        wrapper.intervention_strength = 1.0
        steered = wrapper(tokens).logits
        assert torch.isfinite(steered).all()
        assert not torch.allclose(steered, baseline)
        generated = wrapper.generate(tokens, max_new_tokens=2, do_sample=False,
                                     attention_mask=torch.ones_like(tokens), eos_token_id=None)
        assert generated.shape == (1, 5)
        wrapper._remove_hook()
        assert torch.equal(model(tokens).logits, baseline)


def make_evaluator():
    from eval.unified_evaluator import UnifiedBiasEvaluator
    evaluator = object.__new__(UnifiedBiasEvaluator)
    evaluator.loaded_datasets = {"toy": SimpleNamespace(compute_metrics=lambda p, t: {"accuracy": 1.0})}
    evaluator.dataset_configs = {"toy": {}}
    evaluator.integration_config = {"skip_failed_datasets": False}
    evaluator.model_handler = object()
    evaluator._evaluate_sample_with_timeout = lambda *args, **kwargs: "prediction"
    return evaluator


def test_all_item_predictions_and_ids_are_retained():
    evaluator = make_evaluator()
    samples = [{"text": str(i), "target": "answer"} for i in range(12)]
    result = evaluator.evaluate_model_on_dataset(None, None, "toy", samples)
    assert len(result["predictions"]) == len(result["targets"]) == len(result["sample_ids"]) == 12
    assert len(set(result["sample_ids"])) == 12


def test_empty_prediction_set_is_not_a_zero_bias_result():
    evaluator = make_evaluator()
    evaluator._evaluate_sample_with_timeout = Mock(side_effect=RuntimeError("model failed"))
    with pytest.raises(RuntimeError, match="no successful"):
        evaluator.evaluate_model_on_dataset(None, None, "toy", [{"text": "sample"}])
