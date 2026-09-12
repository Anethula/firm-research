#!/usr/bin/env python3
"""Check an experiment's environment before downloading model weights."""
import argparse
import importlib.metadata
import importlib.util
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def check_storage(path, minimum_gb):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path) as stream:
        stream.write(b"preflight")
        stream.flush()
    free = shutil.disk_usage(path).free / 1024**3
    if free < minimum_gb:
        raise RuntimeError(f"{path}: {free:.1f} GiB free; require {minimum_gb:g} GiB")
    return f"{path}: {free:.1f} GiB free"


def run_preflight(config_path, model_name=None, output_dir=None, min_free_gb=20,
                  check_hub=True, require_sycophancy=False):
    checks = []

    def check(name, action):
        try:
            detail = action()
            checks.append({"check": name, "ok": True, "detail": str(detail)})
        except Exception as exc:
            checks.append({"check": name, "ok": False, "detail": str(exc)})

    def python_check():
        if not (3, 10) <= sys.version_info[:2] < (3, 13):
            raise RuntimeError("Use Python 3.11 (supported setup range: 3.10–3.12)")
        return sys.version.split()[0]

    check("python", python_check)
    for module, package in [("torch", "torch"), ("transformers", "transformers"),
                            ("peft", "peft"), ("yaml", "PyYAML"),
                            ("numpy", "numpy"), ("scipy", "scipy"),
                            ("pandas", "pandas"), ("sklearn", "scikit-learn"),
                            ("huggingface_hub", "huggingface-hub")]:
        def dependency(module=module, package=package):
            if importlib.util.find_spec(module) is None:
                raise RuntimeError(f"Missing dependency: {package}")
            importlib.import_module(module)
            return importlib.metadata.version(package)
        check(package, dependency)

    config = {}
    def read_config():
        import yaml
        config.update(yaml.safe_load(Path(config_path).read_text()))
        configured = config["model"]["name"]
        if model_name and model_name != configured:
            raise ValueError(f"CLI model {model_name!r} differs from config {configured!r}")
        return configured
    check("model_config", read_config)

    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    hub_cache = os.environ.get("HF_HUB_CACHE", str(hf_home / "hub"))
    check("model_cache", lambda: check_storage(hub_cache, min_free_gb))
    check("xet_cache", lambda: check_storage(os.environ.get("HF_XET_CACHE", str(hf_home / "xet")), min_free_gb))
    check("outputs", lambda: check_storage(output_dir or ROOT / "unified_pipeline/robust_evaluation_results", min_free_gb))
    check("temporary_files", lambda: check_storage(tempfile.gettempdir(), 1))

    def gpu_check():
        import torch
        if torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)}; {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB VRAM"
        return "CUDA unavailable; CPU smoke tests only, large-model training not validated"
    check("device", gpu_check)

    if require_sycophancy:
        def external_repo():
            repo = ROOT / "sycophancy-interpretability"
            required = [repo / "path_patching/path_patching_hf.py", repo / "pinpoint_tuning/train.py"]
            missing = [str(p) for p in required if not p.is_file()]
            if missing:
                raise FileNotFoundError("External implementation missing: " + ", ".join(missing))
            return str(repo)
        check("sycophancy_repository", external_repo)

    if check_hub and config.get("model", {}).get("name"):
        def model_access():
            from transformers import AutoConfig, AutoTokenizer
            name = config["model"]["name"]
            try:
                AutoConfig.from_pretrained(name, trust_remote_code=False)
                AutoTokenizer.from_pretrained(name, trust_remote_code=False)
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot access config/tokenizer for {name}. For gated models, accept "
                    "the license and run `hf auth login` in this environment. "
                    "Also check cache space and network access. " + str(exc)
                ) from exc
            return f"{name}: config/tokenizer accessible; weight download not tested"
        check("model_access", model_access)
    from research_status import FIRM_BLOCKERS
    return {"environment_ok": all(c["ok"] for c in checks), "checks": checks,
            "firm_research_ready": False, "firm_blockers": FIRM_BLOCKERS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--min-free-gb", type=float, default=20)
    parser.add_argument("--offline", action="store_true", help="Skip model access check")
    parser.add_argument("--require-sycophancy", action="store_true")
    args = parser.parse_args()
    report = run_preflight(args.model_config, args.model_name, args.output_dir,
                           args.min_free_gb, not args.offline, args.require_sycophancy)
    print(json.dumps(report, indent=2))
    return 0 if report["environment_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
