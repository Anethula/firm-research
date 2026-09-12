#!/usr/bin/env python3
"""Generate static contrastive steering artifacts for a configured model.

This is an exploratory DSV baseline, not a reproduction of dynamic FairSteer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "steering_vectors"))
    parser.add_argument("--pairs", help="JSON list of [biased_text, neutral_text] training pairs")
    parser.add_argument("--train-only", action="store_true", help="Compatibility flag; this command only computes vectors")
    args = parser.parse_args()
    import yaml
    import numpy as np
    import torch
    from steer.model_agnostic_fairsteer import ModelAgnosticFairSteer, generate_bias_pairs

    config = yaml.safe_load(Path(args.config).read_text())
    name = config["model"]["name"]
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pairs = json.loads(Path(args.pairs).read_text()) if args.pairs else generate_bias_pairs()
    if not pairs or any(not isinstance(pair, (list, tuple)) or len(pair) != 2 or
                        any(not isinstance(text, str) or not text.strip() for text in pair)
                        for pair in pairs):
        raise ValueError("Expected nonempty [biased_text, neutral_text] pairs")
    if not args.pairs:
        print("EXPLORATORY ONLY: using built-in toy pairs; supply --pairs for a curated training split")
    computer = ModelAgnosticFairSteer(name, device=config["model"].get("device", "auto"))
    vectors = computer.compute_steering_vectors(pairs)
    if not vectors:
        raise RuntimeError("No steering vectors were computed")
    computer.artifact_metadata = {
        "seed": seed, "method": "static_contrastive_dsv",
        "pair_count": len(pairs),
        "pairs_sha256": hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode()).hexdigest(),
        "toy_pairs": args.pairs is None,
        "layer_selection": "largest_vector_norm_in_configured_range",
        "publication_validated": False,
    }
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    safe = name.replace("/", "_").replace("-", "_").lower()
    path = output / f"fairsteer_{safe}.pkl"
    computer.save_steering_vectors(str(path))
    print(f"Vector artifact: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
