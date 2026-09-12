#!/usr/bin/env python3
"""
Model Variant Loader for Unified Pipeline

Handles loading different model variants:
- baseline: Original model without modifications
- fairsteer: Model with FairSteer steering vectors applied
- sycophancy: Model with pinpoint tuning fine-tuning
- firm: Model with combined FIRM interventions
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent))

warnings.filterwarnings('ignore')


class ModelVariantLoader:
    """
    Loads different variants of models with appropriate interventions applied.
    """
    
    def __init__(self, base_model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: Dict[str, Any]):
        """
        Initialize the model variant loader.
        
        Args:
            base_model: The base model to modify
            tokenizer: The tokenizer for the model
            config: Configuration dictionary containing variant settings
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config
        # Normalize model name to fix character encoding issues
        model_name = config.get('model_name') or config.get('model', {}).get('name', '')
        if not model_name:
            raise ValueError("Variant loading requires model_name or model.name in the configuration")
        model_name = model_name.replace('–', '-').replace('—', '-')  # Replace em-dashes with hyphens
        model_name = model_name.replace('\u2013', '-').replace('\u2014', '-')  # Unicode em-dashes
        model_name = model_name.strip('"').strip("'").strip('\u201c').strip('\u201d')  # Remove quotes
        self.model_name = model_name
        self.model_variant = config.get('model_variant', 'baseline')
    
    def load_variant_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the specified model variant.
        
        Returns:
            Tuple of (modified_model, tokenizer)
        """
        if self.model_variant == 'baseline':
            return self.base_model, self.tokenizer
        elif self.model_variant == 'fairsteer':
            return self._load_fairsteer_variant()
        elif self.model_variant == 'sycophancy':
            return self._load_sycophancy_variant()
        elif self.model_variant == 'firm':
            return self._load_firm_variant()
        else:
            raise ValueError(f"Unknown model variant: {self.model_variant}")
    
    def _load_fairsteer_variant(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load FairSteer variant with steering vectors."""
        print("   🎯 Applying FairSteer steering vectors...")
        
        # Look for existing steering vectors
        fairsteer_path = self._find_fairsteer_path()
        if fairsteer_path is None:
            raise RuntimeError(
                "FairSteer was requested but no usable steering vectors could be found or generated"
            )
        
        try:
            with open(fairsteer_path, 'rb') as f:
                fairsteer_data = pickle.load(f)
            
            if fairsteer_data.get('model_name') != self.model_name:
                raise ValueError("Steering artifact model_name does not match the requested model")
            if fairsteer_data.get('vector_convention') != 'neutral_minus_biased':
                raise ValueError("Legacy/unknown steering direction; regenerate vectors using train_fairsteer.py")
            steering_vectors = fairsteer_data.get('steering_vectors', {})
            optimal_layer = fairsteer_data.get('optimal_layer', 15)
            
            if not steering_vectors:
                raise RuntimeError(f"FairSteer vector file is empty: {fairsteer_path}")
            
            # Create a simple steering wrapper for FairSteer
            from steer.simple_fairsteer_wrapper import SimpleFairSteerWrapper
            das_model = SimpleFairSteerWrapper(
                model=self.base_model,
                steering_vectors=steering_vectors,
                optimal_layer=optimal_layer,
                intervention_strength=self.config.get('fairsteer', {}).get('intervention_strength', 1.0)
            )
            
            print(f"   ✓ FairSteer applied at layer {optimal_layer}")
            return das_model, self.tokenizer
            
        except Exception as e:
            raise RuntimeError(f"Failed to load FairSteer variant: {e}") from e
    
    def _load_sycophancy_variant(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load sycophancy variant with pinpoint tuning."""
        print("   🎯 Applying sycophancy pinpoint tuning...")
        
        # Look for sycophancy fine-tuned model
        sycophancy_path = self._find_sycophancy_path()
        if sycophancy_path is None:
            raise RuntimeError("Sycophancy variant was requested but no trained model was found")
        
        try:
            # Load the fine-tuned model
            sycophancy_model = AutoModelForCausalLM.from_pretrained(
                sycophancy_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            
            print(f"   ✓ Sycophancy model loaded from {sycophancy_path}")
            return sycophancy_model, self.tokenizer
            
        except Exception as e:
            raise RuntimeError(f"Failed to load sycophancy variant: {e}") from e
    
    def _load_firm_variant(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        from research_status import require_firm_implementation
        require_firm_implementation()
    
    def _find_fairsteer_path(self) -> Optional[str]:
        """Find model-agnostic FairSteer steering vectors file."""
        explicit = self.config.get('fairsteer', {}).get('vectors_path')
        if explicit:
            path = Path(explicit).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Steering vector file not found: {path}")
            return str(path)
        # Generate model-specific filename based on actual model name
        safe_model_name = self.model_name.replace('/', '_').replace('-', '_').lower()
        
        # Primary check: Look for existing model-specific steering vectors
        steering_vectors_dir = Path(__file__).parent / "steering_vectors"
        primary_path = steering_vectors_dir / f"fairsteer_{safe_model_name}.pkl"
        
        if primary_path.exists():
            print(f"   📁 Found model-specific FairSteer vectors: {primary_path}")
            return str(primary_path)
        
        # Secondary check: Look in common locations with various naming conventions
        possible_paths = [
            f"steering_vectors/fairsteer_{safe_model_name}.pkl",
            f"fairsteer_{safe_model_name}.pkl",
            f"fairsteer_{self.model_name.split('/')[-1].lower()}.pkl",
            f"fairsteer_{self.model_name.split('/')[-1].lower().replace('-', '_')}.pkl",
        ]
        
        # Check current directory and parent directories
        for base_path in [Path.cwd(), Path.cwd().parent, Path(__file__).parent.parent]:
            for possible_path in possible_paths:
                full_path = base_path / possible_path
                if full_path.exists():
                    print(f"   📁 Found FairSteer vectors: {full_path}")
                    return str(full_path)
        
        # If no model-specific vectors found, create them using model-agnostic FairSteer
        print(f"   ⚠️  No FairSteer vectors found for {self.model_name}")
        print(f"   🔧 Creating model-specific FairSteer vectors using model-agnostic implementation...")
        
        try:
            # Import and use model-agnostic FairSteer
            sys.path.append(str(Path(__file__).parent))
            from steer.model_agnostic_fairsteer import ModelAgnosticFairSteer, generate_bias_pairs
            
            # Create steering vectors for this specific model
            fairsteer = ModelAgnosticFairSteer(self.model_name, self.base_model, self.tokenizer)
            bias_pairs = generate_bias_pairs()
            steering_vectors = fairsteer.compute_steering_vectors(bias_pairs)
            
            if steering_vectors and hasattr(fairsteer, 'optimal_layer'):
                # Save for future use
                steering_vectors_dir.mkdir(exist_ok=True)
                output_path = steering_vectors_dir / f"fairsteer_{safe_model_name}.pkl"
                fairsteer.save_steering_vectors(str(output_path))
                print(f"   ✅ Created and saved model-specific FairSteer vectors: {output_path}")
                return str(output_path)
            else:
                print(f"   ❌ Failed to generate valid steering vectors for {self.model_name}")
        except Exception as e:
            print(f"   ❌ Failed to create FairSteer vectors: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def _find_sycophancy_path(self) -> Optional[str]:
        path = self.config.get("sycophancy", {}).get("model_path")
        if not path:
            raise ValueError("Set sycophancy.model_path to the exact trained checkpoint; automatic latest-run selection is disabled")
        checkpoint = Path(path).expanduser().resolve()
        if not checkpoint.is_dir() or not any((checkpoint / name).is_file() for name in
                ("adapter_config.json", "config.json")):
            raise ValueError(f"Not a model/adapter checkpoint: {checkpoint}")
        return str(checkpoint)
