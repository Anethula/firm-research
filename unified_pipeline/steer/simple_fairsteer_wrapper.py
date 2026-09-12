#!/usr/bin/env python3
"""
Simple FairSteer Wrapper for Model Variant Loading

Provides a lightweight wrapper that applies FairSteer steering vectors
during inference without the complexity of the full DAS system.
This is specifically designed for the model variant loader interface.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from transformers import AutoModelForCausalLM


class SimpleFairSteerWrapper(nn.Module):
    """
    Simple wrapper that applies FairSteer steering vectors at a specific layer.
    
    This wrapper intercepts activations at the optimal layer and applies
    the appropriate steering vector based on the intervention strength.
    """
    
    def __init__(self, 
                 model: AutoModelForCausalLM,
                 steering_vectors: Dict[str, torch.Tensor],
                 optimal_layer: int,
                 intervention_strength: float = 1.0):
        """
        Initialize the FairSteer wrapper.
        
        Args:
            model: Base model to wrap
            steering_vectors: Dictionary of steering vectors by layer
            optimal_layer: Layer index to apply steering
            intervention_strength: Scaling factor for steering vectors
        """
        super().__init__()
        self.model = model
        self.steering_vectors = steering_vectors
        self.optimal_layer = optimal_layer
        self.intervention_strength = intervention_strength
        
        # Convert steering vectors to proper format and device
        self._prepare_steering_vectors()
        
        # Install the forward hook
        self.hook_handle = None
        self._install_hook()
    
    def _prepare_steering_vectors(self):
        # Artifact keys may be integers (pickle) or strings (JSON).
        vector = self.steering_vectors.get(self.optimal_layer)
        if vector is None:
            vector = self.steering_vectors.get(str(self.optimal_layer))
        if vector is None:
            raise ValueError(f"No steering vector for layer {self.optimal_layer}")
        vector = torch.as_tensor(vector).detach()
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.ndim != 1 or not torch.isfinite(vector).all():
            raise ValueError("Steering vector must be a finite hidden-size vector")
        expected = self.model.config.hidden_size
        if vector.numel() != expected:
            raise ValueError(f"Steering width {vector.numel()} does not match model hidden size {expected}")
        self.register_buffer("steering_vector", vector)

    def _install_hook(self):
        if self.hook_handle is not None:
            return
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        else:
            raise ValueError("Unsupported decoder layer structure for steering")
        if not 0 <= self.optimal_layer < len(layers):
            raise ValueError(f"Steering layer {self.optimal_layer} outside model with {len(layers)} layers")

        def steering_hook(module, inputs, output):
            states = output[0] if isinstance(output, tuple) else output
            if not isinstance(states, torch.Tensor) or states.ndim != 3:
                raise ValueError("Expected decoder residual output [batch, sequence, hidden]")
            vector = self.steering_vector.to(device=states.device, dtype=states.dtype)
            # Vectors use neutral-minus-biased; add at decoder residual output.
            steered = states + self.intervention_strength * vector.view(1, 1, -1)
            return (steered,) + output[1:] if isinstance(output, tuple) else steered

        self.hook_handle = layers[self.optimal_layer].register_forward_hook(steering_hook)

    def _remove_hook(self):
        """Remove the forward hook."""
        if self.__dict__.get("hook_handle") is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generate method that uses the wrapped model."""
        return self.model.generate(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    def __del__(self):
        """Clean up the hook when the wrapper is destroyed."""
        self._remove_hook()
