#!/usr/bin/env python3
"""
PyTorch Compilation Fix for Unified Pipeline

This module provides comprehensive PyTorch compilation suppression
to prevent recompilation limit errors during model training.
"""

import os
import torch
import warnings
from typing import Any

def apply_pytorch_compilation_fixes():
    """
    Apply comprehensive PyTorch compilation fixes to prevent recompilation errors.
    Should be called at the very beginning of any script that uses PyTorch models.
    """
    print("🔧 Applying PyTorch compilation fixes...")
    
    # 1. Set environment variables BEFORE any torch operations
    os.environ['TORCH_DYNAMO_DISABLE'] = '1'
    os.environ['TORCH_COMPILE_DEBUG'] = '0'
    os.environ['TORCH_COMPILE_CACHE_SIZE_LIMIT'] = '0'
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    os.environ['PYTORCH_DISABLE_AUTOGRAD_FUNCTION_CACHE'] = '1'
    
    # 2. Increase recompile limits
    try:
        for key, value in {"cache_size_limit": 128,
                           "accumulated_cache_size_limit": 256,
                           "recompile_limit": 50, "suppress_errors": True,
                           "disable": True}.items():
            if hasattr(torch._dynamo.config, key):
                setattr(torch._dynamo.config, key, value)
    except Exception as e:
        print(f"  ⚠ Could not configure torch._dynamo: {e}")
    
    # 3. Reset any existing dynamo state
    try:
        torch._dynamo.reset()
        print("  ✓ Reset torch._dynamo state")
    except Exception as e:
        print(f"  ⚠ Could not reset torch._dynamo: {e}")
    
    # 4. Disable torch.compile globally
    try:
        if hasattr(torch, 'compiler'):
            torch.compiler.disable()
            print("  ✓ Disabled torch.compiler")
    except Exception as e:
        print(f"  ⚠ Could not disable torch.compiler: {e}")
    
    # 5. Set torch compile mode to None/disable
    try:
        if hasattr(torch, '_C') and hasattr(torch._C, '_set_compile_mode'):
            torch._C._set_compile_mode(False)
    except:
        pass
    
    # 6. Suppress all compilation-related warnings
    warnings.filterwarnings('ignore', message='.*recompile_limit.*')
    warnings.filterwarnings('ignore', message='.*dynamo.*')
    warnings.filterwarnings('ignore', message='.*torch.compile.*')
    warnings.filterwarnings('ignore', module='torch._dynamo')
    warnings.filterwarnings('ignore', module='torch.fx')
    
    print("  ✓ Applied compilation warning suppressions")
    print("🔧 PyTorch compilation fixes applied successfully!")

def disable_model_compilation(model: Any) -> Any:
    """
    Disable compilation for a specific model instance.
    
    Args:
        model: PyTorch model instance
        
    Returns:
        Model with compilation disabled
    """
    try:
        # Disable compilation on specific model methods
        if hasattr(model, 'forward'):
            if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'disable'):
                model.forward = torch.compiler.disable(model.forward)
        
        if hasattr(model, 'generate'):
            if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'disable'):
                model.generate = torch.compiler.disable(model.generate)
                
        # Set model to not use compilation
        if hasattr(model, '_dynamo_compile'):
            model._dynamo_compile = False
            
        print(f"  ✓ Disabled compilation for {type(model).__name__}")
    except Exception as e:
        print(f"  ⚠ Could not disable compilation for model: {e}")
    
    return model

def create_compilation_safe_forward_hook():
    """
    Create a forward hook that prevents compilation issues.
    
    Returns:
        Hook function that can be registered on modules
    """
    def safe_forward_hook(module, input, output):
        # Ensure tensors are not compiled
        if isinstance(output, torch.Tensor):
            # Detach from computation graph to prevent compilation tracking
            if output.requires_grad:
                output = output.detach().requires_grad_(True)
        elif isinstance(output, (tuple, list)):
            # Handle multiple outputs
            safe_outputs = []
            for item in output:
                if isinstance(item, torch.Tensor) and item.requires_grad:
                    safe_outputs.append(item.detach().requires_grad_(True))
                else:
                    safe_outputs.append(item)
            output = type(output)(safe_outputs)
        
        return output
    
    return safe_forward_hook

# Apply fixes immediately when module is imported
if __name__ != "__main__":
    apply_pytorch_compilation_fixes()
