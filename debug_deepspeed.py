# debug_deepspeed.py
# Utility script to diagnose DeepSpeed issues
import os
import sys
import torch
import traceback
from datetime import datetime

def diagnose_deepspeed():
    """Run diagnostics on DeepSpeed installation and report issues"""
    log_file = f"deepspeed_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    with open(log_file, "w") as f:
        f.write(f"DeepSpeed Diagnostics - {datetime.now()}\n")
        f.write("="*80 + "\n\n")
        
        # System info
        f.write("System Information:\n")
        f.write("-"*50 + "\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA version: {torch.version.cuda}\n")
            f.write(f"CUDA device count: {torch.cuda.device_count()}\n")
            f.write(f"Current CUDA device: {torch.cuda.current_device()}\n")
            f.write(f"Device name: {torch.cuda.get_device_name()}\n")
        f.write("\n")
        
        # Check for DeepSpeed
        f.write("DeepSpeed Status:\n")
        f.write("-"*50 + "\n")
        try:
            import deepspeed
            f.write(f"DeepSpeed is installed - version: {deepspeed.__version__}\n")
            
            # Check for required components
            f.write("\nChecking DeepSpeed components:\n")
            try:
                from deepspeed.ops.sparse_attention import SparseSelfAttention
                f.write("✓ SparseSelfAttention module is available\n")
                
                # Test sparse attention initialization
                f.write("\nTesting SparseSelfAttention initialization:\n")
                try:
                    # Simple test configuration
                    sparse_attn_config = {
                        "mode": "fixed", 
                        "block": 16,
                        "different_layout_per_head": True,
                        "num_local_blocks": 4,
                        "num_global_blocks": 1,
                        "attention_dropout": 0.1,
                        "horizontal_global_attention": False,
                        "num_different_global_patterns": 4,
                    }
                    
                    # Try to initialize with a small sequence length
                    sparse_attn = SparseSelfAttention(
                        sparsity_config=sparse_attn_config,
                        max_seq_length=256,
                        attn_mask_mode='add',
                        attention_dropout=0.1
                    )
                    f.write("✓ Successfully initialized SparseSelfAttention\n")
                except Exception as e:
                    f.write(f"✗ Failed to initialize SparseSelfAttention: {str(e)}\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
            except ImportError as e:
                f.write(f"✗ SparseSelfAttention module not available: {str(e)}\n")
                
            # Check for CUDA extensions
            f.write("\nChecking DeepSpeed CUDA extensions:\n")
            try:
                import deepspeed.ops.op_builder as op_builder
                sparse_attn_builder = op_builder.SparseAttnBuilder()
                if sparse_attn_builder.is_compatible():
                    f.write("✓ DeepSpeed SparseAttn CUDA extension is compatible\n")
                else:
                    f.write("✗ DeepSpeed SparseAttn CUDA extension is NOT compatible\n")
                
                # Check available builders
                f.write("\nAvailable DeepSpeed CUDA extensions:\n")
                available_ops = [builder_name for builder_name in dir(op_builder) 
                                if "Builder" in builder_name and not builder_name.startswith("_")]
                for op in available_ops:
                    f.write(f"- {op}\n")
                
            except Exception as e:
                f.write(f"✗ Failed to check CUDA extensions: {str(e)}\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
                
        except ImportError as e:
            f.write(f"✗ DeepSpeed is NOT installed: {str(e)}\n")
            f.write("\nTrying to diagnose why DeepSpeed may not be installed or importable:\n")
            
            # Check if deepspeed is installed via pip
            import subprocess
            try:
                result = subprocess.run(["pip", "list"], capture_output=True, text=True)
                packages = result.stdout
                f.write("\nInstalled packages:\n")
                for line in packages.split('\n'):
                    if "deep" in line.lower():
                        f.write(f"{line}\n")
            except Exception as import_e:
                f.write(f"Could not check pip packages: {str(import_e)}\n")
            
            # Check system path
            f.write("\nPython path:\n")
            for path in sys.path:
                f.write(f"{path}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Diagnostics complete. Please check the output above for any issues.\n")
    
    print(f"DeepSpeed diagnostics completed. Results saved to {log_file}")
    print(f"To view the file, use: cat {log_file}")

if __name__ == "__main__":
    diagnose_deepspeed()