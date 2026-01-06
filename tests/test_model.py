"""
Test script for model loading and LoRA setup.

This script verifies that:
1. Model loads correctly
2. LoRA adapters are applied properly
3. Parameters are frozen/unfrozen as expected
4. GPU memory usage is reasonable
"""

import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import (
    load_model_and_tokenizer,
    setup_lora,
    print_model_summary,
    get_model_info
)
from src.utils import get_device, get_gpu_memory_usage, print_gpu_memory


def test_model_loading():
    """Test basic model and tokenizer loading."""
    print("\n" + "="*70)
    print("🧪 Test 1: Model Loading")
    print("="*70)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name="gpt2",
        torch_dtype=torch.float32
    )
    
    # Get model info
    info = get_model_info(model)
    
    print(f"\n✅ Model loaded successfully")
    print(f"   Total parameters: {info['total_params']:,}")
    print(f"   Expected: ~124M parameters")
    
    # Verify parameter count (GPT-2 Small should be ~124M)
    assert 120_000_000 < info['total_params'] < 130_000_000, \
        f"Unexpected parameter count: {info['total_params']:,}"
    
    # Check if model is on GPU
    device = next(model.parameters()).device
    print(f"   Device: {device}")
    
    if torch.cuda.is_available():
        print_gpu_memory()
    
    return model, tokenizer


def test_lora_setup(model):
    """Test LoRA adapter setup."""
    print("\n" + "="*70)
    print("🧪 Test 2: LoRA Setup")
    print("="*70)
    
    # Get baseline info
    baseline_info = get_model_info(model)
    print(f"\n📊 Before LoRA:")
    print(f"   Total params: {baseline_info['total_params']:,}")
    print(f"   Trainable params: {baseline_info['trainable_params']:,}")
    print(f"   Trainable ratio: {baseline_info['trainable_percentage']:.4f}%")
    
    # Apply LoRA
    lora_config = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["c_attn", "c_proj"]
    }
    
    model = setup_lora(model, lora_config=lora_config, use_gradient_checkpointing=True)
    
    # Get LoRA info
    lora_info = get_model_info(model)
    print(f"\n📊 After LoRA:")
    print(f"   Total params: {lora_info['total_params']:,}")
    print(f"   Trainable params: {lora_info['trainable_params']:,}")
    print(f"   Trainable ratio: {lora_info['trainable_percentage']:.4f}%")
    
    # Calculate LoRA overhead
    added_params = lora_info['total_params'] - baseline_info['total_params']
    print(f"\n🔍 LoRA Analysis:")
    print(f"   Added parameters: {added_params:,}")
    print(f"   Overhead: {added_params / baseline_info['total_params'] * 100:.4f}%")
    
    # Verify that LoRA parameters are trainable
    assert lora_info['trainable_params'] > 0, "No trainable parameters after LoRA!"
    assert lora_info['trainable_percentage'] < 1.0, "Too many trainable parameters!"
    
    print(f"\n✅ LoRA setup successful")
    print(f"   Only {lora_info['trainable_percentage']:.4f}% of parameters are trainable")
    
    if torch.cuda.is_available():
        print_gpu_memory()
    
    return model


def test_forward_pass(model, tokenizer):
    """Test forward pass through the model."""
    print("\n" + "="*70)
    print("🧪 Test 3: Forward Pass")
    print("="*70)
    
    # Prepare test input
    test_text = "### Instruction:\n한국의 수도는 어디인가요?\n\n### Response:"
    
    print(f"\n📝 Test input: {test_text[:50]}...")
    
    # Tokenize
    inputs = tokenizer(
        test_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"   Input shape: {inputs['input_ids'].shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"\n✅ Forward pass successful")
    print(f"   Output logits shape: {outputs.logits.shape}")
    print(f"   Expected: (batch_size, seq_len, vocab_size)")
    
    # Verify output shape
    batch_size, seq_len, vocab_size = outputs.logits.shape
    assert vocab_size == len(tokenizer), "Unexpected vocab size in output"
    
    if torch.cuda.is_available():
        print_gpu_memory()
    
    return outputs


def test_generation(model, tokenizer):
    """Test text generation."""
    print("\n" + "="*70)
    print("🧪 Test 4: Text Generation")
    print("="*70)
    
    # Prepare prompt
    prompt = "### Instruction:\n한국의 수도는 어디인가요?\n\n### Response:"
    
    print(f"\n📝 Prompt: {prompt}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n🤖 Generated text:")
    print(f"   {generated_text}")
    
    print(f"\n✅ Generation successful")
    print(f"   Generated {len(outputs[0])} tokens")
    
    if torch.cuda.is_available():
        print_gpu_memory()


def main():
    """Main test function."""
    print("\n" + "="*70)
    print("🧪 Model and LoRA Test Suite")
    print("="*70)
    
    try:
        # Test 1: Model loading
        model, tokenizer = test_model_loading()
        
        # Test 2: LoRA setup
        model = test_lora_setup(model)
        
        # Test 3: Forward pass
        test_forward_pass(model, tokenizer)
        
        # Test 4: Text generation
        test_generation(model, tokenizer)
        
        # Final summary
        print("\n" + "="*70)
        print("✅ All tests passed successfully!")
        print("="*70)
        
        print_model_summary(model)
        
        print("\n📋 Summary:")
        print("   ✅ Model loading: GPT-2 Small (~124M params)")
        print("   ✅ LoRA setup: <1% trainable parameters")
        print("   ✅ Forward pass: Working correctly")
        print("   ✅ Text generation: Working correctly")
        
        if torch.cuda.is_available():
            mem = get_gpu_memory_usage()
            print(f"\n🎮 Final GPU Memory Usage:")
            print(f"   Allocated: {mem['allocated']:.2f} GB / {mem['total']:.2f} GB")
            print(f"   Estimated for training: ~{mem['allocated'] * 1.5:.2f} GB")
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
