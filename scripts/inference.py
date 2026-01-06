"""
Inference script for testing the fine-tuned model.

Usage:
    python scripts/inference.py --prompt "Your prompt here"
"""

import sys
import os
import argparse
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import load_model_and_tokenizer, load_lora_weights
from src.utils import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with LoRA model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt2",
        help="Base model name or path"
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Path to LoRA weights (optional, for fine-tuned model)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="### Instruction:\n한국의 수도는 어디인가요?\n\n### Response:",
        help="Input prompt"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🤖 LoRA Inference Script")
    print("="*70)
    
    # Set seed
    set_seed(args.seed)
    
    # Load model and tokenizer
    print("\n📥 Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.base_model,
        torch_dtype=torch.float32
    )
    
    # Load LoRA weights if specified
    if args.lora_weights:
        print(f"\n🔧 Loading LoRA weights from {args.lora_weights}...")
        model = load_lora_weights(model, args.lora_weights)
        print("✅ LoRA weights loaded")
    else:
        print("\n⚠️  No LoRA weights specified, using base model")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"✅ Model ready on {device}")
    
    # Prepare prompt
    print("\n" + "="*70)
    print("📝 Input Prompt:")
    print("="*70)
    print(args.prompt)
    print("="*70)
    
    # Tokenize
    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    print("\n🔄 Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and print results
    print("\n" + "="*70)
    print("🤖 Generated Output:")
    print("="*70)
    
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        
        if args.num_return_sequences > 1:
            print(f"\n--- Sequence {i+1} ---")
        
        print(generated_text)
        
        if args.num_return_sequences > 1:
            print("-" * 70)
    
    print("="*70 + "\n")
    
    # Print generation info
    print("📊 Generation Info:")
    print(f"   Input tokens: {inputs['input_ids'].shape[1]}")
    print(f"   Output tokens: {outputs.shape[1]}")
    print(f"   Generated tokens: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Top-p: {args.top_p}")
    print(f"   Top-k: {args.top_k}")
    print()


if __name__ == "__main__":
    main()
