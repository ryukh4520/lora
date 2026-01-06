"""
Comparison script for Baseline vs Fine-tuned models.

This script generates outputs from both models and compares them side-by-side.
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import load_model_and_tokenizer, load_lora_weights
from src.utils import set_seed


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7):
    """Generate text from a model."""
    device = next(model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    """Main comparison function."""
    print("\n" + "="*80)
    print("🔍 Baseline vs Fine-tuned Model Comparison")
    print("="*80)
    
    # Set seed
    set_seed(42)
    
    # Test prompts
    test_prompts = [
        "### Instruction:\n한국의 수도는 어디인가요?\n\n### Response:",
        "### Instruction:\n김치의 효능에 대해 설명해주세요.\n\n### Response:",
        "### Instruction:\n서울의 주요 관광지를 추천해주세요.\n\n### Response:",
        "### Instruction:\nPython에서 리스트와 튜플의 차이점은 무엇인가요?\n\n### Response:",
        "### Instruction:\n인공지능이란 무엇인가요?\n\n### Response:"
    ]
    
    # Load baseline model
    print("\n📥 Loading Baseline Model (GPT-2 without LoRA)...")
    baseline_model, tokenizer = load_model_and_tokenizer(
        model_name="gpt2",
        torch_dtype=torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model.to(device)
    print("✅ Baseline model loaded")
    
    # Load fine-tuned model
    print("\n📥 Loading Fine-tuned Model (GPT-2 + LoRA)...")
    finetuned_model, _ = load_model_and_tokenizer(
        model_name="gpt2",
        torch_dtype=torch.float32
    )
    finetuned_model = load_lora_weights(finetuned_model, "outputs/checkpoints/final")
    finetuned_model.to(device)
    print("✅ Fine-tuned model loaded")
    
    # Compare on test prompts
    print("\n" + "="*80)
    print("📊 Comparison Results")
    print("="*80)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/5")
        print(f"{'='*80}")
        
        # Extract instruction
        instruction = prompt.split("### Instruction:\n")[1].split("\n\n### Response:")[0]
        print(f"\n📝 Instruction: {instruction}")
        
        # Generate from baseline
        print(f"\n🔵 Baseline (GPT-2 without LoRA):")
        print("-" * 80)
        baseline_output = generate_text(baseline_model, tokenizer, prompt, max_new_tokens=80)
        # Extract only the response part
        if "### Response:" in baseline_output:
            baseline_response = baseline_output.split("### Response:")[-1].strip()
        else:
            baseline_response = baseline_output
        print(baseline_response[:200])
        
        # Generate from fine-tuned
        print(f"\n🟢 Fine-tuned (GPT-2 + LoRA):")
        print("-" * 80)
        finetuned_output = generate_text(finetuned_model, tokenizer, prompt, max_new_tokens=80)
        # Extract only the response part
        if "### Response:" in finetuned_output:
            finetuned_response = finetuned_output.split("### Response:")[-1].strip()
        else:
            finetuned_response = finetuned_output
        print(finetuned_response[:200])
        
        print("\n" + "-" * 80)
    
    print("\n" + "="*80)
    print("✅ Comparison Complete!")
    print("="*80)
    
    print("\n💡 Observations:")
    print("   - Baseline model: Trained on general English text")
    print("   - Fine-tuned model: Trained on Korean instruction-following data")
    print("   - Expected: Fine-tuned model should better follow Korean instructions")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
