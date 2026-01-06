"""
Evaluation script to calculate perplexity and other metrics.

This script evaluates both baseline and fine-tuned models on the test set.
"""

import sys
import os
import torch
import json
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import load_model_and_tokenizer, load_lora_weights
from src.dataset import InstructionDataset
from src.utils import set_seed
from torch.utils.data import DataLoader


def calculate_perplexity(model, dataloader, device):
    """Calculate perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating perplexity"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Count non-padding tokens
            labels = batch['labels']
            num_tokens = (labels != -100).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = np.exp(avg_loss)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens
    }


def main():
    """Main evaluation function."""
    print("\n" + "="*80)
    print("📊 Model Evaluation: Baseline vs Fine-tuned")
    print("="*80)
    
    # Set seed
    set_seed(42)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    # Load test dataset
    print("\n📚 Loading test dataset...")
    _, tokenizer = load_model_and_tokenizer("gpt2", torch_dtype=torch.float32)
    
    test_dataset = InstructionDataset(
        data_path="data/processed/test.json",
        tokenizer=tokenizer,
        max_length=512
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
    
    # Evaluate baseline model
    print("\n" + "="*80)
    print("🔵 Evaluating Baseline Model (GPT-2 without LoRA)")
    print("="*80)
    
    baseline_model, _ = load_model_and_tokenizer("gpt2", torch_dtype=torch.float32)
    baseline_model.to(device)
    
    baseline_metrics = calculate_perplexity(baseline_model, test_loader, device)
    
    print(f"\n📊 Baseline Results:")
    print(f"   Loss: {baseline_metrics['loss']:.4f}")
    print(f"   Perplexity: {baseline_metrics['perplexity']:.4f}")
    print(f"   Total tokens: {baseline_metrics['total_tokens']:,}")
    
    # Clean up
    del baseline_model
    torch.cuda.empty_cache()
    
    # Evaluate fine-tuned model
    print("\n" + "="*80)
    print("🟢 Evaluating Fine-tuned Model (GPT-2 + LoRA)")
    print("="*80)
    
    finetuned_model, _ = load_model_and_tokenizer("gpt2", torch_dtype=torch.float32)
    finetuned_model = load_lora_weights(finetuned_model, "outputs/checkpoints/final")
    finetuned_model.to(device)
    
    finetuned_metrics = calculate_perplexity(finetuned_model, test_loader, device)
    
    print(f"\n📊 Fine-tuned Results:")
    print(f"   Loss: {finetuned_metrics['loss']:.4f}")
    print(f"   Perplexity: {finetuned_metrics['perplexity']:.4f}")
    print(f"   Total tokens: {finetuned_metrics['total_tokens']:,}")
    
    # Compare results
    print("\n" + "="*80)
    print("📈 Comparison Summary")
    print("="*80)
    
    loss_improvement = (baseline_metrics['loss'] - finetuned_metrics['loss']) / baseline_metrics['loss'] * 100
    ppl_improvement = (baseline_metrics['perplexity'] - finetuned_metrics['perplexity']) / baseline_metrics['perplexity'] * 100
    
    print(f"\n{'Metric':<20} {'Baseline':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 80)
    print(f"{'Loss':<20} {baseline_metrics['loss']:<15.4f} {finetuned_metrics['loss']:<15.4f} {loss_improvement:>13.2f}%")
    print(f"{'Perplexity':<20} {baseline_metrics['perplexity']:<15.4f} {finetuned_metrics['perplexity']:<15.4f} {ppl_improvement:>13.2f}%")
    
    # Save results
    results = {
        'baseline': {
            'loss': float(baseline_metrics['loss']),
            'perplexity': float(baseline_metrics['perplexity']),
            'total_tokens': int(baseline_metrics['total_tokens'])
        },
        'finetuned': {
            'loss': float(finetuned_metrics['loss']),
            'perplexity': float(finetuned_metrics['perplexity']),
            'total_tokens': int(finetuned_metrics['total_tokens'])
        },
        'improvement': {
            'loss_reduction': float(loss_improvement),
            'perplexity_reduction': float(ppl_improvement)
        }
    }
    
    output_file = "outputs/eval/evaluation_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Conclusion
    print("\n" + "="*80)
    print("🎯 Conclusion")
    print("="*80)
    
    if loss_improvement > 0:
        print(f"✅ Fine-tuning IMPROVED the model!")
        print(f"   - Loss reduced by {loss_improvement:.2f}%")
        print(f"   - Perplexity reduced by {ppl_improvement:.2f}%")
    else:
        print(f"⚠️  Fine-tuning did not improve the model on this test set")
        print(f"   - This may be due to limited training data or overfitting")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
