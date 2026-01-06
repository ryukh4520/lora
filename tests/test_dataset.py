"""
Test script for tokenizer and dataset.

This script verifies that:
1. Tokenizer loads correctly
2. Dataset loads and processes data properly
3. DataLoader works as expected
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer
from src.dataset import InstructionDataset, create_dataloaders
from src.utils import load_config


def test_tokenizer():
    """Test tokenizer loading and basic functionality."""
    print("\n" + "="*60)
    print("🔤 Testing Tokenizer")
    print("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # GPT-2 doesn't have a pad token by default, so we add one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ Set pad_token to eos_token")
    
    print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Model max length: {tokenizer.model_max_length}")
    print(f"   PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"   EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"   BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    
    # Test tokenization
    test_text = "안녕하세요! 한국어 테스트입니다."
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text)
    
    print(f"\n📝 Test Text: {test_text}")
    print(f"   Tokens: {tokens[:10]}...")
    print(f"   Token IDs: {token_ids[:10]}...")
    print(f"   Number of tokens: {len(tokens)}")
    
    # Test decoding
    decoded = tokenizer.decode(token_ids)
    print(f"   Decoded: {decoded}")
    
    return tokenizer


def test_dataset(tokenizer):
    """Test dataset loading and processing."""
    print("\n" + "="*60)
    print("📚 Testing Dataset")
    print("="*60)
    
    # Load dataset
    train_dataset = InstructionDataset(
        data_path="data/processed/train.json",
        tokenizer=tokenizer,
        max_length=512
    )
    
    val_dataset = InstructionDataset(
        data_path="data/processed/validation.json",
        tokenizer=tokenizer,
        max_length=512
    )
    
    # Test getting an item
    print(f"\n🔍 Testing dataset item retrieval...")
    sample = train_dataset[0]
    
    print(f"   Keys: {list(sample.keys())}")
    print(f"   Input IDs shape: {sample['input_ids'].shape}")
    print(f"   Attention mask shape: {sample['attention_mask'].shape}")
    print(f"   Labels shape: {sample['labels'].shape}")
    
    # Decode the sample
    print(f"\n📄 Sample decoded text:")
    decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    print(f"   {decoded_text[:200]}...")
    
    # Check labels
    non_masked_labels = sample['labels'][sample['labels'] != -100]
    print(f"\n🏷️  Labels info:")
    print(f"   Total tokens: {len(sample['labels'])}")
    print(f"   Non-masked tokens: {len(non_masked_labels)}")
    print(f"   Masked tokens (padding): {(sample['labels'] == -100).sum().item()}")
    
    return train_dataset, val_dataset


def test_dataloader(train_dataset, val_dataset, tokenizer):
    """Test DataLoader creation and iteration."""
    print("\n" + "="*60)
    print("🔄 Testing DataLoader")
    print("="*60)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=2,
        num_workers=0
    )
    
    # Test iteration
    print(f"\n🔍 Testing batch retrieval...")
    batch = next(iter(train_loader))
    
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   Input IDs shape: {batch['input_ids'].shape}")
    print(f"   Attention mask shape: {batch['attention_mask'].shape}")
    print(f"   Labels shape: {batch['labels'].shape}")
    
    # Calculate some statistics
    total_tokens = batch['input_ids'].numel()
    padding_tokens = (batch['input_ids'] == tokenizer.pad_token_id).sum().item()
    
    print(f"\n📊 Batch statistics:")
    print(f"   Total tokens: {total_tokens}")
    print(f"   Padding tokens: {padding_tokens}")
    print(f"   Actual tokens: {total_tokens - padding_tokens}")
    print(f"   Padding ratio: {padding_tokens / total_tokens * 100:.2f}%")
    
    return train_loader, val_loader


def main():
    """Main test function."""
    print("\n" + "="*70)
    print("🧪 Dataset and Tokenizer Test Suite")
    print("="*70)
    
    try:
        # Test tokenizer
        tokenizer = test_tokenizer()
        
        # Test dataset
        train_dataset, val_dataset = test_dataset(tokenizer)
        
        # Test dataloader
        train_loader, val_loader = test_dataloader(train_dataset, val_dataset, tokenizer)
        
        print("\n" + "="*70)
        print("✅ All tests passed successfully!")
        print("="*70)
        
        # Summary
        print("\n📋 Summary:")
        print(f"   ✅ Tokenizer: GPT-2 ({len(tokenizer)} vocab)")
        print(f"   ✅ Train dataset: {len(train_dataset)} samples")
        print(f"   ✅ Val dataset: {len(val_dataset)} samples")
        print(f"   ✅ Train batches: {len(train_loader)}")
        print(f"   ✅ Val batches: {len(val_loader)}")
        
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
