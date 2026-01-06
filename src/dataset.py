"""
Dataset classes for LoRA fine-tuning.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer


class InstructionDataset(Dataset):
    """
    Dataset for instruction-following tasks.
    
    Expected data format:
    [
        {
            "instruction": "질문 또는 지시사항",
            "input": "추가 입력 (선택사항)",
            "output": "기대되는 출력"
        },
        ...
    ]
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        instruction_template: Optional[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to JSON data file
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            instruction_template: Template for formatting instructions
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Default instruction template
        if instruction_template is None:
            self.instruction_template = (
                "### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n{output}"
            )
        else:
            self.instruction_template = instruction_template
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"✅ Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        item = self.data[idx]
        
        # Format the instruction
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        # Create full text using template
        if input_text:
            full_text = self.instruction_template.format(
                instruction=instruction,
                input=input_text,
                output=output
            )
        else:
            # If no input, use simpler format
            full_text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Prepare labels (same as input_ids for causal LM)
        labels = encodings["input_ids"].clone()
        
        # Mask padding tokens in labels (-100 is ignored by loss function)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }


class SimpleTextDataset(Dataset):
    """
    Simple dataset for text generation tasks.
    
    Expected data format:
    [
        {"text": "샘플 텍스트 1"},
        {"text": "샘플 텍스트 2"},
        ...
    ]
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to JSON data file
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"✅ Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        text = self.data[idx]["text"]
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Prepare labels
        labels = encodings["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 1,
    num_workers: int = 0
) -> tuple:
    """
    Create DataLoaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    print(f"✅ Created DataLoaders:")
    print(f"   Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    if val_loader:
        print(f"   Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    
    return train_loader, val_loader
