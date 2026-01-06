"""
Training script for LoRA fine-tuning.

Usage:
    python scripts/train.py [--config CONFIG_PATH]
"""

import sys
import os
import argparse
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import load_model_and_tokenizer, setup_lora, print_model_summary
from src.dataset import InstructionDataset, create_dataloaders
from src.trainer import LoRATrainer
from src.utils import load_config, set_seed, get_device, print_gpu_memory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LoRA model")
    parser.add_argument(
        "--model_config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    print("\n" + "="*70)
    print("🚀 LoRA Fine-Tuning Training Script")
    print("="*70)
    
    # Load configurations
    print("\n📋 Loading configurations...")
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)
    
    print(f"✅ Model config loaded from {args.model_config}")
    print(f"✅ Training config loaded from {args.training_config}")
    
    # Set random seed
    seed = training_config['training'].get('seed', 42)
    set_seed(seed)
    
    # Get device
    device = get_device()
    
    # Print initial GPU memory
    if torch.cuda.is_available():
        print_gpu_memory()
    
    # Load model and tokenizer
    print("\n" + "="*70)
    print("📥 Loading Model and Tokenizer")
    print("="*70)
    
    model_name = model_config['model']['pretrained_model_name']
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        load_in_8bit=model_config['quantization'].get('load_in_8bit', False),
        load_in_4bit=model_config['quantization'].get('load_in_4bit', False),
        torch_dtype=torch.float32
    )
    
    # Setup LoRA
    print("\n" + "="*70)
    print("🔧 Setting up LoRA")
    print("="*70)
    
    lora_config = {
        "r": model_config['lora']['r'],
        "lora_alpha": model_config['lora']['lora_alpha'],
        "lora_dropout": model_config['lora']['lora_dropout'],
        "bias": model_config['lora']['bias'],
        "target_modules": model_config['lora']['target_modules']
    }
    
    use_gradient_checkpointing = training_config['training'].get('gradient_checkpointing', True)
    model = setup_lora(model, lora_config=lora_config, use_gradient_checkpointing=use_gradient_checkpointing)
    
    # Print model summary
    print_model_summary(model)
    
    # Load datasets
    print("\n" + "="*70)
    print("📚 Loading Datasets")
    print("="*70)
    
    train_file = training_config['training']['train_file']
    val_file = training_config['training']['validation_file']
    max_seq_length = training_config['data_processing']['max_seq_length']
    
    train_dataset = InstructionDataset(
        data_path=train_file,
        tokenizer=tokenizer,
        max_length=max_seq_length
    )
    
    val_dataset = InstructionDataset(
        data_path=val_file,
        tokenizer=tokenizer,
        max_length=max_seq_length
    )
    
    # Create dataloaders
    batch_size = training_config['training']['batch_size']
    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=0
    )
    
    # Setup optimizer
    print("\n" + "="*70)
    print("⚙️  Setting up Optimizer and Scheduler")
    print("="*70)
    
    learning_rate = training_config['training']['learning_rate']
    weight_decay = training_config['training']['weight_decay']
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    print(f"✅ Optimizer: AdamW")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Weight decay: {weight_decay}")
    
    # Setup scheduler
    num_epochs = training_config['training']['num_epochs']
    warmup_steps = training_config['training']['warmup_steps']
    total_steps = len(train_loader) * num_epochs // training_config['training']['gradient_accumulation_steps']
    
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
    from torch.optim.lr_scheduler import SequentialLR
    
    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Main scheduler
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=learning_rate * 0.1
    )
    
    # Combined scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps]
    )
    
    print(f"✅ Scheduler: Warmup + CosineAnnealing")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Total steps: {total_steps}")
    
    # Create trainer
    print("\n" + "="*70)
    print("🎯 Creating Trainer")
    print("="*70)
    
    trainer_config = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_accumulation_steps": training_config['training']['gradient_accumulation_steps'],
        "max_grad_norm": training_config['training']['max_grad_norm'],
        "logging_steps": training_config['training']['logging_steps'],
        "save_steps": training_config['training']['save_steps'],
        "save_total_limit": training_config['training']['save_total_limit'],
        "eval_steps": training_config['training']['eval_steps'],
        "output_dir": training_config['training']['output_dir'],
        "logging_dir": training_config['training']['logging_dir']
    }
    
    trainer = LoRATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=trainer_config
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"\n📂 Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Train
    print("\n" + "="*70)
    print("🏋️  Starting Training")
    print("="*70)
    
    trainer.train(num_epochs=num_epochs)
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 Training Complete!")
    print("="*70)
    print(f"✅ Model saved to: {training_config['training']['output_dir']}")
    print(f"✅ Logs saved to: {training_config['training']['logging_dir']}")
    
    if torch.cuda.is_available():
        print("\n🎮 Final GPU Memory Usage:")
        print_gpu_memory()
    
    print("\n💡 Next steps:")
    print("   1. Evaluate the model: python scripts/evaluate.py")
    print("   2. Test inference: python scripts/inference.py")
    print("   3. Merge LoRA weights: python scripts/merge_lora.py")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
