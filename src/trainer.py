"""
Trainer class for LoRA fine-tuning.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any
from tqdm import tqdm

from .utils import ensure_dir, format_time, get_gpu_memory_usage


class LoRATrainer:
    """
    Trainer for LoRA fine-tuning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            optimizer: Optimizer (optional, will create default if None)
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            lr = self.config.get("learning_rate", 2e-4)
            weight_decay = self.config.get("weight_decay", 0.01)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        self.scheduler = scheduler
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.log_dir = self.config.get("logging_dir", "outputs/logs")
        ensure_dir(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Checkpointing
        self.output_dir = self.config.get("output_dir", "outputs/checkpoints")
        ensure_dir(self.output_dir)
        self.save_steps = self.config.get("save_steps", 500)
        self.save_total_limit = self.config.get("save_total_limit", 3)
        
        # Gradient accumulation
        self.gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        
        # Gradient clipping
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)
        
        # Logging frequency
        self.logging_steps = self.config.get("logging_steps", 10)
        
        # Evaluation
        self.eval_steps = self.config.get("eval_steps", 500)
        
        print(f"✅ Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Optimizer: {self.optimizer.__class__.__name__}")
        print(f"   Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"   Gradient accumulation steps: {self.gradient_accumulation_steps}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=True
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar("train/loss", loss.item() * self.gradient_accumulation_steps, self.global_step)
                    self.writer.add_scalar("train/learning_rate", current_lr, self.global_step)
                    
                    if torch.cuda.is_available():
                        mem = get_gpu_memory_usage()
                        self.writer.add_scalar("train/gpu_memory_allocated", mem['allocated'], self.global_step)
                
                # Evaluation
                if self.val_loader is not None and self.global_step % self.eval_steps == 0:
                    val_metrics = self.evaluate()
                    self.writer.add_scalar("val/loss", val_metrics['loss'], self.global_step)
                    
                    # Save best model
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.save_checkpoint(is_best=True)
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint()
            
            # Accumulate loss
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'num_batches': num_batches
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        self.model.train()
        
        return {
            'loss': avg_loss,
            'num_batches': num_batches
        }
    
    def train(self, num_epochs: int):
        """
        Train for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print("\n" + "="*60)
        print(f"🚀 Starting Training")
        print("="*60)
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.train_loader.batch_size * self.gradient_accumulation_steps}")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Evaluate
            if self.val_loader is not None:
                val_metrics = self.evaluate()
                
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  Val Loss:   {val_metrics['loss']:.4f}")
                
                # Log epoch metrics
                self.writer.add_scalar("epoch/train_loss", train_metrics['loss'], epoch)
                self.writer.add_scalar("epoch/val_loss", val_metrics['loss'], epoch)
            else:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                
                self.writer.add_scalar("epoch/train_loss", train_metrics['loss'], epoch)
        
        # Save final checkpoint
        self.save_checkpoint(is_final=True)
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print("="*60)
        print(f"Total time: {format_time(total_time)}")
        print(f"Average time per epoch: {format_time(total_time / num_epochs)}")
        if self.val_loader:
            print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60 + "\n")
        
        self.writer.close()
    
    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            is_final: Whether this is the final checkpoint
        """
        if is_final:
            checkpoint_dir = os.path.join(self.output_dir, "final")
        elif is_best:
            checkpoint_dir = os.path.join(self.output_dir, "best")
        else:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
        
        ensure_dir(checkpoint_dir)
        
        # Save model (LoRA adapters only)
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'optimizer': self.optimizer.state_dict(),
        }
        
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        
        torch.save(state, os.path.join(checkpoint_dir, "trainer_state.pt"))
        
        if not is_best and not is_final:
            print(f"💾 Checkpoint saved: {checkpoint_dir}")
        elif is_best:
            print(f"🏆 Best model saved: {checkpoint_dir}")
        elif is_final:
            print(f"🎯 Final model saved: {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_dir: Directory containing checkpoint
        """
        # Load training state
        state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path)
            self.current_epoch = state['epoch']
            self.global_step = state['global_step']
            self.best_val_loss = state['best_val_loss']
            self.optimizer.load_state_dict(state['optimizer'])
            
            if self.scheduler is not None and 'scheduler' in state:
                self.scheduler.load_state_dict(state['scheduler'])
            
            print(f"✅ Checkpoint loaded from {checkpoint_dir}")
            print(f"   Epoch: {self.current_epoch}")
            print(f"   Global step: {self.global_step}")
