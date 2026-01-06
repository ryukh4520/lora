"""
Model loading and LoRA setup utilities.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from typing import Tuple, Optional, Dict, Any


def load_model_and_tokenizer(
    model_name: str = "gpt2",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load pretrained model and tokenizer.
    
    Args:
        model_name: Name or path of the pretrained model
        load_in_8bit: Whether to load model in 8-bit precision
        load_in_4bit: Whether to load model in 4-bit precision
        device_map: Device mapping strategy
        torch_dtype: Torch data type (default: float32)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print("\n" + "="*60)
    print(f"📥 Loading Model: {model_name}")
    print("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # GPT-2 doesn't have pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("✅ Set pad_token to eos_token")
    
    print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"   Vocab size: {len(tokenizer):,}")
    
    # Prepare quantization config if needed
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        print("🔧 Using 4-bit quantization (QLoRA)")
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        print("🔧 Using 8-bit quantization")
    
    # Set default dtype if not specified
    if torch_dtype is None:
        torch_dtype = torch.float32
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    print(f"✅ Model loaded: {model.__class__.__name__}")
    print(f"   Device: {next(model.parameters()).device}")
    print(f"   Dtype: {next(model.parameters()).dtype}")
    
    return model, tokenizer


def setup_lora(
    model: AutoModelForCausalLM,
    lora_config: Optional[Dict[str, Any]] = None,
    use_gradient_checkpointing: bool = True
) -> AutoModelForCausalLM:
    """
    Setup LoRA (Low-Rank Adaptation) for the model.
    
    Args:
        model: Pretrained model
        lora_config: LoRA configuration dictionary
        use_gradient_checkpointing: Whether to use gradient checkpointing
        
    Returns:
        Model with LoRA adapters
    """
    print("\n" + "="*60)
    print("🔧 Setting up LoRA")
    print("="*60)
    
    # Default LoRA config for GPT-2
    if lora_config is None:
        lora_config = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": ["c_attn", "c_proj"]  # GPT-2 specific
        }
    
    # Print LoRA configuration
    print("📋 LoRA Configuration:")
    for key, value in lora_config.items():
        if key != "task_type":
            print(f"   {key}: {value}")
    
    # Prepare model for k-bit training if quantized
    if hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit:
        model = prepare_model_for_kbit_training(model)
        print("✅ Prepared model for 8-bit training")
    elif hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model)
        print("✅ Prepared model for 4-bit training")
    
    # Enable gradient checkpointing if requested
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")
    
    # Create LoRA config
    peft_config = LoraConfig(**lora_config)
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    
    print("✅ LoRA adapters applied")
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def get_model_info(model) -> Dict[str, Any]:
    """
    Get detailed model information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
        "model_class": model.__class__.__name__,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype)
    }
    
    return info


def print_model_summary(model):
    """
    Print a detailed summary of the model.
    
    Args:
        model: PyTorch model
    """
    info = get_model_info(model)
    
    print("\n" + "="*60)
    print("📊 Model Summary")
    print("="*60)
    print(f"Model Class:          {info['model_class']}")
    print(f"Device:               {info['device']}")
    print(f"Dtype:                {info['dtype']}")
    print(f"Total Parameters:     {info['total_params']:,}")
    print(f"Trainable Parameters: {info['trainable_params']:,}")
    print(f"Frozen Parameters:    {info['frozen_params']:,}")
    print(f"Trainable Ratio:      {info['trainable_percentage']:.4f}%")
    print("="*60 + "\n")
    
    # Calculate memory usage if on CUDA
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"🎮 GPU Memory:")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved:  {reserved:.2f} GB")
        print("="*60 + "\n")


def save_lora_weights(model, output_dir: str):
    """
    Save only LoRA adapter weights.
    
    Args:
        model: PEFT model with LoRA adapters
        output_dir: Directory to save weights
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    print(f"✅ LoRA weights saved to {output_dir}")


def load_lora_weights(model, lora_weights_path: str):
    """
    Load LoRA adapter weights into a model.
    
    Args:
        model: Base model
        lora_weights_path: Path to LoRA weights
        
    Returns:
        Model with loaded LoRA weights
    """
    from peft import PeftModel
    
    model = PeftModel.from_pretrained(model, lora_weights_path)
    print(f"✅ LoRA weights loaded from {lora_weights_path}")
    
    return model


def merge_lora_weights(model):
    """
    Merge LoRA weights into the base model.
    
    Args:
        model: PEFT model with LoRA adapters
        
    Returns:
        Merged model
    """
    model = model.merge_and_unload()
    print("✅ LoRA weights merged into base model")
    
    return model
