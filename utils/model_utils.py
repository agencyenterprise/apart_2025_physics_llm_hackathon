"""
Utility functions for loading models with different weight initialization strategies.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_random_weight_model(model_name: str, device: str = "auto", torch_dtype=None):
    """
    Load a model with the same architecture as the pretrained model but with randomly initialized weights.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
        device: Device to load model on ("auto", "cpu", "cuda", etc.)
        torch_dtype: Data type for model weights (e.g., torch.float16, torch.float32)
    
    Returns:
        model: Model with random weights
        tokenizer: Associated tokenizer
        config: Model configuration
    """
    # Determine device
    if device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"🔄 Creating random-weight baseline from {model_name}")
    print(f"📱 Device: {device}")
    
    # Load configuration and tokenizer (these don't contain weights)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Create model from config - this initializes with random weights
    model = AutoModelForCausalLM.from_config(
        config, 
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)
    
    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    print(f"✅ Random-weight model created successfully")
    print(f"   Model type: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Hidden size: {config.hidden_size}")
    
    return model, tokenizer, config


def reinitialize_model_weights(model, seed: int = 42):
    """
    Reinitialize all weights in a pretrained model to random values.
    
    Args:
        model: Pretrained model to reinitialize
        seed: Random seed for reproducibility
    
    Returns:
        model: Model with reinitialized weights
    """
    torch.manual_seed(seed)
    
    print(f"🔄 Reinitializing all model weights with seed {seed}")
    
    # Reinitialize all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if len(param.shape) >= 2:
                # Use Xavier/Glorot initialization for weight matrices
                torch.nn.init.xavier_uniform_(param)
            else:
                # Use small random values for biases and 1D parameters
                torch.nn.init.uniform_(param, -0.1, 0.1)
    
    print(f"✅ Model weights reinitialized")
    return model


def compare_models(pretrained_model, random_model, tokenizer, test_text: str = "The capital of France is"):
    """
    Compare outputs between pretrained and random-weight models.
    
    Args:
        pretrained_model: Model with pretrained weights
        random_model: Model with random weights
        tokenizer: Tokenizer for both models
        test_text: Text to test on
    """
    print(f"🔍 Comparing model outputs on: '{test_text}'")
    
    # Tokenize input
    inputs = tokenizer(test_text, return_tensors="pt")
    inputs = {k: v.to(pretrained_model.device) for k, v in inputs.items()}
    
    # Get outputs from both models
    with torch.no_grad():
        pretrained_outputs = pretrained_model(**inputs)
        random_outputs = random_model(**inputs)
    
    # Get predicted next tokens
    pretrained_next_token = torch.argmax(pretrained_outputs.logits[0, -1, :]).item()
    random_next_token = torch.argmax(random_outputs.logits[0, -1, :]).item()
    
    pretrained_completion = tokenizer.decode(pretrained_next_token)
    random_completion = tokenizer.decode(random_next_token)
    
    print(f"   Pretrained model next token: '{pretrained_completion}'")
    print(f"   Random model next token: '{random_completion}'")
    print(f"   Same prediction: {pretrained_next_token == random_next_token}")
    
    # Compare logit distributions
    pretrained_probs = torch.softmax(pretrained_outputs.logits[0, -1, :], dim=-1)
    random_probs = torch.softmax(random_outputs.logits[0, -1, :], dim=-1)
    
    # KL divergence
    kl_div = torch.nn.functional.kl_div(
        torch.log(random_probs + 1e-8), 
        pretrained_probs, 
        reduction='sum'
    ).item()
    
    print(f"   KL divergence (random || pretrained): {kl_div:.4f}")
    
    return {
        'pretrained_next_token': pretrained_completion,
        'random_next_token': random_completion,
        'kl_divergence': kl_div
    } 