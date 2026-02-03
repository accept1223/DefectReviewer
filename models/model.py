import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaTokenizer,LlamaForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

# load codellama or deepseek-coder
def load_model(config):
    model_path = config["model_path"]
    use_lora = config.get("use_lora", False)
    lora_params = config.get("lora", {})

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # set padding token 

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Apply LoRA if enabled
    if use_lora:
        lora_config = LoraConfig(
            r=lora_params.get("r", 8),
            lora_alpha=lora_params.get("alpha", 32),
            lora_dropout=lora_params.get("dropout", 0.1),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_params.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer

