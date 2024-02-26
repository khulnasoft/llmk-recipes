# Copyright (c) Khulnasoft Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from peft import PeftModel
from transformers import AutoModelForCausalLM, LlmkForCausalLM, LlmkConfig

# Function to load the main model for text generation
def load_model(model_name, quantization, use_fast_kernels):
    print(f"use_fast_kernels{use_fast_kernels}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" if use_fast_kernels else None,
    )
    return model


# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

# Loading the model from config to load FSDP checkpoints into that
def load_llmk_from_config(config_path):
    model_config = LlmkConfig.from_pretrained(config_path) 
    model = LlmkForCausalLM(config=model_config)
    return model
    
    