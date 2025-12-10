#!/usr/bin/env python3
"""Merge LoRA adapter into base model for vLLM serving."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_save(base_model_name: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter into base model and save."""
    print(f"Loading base model: {base_model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load on CPU for merging
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging LoRA weights into base model...")
    # Merge and unload LoRA
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    # Save merged model
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("Done! Merged model saved successfully.")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for merged model",
    )
    args = parser.parse_args()

    merge_and_save(args.base_model, args.adapter_path, args.output_path)


if __name__ == "__main__":
    main()
