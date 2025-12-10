#!/usr/bin/env python3
"""Inference script for Qwen3-8B-Instruct with LoRA adapters."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model_with_lora(base_model_name: str, adapter_path: str, device: str = "cuda"):
    """Load base model and apply LoRA adapter."""
    print(f"Loading base model: {base_model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    # Load and merge LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()  # Merge for faster inference
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512):
    """Generate response for a given prompt."""
    messages = [{"role": "user", "content": prompt}]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response


def interactive_chat(model, tokenizer):
    """Interactive chat loop."""
    print("\n" + "="*50)
    print("Interactive Chat Mode")
    print("Type 'quit' or 'exit' to end the conversation")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            response = generate_response(model, tokenizer, user_input)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Qwen3 LoRA Inference")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-8B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for one-shot inference (if not provided, starts interactive mode)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_with_lora(args.base_model, args.adapter_path)

    if args.prompt:
        # Single inference
        print(f"\nPrompt: {args.prompt}")
        response = generate_response(model, tokenizer, args.prompt, args.max_new_tokens)
        print(f"\nResponse: {response}")
    else:
        # Interactive mode
        interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()
