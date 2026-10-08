#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoProcessor


def resolve_model_class():
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration
    except ImportError:
        try:
            from transformers import Qwen2VLForConditionalGeneration

            return Qwen2VLForConditionalGeneration
        except ImportError:
            return AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge an SFT LoRA adapter into the base multimodal model."
    )
    parser.add_argument(
        "--base_model_path",
        default="model/qwen/Qwen2___5-VL-7B-Instruct",
        help="Base Qwen-VL model path or Hugging Face/modelscope id.",
    )
    parser.add_argument(
        "--adapter_path",
        required=True,
        help="SFT LoRA checkpoint directory, e.g. checkpoints/sft_lora/.../checkpoint-300.",
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints/sft_merged",
        help="Directory where the merged SFT model will be written.",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help="Transformers device_map used during merge. Use 'cpu' if GPU memory is tight.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_class = resolve_model_class()
    base_model = model_class.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=args.device_map,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)

    processor = AutoProcessor.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
    )
    processor.save_pretrained(output_dir)
    print(f"Merged SFT model saved to: {output_dir}")


if __name__ == "__main__":
    main()
