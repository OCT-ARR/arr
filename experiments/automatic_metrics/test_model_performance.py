'''
Usage
# Use one GPU (GPU 0) and test sample 0.
python experiments/automatic_metrics/test_model_performance.py --gpus 0 --sample_idx 0

# Use two GPUs (GPU 0 and 1) and test sample 5.
python experiments/automatic_metrics/test_model_performance.py --gpus 0,1 --sample_idx 5

# Use GPUs 2 and 3 and test sample 10.
python experiments/automatic_metrics/test_model_performance.py --gpus 2,3 --sample_idx 10

# Full example.
python experiments/automatic_metrics/test_model_performance.py \
    --model_path checkpoints/grpo_baseline \
    --test_file data/MESC-FT/test.jsonl \
    --gpus 0,1 \
    --sample_idx 3
'''

import os
import argparse
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MAX_IMAGES = 8

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test a local FACES or FIRES checkpoint")
    parser.add_argument("--model_path", type=str, default="checkpoints/grpo_baseline",
                        help="Model path; checkpoints/grpo_baseline is FIRES (Baseline)")
    parser.add_argument("--test_file", type=str, default="data/MESC-FT/test.jsonl",
                        help="Test file path")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Test sample index")
    parser.add_argument("--gpus", type=str, default="0",
                        help="GPU IDs to use, for example '0', '0,1', or '2,3'")
    return parser.parse_args()

def load_model(model_path):
    """Load model and processor."""
    print(f"Loading model: {model_path}")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model.eval()
    print("Model loaded.")
    return model, processor

def load_single_sample(test_file, sample_idx):
    """Load only the sample at the requested index."""
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                return json.loads(line.strip())
    raise IndexError(f"Sample index {sample_idx} is out of range")

def build_messages(sample):
    """Build chat messages from one sample."""
    system_prompt = sample['system']
    conversations = sample['conversations']
    
    user_conv = conversations[0]
    user_text = user_conv['value']
    image_paths = user_conv.get('images', [])
    ground_truth = conversations[1]['value']
    
    messages = [{"role": "system", "content": system_prompt}]
    
    user_content = []
    for img_path in image_paths[:MAX_IMAGES]:
        user_content.append({"type": "image", "image": img_path})
    
    text_only = user_text.replace("<image>", "").strip()
    user_content.append({"type": "text", "text": text_only})
    
    messages.append({"role": "user", "content": user_content})
    
    return messages, ground_truth, image_paths, user_text

def generate_response(model, processor, messages, max_new_tokens=256):
    """Generate a model response with the paper inference token limit."""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

def display_comparison(model_output, ground_truth, sample_idx, image_paths, user_text):
    """Display model output and ground truth side by side."""
    separator = "=" * 80
    
    print(f"\n{separator}")
    print(f"Test sample index: {sample_idx}")
    print(f"Number of images: {len(image_paths)}")
    print(separator)
    
    print("\n[User Input]")
    print("-" * 40)
    print(user_text.replace("<image>", "[IMG]"))
    
    print(f"\n{separator}")
    
    print("\n[Model Output]")
    print("-" * 40)
    print(model_output)
    
    print(f"\n{separator}")
    
    print("\n[Ground Truth]")
    print("-" * 40)
    print(ground_truth)
    
    print(f"\n{separator}\n")

def main():
    # Parse command-line arguments.
    args = parse_args()
    
    # Set visible GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print(f"Using GPUs: {args.gpus}")
    
    # Load model.
    model, processor = load_model(args.model_path)
    
    # Load the requested sample only.
    print(f"Loading test sample {args.sample_idx} ...")
    sample = load_single_sample(args.test_file, args.sample_idx)
    
    # Build messages.
    messages, ground_truth, image_paths, user_text = build_messages(sample)
    
    print("Generating response...")
    
    # Generate response.
    model_output = generate_response(model, processor, messages)
    
    # Display comparison.
    display_comparison(model_output, ground_truth, args.sample_idx, image_paths, user_text)

if __name__ == "__main__":
    main()
