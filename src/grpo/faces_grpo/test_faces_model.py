'''
Usage:
python3 src/grpo/faces_grpo/test_faces_model.py
'''

import gc
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info

    HAS_QWEN_VL_UTILS = True
except Exception:
    HAS_QWEN_VL_UTILS = False


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

# ============================ Hyperparameters ============================
TEST_FILE = PROJECT_ROOT / "data" / "MESC-FT" / "test.jsonl"
# BASE_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "warmup_merged"
# BASE_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "sft_merged"
BASE_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "grpo_baseline"
GRPO_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "faces_main"

# Sequential inference: only 1 GPU is needed in general.
# If you want to split model tests to different cards, set different devices.
REQUIRED_GPU_COUNT = 1
BASE_DEVICE = "cuda:1"
GRPO_DEVICE = "cuda:1"

# Sample control:
# - Set SAMPLE_INDEX to an integer >= 1 to pick that sample (1-based).
# - Set SAMPLE_INDEX = None to use random sampling.
SAMPLE_INDEX = 18
RANDOM_SEED = 42

MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_FRAMES = 8
# =======================================================================

RESPONSE_PATTERN = re.compile(r"<response>(.*?)</response>", re.IGNORECASE | re.DOTALL)


@dataclass
class ConversationSample:
    system: str
    user_text: str
    image_paths: List[str]
    ground_truth_full: str


def extract_response_text(text: str) -> str:
    if not text:
        return ""
    match = RESPONSE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def load_model_config(model_path: str):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = getattr(config, "text_config", None)
    rope_scaling = getattr(text_config, "rope_scaling", None)
    rope_parameters = getattr(text_config, "rope_parameters", None)

    if rope_scaling is None and isinstance(rope_parameters, dict):
        normalized_rope_scaling = dict(rope_parameters)
        if normalized_rope_scaling.get("type") == "mrope":
            normalized_rope_scaling["type"] = "default"
        if "rope_type" not in normalized_rope_scaling and "type" in normalized_rope_scaling:
            normalized_rope_scaling["rope_type"] = normalized_rope_scaling["type"]
        text_config.rope_scaling = normalized_rope_scaling
        if getattr(text_config, "rope_theta", None) is None and rope_parameters.get("rope_theta") is not None:
            text_config.rope_theta = rope_parameters["rope_theta"]

    return config


def resolve_model_class(config):
    model_type = str(getattr(config, "model_type", "")).lower()

    import transformers as tf

    candidates = [
        (("qwen3", "vl", "moe"), "Qwen3VLMoeForConditionalGeneration"),
        (("qwen3", "vl"), "Qwen3VLForConditionalGeneration"),
        (("qwen2_5_vl",), "Qwen2_5_VLForConditionalGeneration"),
        (("qwen2_vl",), "Qwen2VLForConditionalGeneration"),
    ]
    for parts, cls_name in candidates:
        if all(part in model_type for part in parts):
            cls = getattr(tf, cls_name, None)
            if cls is not None:
                return cls
    return AutoModelForCausalLM


def load_model_with_fallback(model_class, model_path: str, kwargs: Dict[str, Any]):
    try:
        return model_class.from_pretrained(model_path, **kwargs)
    except TypeError as exc:
        if "dtype" in kwargs and "unexpected keyword argument 'dtype'" in str(exc):
            retry_kwargs = dict(kwargs)
            retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
            return load_model_with_fallback(model_class, model_path, retry_kwargs)
        raise


def ensure_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available, but device={device}")


def resolve_device(device: str) -> str:
    if device != "cuda:auto":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but device=cuda:auto")

    best_idx = 0
    best_free_bytes = -1
    for idx in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        if free_bytes > best_free_bytes:
            best_free_bytes = free_bytes
            best_idx = idx
    return f"cuda:{best_idx}"


def move_to_device(data: Dict[str, Any], device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def get_processor_load_kwargs(model_path: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    tokenizer_config_path = Path(model_path) / "tokenizer_config.json"
    if not tokenizer_config_path.is_file():
        return kwargs

    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except Exception:
        return kwargs

    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if isinstance(extra_special_tokens, list):
        # transformers>=4.57 expects extra_special_tokens to be a dict.
        kwargs["extra_special_tokens"] = {}
        if not tokenizer_config.get("additional_special_tokens"):
            kwargs["additional_special_tokens"] = extra_special_tokens
    return kwargs


def load_images(image_paths: Sequence[str], max_frames: int) -> List[Image.Image]:
    images: List[Image.Image] = []
    for image_path in list(image_paths)[:max_frames]:
        try:
            images.append(Image.open(image_path).convert("RGB"))
        except Exception:
            images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
    return images


def build_messages(sample: ConversationSample, images: Sequence[Image.Image]) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = []
    for img in images:
        user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": sample.user_text})
    return [
        {"role": "system", "content": [{"type": "text", "text": sample.system}]},
        {"role": "user", "content": user_content},
    ]


def prepare_inputs(
    processor,
    messages: List[Dict[str, Any]],
    images: Sequence[Image.Image],
    device: str,
) -> Dict[str, Any]:
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    tokenizer = processor.tokenizer
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        if HAS_QWEN_VL_UTILS:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=[text_prompt],
                images=list(images) if images else None,
                padding=True,
                return_tensors="pt",
            )
    finally:
        tokenizer.padding_side = old_padding_side

    return move_to_device(dict(inputs), device)


def generate_one(
    model_path: str,
    sample: ConversationSample,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    max_frames: int,
) -> str:
    device = resolve_device(device)
    ensure_device(device)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()

    config = load_model_config(model_path)
    model_class = resolve_model_class(config)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model_kwargs: Dict[str, Any] = {
        "config": config,
        "dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": {"": device},
    }

    model = load_model_with_fallback(model_class, model_path, model_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, **get_processor_load_kwargs(model_path))

    if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    images = load_images(sample.image_paths, max_frames=max_frames)
    messages = build_messages(sample, images)
    inputs = prepare_inputs(processor, messages, images, device)

    generation_kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

    input_len = int(inputs["attention_mask"].sum(dim=1)[0].item())
    new_tokens = output_ids[0, input_len:]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    del model, processor, images, messages, inputs, output_ids, new_tokens
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return text


def parse_sample_from_item(item: Dict[str, Any]) -> ConversationSample:
    system = str(item.get("system", ""))
    user_text = ""
    image_paths: List[str] = []
    gt_full = ""

    for turn in item.get("conversations", []):
        role = turn.get("from")
        if role == "user":
            user_text = str(turn.get("value", ""))
            image_paths = list(turn.get("images", []) or [])
        elif role == "assistant":
            gt_full = str(turn.get("value", ""))

    if not user_text or not gt_full:
        raise ValueError("invalid sample: missing user_text or assistant answer")

    return ConversationSample(
        system=system,
        user_text=user_text,
        image_paths=image_paths,
        ground_truth_full=gt_full,
    )


def load_valid_samples(test_file: Path) -> List[ConversationSample]:
    samples: List[ConversationSample] = []
    with test_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                sample = parse_sample_from_item(item)
            except Exception:
                continue
            samples.append(sample)
    return samples


def pick_sample(test_file: Path, sample_index: int | None, random_seed: int) -> ConversationSample:
    samples = load_valid_samples(test_file)
    if not samples:
        raise RuntimeError(f"no valid samples in {test_file}")

    if sample_index is not None:
        if sample_index < 1 or sample_index > len(samples):
            raise IndexError(
                f"SAMPLE_INDEX={sample_index} out of range. valid range: 1..{len(samples)}"
            )
        return samples[sample_index - 1]

    rng = random.Random(random_seed)
    return rng.choice(samples)


def main():
    sample = pick_sample(
        test_file=TEST_FILE,
        sample_index=SAMPLE_INDEX,
        random_seed=RANDOM_SEED,
    )

    base_output = generate_one(
        model_path=str(BASE_MODEL_PATH),
        sample=sample,
        device=BASE_DEVICE,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_frames=MAX_FRAMES,
    )

    grpo_output = generate_one(
        model_path=str(GRPO_MODEL_PATH),
        sample=sample,
        device=GRPO_DEVICE,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_frames=MAX_FRAMES,
    )

    ground_truth = sample.ground_truth_full
    if not ground_truth.strip():
        ground_truth = extract_response_text(sample.ground_truth_full)

    print("=" * 100)
    print("[1] Base Model Output")
    print(base_output)
    print("=" * 100)
    print("[2] GRPO Merged Model Output")
    print(grpo_output)
    print("=" * 100)
    print("[3] Ground Truth")
    print(ground_truth)
    print("=" * 100)


if __name__ == "__main__":
    main()
