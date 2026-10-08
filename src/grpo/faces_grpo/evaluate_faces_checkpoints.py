#!/usr/bin/env python3
"""Two-round automatic evaluation for psychotherapy dialogue models.

Usage:
    python3 src/grpo/faces_grpo/evaluate_faces_checkpoints.py
"""

from __future__ import annotations

import contextlib
import concurrent.futures
import gc
import json
import multiprocessing as mp
import os
import queue
import random
import re
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
# Respect CUDA_VISIBLE_DEVICES from the shell. If it is not set, fall back to
# AUTO_TEST_VISIBLE_CUDA_DEVICES, then to a safe default mask.
if "AUTO_TEST_VISIBLE_CUDA_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["AUTO_TEST_VISIBLE_CUDA_DEVICES"]
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6,3,4")

warnings.filterwarnings(
    "ignore",
    message=r"Failed to load image Python extension:.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"The following generation flags are not valid and may be ignored:.*",
)

import torch
import torch.nn.functional as F
from PIL import Image


def patch_torchvision_fake_registration() -> None:
    register_fake = getattr(torch.library, "register_fake", None)
    if register_fake is None:
        return

    if getattr(register_fake, "__name__", "") == "safe_register_fake":
        return

    def safe_register_fake(op_name, *args, **kwargs):
        decorator = register_fake(op_name, *args, **kwargs)

        def wrapper(fn):
            try:
                return decorator(fn)
            except RuntimeError as exc:
                if "torchvision::nms does not exist" in str(exc):
                    return fn
                raise

        return wrapper

    torch.library.register_fake = safe_register_fake


patch_torchvision_fake_registration()

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
)
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

try:
    from qwen_vl_utils import process_vision_info

    HAS_QWEN_VL_UTILS = True
except Exception:
    HAS_QWEN_VL_UTILS = False


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

TEST_FILE = PROJECT_ROOT / "data" / "MESC-FT" / "test.jsonl"
FINAL_MODEL_ROOT = PROJECT_ROOT / "checkpoints" / "faces_main"
SCORE_FILE = FINAL_MODEL_ROOT / "score.txt"
SCORE_EVERY_DIMENSION_FILE = FINAL_MODEL_ROOT / "score_every_dimension.txt"
JUDGE_MODEL_PATH = PROJECT_ROOT / "models" / "judge-32b"
VISUAL_MODEL_PATH = (
    PROJECT_ROOT / "models" / "clip-vit-large-patch14"
)
EMOTION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
)

ROUND_SIZE = 50
RANDOM_SEED = 42

VISIBLE_CUDA_DEVICE_IDS = tuple(
    int(part.strip())
    for part in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if part.strip()
)
if len(VISIBLE_CUDA_DEVICE_IDS) < 2:
    raise RuntimeError(
        "AUTO_TEST_VISIBLE_CUDA_DEVICES/CUDA_VISIBLE_DEVICES must expose at least 2 GPUs"
    )


def visible_cuda(index: int) -> str:
    return f"cuda:{index}"


def physical_cuda_label(index: int) -> str:
    return f"cuda:{VISIBLE_CUDA_DEVICE_IDS[index]}"


JUDGE_DEVICE = visible_cuda(0)
GENERATION_DEVICES = tuple(
    visible_cuda(index) for index in range(1, len(VISIBLE_CUDA_DEVICE_IDS))
)

MAX_NEW_TOKENS = 256
MAX_FRAMES = 8
JUDGE_MAX_NEW_TOKENS = 256
JUDGE_BATCH_SIZE = 1
JUDGE_MAX_LENGTH = 4096
VISUAL_DEVICE = os.environ.get("AUTO_TEST_VISUAL_DEVICE", "cpu")
EMOTION_DEVICE = os.environ.get("AUTO_TEST_EMOTION_DEVICE", "cpu")
VISUAL_THRESHOLD = 0.60
VISUAL_TOP_K = 4
USE_COLOR_LOG = bool(
    os.environ.get("NO_COLOR") is None and os.environ.get("FORCE_COLOR", "1") != "0"
)

WEIGHTS = {
    "visual": 15.0,
    "emotion": 20.0,
    "situation": 20.0,
    "strategy": 20.0,
    "response": 25.0,
}
DIMENSION_DISPLAY_NAMES = {
    "visual": "visual",
    "emotion": "emotion",
    "situation": "situation",
    "strategy": "strategy",
    "response": "response",
}

RESPONSE_PATTERN = re.compile(r"<response>(.*?)</response>", re.IGNORECASE | re.DOTALL)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
SEGMENT_LABEL_PATTERN = re.compile(
    r"\b(first(?:ly)?|second(?:ly)?|third(?:ly)?|fourth(?:ly)?)\b[\s:.,-]*",
    re.IGNORECASE,
)
IMAGE_TOKEN_PATTERN = re.compile(r"(?:<image>\s*)+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
EPS = 1e-8

JUDGE_PROMPT_VARIANTS: Tuple[Dict[str, str], ...] = (
    {
        "system": "You are a strict evaluator for multimodal empathetic dialogue.",
        "user_template": """Evaluate the candidate by using the dialogue context, any attached visual evidence, and the corresponding reference annotations. Your judgment must consider all of the following:
1. semantic consistency with the corresponding reference part,
2. functional alignment with the intended role of the reference part, and
3. appropriateness to the current dialogue context and available evidence.

Score the candidate on the following three dimensions using integer scores from 1 to 5:
- Situation: whether the candidate situation inference is semantically consistent with the reference situation, functionally appropriate for understanding the user's condition, and grounded in the available evidence.
- Strategy: whether the candidate support strategy is semantically aligned with the reference strategy, functionally appropriate for the user's needs, and psychologically safe in the current context.
- Outcome: whether the final response is semantically and functionally aligned with the reference response, while remaining empathetic, helpful, and contextually appropriate.

Scoring rule:
- 5 = great
- 4 = good
- 3 = acceptable
- 2 = weak
- 1 = poor
- Penalize overly long, verbose, or formulaic candidates with redundant or padded wording. If a candidate is substantially longer than the reference and the extra content does not improve semantic or functional alignment, lower the corresponding score(s).

[CONTEXT]
{context}

[CANDIDATE_SITUATION]
{candidate_situation}

[REFERENCE_SITUATION]
{reference_situation}

[CANDIDATE_STRATEGY]
{candidate_strategy}

[REFERENCE_STRATEGY]
{reference_strategy}

[CANDIDATE_RESPONSE]
{candidate_response}

[REFERENCE_RESPONSE]
{reference_response}

Return JSON only in the following format:
{{"situation_score": 1, "strategy_score": 1, "outcome_score": 1}}
""",
    },
    {
        "system": "You are an expert reviewer of multimodal emotional support outputs.",
        "user_template": """Please assess the candidate by comparing each component with its corresponding reference annotation. Your judgment must consider all of the following:
1. semantic consistency with the corresponding reference part,
2. functional alignment with the intended role of the reference part, and
3. appropriateness to the current dialogue context and available evidence.

Rate the candidate on three aspects with integer scores from 1 to 5:
1. Situation Understanding:
Is the candidate situation inference semantically consistent with the reference situation, functionally useful for understanding the user's state, and well grounded in the current context?
2. Support Strategy:
Is the candidate strategy semantically aligned with the reference strategy, functionally suitable for the user's needs, and safe and appropriate for the current situation?
3. Response Outcome:
Is the final response semantically and functionally consistent with the reference response, while remaining empathetic, relevant, and useful in context?

Scoring rule:
- 5 = great
- 4 = good
- 3 = acceptable
- 2 = weak
- 1 = poor
- Penalize overly long, verbose, or formulaic candidates with redundant or padded wording. If a candidate is substantially longer than the reference and the extra content does not improve semantic or functional alignment, lower the corresponding score(s).

[CONTEXT]
{context}

[CANDIDATE_SITUATION]
{candidate_situation}

[REFERENCE_SITUATION]
{reference_situation}

[CANDIDATE_STRATEGY]
{candidate_strategy}

[REFERENCE_STRATEGY]
{reference_strategy}

[CANDIDATE_RESPONSE]
{candidate_response}

[REFERENCE_RESPONSE]
{reference_response}

Output JSON only:
{{"situation_score": 1, "strategy_score": 1, "outcome_score": 1}}
""",
    },
    {
        "system": "Your task is to judge one candidate multimodal empathetic response.",
        "user_template": """Read the dialogue context, inspect any attached visual evidence, and compare the candidate situation inference, candidate strategy, and candidate response with their corresponding reference parts. Your judgment must consider all of the following:
1. semantic consistency with the corresponding reference part,
2. functional alignment with the intended role of the reference part, and
3. appropriateness to the current dialogue context and available evidence.

Evaluate the candidate from three perspectives, each on a 1-5 integer scale:
- Situation Score:
How well does the candidate situation inference match the meaning of the reference situation, serve the same interpretive function, and remain grounded in the dialogue context and visual evidence?
- Strategy Score:
How well does the candidate strategy match the meaning and supportive intent of the reference strategy, while remaining appropriate and safe for the user's situation?
- Outcome Score:
How well does the candidate response match the meaning and support function of the reference response, while remaining coherent, empathetic, and contextually appropriate?

Scoring rule:
- 5 = great
- 4 = good
- 3 = acceptable
- 2 = weak
- 1 = poor
- Penalize overly long, verbose, or formulaic candidates with redundant or padded wording. If a candidate is substantially longer than the reference and the extra content does not improve semantic or functional alignment, lower the corresponding score(s).

[CONTEXT]
{context}

[CANDIDATE_SITUATION]
{candidate_situation}

[REFERENCE_SITUATION]
{reference_situation}

[CANDIDATE_STRATEGY]
{candidate_strategy}

[REFERENCE_STRATEGY]
{reference_strategy}

[CANDIDATE_RESPONSE]
{candidate_response}

[REFERENCE_RESPONSE]
{reference_response}

Return JSON only and do not output any explanation:
{{"situation_score": 1, "strategy_score": 1, "outcome_score": 1}}
""",
    },
)


@dataclass
class ConversationSample:
    sample_id: int
    system: str
    user_text: str
    image_paths: List[str]
    ground_truth_full: str
    ground_truth_response: str
    ground_truth_first: str
    ground_truth_second: str
    ground_truth_third: str
    ground_truth_fourth: str


@dataclass
class ModelSpec:
    name: str
    path: str


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _c(text: str, code: str) -> str:
    if not USE_COLOR_LOG:
        return text
    return f"\033[{code}m{text}\033[0m"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def render_progress_bar(completed: int, total: int) -> str:
    completed = max(0, min(completed, total))
    filled = _c("#" * completed, "1;38;5;82")
    empty = _c("-" * (total - completed), "38;5;240")
    return f"{filled}{empty}"


def print_progress(stage: str, completed: int, total: int, start_time: float, last_item: str) -> None:
    elapsed = time.time() - start_time
    eta = 0.0
    if completed > 0 and completed < total:
        eta = (elapsed / completed) * (total - completed)

    stage_color = "1;38;5;45" if stage == "GEN" else "1;38;5;214"
    label = _c(f"[{stage}]", stage_color)
    count = _c(f"{completed}/{total}", "1;38;5;190")
    bar = render_progress_bar(completed, total)
    elapsed_s = _c(format_duration(elapsed), "1;38;5;117")
    eta_s = _c(format_duration(eta), "1;38;5;208")
    item = _c(last_item, "1;38;5;250")
    print(
        f"{label} {count} |{bar}| elapsed={elapsed_s} eta={eta_s} last={item}",
        flush=True,
    )


def extract_response_text(text: str) -> str:
    if not text:
        return ""
    match = RESPONSE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_think_text(text: str) -> str:
    if not text:
        return ""
    match = THINK_PATTERN.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def _canonical_label(label: str) -> Optional[str]:
    if not label:
        return None
    label = label.strip().lower()
    if label.startswith("first"):
        return "first"
    if label.startswith("second"):
        return "second"
    if label.startswith("third"):
        return "third"
    if label.startswith("fourth"):
        return "fourth"
    return None


def parse_think_segments(text: str) -> Dict[str, str]:
    think = extract_think_text(text)
    if not think:
        return {}

    matches = list(SEGMENT_LABEL_PATTERN.finditer(think))
    if not matches:
        return {}

    out: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = _canonical_label(match.group(1))
        if label is None:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(think)
        segment = think[start:end].strip()
        if segment:
            out[label] = segment
    return out


def parse_sample_from_item(item: Dict[str, Any], sample_id: int) -> ConversationSample:
    system = str(item.get("system", ""))
    user_text = ""
    image_paths: List[str] = []
    ground_truth_full = ""

    for turn in item.get("conversations", []):
        role = turn.get("from")
        if role == "user":
            user_text = str(turn.get("value", ""))
            image_paths = list(turn.get("images", []) or [])
        elif role == "assistant":
            ground_truth_full = str(turn.get("value", ""))

    if not user_text or not ground_truth_full:
        raise ValueError("invalid sample: missing user or assistant content")

    gt_segments = parse_think_segments(ground_truth_full)
    ground_truth_response = extract_response_text(ground_truth_full)

    return ConversationSample(
        sample_id=sample_id,
        system=system,
        user_text=user_text,
        image_paths=image_paths,
        ground_truth_full=ground_truth_full,
        ground_truth_response=ground_truth_response,
        ground_truth_first=gt_segments.get("first", ""),
        ground_truth_second=gt_segments.get("second", ""),
        ground_truth_third=gt_segments.get("third", ""),
        ground_truth_fourth=gt_segments.get("fourth", ""),
    )


def load_valid_samples(test_file: Path) -> List[ConversationSample]:
    samples: List[ConversationSample] = []
    with test_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                sample = parse_sample_from_item(item, sample_id=idx)
            except Exception:
                continue
            samples.append(sample)
    return samples


def discover_candidate_models() -> List[ModelSpec]:
    step_dirs = [p for p in FINAL_MODEL_ROOT.iterdir() if p.is_dir() and p.name.startswith("step-")]
    step_dirs.sort(key=lambda p: int(p.name.split("-", 1)[1]))

    specs = [
        ModelSpec(name=p.name, path=str(p))
        for p in step_dirs
    ]
    specs.extend(
        [
            ModelSpec(
                name="merge_245",
                path=str(PROJECT_ROOT / "checkpoints" / "warmup_merged"),
            ),
            ModelSpec(
                name="fixed_sft_380",
                path=str(PROJECT_ROOT / "checkpoints" / "sft_merged"),
            ),
            ModelSpec(
                name="grpo_baseline",
                path=str(PROJECT_ROOT / "checkpoints" / "grpo_baseline"),
            ),
        ]
    )
    return specs


def choose_round_samples(
    samples: Sequence[ConversationSample],
    round_size: int,
    seed: int,
) -> Tuple[List[ConversationSample], Dict[str, List[int]]]:
    required = round_size * 2
    if len(samples) < required:
        raise RuntimeError(f"need at least {required} valid samples, found {len(samples)}")

    rng = random.Random(seed)
    picked = rng.sample(range(len(samples)), required)
    round1_indices = picked[:round_size]
    round2_indices = picked[round_size:]

    selected_samples = [samples[i] for i in picked]
    round_map = {
        "round1": [samples[i].sample_id for i in round1_indices],
        "round2": [samples[i].sample_id for i in round2_indices],
    }
    return selected_samples, round_map


def serialize_sample(sample: ConversationSample) -> Dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "system": sample.system,
        "user_text": sample.user_text,
        "image_paths": list(sample.image_paths),
    }


def load_model_config(model_path: str):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = getattr(config, "text_config", None)
    rope_scaling = getattr(text_config, "rope_scaling", None) if text_config is not None else None
    rope_parameters = (
        getattr(text_config, "rope_parameters", None) if text_config is not None else None
    )

    if rope_scaling is None and isinstance(rope_parameters, dict) and text_config is not None:
        normalized_rope_scaling = dict(rope_parameters)
        if normalized_rope_scaling.get("type") == "mrope":
            normalized_rope_scaling["type"] = "default"
        if "rope_type" not in normalized_rope_scaling and "type" in normalized_rope_scaling:
            normalized_rope_scaling["rope_type"] = normalized_rope_scaling["type"]
        text_config.rope_scaling = normalized_rope_scaling
        if getattr(text_config, "rope_theta", None) is None and rope_parameters.get(
            "rope_theta"
        ) is not None:
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


def build_model_load_attempts(config: Any, device: str) -> List[Dict[str, Any]]:
    common: Dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": {"": device},
    }
    quant_cfg = getattr(config, "quantization_config", None)
    if isinstance(quant_cfg, dict) and quant_cfg.get("quant_method"):
        return [
            dict(common),
            dict(common, torch_dtype="auto"),
            dict(common, dtype="auto"),
        ]

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return [
        dict(common, dtype=dtype),
        dict(common, torch_dtype=dtype),
    ]


def load_model_with_attempts(model_class, model_path: str, attempts: Sequence[Dict[str, Any]]):
    errors: List[str] = []
    for kwargs in attempts:
        try:
            with suppress_stdout_stderr():
                return model_class.from_pretrained(model_path, **kwargs)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
    joined = "\n".join(errors)
    raise RuntimeError(f"failed to load model from {model_path}:\n{joined}")


def load_pretrained_with_fallback(model_class, model_path: str, model_kwargs: Dict[str, Any]):
    kwargs = dict(model_kwargs)
    try:
        with suppress_stdout_stderr():
            return model_class.from_pretrained(model_path, **kwargs)
    except TypeError as exc:
        if "dtype" in kwargs and "unexpected keyword argument 'dtype'" in str(exc):
            retry_kwargs = dict(kwargs)
            retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
            return load_pretrained_with_fallback(model_class, model_path, retry_kwargs)
        raise


def prepare_model_for_greedy_generation(model) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return

    generation_config.do_sample = False
    for attr in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "typical_p",
        "epsilon_cutoff",
        "eta_cutoff",
    ):
        if hasattr(generation_config, attr):
            try:
                setattr(generation_config, attr, None)
            except Exception:
                continue


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
        kwargs["extra_special_tokens"] = {}
        if not tokenizer_config.get("additional_special_tokens"):
            kwargs["additional_special_tokens"] = extra_special_tokens
    return kwargs


def move_to_device(data: Dict[str, Any], device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def load_images(image_paths: Sequence[str], max_frames: int) -> List[Image.Image]:
    images: List[Image.Image] = []
    for image_path in list(image_paths)[:max_frames]:
        try:
            images.append(Image.open(image_path).convert("RGB"))
        except Exception:
            images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
    return images


def build_messages(system: str, user_text: str, images: Sequence[Image.Image]) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = []
    for img in images:
        user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": user_content},
    ]


def prepare_multimodal_inputs(
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


def decode_generated_text(tokenizer, inputs: Dict[str, Any], output_ids: torch.Tensor) -> str:
    input_len = int(inputs["attention_mask"].sum(dim=1)[0].item())
    new_tokens = output_ids[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_outputs_for_models(
    device: str,
    model_specs: Sequence[ModelSpec],
    sample_payloads: Sequence[Dict[str, Any]],
    max_new_tokens: int,
    max_frames: int,
    progress_queue=None,
) -> List[str]:
    completed_models: List[str] = []

    for spec in model_specs:
        model = None
        processor = None
        outputs: Dict[int, str] = {}
        failed_sample_ids: List[int] = []
        oom_sample_ids: List[int] = []
        model_error = ""

        try:
            config = load_model_config(spec.path)
            model_class = resolve_model_class(config)
            load_attempts = build_model_load_attempts(config, device)
            model = load_model_with_attempts(model_class, spec.path, load_attempts)
            model.eval()
            if hasattr(model, "requires_grad_"):
                model.requires_grad_(False)
            prepare_model_for_greedy_generation(model)

            with suppress_stdout_stderr():
                processor = AutoProcessor.from_pretrained(
                    spec.path,
                    **get_processor_load_kwargs(spec.path),
                )
            if (
                hasattr(processor, "tokenizer")
                and processor.tokenizer.pad_token_id is None
                and processor.tokenizer.eos_token_id is not None
            ):
                processor.tokenizer.pad_token = processor.tokenizer.eos_token

            for sample in sample_payloads:
                sample_id = int(sample["sample_id"])
                try:
                    images = load_images(sample["image_paths"], max_frames=max_frames)
                    messages = build_messages(sample["system"], sample["user_text"], images)
                    inputs = prepare_multimodal_inputs(processor, messages, images, device)
                    with torch.inference_mode():
                        output_ids = model.generate(
                            **inputs,
                            do_sample=False,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=processor.tokenizer.pad_token_id,
                            eos_token_id=processor.tokenizer.eos_token_id,
                        )
                    outputs[sample_id] = decode_generated_text(
                        processor.tokenizer, inputs, output_ids
                    )
                    del images, messages, inputs, output_ids
                except Exception as exc:
                    outputs[sample_id] = ""
                    failed_sample_ids.append(sample_id)
                    if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower():
                        oom_sample_ids.append(sample_id)
                    print(
                        f"[GEN-ERR][{device}] sample {sample_id} failed for {spec.name}",
                        flush=True,
                    )
                    traceback.print_exc()
                finally:
                    gc.collect()
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()

        except Exception:
            model_error = traceback.format_exc()
            print(f"[GEN-ERR][{device}] model {spec.name} failed", flush=True)
            print(model_error, flush=True)
            for sample in sample_payloads:
                outputs[int(sample["sample_id"])] = ""
            failed_sample_ids = [int(sample["sample_id"]) for sample in sample_payloads]
        finally:
            del model, processor
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        completed_models.append(spec.name)
        if progress_queue is not None:
            progress_queue.put(
                {
                    "event": "generation_done",
                    "device": device,
                    "model": spec.name,
                    "path": spec.path,
                    "outputs": outputs,
                    "failed_sample_ids": failed_sample_ids,
                    "oom_sample_ids": oom_sample_ids,
                    "model_error": model_error,
                    "failed": len(failed_sample_ids),
                    "has_error": bool(model_error),
                }
            )

    return completed_models


def partition_models(
    model_specs: Sequence[ModelSpec], devices: Sequence[str]
) -> List[Tuple[str, List[ModelSpec]]]:
    buckets: List[List[ModelSpec]] = [[] for _ in devices]
    for idx, spec in enumerate(model_specs):
        buckets[idx % len(devices)].append(spec)
    return list(zip(devices, buckets))


def safe_truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def try_extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


class PsychotherapyJudge:
    def __init__(self, model_path: str, device: str, max_new_tokens: int, max_length: int):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length

        self.processor = None
        try:
            with suppress_stdout_stderr():
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    **get_processor_load_kwargs(model_path),
                )
        except Exception:
            self.processor = None

        if self.processor is not None and hasattr(self.processor, "tokenizer"):
            self.tokenizer = self.processor.tokenizer
        else:
            with suppress_stdout_stderr():
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        config = load_model_config(model_path)
        model_class = resolve_model_class(config)
        load_attempts = build_model_load_attempts(config, device)
        self.model = load_model_with_attempts(model_class, model_path, load_attempts)
        self.model.eval()
        if hasattr(self.model, "requires_grad_"):
            self.model.requires_grad_(False)
        prepare_model_for_greedy_generation(self.model)

    def close(self) -> None:
        del self.model
        if self.processor is not None:
            del self.processor
        del self.tokenizer
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def build_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.processor is not None:
            try:
                return self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return (
            f"[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n"
            f"[USER]\n{user_prompt}\n[/USER]\n[ASSISTANT]\n"
        )

    def score_triplet_batch(
        self,
        contexts: Sequence[str],
        candidate_situations: Sequence[str],
        reference_situations: Sequence[str],
        candidate_strategies: Sequence[str],
        reference_strategies: Sequence[str],
        candidate_responses: Sequence[str],
        reference_responses: Sequence[str],
        batch_size: int,
    ) -> Tuple[List[float], List[float], List[float], List[bool]]:
        assert (
            len(contexts)
            == len(candidate_situations)
            == len(reference_situations)
            == len(candidate_strategies)
            == len(reference_strategies)
            == len(candidate_responses)
            == len(reference_responses)
        )

        n = len(contexts)
        situation_totals = [0.0] * n
        strategy_totals = [0.0] * n
        response_totals = [0.0] * n
        parse_successes = [False] * n

        for variant in JUDGE_PROMPT_VARIANTS:
            variant_situation = [0.0] * n
            variant_strategy = [0.0] * n
            variant_response = [0.0] * n

            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                prompts: List[str] = []
                for idx in range(start, end):
                    user_prompt = variant["user_template"].format(
                        context=(contexts[idx] or "")[-3000:],
                        candidate_situation=(candidate_situations[idx] or "")[:1200],
                        reference_situation=(reference_situations[idx] or "")[:1200],
                        candidate_strategy=(candidate_strategies[idx] or "")[:1200],
                        reference_strategy=(reference_strategies[idx] or "")[:1200],
                        candidate_response=(candidate_responses[idx] or "")[:1200],
                        reference_response=(reference_responses[idx] or "")[:1200],
                    )
                    prompts.append(self.build_chat_prompt(variant["system"], user_prompt))

                model_inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                model_inputs = move_to_device(dict(model_inputs), self.device)

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **model_inputs,
                        do_sample=False,
                        max_new_tokens=self.max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                input_lens = model_inputs["attention_mask"].sum(dim=1).tolist()
                for row in range(output_ids.shape[0]):
                    new_tokens = output_ids[row, int(input_lens[row]) :]
                    raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    parsed, success = parse_triplet_scores(raw)
                    sample_idx = start + row
                    variant_situation[sample_idx] = parsed["situation_score"]
                    variant_strategy[sample_idx] = parsed["strategy_score"]
                    variant_response[sample_idx] = parsed["outcome_score"]
                    parse_successes[sample_idx] = parse_successes[sample_idx] or success

                del model_inputs, output_ids
                gc.collect()
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

            for idx in range(n):
                situation_totals[idx] += variant_situation[idx]
                strategy_totals[idx] += variant_strategy[idx]
                response_totals[idx] += variant_response[idx]

        denom = float(len(JUDGE_PROMPT_VARIANTS))
        return (
            [score / denom for score in situation_totals],
            [score / denom for score in strategy_totals],
            [score / denom for score in response_totals],
            parse_successes,
        )


def normalize_cosine_to_unit(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def normalize_likert_score(value: float) -> float:
    return min(1.0, max(0.0, (float(value) - 1.0) / 4.0))


def parse_triplet_scores(text: str) -> Tuple[Dict[str, float], bool]:
    parsed = try_extract_json(text)
    result = {
        "situation_score": 0.0,
        "strategy_score": 0.0,
        "outcome_score": 0.0,
    }
    valid_keys = 0

    if isinstance(parsed, dict):
        for key in result:
            if key not in parsed:
                continue
            try:
                raw_value = float(parsed[key])
            except (TypeError, ValueError):
                continue

            if 1.0 <= raw_value <= 5.0:
                result[key] = normalize_likert_score(raw_value)
                valid_keys += 1
            elif 0.0 <= raw_value <= 1.0:
                result[key] = raw_value
                valid_keys += 1

    return result, valid_keys > 0


class VisualEvidenceScorer:
    def __init__(self, model_path: str, device: str, threshold: float, top_k: int):
        self.device = device
        self.threshold = float(threshold)
        self.top_k = max(1, int(top_k))
        with suppress_stdout_stderr():
            self.processor = CLIPProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = load_pretrained_with_fallback(
            model_class=CLIPModel,
            model_path=model_path,
            model_kwargs={
                "dtype": dtype,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            },
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        text_cfg = getattr(self.model.config, "text_config", None)
        self.max_text_len = int(getattr(text_cfg, "max_position_embeddings", 77) or 77)
        if self.max_text_len <= 0:
            self.max_text_len = 77

    def close(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def score_single(
        self,
        candidate_visual_text: str,
        image_paths: Sequence[str],
    ) -> Tuple[float, float]:
        candidate_visual_text = (candidate_visual_text or "").strip()
        if not candidate_visual_text:
            return 0.0, 0.0

        images = load_images(image_paths, max_frames=MAX_FRAMES)
        if not images:
            return 0.0, 0.0

        try:
            with torch.no_grad():
                image_inputs = self.processor(images=list(images), return_tensors="pt")
                image_inputs = move_to_device(dict(image_inputs), self.device)
                image_features = self.model.get_image_features(**image_inputs)
                image_features = F.normalize(image_features, p=2, dim=-1)

                text_inputs = self.processor(
                    text=[candidate_visual_text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_text_len,
                )
                text_inputs = move_to_device(dict(text_inputs), self.device)
                text_features = self.model.get_text_features(**text_inputs)
                text_features = F.normalize(text_features, p=2, dim=-1)

                sim = torch.matmul(text_features, image_features.transpose(0, 1))
                k = min(self.top_k, sim.shape[1])
                raw_cos = torch.topk(sim, k=k, dim=1).values.mean(dim=1)
                norm_score = float(normalize_cosine_to_unit(raw_cos)[0].item())
                binary_score = 1.0 if norm_score >= self.threshold else 0.0
                return binary_score, norm_score
        except Exception:
            return 0.0, 0.0
        finally:
            for image in images:
                try:
                    image.close()
                except Exception:
                    pass
            gc.collect()
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()


class EmotionSemanticScorer:
    def __init__(self, model_path: str, device: str):
        self.device = device
        with suppress_stdout_stderr():
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = load_pretrained_with_fallback(
            model_class=AutoModel,
            model_path=model_path,
            model_kwargs={
                "dtype": dtype,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            },
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def _encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 1), dtype=torch.float32, device=self.device)

        model_inputs = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        model_inputs = move_to_device(dict(model_inputs), self.device)

        with torch.no_grad():
            output = self.model(**model_inputs)
            hidden = output.last_hidden_state
            mask = model_inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def score_batch(self, candidates: Sequence[str], references: Sequence[str]) -> List[float]:
        if not candidates:
            return []
        if len(candidates) != len(references):
            raise ValueError("candidates and references must have the same length")
        candidate_embeddings = self._encode(candidates)
        reference_embeddings = self._encode(references)
        cosine = (candidate_embeddings * reference_embeddings).sum(dim=-1)
        scores = normalize_cosine_to_unit(cosine)
        return scores.detach().cpu().tolist()


def resolve_judge_model_path() -> Path:
    override = os.environ.get("AUTO_TEST_JUDGE_MODEL_PATH")
    if override:
        return Path(override).expanduser()
    return JUDGE_MODEL_PATH


def compute_weighted_total(score_obj: Dict[str, Any]) -> float:
    total = 0.0
    for key, weight in WEIGHTS.items():
        value = float(score_obj[key]["score"])
        total += weight * value
    return total


def compute_weighted_dimension_score(key: str, score: float) -> float:
    return WEIGHTS[key] * float(score)


def average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def zero_weighted_dimensions() -> Dict[str, float]:
    return {key: 0.0 for key in WEIGHTS}


def summarize_weighted_dimensions(
    per_sample: Dict[int, Dict[str, Any]],
    sample_ids: Sequence[int],
) -> Dict[str, float]:
    return {
        key: average(
            [
                compute_weighted_dimension_score(key, per_sample[sample_id][key])
                for sample_id in sample_ids
                if sample_id in per_sample
            ]
        )
        for key in WEIGHTS
    }


def average_weighted_dimension_maps(*dimension_maps: Dict[str, float]) -> Dict[str, float]:
    if not dimension_maps:
        return zero_weighted_dimensions()
    return {
        key: average([dimension_map[key] for dimension_map in dimension_maps]) for key in WEIGHTS
    }


def format_weighted_dimensions(weighted_dims: Dict[str, float]) -> str:
    return ", ".join(
        f"{DIMENSION_DISPLAY_NAMES[key]}={weighted_dims[key]:.2f}" for key in WEIGHTS
    )


def score_output_batch(
    judge: PsychotherapyJudge,
    visual_scorer: VisualEvidenceScorer,
    emotion_scorer: EmotionSemanticScorer,
    batch_samples: Sequence[ConversationSample],
    batch_outputs: Sequence[str],
    judge_batch_size: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = [
        {
            "visual": {"score": 0.0, "reason": ""},
            "emotion": {"score": 0.0, "reason": ""},
            "situation": {"score": 0.0, "reason": ""},
            "strategy": {"score": 0.0, "reason": ""},
            "response": {"score": 0.0, "reason": ""},
            "overall_note": "",
        }
        for _ in batch_samples
    ]
    segments_cache: List[Dict[str, str]] = []
    candidate_responses: List[str] = []
    valid_indices: List[int] = []

    for idx, candidate_output in enumerate(batch_outputs):
        candidate_output = candidate_output or ""
        if not candidate_output.strip():
            results[idx]["overall_note"] = "empty_output"
            segments_cache.append({})
            candidate_responses.append("")
            continue
        valid_indices.append(idx)
        segments_cache.append(parse_think_segments(candidate_output))
        candidate_responses.append(extract_response_text(candidate_output).strip())

    for idx in valid_indices:
        first_segment = segments_cache[idx].get("first", "").strip()
        if first_segment:
            visual_binary, _ = visual_scorer.score_single(
                first_segment,
                batch_samples[idx].image_paths,
            )
            results[idx]["visual"]["score"] = float(visual_binary)

    emotion_indices: List[int] = []
    emotion_candidates: List[str] = []
    emotion_references: List[str] = []
    for idx in valid_indices:
        second_segment = segments_cache[idx].get("second", "").strip()
        reference_second = (batch_samples[idx].ground_truth_second or "").strip()
        if not second_segment or not reference_second:
            continue
        emotion_indices.append(idx)
        emotion_candidates.append(second_segment)
        emotion_references.append(reference_second)
    if emotion_indices:
        emotion_scores = emotion_scorer.score_batch(emotion_candidates, emotion_references)
        for local_idx, score in enumerate(emotion_scores):
            results[emotion_indices[local_idx]]["emotion"]["score"] = float(score)

    judge_indices: List[int] = []
    contexts: List[str] = []
    candidate_situations: List[str] = []
    reference_situations: List[str] = []
    candidate_strategies: List[str] = []
    reference_strategies: List[str] = []
    judge_candidate_responses: List[str] = []
    reference_responses: List[str] = []

    for idx in valid_indices:
        judge_indices.append(idx)
        contexts.append(batch_samples[idx].user_text)
        candidate_situations.append(segments_cache[idx].get("third", "").strip())
        reference_situations.append((batch_samples[idx].ground_truth_third or "").strip())
        candidate_strategies.append(segments_cache[idx].get("fourth", "").strip())
        reference_strategies.append((batch_samples[idx].ground_truth_fourth or "").strip())
        judge_candidate_responses.append(candidate_responses[idx])
        reference_responses.append((batch_samples[idx].ground_truth_response or "").strip())

    if judge_indices:
        situation_scores, strategy_scores, response_scores, parse_successes = judge.score_triplet_batch(
            contexts=contexts,
            candidate_situations=candidate_situations,
            reference_situations=reference_situations,
            candidate_strategies=candidate_strategies,
            reference_strategies=reference_strategies,
            candidate_responses=judge_candidate_responses,
            reference_responses=reference_responses,
            batch_size=judge_batch_size,
        )
        for local_idx, sample_idx in enumerate(judge_indices):
            results[sample_idx]["response"]["score"] = float(response_scores[local_idx])
            if candidate_situations[local_idx] and reference_situations[local_idx]:
                results[sample_idx]["situation"]["score"] = float(situation_scores[local_idx])
            if candidate_strategies[local_idx] and reference_strategies[local_idx]:
                results[sample_idx]["strategy"]["score"] = float(strategy_scores[local_idx])
            if not parse_successes[local_idx]:
                results[sample_idx]["overall_note"] = "judge_parse_failed"

    return results


def score_model_outputs(
    judge: PsychotherapyJudge,
    visual_scorer: VisualEvidenceScorer,
    emotion_scorer: EmotionSemanticScorer,
    samples: Sequence[ConversationSample],
    model_outputs: Dict[int, str],
    round_map: Dict[str, List[int]],
    batch_size: int,
    skipped_sample_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    per_sample: Dict[int, Dict[str, Any]] = {}
    skipped_ids = {int(sample_id) for sample_id in (skipped_sample_ids or [])}
    sample_list = [sample for sample in samples if sample.sample_id not in skipped_ids]

    for start in range(0, len(sample_list), batch_size):
        batch_samples = sample_list[start : start + batch_size]
        batch_outputs = [model_outputs.get(sample.sample_id, "") for sample in batch_samples]
        batch_scores = score_output_batch(
            judge=judge,
            visual_scorer=visual_scorer,
            emotion_scorer=emotion_scorer,
            batch_samples=batch_samples,
            batch_outputs=batch_outputs,
            judge_batch_size=batch_size,
        )
        for sample, score_obj in zip(batch_samples, batch_scores):
            weighted_total = compute_weighted_total(score_obj)
            per_sample[sample.sample_id] = {
                "weighted_total": weighted_total,
                "visual": float(score_obj["visual"]["score"]),
                "emotion": float(score_obj["emotion"]["score"]),
                "situation": float(score_obj["situation"]["score"]),
                "strategy": float(score_obj["strategy"]["score"]),
                "response": float(score_obj["response"]["score"]),
                "overall_note": str(score_obj.get("overall_note", "")),
            }

    round_scores: Dict[str, float] = {}
    round_weighted_dims: Dict[str, Dict[str, float]] = {}
    for round_name, sample_ids in round_map.items():
        values = [
            per_sample[sample_id]["weighted_total"]
            for sample_id in sample_ids
            if sample_id in per_sample
        ]
        round_scores[round_name] = average(values)
        round_weighted_dims[round_name] = summarize_weighted_dimensions(per_sample, sample_ids)

    all_ids = [sample.sample_id for sample in sample_list]
    summary = {
        "round1_avg": round_scores["round1"],
        "round2_avg": round_scores["round2"],
        "final_score": (round_scores["round1"] + round_scores["round2"]) / 2.0,
        "round1_weighted_dims": round_weighted_dims["round1"],
        "round2_weighted_dims": round_weighted_dims["round2"],
        "final_weighted_dims": average_weighted_dimension_maps(
            round_weighted_dims["round1"],
            round_weighted_dims["round2"],
        ),
        "visual_avg": average([per_sample[sample_id]["visual"] for sample_id in all_ids]),
        "emotion_avg": average([per_sample[sample_id]["emotion"] for sample_id in all_ids]),
        "situation_avg": average([per_sample[sample_id]["situation"] for sample_id in all_ids]),
        "strategy_avg": average([per_sample[sample_id]["strategy"] for sample_id in all_ids]),
        "response_avg": average([per_sample[sample_id]["response"] for sample_id in all_ids]),
        "judge_parse_failures": sum(
            1
            for sample_id in all_ids
            if per_sample[sample_id]["overall_note"] == "judge_parse_failed"
        ),
        "scored_sample_count": len(all_ids),
        "skipped_sample_count": len(skipped_ids),
        "per_sample": per_sample,
    }
    return summary


def zero_score_summary() -> Dict[str, Any]:
    return {
        "round1_avg": 0.0,
        "round2_avg": 0.0,
        "final_score": 0.0,
        "round1_weighted_dims": zero_weighted_dimensions(),
        "round2_weighted_dims": zero_weighted_dimensions(),
        "final_weighted_dims": zero_weighted_dimensions(),
        "visual_avg": 0.0,
        "emotion_avg": 0.0,
        "situation_avg": 0.0,
        "strategy_avg": 0.0,
        "response_avg": 0.0,
        "judge_parse_failures": 0,
        "scored_sample_count": 0,
        "skipped_sample_count": 0,
    }


def build_evaluation_entry(
    judge: PsychotherapyJudge,
    visual_scorer: VisualEvidenceScorer,
    emotion_scorer: EmotionSemanticScorer,
    samples: Sequence[ConversationSample],
    round_map: Dict[str, List[int]],
    result: Dict[str, Any],
    batch_size: int,
) -> Dict[str, Any]:
    outputs = {int(k): v for k, v in result["outputs"].items()}
    oom_sample_ids = [int(x) for x in result.get("oom_sample_ids", [])]

    if result["model_error"] and not any(text.strip() for text in outputs.values()):
        scored = zero_score_summary()
    else:
        scored = score_model_outputs(
            judge=judge,
            visual_scorer=visual_scorer,
            emotion_scorer=emotion_scorer,
            samples=samples,
            model_outputs=outputs,
            round_map=round_map,
            batch_size=batch_size,
            skipped_sample_ids=oom_sample_ids,
        )

    return {
        "name": result["name"],
        "path": result["path"],
        "round1_avg": scored["round1_avg"],
        "round2_avg": scored["round2_avg"],
        "final_score": scored["final_score"],
        "round1_weighted_dims": scored["round1_weighted_dims"],
        "round2_weighted_dims": scored["round2_weighted_dims"],
        "final_weighted_dims": scored["final_weighted_dims"],
        "visual_avg": scored["visual_avg"],
        "emotion_avg": scored["emotion_avg"],
        "situation_avg": scored["situation_avg"],
        "strategy_avg": scored["strategy_avg"],
        "response_avg": scored["response_avg"],
        "judge_parse_failures": scored["judge_parse_failures"],
        "scored_sample_count": scored["scored_sample_count"],
        "skipped_oom_count": len(oom_sample_ids),
        "failed_generation_count": len(result["failed_sample_ids"]),
        "model_error": result["model_error"],
    }


def write_score_file(
    score_file: Path,
    evaluations: Sequence[Dict[str, Any]],
    round_map: Dict[str, List[int]],
    seed: int,
    total_model_count: Optional[int] = None,
    pending_model_names: Optional[Sequence[str]] = None,
) -> None:
    lines: List[str] = []
    lines.append("Automatic psychotherapy model evaluation")
    lines.append(f"generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"random_seed: {seed}")
    lines.append(f"round_size: {ROUND_SIZE}")
    lines.append(f"round1_sample_ids: {round_map['round1']}")
    lines.append(f"round2_sample_ids: {round_map['round2']}")
    if total_model_count is not None:
        lines.append(f"completed_models: {len(evaluations)}/{total_model_count}")
    if pending_model_names:
        lines.append(f"pending_models: {list(pending_model_names)}")
    lines.append("")

    for rank, item in enumerate(evaluations, start=1):
        lines.append(f"{rank}. {item['name']}")
        lines.append(f"   path: {item['path']}")
        lines.append(f"   round1_avg: {item['round1_avg']:.4f}")
        lines.append(f"   round2_avg: {item['round2_avg']:.4f}")
        lines.append(f"   final_score: {item['final_score']:.4f}")
        lines.append(
            "   dimension_avg_norm: "
            f"visual={item['visual_avg']:.4f}, "
            f"emotion={item['emotion_avg']:.4f}, "
            f"situation={item['situation_avg']:.4f}, "
            f"strategy={item['strategy_avg']:.4f}, "
            f"response={item['response_avg']:.4f}"
        )
        lines.append(f"   failed_generations: {item['failed_generation_count']}")
        lines.append(f"   scored_samples: {item['scored_sample_count']}")
        lines.append(f"   skipped_oom_samples: {item['skipped_oom_count']}")
        lines.append(f"   judge_parse_failures: {item['judge_parse_failures']}")
        if item["model_error"]:
            lines.append("   model_error:")
            for raw_line in item["model_error"].strip().splitlines():
                lines.append(f"      {raw_line}")
        lines.append("")

    score_file.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_score_every_dimension_file(
    score_file: Path,
    evaluations: Sequence[Dict[str, Any]],
    round_map: Dict[str, List[int]],
    seed: int,
    total_model_count: Optional[int] = None,
    pending_model_names: Optional[Sequence[str]] = None,
) -> None:
    lines: List[str] = []
    lines.append("Automatic psychotherapy model evaluation with weighted dimensions")
    lines.append(f"generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"random_seed: {seed}")
    lines.append(f"round_size: {ROUND_SIZE}")
    lines.append(f"round1_sample_ids: {round_map['round1']}")
    lines.append(f"round2_sample_ids: {round_map['round2']}")
    if total_model_count is not None:
        lines.append(f"completed_models: {len(evaluations)}/{total_model_count}")
    if pending_model_names:
        lines.append(f"pending_models: {list(pending_model_names)}")
    lines.append("")

    for rank, item in enumerate(evaluations, start=1):
        lines.append(f"{rank}. {item['name']}")
        lines.append(f"   path: {item['path']}")
        lines.append(
            "   round1_avg: "
            f"{item['round1_avg']:.4f} "
            f"(weighted_dims: {format_weighted_dimensions(item['round1_weighted_dims'])})"
        )
        lines.append(
            "   round2_avg: "
            f"{item['round2_avg']:.4f} "
            f"(weighted_dims: {format_weighted_dimensions(item['round2_weighted_dims'])})"
        )
        lines.append(
            "   final_score: "
            f"{item['final_score']:.4f} "
            f"(weighted_dims: {format_weighted_dimensions(item['final_weighted_dims'])})"
        )
        lines.append(
            "   dimension_avg_norm: "
            f"{DIMENSION_DISPLAY_NAMES['visual']}={item['visual_avg']:.4f}, "
            f"{DIMENSION_DISPLAY_NAMES['emotion']}={item['emotion_avg']:.4f}, "
            f"{DIMENSION_DISPLAY_NAMES['situation']}={item['situation_avg']:.4f}, "
            f"{DIMENSION_DISPLAY_NAMES['strategy']}={item['strategy_avg']:.4f}, "
            f"{DIMENSION_DISPLAY_NAMES['response']}={item['response_avg']:.4f}"
        )
        lines.append(f"   failed_generations: {item['failed_generation_count']}")
        lines.append(f"   scored_samples: {item['scored_sample_count']}")
        lines.append(f"   skipped_oom_samples: {item['skipped_oom_count']}")
        lines.append(f"   judge_parse_failures: {item['judge_parse_failures']}")
        if item["model_error"]:
            lines.append("   model_error:")
            for raw_line in item["model_error"].strip().splitlines():
                lines.append(f"      {raw_line}")
        lines.append("")

    score_file.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    if not GENERATION_DEVICES:
        raise RuntimeError("GENERATION_DEVICES must not be empty")
    if JUDGE_DEVICE in GENERATION_DEVICES:
        raise RuntimeError(
            f"judge device {JUDGE_DEVICE} must be different from generation devices {GENERATION_DEVICES}"
        )
    judge_path = resolve_judge_model_path()
    if not judge_path.exists():
        raise RuntimeError(f"judge model path does not exist: {judge_path}")
    if not VISUAL_MODEL_PATH.exists():
        raise RuntimeError(f"visual scorer model path does not exist: {VISUAL_MODEL_PATH}")
    if not EMOTION_MODEL_PATH.exists():
        raise RuntimeError(f"emotion scorer model path does not exist: {EMOTION_MODEL_PATH}")

    judge: Optional[PsychotherapyJudge] = None
    visual_scorer: Optional[VisualEvidenceScorer] = None
    emotion_scorer: Optional[EmotionSemanticScorer] = None

    try:
        print(
            f"[AUTO-TEST] preloading judge on {JUDGE_DEVICE} "
            f"(physical {physical_cuda_label(0)}): {judge_path}"
        )
        judge = PsychotherapyJudge(
            model_path=str(judge_path),
            device=JUDGE_DEVICE,
            max_new_tokens=JUDGE_MAX_NEW_TOKENS,
            max_length=JUDGE_MAX_LENGTH,
        )
        print(f"[AUTO-TEST] judge ready: {judge_path}")
        print(f"[AUTO-TEST] preloading visual scorer on {VISUAL_DEVICE}: {VISUAL_MODEL_PATH}")
        visual_scorer = VisualEvidenceScorer(
            model_path=str(VISUAL_MODEL_PATH),
            device=VISUAL_DEVICE,
            threshold=VISUAL_THRESHOLD,
            top_k=VISUAL_TOP_K,
        )
        print(f"[AUTO-TEST] visual scorer ready: {VISUAL_MODEL_PATH}")
        print(f"[AUTO-TEST] preloading emotion scorer on {EMOTION_DEVICE}: {EMOTION_MODEL_PATH}")
        emotion_scorer = EmotionSemanticScorer(
            model_path=str(EMOTION_MODEL_PATH),
            device=EMOTION_DEVICE,
        )
        print(f"[AUTO-TEST] emotion scorer ready: {EMOTION_MODEL_PATH}")

        print("[AUTO-TEST] loading samples")
        samples = load_valid_samples(TEST_FILE)
        selected_samples, round_map = choose_round_samples(samples, ROUND_SIZE, RANDOM_SEED)
        sample_payloads = [serialize_sample(sample) for sample in selected_samples]

        model_specs = discover_candidate_models()
        if not model_specs:
            raise RuntimeError(f"no candidate models found under {FINAL_MODEL_ROOT}")

        print(f"[AUTO-TEST] selected {len(selected_samples)} samples for 2 rounds")
        print(f"[AUTO-TEST] discovered {len(model_specs)} candidate models")
        print(
            "[AUTO-TEST] visible physical GPUs: "
            + ", ".join(f"cuda:{gpu_id}" for gpu_id in VISIBLE_CUDA_DEVICE_IDS)
        )
        print(
            "[AUTO-TEST] generation devices: "
            + ", ".join(
                f"{device} (physical {physical_cuda_label(index + 1)})"
                for index, device in enumerate(GENERATION_DEVICES)
            )
        )
        print(f"[AUTO-TEST] judge device: {JUDGE_DEVICE} (physical {physical_cuda_label(0)})")

        assignments = partition_models(model_specs, GENERATION_DEVICES)
        total_models = len(model_specs)
        evaluations_by_name: Dict[str, Dict[str, Any]] = {}
        SCORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        write_score_file(
            score_file=SCORE_FILE,
            evaluations=[],
            round_map=round_map,
            seed=RANDOM_SEED,
            total_model_count=total_models,
            pending_model_names=[spec.name for spec in model_specs],
        )
        write_score_every_dimension_file(
            score_file=SCORE_EVERY_DIMENSION_FILE,
            evaluations=[],
            round_map=round_map,
            seed=RANDOM_SEED,
            total_model_count=total_models,
            pending_model_names=[spec.name for spec in model_specs],
        )

        ctx = mp.get_context("spawn")
        generation_progress_count = 0
        eval_progress_count = 0
        generation_started_at = time.time()
        eval_started_at: Optional[float] = None
        print_progress("GEN", 0, total_models, generation_started_at, "waiting")

        with mp.Manager() as manager:
            progress_queue = manager.Queue()
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(GENERATION_DEVICES),
                mp_context=ctx,
            ) as executor:
                future_map = {
                    executor.submit(
                        generate_outputs_for_models,
                        device,
                        assigned_specs,
                        sample_payloads,
                        MAX_NEW_TOKENS,
                        MAX_FRAMES,
                        progress_queue,
                    ): device
                    for device, assigned_specs in assignments
                    if assigned_specs
                }

                pending = set(future_map)
                while pending or eval_progress_count < total_models:
                    done = set()
                    if pending:
                        done, pending = concurrent.futures.wait(
                            pending,
                            timeout=1.0,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )

                    drained_event = False
                    while True:
                        try:
                            event = progress_queue.get_nowait()
                        except queue.Empty:
                            break

                        drained_event = True
                        if event.get("event") == "generation_done":
                            generation_progress_count += 1
                            suffix = event.get("model", "")
                            if event.get("has_error"):
                                suffix = f"{suffix} (error)"
                            elif event.get("failed"):
                                suffix = f"{suffix} (partial)"
                            print_progress(
                                "GEN",
                                generation_progress_count,
                                total_models,
                                generation_started_at,
                                suffix,
                            )

                            if eval_started_at is None:
                                eval_started_at = time.time()
                                print_progress("EVAL", 0, total_models, eval_started_at, "waiting")

                            result = {
                                "name": str(event["model"]),
                                "path": str(event["path"]),
                                "outputs": dict(event["outputs"]),
                                "failed_sample_ids": [int(x) for x in event["failed_sample_ids"]],
                                "oom_sample_ids": [int(x) for x in event.get("oom_sample_ids", [])],
                                "model_error": str(event.get("model_error", "")),
                            }
                            print(f"[AUTO-TEST] scoring {result['name']}", flush=True)
                            evaluation = build_evaluation_entry(
                                judge=judge,
                                visual_scorer=visual_scorer,
                                emotion_scorer=emotion_scorer,
                                samples=selected_samples,
                                round_map=round_map,
                                result=result,
                                batch_size=JUDGE_BATCH_SIZE,
                            )
                            evaluations_by_name[evaluation["name"]] = evaluation
                            eval_progress_count = len(evaluations_by_name)

                            ranked_evaluations = sorted(
                                evaluations_by_name.values(),
                                key=lambda item: (
                                    item["final_score"],
                                    item["round2_avg"],
                                    item["round1_avg"],
                                ),
                                reverse=True,
                            )
                            pending_model_names = [
                                spec.name for spec in model_specs if spec.name not in evaluations_by_name
                            ]
                            write_score_file(
                                score_file=SCORE_FILE,
                                evaluations=ranked_evaluations,
                                round_map=round_map,
                                seed=RANDOM_SEED,
                                total_model_count=total_models,
                                pending_model_names=pending_model_names,
                            )
                            write_score_every_dimension_file(
                                score_file=SCORE_EVERY_DIMENSION_FILE,
                                evaluations=ranked_evaluations,
                                round_map=round_map,
                                seed=RANDOM_SEED,
                                total_model_count=total_models,
                                pending_model_names=pending_model_names,
                            )
                            print(
                                "[AUTO-TEST] updated ranking after "
                                f"{evaluation['name']} -> {SCORE_FILE}, "
                                f"{SCORE_EVERY_DIMENSION_FILE}",
                                flush=True,
                            )

                            eval_suffix = evaluation["name"]
                            if evaluation["model_error"]:
                                eval_suffix = f"{eval_suffix} (error)"
                            elif evaluation["failed_generation_count"]:
                                eval_suffix = f"{eval_suffix} (partial)"
                            print_progress(
                                "EVAL",
                                eval_progress_count,
                                total_models,
                                eval_started_at,
                                eval_suffix,
                            )

                    for future in done:
                        device = future_map[future]
                        completed_models = future.result()
                        print(
                            f"[AUTO-TEST] generation worker finished on {device}: "
                            f"{len(completed_models)} models",
                            flush=True,
                        )

                    if not pending and not drained_event and eval_progress_count < total_models:
                        raise RuntimeError(
                            f"missing generation results: scored {eval_progress_count} of {total_models} models"
                        )

        evaluations = sorted(
            evaluations_by_name.values(),
            key=lambda item: (item["final_score"], item["round2_avg"], item["round1_avg"]),
            reverse=True,
        )
        write_score_file(
            score_file=SCORE_FILE,
            evaluations=evaluations,
            round_map=round_map,
            seed=RANDOM_SEED,
            total_model_count=total_models,
            pending_model_names=[],
        )
        write_score_every_dimension_file(
            score_file=SCORE_EVERY_DIMENSION_FILE,
            evaluations=evaluations,
            round_map=round_map,
            seed=RANDOM_SEED,
            total_model_count=total_models,
            pending_model_names=[],
        )
        print(
            f"[AUTO-TEST] wrote ranking to {SCORE_FILE} and {SCORE_EVERY_DIMENSION_FILE}"
        )
    finally:
        if judge is not None:
            judge.close()
        if visual_scorer is not None:
            visual_scorer.close()
        if emotion_scorer is not None:
            emotion_scorer.close()


if __name__ == "__main__":
    main()
