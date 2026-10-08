#!/usr/bin/env python3
"""
StickerConv-style external evaluation for image-grounded CoT + response.

Quick run example:
CUDA_VISIBLE_DEVICES=0,1 python \
  experiments/llm_blind_judge/stickerconv_eval.py

With the default config, candidate generation runs in parallel:
faces_main on cuda:0 and grpo_baseline (FIRES (Baseline)) on cuda:1. The judge model is loaded
after generation. To reserve a third visible GPU for the 32B judge, set
JUDGE_DEVICE = "cuda:2" and run with CUDA_VISIBLE_DEVICES=0,1,2.
The judge stage can also run in data parallel mode: each device in
JUDGE_DEVICES loads one judge copy and scores a slice of the samples.

You can switch GPUs / judge model / candidate models directly in the config block
below without touching the rest of the script.
"""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import re
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info

    HAS_QWEN_VL_UTILS = True
except Exception:
    HAS_QWEN_VL_UTILS = False


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def patch_transformers_awq_compat() -> None:
    # autoawq still imports the old activation symbol name on some versions.
    try:
        import transformers.activations as tf_activations
    except Exception:
        return
    if not hasattr(tf_activations, "PytorchGELUTanh") and hasattr(tf_activations, "GELUTanh"):
        tf_activations.PytorchGELUTanh = tf_activations.GELUTanh


def is_cudnn_init_error(exc: RuntimeError) -> bool:
    text = str(exc)
    return "CUDNN_STATUS_NOT_INITIALIZED" in text or "cuDNN error" in text


# ==============================================================================
# Quick Switch Config
# ==============================================================================
TEST_FILE = PROJECT_ROOT / "data" / "MESC-FT" / "test.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Candidate models to be evaluated. Edit names/paths here for new experiments.
CANDIDATE_MODELS: Dict[str, Path] = {
    "faces_main": PROJECT_ROOT / "checkpoints" / "faces_main",
    "grpo_baseline": PROJECT_ROOT / "checkpoints" / "grpo_baseline",
}
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "faces_main": "FACES",
    "grpo_baseline": "FIRES (Baseline)",
}

# Blind pairwise comparisons. Each tuple is (left_model_name, right_model_name).
PAIRWISE_MODEL_PAIRS: List[Tuple[str, str]] = [
    ("faces_main", "grpo_baseline"),
]

# GPU / model selection for generation and judging.
# Candidate generation runs in parallel workers. With CUDA_VISIBLE_DEVICES=0,1,
# cuda:0 maps to the first visible GPU and cuda:1 maps to the second visible GPU.
CANDIDATE_DEVICES: Dict[str, str] = {
    "faces_main": "cuda:0",
    "grpo_baseline": "cuda:1",
}
CANDIDATE_DEFAULT_DEVICE = "cuda:0"
PARALLEL_CANDIDATE_GENERATION = True
CANDIDATE_PARALLEL_WORKERS = 0  # 0 means one worker per candidate model.
JUDGE_MODEL_PATH = PROJECT_ROOT / "models" / "judge-32b"
JUDGE_DEVICE = "cuda:1"
PARALLEL_JUDGE_EVALUATION = True
JUDGE_DEVICES: List[str] = ["cuda:0", "cuda:1"]
JUDGE_PARALLEL_WORKERS = 0  # 0 means one worker per judge device.

# Generation settings for candidate models.
GEN_MAX_NEW_TOKENS = 256
GEN_DO_SAMPLE = False
GEN_TEMPERATURE = 0.0
GEN_TOP_P = 1.0

# Judge settings.
JUDGE_MAX_NEW_TOKENS = 256
JUDGE_DO_SAMPLE = False
JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 1.0

# Evaluation scope.
MAX_IMAGES = 8


def display_model_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


SAMPLE_LIMIT = 0  # 0 means use the whole test set.
FORCE_REGENERATE = False
RUN_POINTWISE = True
RUN_PAIRWISE = True
RUN_GENERATION = True  # False means use cached predictions only; missing cache raises.
# ==============================================================================


STRICT_FORMAT_PATTERN = re.compile(
    r"<think>[\s\S]*?</think>\s*<response>[\s\S]*?</response>",
    re.IGNORECASE,
)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
RESPONSE_PATTERN = re.compile(r"<response>(.*?)</response>", re.IGNORECASE | re.DOTALL)
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"(?:<image>\s*)+", re.IGNORECASE)

VERBOSITY_PENALTY_RULE = (
    "CRITICAL RULE: Penalize overly long, verbose, or formulaic candidates with "
    "redundant or padded wording. If a candidate is substantially longer than "
    "necessary but does not improve functional alignment or empathy, you MUST "
    "lower its score or make it lose."
)

THINK_MISSING_NOTE = (
    "Note: Some candidates may not have a [Think] section. Judge them primarily "
    "on the quality of the [Response], while using available reasoning only when "
    "present."
)

POINTWISE_JSON_EXAMPLES = {
    "empathy_text": '{"empathy_text": 1}',
    "multimodal_appropriateness": '{"multimodal_appropriateness": 1}',
    "context_response_consistency": '{"context_response_consistency": 1}',
}

PAIRWISE_JSON_EXAMPLE = '{"winner": "A"}'


POINTWISE_METRIC_DEFINITIONS = {
    "empathy_text": (
        "Whether the final response expresses appropriate empathy, emotional "
        "validation, and supportive wording."
    ),
    "multimodal_appropriateness": (
        "Whether the response is consistent with the dialogue context and "
        "available visual evidence."
    ),
    "context_response_consistency": (
        "Whether the final response is consistent with the dialogue context, "
        "visual evidence, and support direction."
    ),
}

POINTWISE_METRIC_NAMES = {
    "empathy_text": "Empathy",
    "multimodal_appropriateness": "Multimodal appropriateness",
    "context_response_consistency": "Context-Response Consistency",
}


def escaped_json_format(metric: str) -> str:
    return POINTWISE_JSON_EXAMPLES[metric].replace("{", "{{").replace("}", "}}")


def build_pointwise_prompt(metric: str, wording: str) -> str:
    return (
        f"{wording}\n"
        "Use the dialogue context, any attached visual evidence, the model's "
        "think section if available, and the final response. If the think "
        "section is missing, it will be shown as Not provided.\n"
        "Your judgment must consider:\n"
        "1. whether the candidate is appropriate to the user's emotional state "
        "and dialogue context;\n"
        "2. whether it is grounded in the available multimodal evidence;\n"
        "3. whether it avoids unsupported, unsafe, overly verbose, repetitive, "
        "or formulaic content;\n"
        "4. whether it satisfies the specific evaluation metric.\n"
        "[EVALUATION_METRIC]\n"
        f"{POINTWISE_METRIC_NAMES[metric]}\n"
        "[METRIC_DEFINITION]\n"
        f"{POINTWISE_METRIC_DEFINITIONS[metric]}\n"
        "[CONTEXT]\n"
        "{conversation}\n"
        "[MODEL_THINK]\n"
        "{think}\n"
        "[CANDIDATE_RESPONSE]\n"
        "{response}\n"
        "Score the candidate from 1 to 5. Return JSON only in the following format:\n"
        f"{escaped_json_format(metric)}"
    )


POINTWISE_PROMPTS = {
    metric: (
        {
            "system": "You are an impartial evaluator for multimodal emotional support dialogue.",
            "user": build_pointwise_prompt(
                metric,
                "Evaluate the candidate output according to the specified metric.",
            ),
        },
        {
            "system": "You are an impartial judge for multimodal emotional support dialogue.",
            "user": build_pointwise_prompt(
                metric,
                "Assess the candidate response according to the specified metric.",
            ),
        },
        {
            "system": "You are a neutral evaluator of multimodal emotional support dialogue.",
            "user": build_pointwise_prompt(
                metric,
                "Score the candidate output using the specified evaluation metric.",
            ),
        },
    )
    for metric in POINTWISE_JSON_EXAMPLES
}


PAIRWISE_PROMPTS = (
    {
        "system": "You are an impartial judge for multimodal emotional support dialogue.",
        "user": (
            "Compare Candidate A and Candidate B based on the dialogue context, "
            "any attached visual evidence, the provided think sections if available, "
            "and the final responses. The model identities are hidden. If a think "
            "section is missing, it will be shown as Not provided.\n"
            "Choose the better candidate according to the following criteria:\n"
            "1. empathy and emotional validation;\n"
            "2. contextual relevance to the user's situation;\n"
            "3. consistency with the visual evidence when available;\n"
            "4. appropriateness and safety of the support strategy;\n"
            "5. coherence between the think section and the final response;\n"
            "6. avoidance of unnecessarily long, repetitive, verbose, or formulaic wording.\n"
            "If both candidates are comparable in overall quality, choose TIE.\n"
            "[CONTEXT]\n{conversation}\n"
            "[CANDIDATE A THINK]\n{think_a}\n"
            "[CANDIDATE A RESPONSE]\n{response_a}\n"
            "[CANDIDATE B THINK]\n{think_b}\n"
            "[CANDIDATE B RESPONSE]\n{response_b}\n"
            "Return JSON only in the following format:\n"
            '{{"winner": "A"}}\n'
            "The value of winner must be one of A, B, or TIE."
        ),
    },
    {
        "system": "You are a neutral judge for multimodal emotional support dialogue.",
        "user": (
            "Compare Candidate A and Candidate B for the same multimodal dialogue. "
            "Use the context, visual evidence, available think sections, and final "
            "responses. Model identities are hidden.\n"
            "Prefer the candidate with stronger empathy, contextual relevance, "
            "visual consistency, support-strategy safety, think-response coherence, "
            "and less repetitive or formulaic wording. If both are comparable, choose TIE.\n"
            "[CONTEXT]\n{conversation}\n"
            "[CANDIDATE A THINK]\n{think_a}\n"
            "[CANDIDATE A RESPONSE]\n{response_a}\n"
            "[CANDIDATE B THINK]\n{think_b}\n"
            "[CANDIDATE B RESPONSE]\n{response_b}\n"
            'Return JSON only in the following format: {{"winner": "A"}}\n'
            "The value of winner must be one of A, B, or TIE."
        ),
    },
    {
        "system": "You are an impartial evaluator for pairwise multimodal emotional support quality.",
        "user": (
            "Choose the better candidate for the multimodal emotional support dialogue. "
            "Judge empathy, emotional validation, context relevance, grounding in "
            "visual evidence, support-strategy appropriateness and safety, coherence "
            "between think content and response, and avoidance of excessive verbosity. "
            "Use TIE if the two candidates are comparable.\n"
            "[CONTEXT]\n{conversation}\n"
            "[CANDIDATE A THINK]\n{think_a}\n"
            "[CANDIDATE A RESPONSE]\n{response_a}\n"
            "[CANDIDATE B THINK]\n{think_b}\n"
            "[CANDIDATE B RESPONSE]\n{response_b}\n"
            'Return JSON only in the following format: {{"winner": "A"}}\n'
            "The value of winner must be one of A, B, or TIE."
        ),
    },
)


@dataclass
class EvalSample:
    sample_id: int
    system: str
    conversation: str
    image_paths: List[str]
    ground_truth_full: str


@dataclass
class ModelPrediction:
    sample_id: int
    model_name: str
    raw_output: str
    think: str
    response: str
    format_valid: bool
    think_missing: bool
    response_missing: bool


def extract_think_text(text: str) -> str:
    match = THINK_PATTERN.search(text or "")
    return match.group(1).strip() if match else ""


def extract_response_text(text: str) -> str:
    match = RESPONSE_PATTERN.search(text or "")
    if match:
        return match.group(1).strip()
    return THINK_PATTERN.sub("", text or "").strip()


def is_format_valid(text: str) -> bool:
    return STRICT_FORMAT_PATTERN.search(text or "") is not None


def strip_image_placeholders(text: str) -> str:
    cleaned = IMAGE_PLACEHOLDER_PATTERN.sub("", text or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def build_model_prediction(sample_id: int, model_name: str, raw_output: str) -> ModelPrediction:
    raw_output = raw_output or ""
    return ModelPrediction(
        sample_id=sample_id,
        model_name=model_name,
        raw_output=raw_output,
        think=extract_think_text(raw_output),
        response=extract_response_text(raw_output),
        format_valid=is_format_valid(raw_output),
        think_missing=THINK_PATTERN.search(raw_output) is None,
        response_missing=RESPONSE_PATTERN.search(raw_output) is None,
    )


def format_think_for_judge(prediction: ModelPrediction) -> str:
    return prediction.think if prediction.think else "Not provided."


def format_response_for_judge(prediction: ModelPrediction) -> str:
    return prediction.response if prediction.response else "(empty response)"


def append_pointwise_judge_rules(user_prompt: str, metric: str) -> str:
    rules = []
    if metric != "empathy_text":
        rules.append(THINK_MISSING_NOTE)
    rules.extend(
        [
            VERBOSITY_PENALTY_RULE,
            f"Return your output EXACTLY as the following JSON format: {POINTWISE_JSON_EXAMPLES[metric]}",
        ]
    )
    return f"{user_prompt}\n\n[Evaluation Rules]\n" + "\n".join(rules)


def append_pairwise_judge_rules(user_prompt: str) -> str:
    rules = [
        THINK_MISSING_NOTE,
        VERBOSITY_PENALTY_RULE,
        f"Return your output EXACTLY as the following JSON format: {PAIRWISE_JSON_EXAMPLE}",
    ]
    return f"{user_prompt}\n\n[Evaluation Rules]\n" + "\n".join(rules)


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


def ensure_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available, but device={device}")


def load_model_config(model_path: str):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = getattr(config, "text_config", None)
    rope_scaling = getattr(text_config, "rope_scaling", None) if text_config is not None else None
    rope_parameters = getattr(text_config, "rope_parameters", None) if text_config is not None else None

    if text_config is not None and rope_scaling is None and isinstance(rope_parameters, dict):
        normalized = dict(rope_parameters)
        if normalized.get("type") == "mrope":
            normalized["type"] = "default"
        if "rope_type" not in normalized and "type" in normalized:
            normalized["rope_type"] = normalized["type"]
        text_config.rope_scaling = normalized
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


def is_quantized_config(config) -> bool:
    quant_cfg = getattr(config, "quantization_config", None)
    return isinstance(quant_cfg, dict) and bool(quant_cfg.get("quant_method"))


class VisionLanguageRunner:
    def __init__(self, model_path: str, device: str):
        self.model_path = str(model_path)
        self.device = resolve_device(device)
        ensure_device(self.device)
        patch_transformers_awq_compat()

        config = load_model_config(self.model_path)
        model_class = resolve_model_class(config)
        model_kwargs: Dict[str, Any] = {
            "config": config,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": {"": self.device},
        }
        if not is_quantized_config(config):
            model_kwargs["dtype"] = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        self.model = load_model_with_fallback(model_class, self.model_path, model_kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            **get_processor_load_kwargs(self.model_path),
        )
        if getattr(self.processor, "tokenizer", None) is not None:
            tokenizer = self.processor.tokenizer
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token

    def close(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    @staticmethod
    def _load_images(image_paths: Sequence[str], max_images: int) -> List[Image.Image]:
        images: List[Image.Image] = []
        for image_path in list(image_paths)[:max_images]:
            try:
                with Image.open(image_path) as img:
                    images.append(img.convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
        return images

    @staticmethod
    def _close_images(images: Sequence[Image.Image]) -> None:
        for img in images:
            try:
                img.close()
            except Exception:
                pass

    def _prepare_inputs(
        self,
        messages: List[Dict[str, Any]],
        raw_images: Sequence[Image.Image],
    ) -> Dict[str, Any]:
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        tokenizer = self.processor.tokenizer
        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            if HAS_QWEN_VL_UTILS:
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=[text_prompt],
                    images=list(raw_images) if raw_images else None,
                    padding=True,
                    return_tensors="pt",
                )
        finally:
            tokenizer.padding_side = old_padding_side

        out: Dict[str, Any] = {}
        for key, value in dict(inputs).items():
            out[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value
        return out

    def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        image_paths: Sequence[str],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        max_images: int,
    ) -> str:
        raw_images = self._load_images(image_paths, max_images)
        try:
            inputs = self._prepare_inputs(messages, raw_images)
            generate_kwargs = dict(
                inputs,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            if do_sample:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = top_p

            try:
                with torch.inference_mode():
                    output_ids = self.model.generate(**generate_kwargs)
            except RuntimeError as exc:
                if not image_paths or not is_cudnn_init_error(exc):
                    raise
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                # Some Qwen2.5-VL AWQ judge runs fail in the visual conv3d path when
                # cuDNN does not initialize cleanly. Retrying once with cuDNN disabled
                # is slower but much more stable for long evaluation jobs.
                with torch.backends.cudnn.flags(enabled=False):
                    with torch.inference_mode():
                        output_ids = self.model.generate(**generate_kwargs)

            input_len = int(inputs["attention_mask"].sum(dim=1)[0].item())
            new_tokens = output_ids[0, input_len:]
            return self.processor.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()
        finally:
            self._close_images(raw_images)


def build_candidate_messages(sample: EvalSample) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = []
    for img_path in sample.image_paths[:MAX_IMAGES]:
        user_content.append({"type": "image", "image": img_path})
    user_content.append({"type": "text", "text": sample.conversation})
    return [
        {"role": "system", "content": [{"type": "text", "text": sample.system}]},
        {"role": "user", "content": user_content},
    ]


def build_judge_messages(
    system_prompt: str,
    user_prompt: str,
    image_paths: Sequence[str],
) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = []
    for img_path in list(image_paths)[:MAX_IMAGES]:
        user_content.append({"type": "image", "image": img_path})
    user_content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


def parse_sample(item: Dict[str, Any], sample_id: int) -> Optional[EvalSample]:
    system = str(item.get("system", "")).strip()
    conversation = ""
    image_paths: List[str] = []
    ground_truth_full = ""

    for turn in item.get("conversations", []):
        role = turn.get("from")
        if role == "user":
            conversation = str(turn.get("value", "")).strip()
            image_paths = list(turn.get("images", []) or [])
        elif role == "assistant":
            ground_truth_full = str(turn.get("value", "")).strip()

    if not conversation:
        return None

    return EvalSample(
        sample_id=sample_id,
        system=system or "You are a helpful therapist.",
        conversation=conversation,
        image_paths=image_paths,
        ground_truth_full=ground_truth_full,
    )


def load_test_samples(test_file: Path, sample_limit: int) -> List[EvalSample]:
    samples: List[EvalSample] = []
    with test_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                sample = parse_sample(item, idx)
            except Exception:
                sample = None
            if sample is None:
                continue
            samples.append(sample)
            if sample_limit > 0 and len(samples) >= sample_limit:
                break
    return samples


def ensure_output_dirs() -> Dict[str, Path]:
    generation_dir = OUTPUT_DIR / "generations"
    pointwise_dir = OUTPUT_DIR / "pointwise"
    pairwise_dir = OUTPUT_DIR / "pairwise"
    for path in (OUTPUT_DIR, generation_dir, pointwise_dir, pairwise_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": OUTPUT_DIR,
        "generation": generation_dir,
        "pointwise": pointwise_dir,
        "pairwise": pairwise_dir,
    }


def load_prediction_cache(cache_path: Path) -> Dict[int, ModelPrediction]:
    if not cache_path.is_file():
        return {}
    records: Dict[int, ModelPrediction] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                raw_output = str(obj.get("raw_output", ""))
                pred = build_model_prediction(
                    sample_id=int(obj["sample_id"]),
                    model_name=str(obj["model_name"]),
                    raw_output=raw_output,
                )
                pred.think = str(obj.get("think") or pred.think)
                cached_response = str(obj.get("response") or "")
                if cached_response and THINK_PATTERN.search(cached_response) is None:
                    pred.response = cached_response
                pred.format_valid = bool(obj.get("format_valid", pred.format_valid))
                pred.think_missing = bool(obj.get("think_missing", pred.think_missing))
                pred.response_missing = bool(obj.get("response_missing", pred.response_missing))
            except Exception:
                continue
            records[pred.sample_id] = pred
    return records


def save_predictions(cache_path: Path, predictions: Sequence[ModelPrediction]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(asdict(pred), ensure_ascii=False) + "\n")


def append_prediction(cache_path: Path, prediction: ModelPrediction) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(prediction), ensure_ascii=False) + "\n")
        f.flush()


def generate_predictions_for_model(
    model_name: str,
    model_path: Path,
    samples: Sequence[EvalSample],
    device: str,
    cache_path: Path,
    force_regenerate: bool,
    allow_generate: bool,
    progress_position: int = 0,
) -> List[ModelPrediction]:
    cached = load_prediction_cache(cache_path) if not force_regenerate else {}
    missing = [sample for sample in samples if sample.sample_id not in cached]
    if not missing:
        ordered = [cached[sample.sample_id] for sample in samples]
        return ordered
    if not allow_generate:
        missing_ids = [sample.sample_id for sample in missing[:10]]
        raise RuntimeError(
            f"Missing {len(missing)} cached predictions for {model_name} at {cache_path}; "
            f"RUN_GENERATION=False, so generation is disabled. First missing sample ids: {missing_ids}"
        )
    if force_regenerate:
        cache_path.unlink(missing_ok=True)

    runner = VisionLanguageRunner(str(model_path), device)
    try:
        for sample in tqdm(
            missing,
            desc=f"Generate {model_name} on {device}",
            ncols=100,
            position=progress_position,
            leave=True,
        ):
            raw_output = runner.generate_chat(
                messages=build_candidate_messages(sample),
                image_paths=sample.image_paths,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=GEN_DO_SAMPLE,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                max_images=MAX_IMAGES,
            )
            prediction = build_model_prediction(sample.sample_id, model_name, raw_output)
            cached[sample.sample_id] = prediction
            append_prediction(cache_path, prediction)
    finally:
        runner.close()

    ordered = [cached[sample.sample_id] for sample in samples]
    # Compact the incremental cache into deterministic sample order after a clean finish.
    save_predictions(cache_path, ordered)
    return ordered


def _generate_predictions_worker(
    task: Tuple[str, str, List[EvalSample], str, str, bool, bool, int],
) -> Tuple[str, List[ModelPrediction]]:
    (
        model_name,
        model_path,
        samples,
        device,
        cache_path,
        force_regenerate,
        allow_generate,
        progress_position,
    ) = task
    predictions = generate_predictions_for_model(
        model_name=model_name,
        model_path=Path(model_path),
        samples=samples,
        device=device,
        cache_path=Path(cache_path),
        force_regenerate=force_regenerate,
        allow_generate=allow_generate,
        progress_position=progress_position,
    )
    return model_name, predictions


def get_candidate_device(model_name: str) -> str:
    return CANDIDATE_DEVICES.get(model_name, CANDIDATE_DEFAULT_DEVICE)


def generate_all_predictions(
    candidate_models: Dict[str, Path],
    samples: Sequence[EvalSample],
    generation_dir: Path,
) -> Dict[str, List[ModelPrediction]]:
    tasks: List[Tuple[str, str, List[EvalSample], str, str, bool, bool, int]] = []
    sample_list = list(samples)
    for position, (model_name, model_path) in enumerate(candidate_models.items()):
        tasks.append(
            (
                model_name,
                str(model_path),
                sample_list,
                get_candidate_device(model_name),
                str(generation_dir / f"{model_name}.jsonl"),
                FORCE_REGENERATE and RUN_GENERATION,
                RUN_GENERATION,
                position,
            )
        )

    if not PARALLEL_CANDIDATE_GENERATION or len(tasks) <= 1:
        predictions: Dict[str, List[ModelPrediction]] = {}
        for task in tasks:
            model_name, model_predictions = _generate_predictions_worker(task)
            predictions[model_name] = model_predictions
        return {name: predictions[name] for name in candidate_models}

    worker_count = CANDIDATE_PARALLEL_WORKERS or len(tasks)
    worker_count = max(1, min(worker_count, len(tasks)))
    plan = ", ".join(f"{task[0]}->{task[3]}" for task in tasks)
    print(f"[GEN] parallel candidate generation with {worker_count} workers: {plan}")

    predictions = {}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = {executor.submit(_generate_predictions_worker, task): task[0] for task in tasks}
        for future in as_completed(futures):
            model_name, model_predictions = future.result()
            predictions[model_name] = model_predictions
            print(f"[GEN] finished {model_name}: {len(model_predictions)} samples")

    return {name: predictions[name] for name in candidate_models}


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def normalize_likert(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 1.0 or score > 5.0:
        return None
    return score


def parse_winner(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"A", "B", "TIE"}:
        return text
    return "TIE"


class StickerConvJudge:
    def __init__(self, model_path: str, device: str):
        self.runner = VisionLanguageRunner(model_path, device)

    def close(self) -> None:
        self.runner.close()

    def _judge_once(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[str],
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        raw_text = self.runner.generate_chat(
            messages=build_judge_messages(system_prompt, user_prompt, image_paths),
            image_paths=image_paths,
            max_new_tokens=JUDGE_MAX_NEW_TOKENS,
            do_sample=JUDGE_DO_SAMPLE,
            temperature=JUDGE_TEMPERATURE,
            top_p=JUDGE_TOP_P,
            max_images=MAX_IMAGES,
        )
        return extract_first_json_object(raw_text), raw_text

    def score_pointwise(
        self,
        sample: EvalSample,
        prediction: ModelPrediction,
        task_name: str,
    ) -> Dict[str, Any]:
        prompts = POINTWISE_PROMPTS[task_name]
        scores: List[float] = []
        variant_records: List[Dict[str, Any]] = []
        for idx, prompt in enumerate(prompts):
            if task_name == "empathy_text":
                image_paths: Sequence[str] = []
                conversation = strip_image_placeholders(sample.conversation)
            else:
                image_paths = sample.image_paths
                conversation = sample.conversation
            base_prompt = prompt["user"].format(
                conversation=conversation,
                think=format_think_for_judge(prediction),
                response=format_response_for_judge(prediction),
            )
            user_prompt = append_pointwise_judge_rules(base_prompt, task_name)
            parsed, raw_text = self._judge_once(prompt["system"], user_prompt, image_paths)
            score_value = None if parsed is None else normalize_likert(parsed.get(task_name))
            if score_value is not None:
                scores.append(score_value)
            variant_records.append(
                {
                    "variant_id": idx,
                    "parsed": parsed,
                    "raw_output": raw_text,
                    "score": score_value,
                    "valid": score_value is not None,
                }
            )

        avg_score = statistics.mean(scores) if scores else None
        return {
            "valid": avg_score is not None,
            "valid_variant_count": len(scores),
            "invalid_variant_count": len(prompts) - len(scores),
            "score_1to5": avg_score,
            "score_0to1": None if avg_score is None else (avg_score - 1.0) / 4.0,
            "variants": variant_records,
        }

    def score_pairwise(
        self,
        sample: EvalSample,
        left_name: str,
        left_pred: ModelPrediction,
        right_name: str,
        right_pred: ModelPrediction,
    ) -> Dict[str, Any]:
        votes: List[str] = []
        variant_records: List[Dict[str, Any]] = []

        def judge_round(
            prompt: Dict[str, str],
            a_name: str,
            a_pred: ModelPrediction,
            b_name: str,
            b_pred: ModelPrediction,
        ) -> Dict[str, Any]:
            base_prompt = prompt["user"].format(
                conversation=sample.conversation,
                think_a=format_think_for_judge(a_pred),
                response_a=format_response_for_judge(a_pred),
                think_b=format_think_for_judge(b_pred),
                response_b=format_response_for_judge(b_pred),
            )
            user_prompt = append_pairwise_judge_rules(base_prompt)
            parsed, raw_text = self._judge_once(prompt["system"], user_prompt, sample.image_paths)
            raw_winner = None if parsed is None else parsed.get("winner")
            winner_label = "TIE" if parsed is None else parse_winner(raw_winner)
            valid = parsed is not None and str(raw_winner or "").strip().upper() in {"A", "B", "TIE"}
            if winner_label == "A":
                mapped_winner = a_name
            elif winner_label == "B":
                mapped_winner = b_name
            else:
                mapped_winner = "TIE"
            return {
                "candidate_a_name": a_name,
                "candidate_b_name": b_name,
                "parsed": parsed,
                "raw_output": raw_text,
                "winner_label": winner_label,
                "mapped_winner": mapped_winner,
                "valid": valid,
            }

        for idx, prompt in enumerate(PAIRWISE_PROMPTS):
            round1 = judge_round(
                prompt=prompt,
                a_name=left_name,
                a_pred=left_pred,
                b_name=right_name,
                b_pred=right_pred,
            )
            round2 = judge_round(
                prompt=prompt,
                a_name=right_name,
                a_pred=right_pred,
                b_name=left_name,
                b_pred=left_pred,
            )

            mapped_winner_1 = round1["mapped_winner"]
            mapped_winner_2 = round2["mapped_winner"]
            if mapped_winner_1 == mapped_winner_2:
                final_variant_winner = mapped_winner_1
            elif mapped_winner_1 == "TIE":
                final_variant_winner = mapped_winner_2
            elif mapped_winner_2 == "TIE":
                final_variant_winner = mapped_winner_1
            else:
                final_variant_winner = "TIE"

            votes.append(final_variant_winner)
            variant_records.append(
                {
                    "variant_id": idx,
                    "round1": round1,
                    "round2": round2,
                    "final_variant_winner": final_variant_winner,
                }
            )

        count_left = votes.count(left_name)
        count_right = votes.count(right_name)
        if count_left > count_right:
            final_winner = left_name
            final_label = "LEFT"
        elif count_right > count_left:
            final_winner = right_name
            final_label = "RIGHT"
        else:
            final_winner = "TIE"
            final_label = "TIE"

        round_count = len(variant_records) * 2
        valid_round_count = sum(
            1
            for record in variant_records
            for round_record in (record["round1"], record["round2"])
            if round_record["valid"]
        )
        return {
            "sample_id": sample.sample_id,
            "left_name": left_name,
            "right_name": right_name,
            "bidirectional_swap": True,
            "winner_label": final_label,
            "winner_name": final_winner,
            "valid_round_count": valid_round_count,
            "invalid_round_count": round_count - valid_round_count,
            "variants": variant_records,
        }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any], lock: Any = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def write_one() -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    if lock is None:
        write_one()
        return
    with lock:
        write_one()


def load_jsonl_records(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.is_file():
        return {}
    records: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                sample_id = int(row["sample_id"])
            except Exception:
                continue
            records[sample_id] = row
    return records


def ordered_cached_records(
    cache: Dict[int, Dict[str, Any]],
    samples: Sequence[EvalSample],
) -> List[Dict[str, Any]]:
    return [cache[sample.sample_id] for sample in samples if sample.sample_id in cache]


def missing_sample_ids(cache: Dict[int, Dict[str, Any]], samples: Sequence[EvalSample]) -> Set[int]:
    return {sample.sample_id for sample in samples if sample.sample_id not in cache}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_pointwise(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"num_samples": len(records)}
    for metric in POINTWISE_PROMPTS:
        metric_rows = [row[metric] for row in records]
        valid_rows = [row for row in metric_rows if row.get("valid")]
        scores_1to5 = [float(row["score_1to5"]) for row in valid_rows if row["score_1to5"] is not None]
        scores_0to1 = [float(row["score_0to1"]) for row in valid_rows if row["score_0to1"] is not None]
        variant_total = sum(
            int(row.get("valid_variant_count", 0)) + int(row.get("invalid_variant_count", 0))
            for row in metric_rows
        )
        invalid_variants = sum(int(row.get("invalid_variant_count", 0)) for row in metric_rows)
        denom = len(records) if records else 1
        variant_denom = variant_total if variant_total > 0 else 1
        summary[metric] = {
            "mean_1to5": statistics.mean(scores_1to5) if scores_1to5 else None,
            "mean_0to1": statistics.mean(scores_0to1) if scores_0to1 else None,
            "valid_count": len(scores_1to5),
            "invalid_count": len(records) - len(scores_1to5),
            "invalid_rate": (len(records) - len(scores_1to5)) / denom,
            "invalid_variant_count": invalid_variants,
            "invalid_variant_rate": invalid_variants / variant_denom,
        }
    return summary


def summarize_pairwise(records: Sequence[Dict[str, Any]], left_name: str, right_name: str) -> Dict[str, Any]:
    total = len(records)
    left_win = sum(1 for row in records if row["winner_name"] == left_name)
    right_win = sum(1 for row in records if row["winner_name"] == right_name)
    tie = sum(1 for row in records if row["winner_name"] == "TIE")
    invalid_rounds = sum(int(row.get("invalid_round_count", 0)) for row in records)
    total_rounds = sum(
        int(row.get("valid_round_count", 0)) + int(row.get("invalid_round_count", 0))
        for row in records
    )
    denom = total if total > 0 else 1
    round_denom = total_rounds if total_rounds > 0 else 1
    return {
        "left_name": left_name,
        "right_name": right_name,
        "left_display_name": display_model_name(left_name),
        "right_display_name": display_model_name(right_name),
        "num_samples": total,
        "win_rate_left": left_win / denom,
        "win_rate_right": right_win / denom,
        "tie_rate": tie / denom,
        "left_wins": left_win,
        "right_wins": right_win,
        "ties": tie,
        "invalid_round_count": invalid_rounds,
        "invalid_round_rate": invalid_rounds / round_denom,
    }


def build_pointwise_record(
    judge: StickerConvJudge,
    sample: EvalSample,
    pred: ModelPrediction,
) -> Dict[str, Any]:
    row = {
        "sample_id": sample.sample_id,
        "model_name": pred.model_name,
        "raw_output": pred.raw_output,
        "think": pred.think,
        "response": pred.response,
        "format_valid": pred.format_valid,
        "think_missing": pred.think_missing,
        "response_missing": pred.response_missing,
    }
    for metric in POINTWISE_PROMPTS:
        row[metric] = judge.score_pointwise(sample, pred, metric)
    return row


def split_aligned(*items: Sequence[Any], parts: int) -> List[Tuple[List[Any], ...]]:
    if not items:
        return []
    total = len(items[0])
    parts = max(1, min(parts, total))
    chunks: List[Tuple[List[Any], ...]] = []
    for idx in range(parts):
        start = total * idx // parts
        end = total * (idx + 1) // parts
        chunks.append(tuple(list(seq[start:end]) for seq in items))
    return chunks


def get_judge_devices() -> List[str]:
    if not PARALLEL_JUDGE_EVALUATION:
        return [JUDGE_DEVICE]
    devices = [device for device in JUDGE_DEVICES if device]
    return devices or [JUDGE_DEVICE]


def run_pointwise(
    judge: StickerConvJudge,
    samples: Sequence[EvalSample],
    predictions: Sequence[ModelPrediction],
    output_path: Path,
    summary_path: Path,
) -> None:
    cached = load_jsonl_records(output_path)
    missing_ids = missing_sample_ids(cached, samples)
    if not missing_ids:
        records = ordered_cached_records(cached, samples)
        write_jsonl(output_path, records)
        write_json(summary_path, summarize_pointwise(records))
        print(f"[JUDGE] pointwise cache hit: {output_path}")
        return

    for sample, pred in tqdm(
        [(sample, pred) for sample, pred in zip(samples, predictions) if sample.sample_id in missing_ids],
        desc=f"Pointwise {predictions[0].model_name if predictions else 'model'}",
        ncols=100,
    ):
        row = build_pointwise_record(judge, sample, pred)
        cached[sample.sample_id] = row
        append_jsonl(output_path, row)

    records = ordered_cached_records(cached, samples)
    write_jsonl(output_path, records)
    write_json(summary_path, summarize_pointwise(records))


def run_pairwise(
    judge: StickerConvJudge,
    samples: Sequence[EvalSample],
    left_predictions: Sequence[ModelPrediction],
    right_predictions: Sequence[ModelPrediction],
    left_name: str,
    right_name: str,
    output_path: Path,
    summary_path: Path,
) -> None:
    cached = load_jsonl_records(output_path)
    missing_ids = missing_sample_ids(cached, samples)
    if not missing_ids:
        records = ordered_cached_records(cached, samples)
        write_jsonl(output_path, records)
        write_json(summary_path, summarize_pairwise(records, left_name, right_name))
        print(f"[JUDGE] pairwise cache hit: {output_path}")
        return

    for sample, left_pred, right_pred in tqdm(
        [
            (sample, left_pred, right_pred)
            for sample, left_pred, right_pred in zip(samples, left_predictions, right_predictions)
            if sample.sample_id in missing_ids
        ],
        desc=f"Pairwise {left_name} vs {right_name}",
        ncols=100,
    ):
        row = judge.score_pairwise(sample, left_name, left_pred, right_name, right_pred)
        cached[sample.sample_id] = row
        append_jsonl(output_path, row)

    records = ordered_cached_records(cached, samples)
    write_jsonl(output_path, records)
    write_json(summary_path, summarize_pairwise(records, left_name, right_name))


def _run_pointwise_judge_worker(
    task: Tuple[int, str, str, List[EvalSample], List[ModelPrediction], str, str, Any],
) -> Tuple[int, int]:
    shard_id, model_name, judge_model_path, sample_chunk, pred_chunk, device, output_path, write_lock = task
    print(f"[JUDGE:{model_name}] worker {shard_id} loading judge on {device}")
    judge = StickerConvJudge(judge_model_path, device)
    try:
        completed = 0
        for sample, pred in tqdm(
            list(zip(sample_chunk, pred_chunk)),
            desc=f"Pointwise {model_name} {device}",
            ncols=100,
            position=shard_id,
            leave=True,
        ):
            row = build_pointwise_record(judge, sample, pred)
            append_jsonl(Path(output_path), row, write_lock)
            completed += 1
        return shard_id, completed
    finally:
        judge.close()


def run_pointwise_parallel(
    samples: Sequence[EvalSample],
    predictions: Sequence[ModelPrediction],
    output_path: Path,
    summary_path: Path,
    judge_devices: Sequence[str],
) -> None:
    model_name = predictions[0].model_name if predictions else "model"
    cached = load_jsonl_records(output_path)
    missing_ids = missing_sample_ids(cached, samples)
    if not missing_ids:
        records = ordered_cached_records(cached, samples)
        write_jsonl(output_path, records)
        write_json(summary_path, summarize_pointwise(records))
        print(f"[JUDGE] pointwise cache hit: {output_path}")
        return

    missing_samples: List[EvalSample] = []
    missing_predictions: List[ModelPrediction] = []
    for sample, pred in zip(samples, predictions):
        if sample.sample_id in missing_ids:
            missing_samples.append(sample)
            missing_predictions.append(pred)

    worker_count = JUDGE_PARALLEL_WORKERS or len(judge_devices)
    worker_count = max(1, min(worker_count, len(judge_devices), len(missing_samples)))
    chunks = split_aligned(missing_samples, missing_predictions, parts=worker_count)
    context = mp.get_context("spawn")
    with context.Manager() as manager:
        write_lock = manager.Lock()
        tasks = [
            (
                shard_id,
                model_name,
                str(JUDGE_MODEL_PATH),
                sample_chunk,
                pred_chunk,
                judge_devices[shard_id % len(judge_devices)],
                str(output_path),
                write_lock,
            )
            for shard_id, (sample_chunk, pred_chunk) in enumerate(chunks)
        ]
        plan = ", ".join(f"worker{task[0]}->{task[5]}:{len(task[3])}" for task in tasks)
        print(
            f"[JUDGE] parallel pointwise {model_name}: {plan} "
            f"(resume {len(samples) - len(missing_samples)}/{len(samples)} done)"
        )

        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = [executor.submit(_run_pointwise_judge_worker, task) for task in tasks]
            for future in as_completed(futures):
                shard_id, completed = future.result()
                print(f"[JUDGE] finished pointwise shard {shard_id}: {completed} samples")

    records = ordered_cached_records(load_jsonl_records(output_path), samples)
    records.sort(key=lambda row: int(row["sample_id"]))
    write_jsonl(output_path, records)
    write_json(summary_path, summarize_pointwise(records))


def _run_pairwise_judge_worker(
    task: Tuple[
        int,
        str,
        str,
        str,
        List[EvalSample],
        List[ModelPrediction],
        List[ModelPrediction],
        str,
        str,
        Any,
    ],
) -> Tuple[int, int]:
    (
        shard_id,
        left_name,
        right_name,
        judge_model_path,
        sample_chunk,
        left_pred_chunk,
        right_pred_chunk,
        device,
        output_path,
        write_lock,
    ) = task
    print(f"[JUDGE:{left_name}_vs_{right_name}] worker {shard_id} loading judge on {device}")
    judge = StickerConvJudge(judge_model_path, device)
    try:
        completed = 0
        for sample, left_pred, right_pred in tqdm(
            list(zip(sample_chunk, left_pred_chunk, right_pred_chunk)),
            desc=f"Pairwise {left_name} vs {right_name} {device}",
            ncols=100,
            position=shard_id,
            leave=True,
        ):
            row = judge.score_pairwise(sample, left_name, left_pred, right_name, right_pred)
            append_jsonl(Path(output_path), row, write_lock)
            completed += 1
        return shard_id, completed
    finally:
        judge.close()


def run_pairwise_parallel(
    samples: Sequence[EvalSample],
    left_predictions: Sequence[ModelPrediction],
    right_predictions: Sequence[ModelPrediction],
    left_name: str,
    right_name: str,
    output_path: Path,
    summary_path: Path,
    judge_devices: Sequence[str],
) -> None:
    cached = load_jsonl_records(output_path)
    missing_ids = missing_sample_ids(cached, samples)
    if not missing_ids:
        records = ordered_cached_records(cached, samples)
        write_jsonl(output_path, records)
        write_json(summary_path, summarize_pairwise(records, left_name, right_name))
        print(f"[JUDGE] pairwise cache hit: {output_path}")
        return

    missing_samples: List[EvalSample] = []
    missing_left_predictions: List[ModelPrediction] = []
    missing_right_predictions: List[ModelPrediction] = []
    for sample, left_pred, right_pred in zip(samples, left_predictions, right_predictions):
        if sample.sample_id in missing_ids:
            missing_samples.append(sample)
            missing_left_predictions.append(left_pred)
            missing_right_predictions.append(right_pred)

    worker_count = JUDGE_PARALLEL_WORKERS or len(judge_devices)
    worker_count = max(1, min(worker_count, len(judge_devices), len(missing_samples)))
    chunks = split_aligned(
        missing_samples,
        missing_left_predictions,
        missing_right_predictions,
        parts=worker_count,
    )
    context = mp.get_context("spawn")
    with context.Manager() as manager:
        write_lock = manager.Lock()
        tasks = [
            (
                shard_id,
                left_name,
                right_name,
                str(JUDGE_MODEL_PATH),
                sample_chunk,
                left_pred_chunk,
                right_pred_chunk,
                judge_devices[shard_id % len(judge_devices)],
                str(output_path),
                write_lock,
            )
            for shard_id, (sample_chunk, left_pred_chunk, right_pred_chunk) in enumerate(chunks)
        ]
        plan = ", ".join(f"worker{task[0]}->{task[7]}:{len(task[4])}" for task in tasks)
        print(
            f"[JUDGE] parallel pairwise {left_name} vs {right_name}: {plan} "
            f"(resume {len(samples) - len(missing_samples)}/{len(samples)} done)"
        )

        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = [executor.submit(_run_pairwise_judge_worker, task) for task in tasks]
            for future in as_completed(futures):
                shard_id, completed = future.result()
                print(f"[JUDGE] finished pairwise shard {shard_id}: {completed} samples")

    records = ordered_cached_records(load_jsonl_records(output_path), samples)
    records.sort(key=lambda row: int(row["sample_id"]))
    write_jsonl(output_path, records)
    write_json(summary_path, summarize_pairwise(records, left_name, right_name))


def main() -> None:
    if not TEST_FILE.is_file():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    missing_models = {name: path for name, path in CANDIDATE_MODELS.items() if not Path(path).is_dir()}
    if missing_models:
        raise FileNotFoundError(f"Missing candidate model dirs: {missing_models}")
    if not Path(JUDGE_MODEL_PATH).is_dir():
        raise FileNotFoundError(f"Judge model not found: {JUDGE_MODEL_PATH}")

    output_dirs = ensure_output_dirs()
    samples = load_test_samples(TEST_FILE, SAMPLE_LIMIT)
    if not samples:
        raise RuntimeError(f"No valid samples loaded from {TEST_FILE}")

    all_predictions: Dict[str, List[ModelPrediction]] = {}
    if RUN_GENERATION or RUN_POINTWISE or RUN_PAIRWISE:
        all_predictions = generate_all_predictions(
            candidate_models=CANDIDATE_MODELS,
            samples=samples,
            generation_dir=output_dirs["generation"],
        )

    if not RUN_POINTWISE and not RUN_PAIRWISE:
        print(f"Done. Predictions cached under: {output_dirs['generation']}")
        return

    judge_devices = get_judge_devices()
    if len(judge_devices) > 1:
        print(f"[JUDGE] parallel evaluation on devices: {', '.join(judge_devices)}")
        if RUN_POINTWISE:
            for model_name, preds in all_predictions.items():
                output_path = output_dirs["pointwise"] / f"{model_name}.jsonl"
                summary_path = output_dirs["pointwise"] / f"{model_name}_summary.json"
                run_pointwise_parallel(samples, preds, output_path, summary_path, judge_devices)

        if RUN_PAIRWISE:
            for left_name, right_name in PAIRWISE_MODEL_PAIRS:
                if left_name not in all_predictions or right_name not in all_predictions:
                    raise KeyError(f"Pairwise models not found in CANDIDATE_MODELS: {(left_name, right_name)}")
                output_path = output_dirs["pairwise"] / f"{left_name}_vs_{right_name}.jsonl"
                summary_path = output_dirs["pairwise"] / f"{left_name}_vs_{right_name}_summary.json"
                run_pairwise_parallel(
                    samples=samples,
                    left_predictions=all_predictions[left_name],
                    right_predictions=all_predictions[right_name],
                    left_name=left_name,
                    right_name=right_name,
                    output_path=output_path,
                    summary_path=summary_path,
                    judge_devices=judge_devices,
                )
    else:
        judge_device = judge_devices[0]
        print(f"[JUDGE] loading judge model on {judge_device}: {JUDGE_MODEL_PATH}")
        judge = StickerConvJudge(str(JUDGE_MODEL_PATH), judge_device)
        try:
            if RUN_POINTWISE:
                for model_name, preds in all_predictions.items():
                    output_path = output_dirs["pointwise"] / f"{model_name}.jsonl"
                    summary_path = output_dirs["pointwise"] / f"{model_name}_summary.json"
                    run_pointwise(judge, samples, preds, output_path, summary_path)

            if RUN_PAIRWISE:
                for left_name, right_name in PAIRWISE_MODEL_PAIRS:
                    if left_name not in all_predictions or right_name not in all_predictions:
                        raise KeyError(f"Pairwise models not found in CANDIDATE_MODELS: {(left_name, right_name)}")
                    output_path = output_dirs["pairwise"] / f"{left_name}_vs_{right_name}.jsonl"
                    summary_path = output_dirs["pairwise"] / f"{left_name}_vs_{right_name}_summary.json"
                    run_pairwise(
                        judge=judge,
                        samples=samples,
                        left_predictions=all_predictions[left_name],
                        right_predictions=all_predictions[right_name],
                        left_name=left_name,
                        right_name=right_name,
                        output_path=output_path,
                        summary_path=summary_path,
                    )
        finally:
            judge.close()

    print(f"Done. Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
