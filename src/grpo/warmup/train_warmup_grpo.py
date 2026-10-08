#!/usr/bin/env python3
"""Warm-up GRPO training for Qwen2.5-VL.

Key properties:
- Multimodal policy model with LoRA updates on both language and vision linear layers.
- Grouped sampling (G=4) with GRPO-style group advantage normalization.
- Reward = format_reward (0/1) + outcome_reward (0~1 from Qwen3-VL judge).
- Clipped GRPO updates on cached on-policy rollouts with a frozen old-policy log-prob baseline.
- KL regularization to frozen SFT reference via adapter disable.
- Full checkpointing every N steps (adapter, optimizer, scheduler, RNG, data cursor state).
- Auto-resume from latest checkpoint and periodic validation + top-20 best tracking.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import re
import shutil
import sys
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

try:
    from qwen_vl_utils import process_vision_info

    HAS_QWEN_VL_UTILS = True
except Exception:
    HAS_QWEN_VL_UTILS = False


def detect_flash_attn() -> Tuple[bool, str]:
    if importlib.util.find_spec("flash_attn") is None:
        return False, "flash_attn module not found"

    # transformers checks importlib.metadata.version("flash_attn") internally when
    # attn_implementation='flash_attention_2' is requested. In some envs, module exists
    # but wheel metadata is missing, which would crash model initialization.
    for dist_name in ("flash_attn", "flash-attn"):
        try:
            ver = importlib.metadata.version(dist_name)
            return True, f"{dist_name}=={ver}"
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception as exc:
            return False, f"flash_attn metadata check failed: {exc}"

    return False, "flash_attn metadata not found"


HAS_FLASH_ATTN, FLASH_ATTN_STATUS = detect_flash_attn()
USE_COLOR_LOG = bool(
    (sys.stdout.isatty() or os.environ.get("FORCE_COLOR", "0") != "0")
    and os.environ.get("NO_COLOR") is None
)


def _c(text: str, code: str) -> str:
    if not USE_COLOR_LOG:
        return text
    return f"\033[{code}m{text}\033[0m"


def _train_step_log_line(step: int, epoch: int, metrics: Dict[str, float]) -> str:
    crown = _c("", "1;38;5;220")
    title = _c("[TRAIN-STEP]", "1;38;5;45")
    step_s = _c(f"step={step}", "1;38;5;82")
    epoch_s = _c(f"epoch={epoch}", "1;38;5;117")
    loss_s = _c(f"loss={metrics['loss']:.6f}", "1;38;5;203")
    reward_s = _c(f"reward={metrics['avg_reward']:.4f}", "1;38;5;214")
    format_s = _c(f"format={metrics['avg_format']:.4f}", "1;38;5;51")
    outcome_s = _c(f"outcome={metrics['avg_outcome']:.4f}", "1;38;5;141")
    kl_s = _c(f"kl={metrics['avg_kl']:.6f}", "1;38;5;39")
    lr_s = _c(f"lr={metrics['lr']:.8f}", "1;38;5;190")
    upd_s = _c(f"updates={int(metrics.get('optimizer_updates', 0))}", "1;38;5;220")
    valid_s = _c(f"valid_samples={metrics['valid_samples']}", "1;38;5;250")
    return (
        f"{crown} {title} {step_s} {epoch_s} | "
        f"{loss_s} {reward_s} {format_s} {outcome_s} {kl_s} {lr_s} {upd_s} {valid_s}"
    )


def _info_line(label: str, message: str, color_code: str = "1;38;5;39") -> str:
    return f"{_c(label, color_code)} {message}"

STRICT_FORMAT_PATTERN = re.compile(
    r"^\s*<think>[\s\S]*?</think>\s*<response>[\s\S]*?</response>\s*$",
    re.IGNORECASE,
)
RESPONSE_PATTERN = re.compile(r"<response>(.*?)</response>", re.IGNORECASE | re.DOTALL) # Pattern used to extract the model response for judge scoring
SCORE_JSON_PATTERN = re.compile(r'"score"\s*:\s*([01](?:\.\d+)?)')
SCORE_NUM_PATTERN = re.compile(r"(?<![\d.])([01](?:\.\d+)?)(?![\d.])")

OUTCOME_SYSTEM_PROMPT = (
    "You are a strict psychotherapy response evaluator. "
    "Score semantic alignment between candidate and reference therapist responses."
)

OUTCOME_USER_TEMPLATE = """Evaluate the candidate response against the reference response for the same client context.

Scoring rubric (continuous 0 to 1):
1) Therapeutic intent alignment (0.00-0.35)
2) Emotional attunement / empathy consistency (0.00-0.25)
3) Intervention strategy consistency (0.00-0.25)
4) Safety and non-harmfulness (0.00-0.15)

Penalties:
- Contradiction with reference meaning
- Missing core therapeutic move
- Harmful / judgmental tone
- If the candidate is overly templated or padded with boilerplate, and its sentence length differs greatly from the reference because of verbose formatted filler, lower the score appropriately.

Client context:
{context}

Reference therapist response:
{reference}

Candidate therapist response:
{candidate}

Return ONLY JSON in one line:
{{"score": <float between 0 and 1>}}
"""

A2_ABLATIONS = ("full", "w_o_format", "w_o_outcome")


def apply_a2_reward_ablation(format_score: float, outcome_score: float, ablation: str) -> float:
    if ablation == "w_o_format":
        return float(outcome_score)
    if ablation == "w_o_outcome":
        return float(format_score)
    return float(format_score + outcome_score)


@dataclass
class TrainConfig:
    base_model_path: str = "checkpoints/sft_merged"
    reward_model_path: str = "models/Qwen3-VL-8B-Instruct"
    train_file: str = "data/MESC-FT/train.jsonl"
    val_file: str = "data/MESC-FT/val.jsonl"
    checkpoint_dir: str = "checkpoints/warmup_grpo"

    policy_device: str = "cuda:0"
    reward_device: str = "cuda:1"
    seed: int = 42

    batch_size: int = 32
    group_size: int = 4
    epochs: int = 3

    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05

    kl_beta: float = 1e-3
    grpo_clip_epsilon: float = 0.2
    grpo_updates_per_rollout: int = 4

    temperature: float = 1.3
    top_p: float = 0.95
    max_new_tokens: int = 192

    reward_temperature: float = 0.0
    reward_top_p: float = 0.95
    reward_max_new_tokens: int = 512
    reward_batch_size: int = 4     # Number of answers scored per judge batch

    lora_rank: int = 8   # LoRA rank; r=8 adds a small number of trainable parameters
    lora_alpha: int = 32   # LoRA scaling factor; updates are scaled by alpha / rank before being added to the base model
    lora_dropout: float = 0.05

    max_image_size: int = 384
    max_frames: int = 16

    eval_every: int = 5
    save_every: int = 5
    log_every: int = 1
    val_max_samples: int = 0

    best_k: int = 20
    resume: str = "auto"  # auto | none | /abs/path/to/checkpoint-xxx
    ablation: str = "full"
    max_steps: int = 0


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Warm-up multimodal GRPO trainer")
    parser.add_argument("--base_model_path", type=str, default=TrainConfig.base_model_path)
    parser.add_argument("--reward_model_path", type=str, default=TrainConfig.reward_model_path)
    parser.add_argument("--train_file", type=str, default=TrainConfig.train_file)
    parser.add_argument("--val_file", type=str, default=TrainConfig.val_file)
    parser.add_argument("--checkpoint_dir", type=str, default=TrainConfig.checkpoint_dir)

    parser.add_argument("--policy_device", type=str, default=TrainConfig.policy_device)
    parser.add_argument("--reward_device", type=str, default=TrainConfig.reward_device)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)

    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--group_size", type=int, default=TrainConfig.group_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)

    parser.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=TrainConfig.warmup_ratio)

    parser.add_argument("--kl_beta", type=float, default=TrainConfig.kl_beta)
    parser.add_argument(
        "--grpo_clip_epsilon",
        type=float,
        default=TrainConfig.grpo_clip_epsilon,
    )
    parser.add_argument(
        "--grpo_updates_per_rollout",
        type=int,
        default=TrainConfig.grpo_updates_per_rollout,
    )

    parser.add_argument("--temperature", type=float, default=TrainConfig.temperature)
    parser.add_argument("--top_p", type=float, default=TrainConfig.top_p)
    parser.add_argument("--max_new_tokens", type=int, default=TrainConfig.max_new_tokens)

    parser.add_argument(
        "--reward_temperature", type=float, default=TrainConfig.reward_temperature
    )
    parser.add_argument("--reward_top_p", type=float, default=TrainConfig.reward_top_p)
    parser.add_argument(
        "--reward_max_new_tokens",
        type=int,
        default=TrainConfig.reward_max_new_tokens,
    )
    parser.add_argument("--reward_batch_size", type=int, default=TrainConfig.reward_batch_size)

    parser.add_argument("--lora_rank", type=int, default=TrainConfig.lora_rank)
    parser.add_argument("--lora_alpha", type=int, default=TrainConfig.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=TrainConfig.lora_dropout)

    parser.add_argument("--max_image_size", type=int, default=TrainConfig.max_image_size)
    parser.add_argument("--max_frames", type=int, default=TrainConfig.max_frames)

    parser.add_argument("--eval_every", type=int, default=TrainConfig.eval_every)
    parser.add_argument("--save_every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--log_every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--val_max_samples", type=int, default=TrainConfig.val_max_samples)

    parser.add_argument("--best_k", type=int, default=TrainConfig.best_k)
    parser.add_argument("--resume", type=str, default=TrainConfig.resume)
    parser.add_argument(
        "--ablation",
        type=str,
        choices=A2_ABLATIONS,
        default=TrainConfig.ablation,
    )
    parser.add_argument("--max_steps", type=int, default=TrainConfig.max_steps)

    ns = parser.parse_args()
    return TrainConfig(**vars(ns))


@dataclass
class ConversationSample:
    system: str
    user_text: str
    image_paths: List[str]
    ground_truth_full: str
    ground_truth_response: str


'''
Load the JSONL dataset and convert each valid record into a ConversationSample.
'''
class JsonlConversationDataset:
    def __init__(self, path: str):
        self.path = path
        self.samples: List[ConversationSample] = []
        self._load()

    def _load(self) -> None:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                system = item.get("system", "")
                user_text = ""
                image_paths: List[str] = []
                gt_full = ""

                for turn in item.get("conversations", []):
                    # Extract user text, image paths, and the assistant reference answer from the conversation turns.
                    role = turn.get("from")
                    if role == "user":
                        user_text = str(turn.get("value", ""))
                        image_paths = list(turn.get("images", []) or [])
                    elif role == "assistant":
                        gt_full = str(turn.get("value", ""))

                if not user_text or not gt_full:
                    continue

                gt_resp = extract_response_text(gt_full)  # Extract plain response text from the full reference answer.
                self.samples.append(
                    ConversationSample(
                        system=system,
                        user_text=user_text,
                        image_paths=image_paths,
                        ground_truth_full=gt_full,
                        ground_truth_response=gt_resp,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)   # Number of valid samples.

    def __getitem__(self, idx: int) -> ConversationSample:
        return self.samples[idx]


def extract_response_text(text: str) -> str:
    if not text:
        return ""
    m = RESPONSE_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: if the model omits <response> tags, score the raw stripped output.
    return text.strip()


def format_reward(text: str) -> float:
    return 1.0 if STRICT_FORMAT_PATTERN.fullmatch(text or "") else 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_class(model_path: str):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
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


def load_pretrained_with_fallback(
    model_class,
    model_path: str,
    model_kwargs: Dict[str, Any],
    model_name: str,
):
    kwargs = dict(model_kwargs)

    try:
        return model_class.from_pretrained(model_path, **kwargs)
    except TypeError as exc:
        # transformers version compatibility: newer versions prefer `dtype`;
        # some older ones only accept `torch_dtype`.
        if "dtype" in kwargs and "unexpected keyword argument 'dtype'" in str(exc):
            retry_kwargs = dict(kwargs)
            retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
            return load_pretrained_with_fallback(
                model_class, model_path, retry_kwargs, model_name
            )
        raise
    except Exception as exc:
        # Runtime fallback if flash-attn is partially installed / incompatible.
        if kwargs.get("attn_implementation") == "flash_attention_2":
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("attn_implementation", None)
            print(
                f"[WARN] {model_name}: flash_attention_2 unavailable "
                f"({exc.__class__.__name__}: {exc}). Retrying without FA2."
            )
            return load_pretrained_with_fallback(
                model_class, model_path, retry_kwargs, model_name
            )
        raise


def load_resized_images(
    image_paths: Sequence[str],
    max_frames: int,
    max_size: int,
) -> List[Image.Image]:
    output: List[Image.Image] = []
    for path in list(image_paths)[:max_frames]:
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            if max(w, h) > max_size:    # Resize only when the longest side exceeds the configured limit.
                scale = max_size / float(max(w, h))
                img = img.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.Resampling.LANCZOS,
                )
            output.append(img)
        # Fallback: if image loading fails, use a black placeholder so training can continue.
        except Exception:
            output.append(Image.new("RGB", (224, 224), (0, 0, 0)))
    return output

# Build Hugging Face / Qwen chat-template messages from images and text.
def build_messages(sample: ConversationSample, images: Sequence[Image.Image]) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = []
    for img in images:
        user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": sample.user_text})
    return [
        {"role": "system", "content": [{"type": "text", "text": sample.system}]},
        {"role": "user", "content": user_content},
    ]


def _move_to_device(obj: Dict[str, Any], device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def prepare_actor_inputs(
    processor,
    messages: List[Dict[str, Any]],
    images: Sequence[Image.Image],
    device: str,
) -> Tuple[Dict[str, Any], str]:
    # The processor converts role dictionaries into the string format expected by Qwen.
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    tokenizer = processor.tokenizer
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"   # Left padding is required for stable autoregressive generation with Qwen.

    try:
        if HAS_QWEN_VL_UTILS:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=[text_prompt],
                images=list(images) if images else None,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
    finally:
        tokenizer.padding_side = old_padding_side

    return _move_to_device(dict(inputs), device), text_prompt


def _parse_judge_score(text: str) -> float:
    if not text:
        return 0.0

    text = text.strip()
    m = SCORE_JSON_PATTERN.search(text)
    if m:
        try:
            return min(1.0, max(0.0, float(m.group(1))))
        except ValueError:
            pass

    m = SCORE_NUM_PATTERN.search(text)
    if m:
        try:
            return min(1.0, max(0.0, float(m.group(1))))
        except ValueError:
            pass

    try:
        maybe_json = json.loads(text)
        if isinstance(maybe_json, dict) and "score" in maybe_json:
            return min(1.0, max(0.0, float(maybe_json["score"])))
    except Exception:
        pass

    return 0.0


class OutcomeJudge:
    def __init__(
        self,
        model_path: str,
        device: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ):
        self.model_path = model_path
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)   # trust_remote_code=True is required for custom model classes such as Qwen.
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only generation expects left padding for batched prompts.
        self.tokenizer.padding_side = "left"

        model_class = resolve_model_class(model_path)
        dtype = torch.bfloat16 if "cuda" in device else torch.float32
        model_kwargs: Dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,   # Load weights lazily to reduce CPU memory pressure.
            "device_map": {"": device},
        }
        if HAS_FLASH_ATTN:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Instantiate the model and switch it to inference mode.
        self.model = load_pretrained_with_fallback(
            model_class=model_class,
            model_path=model_path,
            model_kwargs=model_kwargs,
            model_name="reward_model",
        )
        self.model.eval()
        self.model.requires_grad_(False)

        # Chat templating fallback.
        self.processor = None
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
                if self.processor.tokenizer.pad_token_id is None and self.processor.tokenizer.eos_token_id is not None:
                    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                self.processor.tokenizer.padding_side = "left"
        except Exception:
            self.processor = None

    def _chat_prompt(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": OUTCOME_SYSTEM_PROMPT},
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
            f"[SYSTEM]\n{OUTCOME_SYSTEM_PROMPT}\n[/SYSTEM]\n"
            f"[USER]\n{user_prompt}\n[/USER]\n[ASSISTANT]\n"
        )

    def score_batch(
        self,
        contexts: Sequence[str],
        references: Sequence[str],
        candidates: Sequence[str],
        batch_size: int,
    ) -> List[float]:
        assert len(contexts) == len(references) == len(candidates)
        all_scores: List[float] = []

        for start in range(0, len(candidates), batch_size):
            end = min(len(candidates), start + batch_size)
            prompts: List[str] = []
            for i in range(start, end):
                prompt = OUTCOME_USER_TEMPLATE.format(
                    context=(contexts[i] or "")[-3000:],
                    reference=(references[i] or "")[:1200],
                    candidate=(candidates[i] or "")[:1200],
                )
                prompts.append(self._chat_prompt(prompt))

            model_inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )
            model_inputs = _move_to_device(dict(model_inputs), self.device)

            with torch.no_grad():
                gen_ids = self.model.generate(
                    **model_inputs,
                    # Restore sampling-based judging while keeping NaN/Inf safeguards.
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=50,
                    remove_invalid_values=True,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            input_lens = model_inputs["attention_mask"].sum(dim=1).tolist()
            for row in range(gen_ids.shape[0]):
                new_tokens = gen_ids[row, int(input_lens[row]) :]
                raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                all_scores.append(_parse_judge_score(raw))

            del model_inputs, gen_ids

        return all_scores

# Batch data cursor
'''
BatchCursor replaces a standard DataLoader with a deterministic iterator that supports exact resume after interruptions.
'''
class BatchCursor:
    def __init__(self, dataset_size: int, batch_size: int, total_epochs: int, seed: int):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.seed = seed

        self.current_epoch = 0
        self.position = 0
        self.order: List[int] = self._make_order(0)

    def _make_order(self, epoch: int) -> List[int]:   # Deterministic shuffle order
        rng = random.Random(self.seed + epoch)
        order = list(range(self.dataset_size))
        rng.shuffle(order)
        return order

    def state_dict(self) -> Dict[str, Any]:   # Store epoch, position, and order so checkpoints can resume exactly.
        return {
            "current_epoch": self.current_epoch,
            "position": self.position,
            "order": self.order,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "total_epochs": self.total_epochs,
            "seed": self.seed,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.current_epoch = int(state["current_epoch"])
        self.position = int(state["position"])
        self.order = list(state["order"])

    @property
    def total_steps(self) -> int:
        return math.ceil(self.dataset_size / self.batch_size) * self.total_epochs

    def next_batch(self) -> Optional[Tuple[int, List[int]]]:
        while self.current_epoch < self.total_epochs:
            if self.position >= len(self.order):
                self.current_epoch += 1
                if self.current_epoch >= self.total_epochs:
                    return None
                self.order = self._make_order(self.current_epoch)
                self.position = 0
                continue

            start = self.position
            end = min(len(self.order), start + self.batch_size)
            self.position = end
            return self.current_epoch, self.order[start:end]

        return None


class BestTracker:
    def __init__(self, checkpoint_root: str, keep_top_k: int):
        self.checkpoint_root = Path(checkpoint_root)
        self.keep_top_k = keep_top_k
        self.best_txt = self.checkpoint_root / "best.txt"
        self.best_json = self.checkpoint_root / "best_records.json"
        self.records: List[Dict[str, Any]] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if self.best_json.exists():
            try:
                data = json.loads(self.best_json.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.records = data
            except Exception:
                self.records = []

    def update(self, record: Dict[str, Any]) -> None:
        self.records.append(record)
        self.records.sort(
            key=lambda x: (float(x.get("val_avg_reward", 0.0)), -int(x.get("step", 0))),
            reverse=True,
        )
        self.records = self.records[: self.keep_top_k]    # Keep only the top-k validation records.
        self._flush()

    def _flush(self) -> None:   # Write both JSON and text summaries.
        self.best_json.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        lines = [
            "# Top validation points (sorted by val_avg_reward desc)",
            "rank\tstep\tepoch\tval_avg_reward\tval_avg_format\tval_avg_outcome\tcheckpoint",
        ]
        for i, rec in enumerate(self.records, start=1):
            lines.append(
                f"{i}\t{rec.get('step', 0)}\t{rec.get('epoch', 0)}\t"
                f"{float(rec.get('val_avg_reward', 0.0)):.6f}\t"
                f"{float(rec.get('val_avg_format', 0.0)):.6f}\t"
                f"{float(rec.get('val_avg_outcome', 0.0)):.6f}\t"
                f"{rec.get('checkpoint', '')}"
            )
        self.best_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


class CheckpointManager:
    def __init__(self, root: str, base_model_path: str):
        self.root = Path(root)
        self.base_model_path = Path(base_model_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_file = self.root / "latest_checkpoint.txt"

    def resolve_resume_path(self, resume: str) -> Optional[Path]:
        if resume.lower() == "none":
            return None
        if resume.lower() == "auto":
            if self.latest_file.exists():
                txt = self.latest_file.read_text(encoding="utf-8").strip()
                if txt:
                    p = Path(txt)
                    if p.exists():
                        return p
            return None

        p = Path(resume)
        if p.exists():
            return p
        raise FileNotFoundError(f"Resume checkpoint not found: {resume}")

    def save(
        self,
        tag: str,
        policy_model,
        processor,
        optimizer,
        scheduler,
        trainer_state: Dict[str, Any],
    ) -> Path:
        ckpt_dir = self.root / f"checkpoint-{tag}"
        tmp_dir = self.root / f".tmp-checkpoint-{tag}-{uuid.uuid4().hex[:8]}"     # Atomic save directory
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1) Train-resume artifacts (LoRA adapter + optimizer/scheduler/RNG/cursor state)
            adapter_dir = tmp_dir / "adapter"
            policy_model.save_pretrained(adapter_dir)

            payload = {
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "trainer_state": trainer_state,
            }
            torch.save(payload, tmp_dir / "trainer_state.pt")

            summary = {
                "global_step": int(trainer_state.get("global_step", 0)),
                "epoch": int(trainer_state.get("cursor_state", {}).get("current_epoch", 0)),
                "timestamp": trainer_state.get("timestamp", ""),
                "tag": tag,
            }
            (tmp_dir / "training_state.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # 2) Inference-friendly HF files at checkpoint root (not only README)
            # Save processor/tokenizer files directly into checkpoint root.
            processor.save_pretrained(tmp_dir)

            # Save LoRA files also in root for quick inspection/loading.
            policy_model.save_pretrained(tmp_dir)

            # Copy base model config/template/index metadata for a familiar HF layout.
            for name in (
                "config.json",
                "generation_config.json",
                "chat_template.jinja",
                "model.safetensors.index.json",
                "preprocessor_config.json",
                "processor_config.json",
            ):
                src = self.base_model_path / name
                dst = tmp_dir / name
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)

            (tmp_dir / "base_model_path.txt").write_text(
                str(self.base_model_path) + "\n",
                encoding="utf-8",
            )
            (tmp_dir / "checkpoint_note.txt").write_text(
                "This checkpoint stores LoRA adapter weights (not merged full model).\n"
                "Load with base model path from base_model_path.txt + adapter files.\n",
                encoding="utf-8",
            )

            # Atomic replace to avoid half-written checkpoints on interruption.
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
            tmp_dir.rename(ckpt_dir)
            self.latest_file.write_text(str(ckpt_dir), encoding="utf-8")
            return ckpt_dir

        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def load_payload(self, ckpt_dir: Path) -> Dict[str, Any]:
        payload_file = ckpt_dir / "trainer_state.pt"
        if not payload_file.exists():
            raise FileNotFoundError(f"Missing trainer state: {payload_file}")
        return torch.load(payload_file, map_location="cpu", weights_only=False)


def collect_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception as exc:
            # If CUDA context is already in error state (e.g. device-side assert),
            # snapshot CPU RNGs and continue checkpointing without CUDA RNG.
            print(_info_line("[WARN]", f"Skip CUDA RNG snapshot: {exc}", "1;38;5;214"))
            state["cuda"] = None
    else:
        state["cuda"] = None
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])          # Restore Python random state.
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])

'''
Extract vision-only keyword arguments before forwarding through the vision encoder.
'''
def _extract_forward_vision_kwargs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]
    out = {}
    for k in keys:
        if k in inputs:
            out[k] = inputs[k]
    return out


def selected_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:].clone()
    mask = (shifted_labels != -100).float()
    shifted_labels[shifted_labels == -100] = 0

    log_probs = F.log_softmax(shifted_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, shifted_labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs, mask


@dataclass
class RolloutSequenceCache:
    seq_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    vision_kwargs: Dict[str, Any]
    old_logp: torch.Tensor
    ref_logp: torch.Tensor
    token_mask: torch.Tensor
    token_advantages: torch.Tensor


def _tensor_tree_to_cpu(tree: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in tree.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu()
        else:
            out[key] = value
    return out


def _tensor_tree_to_device(tree: Dict[str, Any], device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in tree.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out

# Run validation by generating policy responses and scoring them with the judge.
def run_validation(
    policy_model,
    processor,
    judge: OutcomeJudge,
    val_dataset: JsonlConversationDataset,
    cfg: TrainConfig,
) -> Dict[str, float]:
    policy_model.eval()
    dataset_size = len(val_dataset)
    if cfg.val_max_samples > 0:
        sampled_indices = random.sample(  # Randomly sample validation examples.
            range(dataset_size), k=min(dataset_size, cfg.val_max_samples)
        )
        print(f"[VAL] sampled {len(sampled_indices)}/{dataset_size}")
    else:
        sampled_indices = list(range(dataset_size))
        print(f"[VAL] using full validation set ({dataset_size}/{dataset_size})")
    val_count = len(sampled_indices)

    total_reward = 0.0
    total_format = 0.0
    total_outcome = 0.0
    valid_samples = 0

    for i, sample_idx in enumerate(sampled_indices, start=1):
        sample = val_dataset[sample_idx]
        images = load_resized_images(sample.image_paths, cfg.max_frames, cfg.max_image_size)
        messages = build_messages(sample, images)

        try:
            inputs, _ = prepare_actor_inputs(processor, messages, images, cfg.policy_device)
            with torch.no_grad(): # Validation uses greedy decoding with do_sample=False.
                gen_ids = policy_model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            prompt_len = int(inputs["input_ids"].shape[1])
            generated_tokens = gen_ids[:, prompt_len:]
            completion = processor.tokenizer.decode(
                generated_tokens[0], skip_special_tokens=True
            )

            r_f = format_reward(completion)
            cand_response = extract_response_text(completion)
            r_out = judge.score_batch(
                contexts=[sample.user_text],
                references=[sample.ground_truth_response],
                candidates=[cand_response],
                batch_size=1,
            )[0]

            total = apply_a2_reward_ablation(r_f, r_out, cfg.ablation)
            total_reward += float(total)
            total_format += float(r_f)
            total_outcome += float(r_out)
            valid_samples += 1

            del inputs, gen_ids

        except Exception as exc:
            print(f"[VAL] sample {sample_idx} failed: {exc}")
            continue

        if i % 20 == 0 or i == val_count:
            print(f"[VAL] processed {i}/{val_count}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if valid_samples == 0:
        return {
            "val_avg_reward": 0.0,
            "val_avg_format": 0.0,
            "val_avg_outcome": 0.0,
            "val_count": 0,
        }

    return {
        "val_avg_reward": total_reward / valid_samples,
        "val_avg_format": total_format / valid_samples,
        "val_avg_outcome": total_outcome / valid_samples,
        "val_count": valid_samples,
    }


def train_one_update_step(
    policy_model,
    processor,
    judge: OutcomeJudge,
    train_dataset: JsonlConversationDataset,
    batch_indices: Sequence[int],
    optimizer,
    scheduler,
    cfg: TrainConfig,
) -> Dict[str, float]:
    def _empty_metrics() -> Dict[str, float]:
        return {
            "loss": 0.0,
            "avg_reward": 0.0,
            "avg_format": 0.0,
            "avg_outcome": 0.0,
            "avg_kl": 0.0,
            "valid_samples": 0,
            "valid_sequences": 0,
            "optimizer_updates": 0,
            "lr": scheduler.get_last_lr()[0],
        }

    policy_model.train()
    optimizer.zero_grad(set_to_none=True)

    batch_total_reward = 0.0
    batch_total_format = 0.0
    batch_total_outcome = 0.0
    batch_total_loss = 0.0
    batch_total_kl = 0.0
    valid_samples = 0
    valid_sequences = 0
    optimizer_updates = 0
    rollout_cache: List[RolloutSequenceCache] = []

    for idx in batch_indices:
        sample = train_dataset[idx]
        images = load_resized_images(sample.image_paths, cfg.max_frames, cfg.max_image_size)
        messages = build_messages(sample, images)

        try:
            inputs, _ = prepare_actor_inputs(processor, messages, images, cfg.policy_device)
        except Exception as exc:
            print(f"[TRAIN] prepare input failed idx={idx}: {exc}")
            continue

        try:
            policy_model.eval()
            with torch.no_grad():
                gen_ids = policy_model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_new_tokens=cfg.max_new_tokens,
                    num_return_sequences=cfg.group_size,   # Generate multiple sampled responses for the same prompt.
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
        except Exception as exc:
            print(f"[TRAIN] generation failed idx={idx}: {exc}")
            policy_model.train()
            continue

        if gen_ids.dim() == 1:
            gen_ids = gen_ids.unsqueeze(0)

        prompt_len = int(inputs["input_ids"].shape[1])
        generated_only = gen_ids[:, prompt_len:]

        completions: List[str] = []
        for row in range(generated_only.shape[0]):
            completion = processor.tokenizer.decode(generated_only[row], skip_special_tokens=True)
            completions.append(completion)  # completions stores G candidate responses for each prompt.

        # Reward = format + outcome
        r_f_list = [format_reward(c) for c in completions]
        cand_resp = [extract_response_text(c) for c in completions]
        r_out_list = judge.score_batch(
            contexts=[sample.user_text] * len(completions),
            references=[sample.ground_truth_response] * len(completions),
            candidates=cand_resp,
            batch_size=cfg.reward_batch_size,
        )
        reward_list = [
            apply_a2_reward_ablation(r_f, r_o, cfg.ablation)
            for r_f, r_o in zip(r_f_list, r_out_list)
        ]
        
        # Compute advantages.
        reward_tensor = torch.tensor(reward_list, dtype=torch.float32, device=cfg.policy_device)
        reward_mean = reward_tensor.mean()
        reward_std = reward_tensor.std(unbiased=False) + 1e-8
        advantages = (reward_tensor - reward_mean) / reward_std

        vision_kwargs = _extract_forward_vision_kwargs(inputs)
        vision_kwargs_cpu = _tensor_tree_to_cpu(vision_kwargs)
        pad_id = processor.tokenizer.pad_token_id

        try:
            for g in range(gen_ids.shape[0]):
                seq_ids = gen_ids[g : g + 1].to(cfg.policy_device)
                labels = seq_ids.clone()
                labels[:, :prompt_len] = -100
                if pad_id is not None:
                    labels[seq_ids == pad_id] = -100
                    attention_mask = (seq_ids != pad_id).long()
                else:
                    attention_mask = torch.ones_like(seq_ids)

                fwd_kwargs = {
                    "input_ids": seq_ids,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                }
                fwd_kwargs.update(vision_kwargs)

                try:
                    with torch.no_grad():
                        old_out = policy_model(**fwd_kwargs)
                        old_logp, _ = selected_log_probs(old_out.logits, labels)
                        with policy_model.disable_adapter():
                            ref_out = policy_model(**fwd_kwargs)
                            ref_logp, token_mask = selected_log_probs(ref_out.logits, labels)
                except Exception as exc:
                    print(f"[TRAIN] rollout forward failed idx={idx} group={g}: {exc}")
                    break

                token_advantages = torch.full_like(
                    token_mask,
                    float(advantages[g].detach().item()),
                    dtype=torch.float32,
                )
                rollout_cache.append(
                    RolloutSequenceCache(
                        seq_ids=seq_ids.detach().cpu(),
                        labels=labels.detach().cpu(),
                        attention_mask=attention_mask.detach().cpu(),
                        vision_kwargs=vision_kwargs_cpu,
                        old_logp=old_logp.detach().cpu(),
                        ref_logp=ref_logp.detach().cpu(),
                        token_mask=token_mask.detach().cpu(),
                        token_advantages=token_advantages.detach().cpu(),
                    )
                )
                valid_sequences += 1

                del seq_ids, labels, attention_mask, old_out, old_logp, ref_out, ref_logp
                del token_mask, token_advantages
        finally:
            policy_model.train()

        batch_total_reward += float(sum(reward_list) / len(reward_list))
        batch_total_format += float(sum(r_f_list) / len(r_f_list))
        batch_total_outcome += float(sum(r_out_list) / len(r_out_list))
        valid_samples += 1

        del inputs, gen_ids, generated_only

    if valid_samples == 0 or not rollout_cache:
        optimizer.zero_grad(set_to_none=True)
        return _empty_metrics()

    updates_per_rollout = max(1, int(cfg.grpo_updates_per_rollout))
    clip_eps = max(0.0, float(cfg.grpo_clip_epsilon))

    for update_idx in range(updates_per_rollout):
        optimizer.zero_grad(set_to_none=True)
        update_loss_value = 0.0
        update_kl_value = 0.0
        update_valid_sequences = 0

        for seq_idx, rollout in enumerate(rollout_cache):
            seq_ids = rollout.seq_ids.to(cfg.policy_device)
            labels = rollout.labels.to(cfg.policy_device)
            attention_mask = rollout.attention_mask.to(cfg.policy_device)
            old_logp = rollout.old_logp.to(cfg.policy_device)
            ref_logp = rollout.ref_logp.to(cfg.policy_device)
            token_mask = rollout.token_mask.to(cfg.policy_device)
            token_advantages = rollout.token_advantages.to(cfg.policy_device)
            fwd_kwargs = {
                "input_ids": seq_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
            }
            fwd_kwargs.update(_tensor_tree_to_device(rollout.vision_kwargs, cfg.policy_device))

            try:
                pol_out = policy_model(**fwd_kwargs)
                pol_logp, _ = selected_log_probs(pol_out.logits, labels)
            except Exception as exc:
                print(f"[TRAIN] clipped update forward failed seq={seq_idx}: {exc}")
                break

            log_ratio = pol_logp - old_logp
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            surrogate = torch.minimum(
                ratio * token_advantages.detach(),
                clipped_ratio * token_advantages.detach(),
            )
            kl_tokens = pol_logp - ref_logp
            token_count = token_mask.sum() + 1e-8
            seq_kl = (kl_tokens * token_mask).sum() / token_count
            seq_objective = surrogate - cfg.kl_beta * kl_tokens
            seq_loss = -(seq_objective * token_mask).sum() / token_count

            if not torch.isfinite(seq_loss):
                print(f"[TRAIN] non-finite clipped seq_loss skipped seq={seq_idx}")
                del seq_ids, labels, attention_mask, old_logp, ref_logp, token_mask
                del token_advantages, pol_out, pol_logp, log_ratio, ratio, clipped_ratio
                del surrogate, kl_tokens, seq_kl, seq_objective, seq_loss
                continue

            scaled_loss = seq_loss / max(1, len(rollout_cache))
            scaled_loss.backward()

            update_loss_value += float(seq_loss.detach().item())
            update_kl_value += float(seq_kl.detach().item())
            update_valid_sequences += 1

            del seq_ids, labels, attention_mask, old_logp, ref_logp, token_mask
            del token_advantages, pol_out, pol_logp, log_ratio, ratio, clipped_ratio
            del surrogate, kl_tokens, seq_kl, seq_objective, seq_loss, scaled_loss

        if update_valid_sequences == 0:
            print(f"[TRAIN] optimizer update {update_idx + 1} had no valid sequences, skipping step().")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        optimizer_updates += 1
        batch_total_loss += update_loss_value / update_valid_sequences
        batch_total_kl += update_kl_value / update_valid_sequences

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "loss": batch_total_loss / max(1, optimizer_updates),
        "avg_reward": batch_total_reward / valid_samples,
        "avg_format": batch_total_format / valid_samples,
        "avg_outcome": batch_total_outcome / valid_samples,
        "avg_kl": batch_total_kl / max(1, optimizer_updates),
        "valid_samples": valid_samples,
        "valid_sequences": valid_sequences,
        "optimizer_updates": optimizer_updates,
        "lr": scheduler.get_last_lr()[0],
    }


def build_policy_model_and_processor(cfg: TrainConfig):
    model_class = resolve_model_class(cfg.base_model_path)
    processor = AutoProcessor.from_pretrained(cfg.base_model_path, trust_remote_code=True)

    if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    # Keep left padding as default for decoder-only generation safety.
    processor.tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if "cuda" in cfg.policy_device else torch.float32
    model_kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": {"": cfg.policy_device},
    }
    if HAS_FLASH_ATTN:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    base_model = load_pretrained_with_fallback(
        model_class=model_class,
        model_path=cfg.base_model_path,
        model_kwargs=model_kwargs,
        model_name="policy_model",
    )
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()  # Gradient checkpointing trades extra compute for lower memory usage.
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    return base_model, processor


def create_new_lora_model(base_model, cfg: TrainConfig):
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear", # Apply LoRA to all linear layers, including attention and MLP projections.
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(base_model, peft_cfg)
    return model


def load_policy_from_checkpoint(base_model, checkpoint_path: Path):
    adapter_dir = checkpoint_path / "adapter"
    if not adapter_dir.exists():
        # fallback: old style where adapter files are directly in checkpoint folder
        adapter_dir = checkpoint_path
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=True)  # Keep LoRA adapters trainable when resuming training.
    return model


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    ckpt_mgr = CheckpointManager(cfg.checkpoint_dir, cfg.base_model_path)
    best_tracker = BestTracker(cfg.checkpoint_dir, cfg.best_k)

    print(_c("=" * 80, "1;38;5;45"))
    print(_c(" Warm-up GRPO Training", "1;38;5;220"))
    print(_c("=" * 80, "1;38;5;45"))
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    print(_info_line("[ENV]", f"HAS_QWEN_VL_UTILS: {HAS_QWEN_VL_UTILS}", "1;38;5;111"))
    print(_info_line("[ENV]", f"HAS_FLASH_ATTN: {HAS_FLASH_ATTN}", "1;38;5;111"))
    print(_info_line("[ENV]", f"FLASH_ATTN_STATUS: {FLASH_ATTN_STATUS}", "1;38;5;111"))

    train_dataset = JsonlConversationDataset(cfg.train_file)
    val_dataset = JsonlConversationDataset(cfg.val_file)
    print(
        _info_line(
            "[DATA]",
            f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}",
            "1;38;5;83",
        )
    )

    cursor = BatchCursor(
        dataset_size=len(train_dataset),
        batch_size=cfg.batch_size,
        total_epochs=cfg.epochs,
        seed=cfg.seed,
    )

    base_model, processor = build_policy_model_and_processor(cfg)

    resume_path = ckpt_mgr.resolve_resume_path(cfg.resume)
    if resume_path is not None:
        print(_info_line("[RESUME]", f"Resuming from checkpoint: {resume_path}", "1;38;5;226"))
        policy_model = load_policy_from_checkpoint(base_model, resume_path)  # Create a new LoRA adapter, or attach the adapter from the resume checkpoint.
    else:
        print(_info_line("[INIT]", "Starting from fresh LoRA initialization.", "1;38;5;226"))
        policy_model = create_new_lora_model(base_model, cfg)

    policy_model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    total_rollout_steps = cursor.total_steps
    if cfg.max_steps > 0:
        total_rollout_steps = min(total_rollout_steps, cfg.max_steps)
    total_training_steps = max(
        1,
        total_rollout_steps * max(1, int(cfg.grpo_updates_per_rollout)),
    )
    '''
    Warm-up gradually increases the learning rate from 0 to the target value to stabilize early training.
    '''
    warmup_steps = int(total_training_steps * cfg.warmup_ratio)
    # After warm-up, cosine scheduling gradually decays the learning rate toward zero.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    global_step = 0
    if resume_path is not None:
        print(_c(f"Loading checkpoint payload from: {resume_path}", "1;38;5;220"))
        payload = ckpt_mgr.load_payload(resume_path)

        # Verify LoRA adapter weights are actually loaded (not fresh init).
        print("--- [DEBUG] LoRA weight check ---")
        check_count = 0
        for name, param in policy_model.named_parameters():
            if "lora_A" in name and param.requires_grad:
                norm_val = float(param.data.norm().item())
                print(f" -> layer {name}: L2 norm = {norm_val:.4f}")
                check_count += 1
                if check_count >= 2:
                    break
        if check_count == 0:
            print(_info_line("[WARN]", "No active LoRA adapter parameter found.", "1;38;5;214"))

        # Warm Start: do not restore optimizer/scheduler/RNG state.
        # Warm-start from weights and step count while resetting optimizer state.
        print(
            _info_line(
                "[RESUME]",
                "Skipping optimizer/scheduler/RNG state load for Warm Start stability.",
                "1;38;5;196",
            )
        )

        trainer_state = payload.get("trainer_state", {})
        if "cursor_state" in trainer_state:
            cursor.load_state_dict(trainer_state["cursor_state"])
        global_step = int(trainer_state.get("global_step", 0))

        if "best_records" in trainer_state and isinstance(trainer_state["best_records"], list):
            best_tracker.records = trainer_state["best_records"]
            best_tracker._flush()  # Flush updated tracker records.

        print(
            _info_line(
                "[RESUME]",
                f"Recovered step={global_step}, epoch={cursor.current_epoch}, "
                f"position={cursor.position}. Optimizer reset.",
                "1;38;5;226",
            )
        )

    judge = OutcomeJudge(
        model_path=cfg.reward_model_path,
        device=cfg.reward_device,
        temperature=cfg.reward_temperature,
        top_p=cfg.reward_top_p,
        max_new_tokens=cfg.reward_max_new_tokens,
    )

    last_val_metrics = {
        "val_avg_reward": 0.0,
        "val_avg_format": 0.0,
        "val_avg_outcome": 0.0,
        "val_count": 0,
    }
    
    # Save the current step, adapter state, and metrics as a checkpoint.
    def save_full_checkpoint(tag: str) -> Path:
        trainer_state = {
            "global_step": global_step,
            "cursor_state": cursor.state_dict(),
            "best_records": best_tracker.records,
            "last_val_metrics": last_val_metrics,
            "config": asdict(cfg),
            "rng_state": collect_rng_state(),
            "timestamp": str(np.datetime64("now")),
        }
        ckpt = ckpt_mgr.save(
            tag=tag,
            policy_model=policy_model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
        )
        print(_info_line("[CKPT]", f"Checkpoint saved: {ckpt}", "1;38;5;220"))
        return ckpt

    try:
        while True:
            next_item = cursor.next_batch()
            if next_item is None:
                break

            epoch_idx, batch_indices = next_item
            step_metrics = train_one_update_step(
                policy_model=policy_model,
                processor=processor,
                judge=judge,
                train_dataset=train_dataset,
                batch_indices=batch_indices,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
            )
            global_step += 1

            if global_step % cfg.log_every == 0:
                print(_train_step_log_line(global_step, epoch_idx, step_metrics))

            need_eval = global_step % cfg.eval_every == 0
            need_save = global_step % cfg.save_every == 0

            if need_eval:
                print(
                    _info_line(
                        "[VAL]",
                        f"Running validation at step={global_step} ...",
                        "1;38;5;117",
                    )
                )
                last_val_metrics = run_validation(
                    policy_model=policy_model,
                    processor=processor,
                    judge=judge,
                    val_dataset=val_dataset,
                    cfg=cfg,
                )
                print(
                    _info_line(
                        "[VAL]",
                        f"step={global_step} avg_reward={last_val_metrics['val_avg_reward']:.4f} "
                        f"format={last_val_metrics['val_avg_format']:.4f} "
                        f"outcome={last_val_metrics['val_avg_outcome']:.4f} "
                        f"count={int(last_val_metrics['val_count'])}",
                        "1;38;5;117",
                    )
                )

            if need_save:
                ckpt_path = save_full_checkpoint(tag=f"step-{global_step}")
                if need_eval:
                    best_tracker.update(
                        {
                            "step": global_step,
                            "epoch": epoch_idx,
                            "val_avg_reward": last_val_metrics["val_avg_reward"],
                            "val_avg_format": last_val_metrics["val_avg_format"],
                            "val_avg_outcome": last_val_metrics["val_avg_outcome"],
                            "checkpoint": str(ckpt_path),
                        }
                    )

            if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                print(
                    _info_line(
                        "[STOP]",
                        f"Reached max_steps={cfg.max_steps}. Stopping early.",
                        "1;38;5;208",
                    )
                )
                break

        print(_info_line("[DONE]", "Training loop finished.", "1;38;5;82"))

    except KeyboardInterrupt:  # Manual interruption guard
        print(_info_line("[INTERRUPT]", "KeyboardInterrupt captured. Saving interrupted checkpoint...", "1;38;5;208"))
        try:
            save_full_checkpoint(tag=f"interrupted-{global_step}")
        except Exception as exc:
            print(_info_line("[WARN]", f"Interrupted checkpoint save failed: {exc}", "1;38;5;214"))
        raise
    except Exception:  # Failure guard
        print(_info_line("[ERROR]", "Unhandled exception during training. Saving crash checkpoint...", "1;38;5;196"))
        traceback.print_exc()
        try:
            save_full_checkpoint(tag=f"crashed-{global_step}")
        except Exception as exc:
            print(_info_line("[WARN]", f"Crash checkpoint save failed: {exc}", "1;38;5;214"))
        raise

    # Always save a final checkpoint after normal finish.
    final_ckpt = save_full_checkpoint(tag=f"final-{global_step}")

    # If training ends on non-eval step, run one final validation for best tracker consistency.
    if global_step % cfg.eval_every != 0:
        print(_info_line("[VAL]", "Running final validation at training end...", "1;38;5;117"))
        last_val_metrics = run_validation(
            policy_model=policy_model,
            processor=processor,
            judge=judge,
            val_dataset=val_dataset,
            cfg=cfg,
        )

    best_tracker.update(
        {
            "step": global_step,
            "epoch": cursor.current_epoch,
            "val_avg_reward": last_val_metrics["val_avg_reward"],
            "val_avg_format": last_val_metrics["val_avg_format"],
            "val_avg_outcome": last_val_metrics["val_avg_outcome"],
            "checkpoint": str(final_ckpt),
        }
    )

    print(_c("=" * 80, "1;38;5;45"))
    print(_c(" Training complete.", "1;38;5;220"))
    print(_info_line("[FINAL]", f"Final checkpoint: {final_ckpt}", "1;38;5;82"))
    print(
        _info_line(
            "[FINAL]",
            f"Final val avg reward: {last_val_metrics['val_avg_reward']:.4f}",
            "1;38;5;82",
        )
    )
    print(
        _info_line(
            "[FINAL]",
            f"Best records saved to: {Path(cfg.checkpoint_dir) / 'best.txt'}",
            "1;38;5;82",
        )
    )
    print(_c("=" * 80, "1;38;5;45"))


if __name__ == "__main__":
    main()
