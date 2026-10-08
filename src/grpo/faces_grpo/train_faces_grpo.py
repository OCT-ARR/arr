#!/usr/bin/env python3
"""FACES GRPO training for Qwen2.5-VL.

Key properties:
- Multimodal policy model with LoRA updates on both language and vision linear layers.
- Grouped sampling (G=4) with GRPO-style group advantage normalization.
- Reward = mean(visual, emotion, situation) * outcome * strategy under strict format check.
- Clipped GRPO updates on cached on-policy rollouts with a frozen old-policy log-prob baseline.
- KL regularization to frozen SFT reference via adapter disable.
- Full checkpointing every N steps (adapter, optimizer, scheduler, RNG, data cursor state).
- Auto-resume from latest checkpoint and periodic validation + top-20 best tracking.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import random
import re
import shutil
import sys
import traceback
import uuid
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

# Quiet down non-critical HF advisory warnings in terminal logs.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings(
    "ignore",
    message=r"The image processor of type `CLIPImageProcessor` is now loaded as a fast processor by default.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch_dtype` is deprecated! Use `dtype` instead!",
)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    LogitsProcessor,
    LogitsProcessorList,
    get_cosine_schedule_with_warmup,
)
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


def patch_transformers_awq_compat() -> None:
    """Compatibility shim for autoawq with newer transformers versions."""
    try:
        import transformers.activations as tf_activations
    except Exception:
        return

    if not hasattr(tf_activations, "PytorchGELUTanh") and hasattr(tf_activations, "GELUTanh"):
        tf_activations.PytorchGELUTanh = tf_activations.GELUTanh


def patch_peft_quantization_backend_probes() -> None:
    """Avoid optional quantization backend probes from breaking standard LoRA."""

    def _not_available() -> bool:
        return False

    try:
        import peft.import_utils as peft_import_utils

        old_fn = getattr(peft_import_utils, "is_gptqmodel_available", None)
        if old_fn is not None and hasattr(old_fn, "cache_clear"):
            old_fn.cache_clear()
        peft_import_utils.is_gptqmodel_available = _not_available
    except Exception:
        pass

    try:
        import peft.tuners.lora.gptq as peft_lora_gptq

        peft_lora_gptq.is_gptqmodel_available = _not_available
    except Exception:
        pass


patch_transformers_awq_compat()
patch_peft_quantization_backend_probes()

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
    os.environ.get("NO_COLOR") is None and os.environ.get("FORCE_COLOR", "1") != "0"
)


@contextlib.contextmanager
def suppress_stdout_stderr():
    sink = io.StringIO()
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


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
    format_s = _c(f"format={metrics['avg_format']:.4f}", "1;38;5;51")
    visual_s = _c(f"visual={metrics['avg_visual']:.4f}", "1;38;5;45")
    emotion_s = _c(f"emotion={metrics['avg_emotion']:.4f}", "1;38;5;141")
    situation_s = _c(f"situation={metrics['avg_situation']:.4f}", "1;38;5;177")
    strategy_s = _c(f"strategy={metrics['avg_strategy']:.4f}", "1;38;5;208")
    outcome_s = _c(f"outcome={metrics['avg_outcome']:.4f}", "1;38;5;111")
    total_s = _c(f"total={metrics['avg_total']:.4f}", "1;38;5;214")
    kl_s = _c(f"kl={metrics['avg_kl']:.6f}", "1;38;5;39")
    lr_s = _c(f"lr={metrics['lr']:.8f}", "1;38;5;190")
    upd_s = _c(f"updates={int(metrics.get('optimizer_updates', 0))}", "1;38;5;220")
    valid_s = _c(f"valid_samples={metrics['valid_samples']}", "1;38;5;250")
    seg_visual_s = _c(f"visual={metrics['seg_adv_visual']:.4f}", "1;38;5;45")
    seg_emotion_s = _c(f"emotion={metrics['seg_adv_emotion']:.4f}", "1;38;5;141")
    seg_situation_s = _c(f"situation={metrics['seg_adv_situation']:.4f}", "1;38;5;177")
    seg_strategy_s = _c(f"strategy={metrics['seg_adv_strategy']:.4f}", "1;38;5;208")
    seg_outcome_s = _c(f"outcome={metrics['seg_adv_outcome']:.4f}", "1;38;5;111")
    coverage_s = _c(f"coverage={metrics['seg_coverage']:.4f}", "1;38;5;159")
    return (
        f"{crown} {title} {step_s} {epoch_s} | "
        f"{loss_s} {format_s} {visual_s} {emotion_s} {situation_s} {strategy_s} {outcome_s} "
        f"{total_s} {kl_s} {lr_s} {upd_s} {valid_s} | "
        f"seg_adv: {seg_visual_s} {seg_emotion_s} {seg_situation_s} {seg_strategy_s} "
        f"{seg_outcome_s} {coverage_s}"
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
STRATEGY_SCORE_PATTERN = re.compile(r'"score"\s*:\s*(0\.5|0(?:\.0+)?|1(?:\.0+)?)')
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
SEGMENT_LABEL_PATTERN = re.compile(
    r"\b(first(?:ly)?|second(?:ly)?|third(?:ly)?|fourth(?:ly)?)\b[\s:.,-]*",
    re.IGNORECASE,
)
EPS = 1e-8


class SanitizeLogitsProcessor(LogitsProcessor):
    """Replace NaN/Inf logits before sampling to avoid CUDA assert in multinomial."""

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if not torch.isfinite(scores).all():
            scores = torch.where(
                torch.isfinite(scores), scores, torch.full_like(scores, -1e9)
            )
        return scores


_SANITIZE_LOGITS = LogitsProcessorList([SanitizeLogitsProcessor()])


def _cuda_is_healthy(device: str) -> bool:
    if not device.startswith("cuda"):
        return True
    try:
        _ = (torch.zeros(1, device=device) + 1.0).item()
        return True
    except RuntimeError:
        return False


def _is_cuda_runtime_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("cuda error" in msg) or ("device-side assert" in msg)


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

A3_ABLATIONS = (
    "full",
    "w_o_visual",
    "w_o_emotion",
    "w_o_situation",
    "w_o_strategy",
    "w_o_fine_grained_credit",
    "w_o_multi_source_reward",
    "w_o_noncomp_gating",
    "w_o_segadv",
    "w_o_segadv_lineardecay",
    "w_o_segadv_expdecay",
)


def fuse_cognitive_reward(reward_parts: "RewardComponents", cfg: "TrainConfig") -> float:
    active_terms: List[Tuple[float, float]] = []
    if cfg.ablation != "w_o_visual":
        active_terms.append((float(cfg.visual_weight), float(reward_parts.visual)))
    if cfg.ablation != "w_o_emotion":
        active_terms.append((float(cfg.emotion_weight), float(reward_parts.emotion)))
    if cfg.ablation != "w_o_situation":
        active_terms.append((float(cfg.situation_weight), float(reward_parts.situation)))
    if not active_terms:
        return 0.0

    weight_sum = sum(weight for weight, _ in active_terms)
    if weight_sum <= EPS:
        return float(sum(value for _, value in active_terms) / len(active_terms))
    return float(sum(weight * value for weight, value in active_terms) / weight_sum)


def compose_a3_total_reward(reward_parts: "RewardComponents", cfg: "TrainConfig") -> float:
    if reward_parts.format < 0.5:
        return float(cfg.format_fail_score)
    if cfg.ablation == "w_o_noncomp_gating":
        return float(
            (
                float(reward_parts.visual)
                + float(reward_parts.emotion)
                + float(reward_parts.situation)
                + float(reward_parts.strategy)
                + float(reward_parts.outcome)
            )
            / 5.0
        )
    total = fuse_cognitive_reward(reward_parts, cfg) * float(reward_parts.outcome)
    if cfg.ablation != "w_o_strategy":
        total *= float(reward_parts.strategy)
    return float(total)


def segment_advantage_gate(
    total_reward: float,
    group_max_total_reward: float,
    cfg: "TrainConfig",
) -> float:
    total_reward = max(0.0, float(total_reward))
    group_max_total_reward = max(0.0, float(group_max_total_reward))
    floor = min(1.0, max(0.0, float(cfg.segment_advantage_gate_floor)))
    if group_max_total_reward <= EPS:
        return float(floor)
    relative_ratio = min(1.0, total_reward / group_max_total_reward)
    return float(floor + (1.0 - floor) * relative_ratio)


def gate_positive_segment_advantages(
    segment_advantages: Dict[str, float],
    total_reward: float,
    group_max_total_reward: float,
    cfg: "TrainConfig",
) -> Dict[str, float]:
    gate = segment_advantage_gate(total_reward, group_max_total_reward, cfg)
    gated: Dict[str, float] = {}
    for name, adv in segment_advantages.items():
        adv = float(adv)
        gated[name] = adv * gate if adv > 0.0 else adv
    return gated


@dataclass
class TrainConfig:
    base_model_path: str = (
        "checkpoints/warmup_merged"
    )
    reward_model_path: str = (
        "models/Qwen3-VL-8B-Instruct"
    )
    visual_model_path: str = (
        "models/clip-vit-large-patch14"
    )
    emotion_model_path: str = (
        "models/paraphrase-multilingual-MiniLM-L12-v2"
    )
    traditional_reward_model_path: str = (
        "checkpoints/reward_model/"
        "roberta_epitome_multitask/final"
    )
    train_file: str = "data/MESC-FT/train.jsonl"
    val_file: str = "data/MESC-FT/val.jsonl"
    checkpoint_dir: str = (
        "checkpoints/faces_grpo"
    )

    policy_device: str = "cuda:0"
    reward_device: str = "cuda:1"
    visual_device: str = "cpu"
    emotion_device: str = "cpu"
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
    visual_threshold: float = 0.60
    visual_top_k: int = 4
    format_fail_score: float = 0.0
    visual_weight: float = 1.0 / 3.0
    emotion_weight: float = 1.0 / 3.0
    situation_weight: float = 1.0 / 3.0
    segment_advantage_gate_floor: float = 0.1
    endpoint_anchor_window: int = 5
    endpoint_anchor_decay: float = 0.8
    traditional_format_weight: float = 0.1
    traditional_rouge_weight: float = 1.0
    traditional_skill_weight: float = 0.1
    traditional_skill_decay: float = 0.5

    lora_rank: int = 8   # LoRA rank; r=8 adds a small number of trainable parameters
    lora_alpha: int = 32   # LoRA scaling factor; updates are scaled by alpha / rank before being added to the base model
    lora_dropout: float = 0.05
    max_grad_norm: float = 1.0

    max_image_size: int = 384
    max_frames: int = 16

    eval_every: int = 5
    save_every: int = 5
    log_every: int = 1
    val_max_samples: int = 100
    val_index_file: str = ""

    best_k: int = 20
    resume: str = "auto"  # auto | none | /abs/path/to/checkpoint-xxx
    ablation: str = "full"
    max_steps: int = 0
    eval_only: bool = False
    token_credit_mode: str = "endpoint"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="FACES multimodal GRPO trainer")
    parser.add_argument("--base_model_path", type=str, default=TrainConfig.base_model_path)
    parser.add_argument("--reward_model_path", type=str, default=TrainConfig.reward_model_path)
    parser.add_argument("--visual_model_path", type=str, default=TrainConfig.visual_model_path)
    parser.add_argument("--emotion_model_path", type=str, default=TrainConfig.emotion_model_path)
    parser.add_argument(
        "--traditional_reward_model_path",
        type=str,
        default=TrainConfig.traditional_reward_model_path,
    )
    parser.add_argument("--train_file", type=str, default=TrainConfig.train_file)
    parser.add_argument("--val_file", type=str, default=TrainConfig.val_file)
    parser.add_argument("--checkpoint_dir", type=str, default=TrainConfig.checkpoint_dir)

    parser.add_argument("--policy_device", type=str, default=TrainConfig.policy_device)
    parser.add_argument("--reward_device", type=str, default=TrainConfig.reward_device)
    parser.add_argument("--visual_device", type=str, default=TrainConfig.visual_device)
    parser.add_argument("--emotion_device", type=str, default=TrainConfig.emotion_device)
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
    parser.add_argument("--visual_threshold", type=float, default=TrainConfig.visual_threshold)
    parser.add_argument("--visual_top_k", type=int, default=TrainConfig.visual_top_k)
    parser.add_argument("--format_fail_score", type=float, default=TrainConfig.format_fail_score)
    parser.add_argument("--visual_weight", type=float, default=TrainConfig.visual_weight)
    parser.add_argument("--emotion_weight", type=float, default=TrainConfig.emotion_weight)
    parser.add_argument("--situation_weight", type=float, default=TrainConfig.situation_weight)
    parser.add_argument(
        "--segment_advantage_gate_floor",
        type=float,
        default=TrainConfig.segment_advantage_gate_floor,
    )
    parser.add_argument(
        "--endpoint_anchor_window",
        type=int,
        default=TrainConfig.endpoint_anchor_window,
    )
    parser.add_argument(
        "--endpoint_anchor_decay",
        type=float,
        default=TrainConfig.endpoint_anchor_decay,
    )
    parser.add_argument(
        "--traditional_format_weight",
        type=float,
        default=TrainConfig.traditional_format_weight,
    )
    parser.add_argument(
        "--traditional_rouge_weight",
        type=float,
        default=TrainConfig.traditional_rouge_weight,
    )
    parser.add_argument(
        "--traditional_skill_weight",
        type=float,
        default=TrainConfig.traditional_skill_weight,
    )
    parser.add_argument(
        "--traditional_skill_decay",
        type=float,
        default=TrainConfig.traditional_skill_decay,
    )

    parser.add_argument("--lora_rank", type=int, default=TrainConfig.lora_rank)
    parser.add_argument("--lora_alpha", type=int, default=TrainConfig.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=TrainConfig.lora_dropout)
    parser.add_argument("--max_grad_norm", type=float, default=TrainConfig.max_grad_norm)

    parser.add_argument("--max_image_size", type=int, default=TrainConfig.max_image_size)
    parser.add_argument("--max_frames", type=int, default=TrainConfig.max_frames)

    parser.add_argument("--eval_every", type=int, default=TrainConfig.eval_every)
    parser.add_argument("--save_every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--log_every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--val_max_samples", type=int, default=TrainConfig.val_max_samples)
    parser.add_argument("--val_index_file", type=str, default=TrainConfig.val_index_file)

    parser.add_argument("--best_k", type=int, default=TrainConfig.best_k)
    parser.add_argument("--resume", type=str, default=TrainConfig.resume)
    parser.add_argument(
        "--ablation",
        type=str,
        choices=A3_ABLATIONS,
        default=TrainConfig.ablation,
    )
    parser.add_argument("--max_steps", type=int, default=TrainConfig.max_steps)

    parser.add_argument("--eval_only", action="store_true", help="Evaluate training rewards without policy updates.")
    parser.add_argument("--token_credit_mode", choices=["endpoint", "linear", "exponential"], default=TrainConfig.token_credit_mode)
    ns = parser.parse_args()
    legacy_modes = {"w_o_segadv_lineardecay": "linear", "w_o_segadv_expdecay": "exponential"}
    if ns.ablation in legacy_modes:
        ns.token_credit_mode = legacy_modes[ns.ablation]
        ns.ablation = "full"
    elif ns.ablation == "w_o_segadv":
        ns.ablation = "w_o_fine_grained_credit"
    return TrainConfig(**vars(ns))


@dataclass
class ConversationSample:
    system: str
    user_text: str
    image_paths: List[str]
    ground_truth_full: str
    ground_truth_response: str
    ground_truth_first: str
    ground_truth_second: str
    ground_truth_third: str
    ground_truth_fourth: str


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
                gt_segments = parse_think_segments(gt_full)
                self.samples.append(
                    ConversationSample(
                        system=system,
                        user_text=user_text,
                        image_paths=image_paths,
                        ground_truth_full=gt_full,
                        ground_truth_response=gt_resp,
                        ground_truth_first=gt_segments.get("first", ""),
                        ground_truth_second=gt_segments.get("second", ""),
                        ground_truth_third=gt_segments.get("third", ""),
                        ground_truth_fourth=gt_segments.get("fourth", ""),
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


def extract_think_text(text: str) -> str:
    if not text:
        return ""
    m = THINK_PATTERN.search(text)
    if not m:
        return ""
    return m.group(1).strip()


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
    for i, m in enumerate(matches):
        label = _canonical_label(m.group(1))
        if label is None:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(think)
        seg = think[start:end].strip()
        if seg:
            out[label] = seg
    return out


@dataclass
class TokenAdvantageStats:
    adv_visual: float = 0.0
    adv_emotion: float = 0.0
    adv_situation: float = 0.0
    adv_strategy: float = 0.0
    adv_outcome: float = 0.0
    coverage: float = 0.0
    covered_tokens: int = 0
    total_gen_tokens: int = 0


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


def _average_desc_ranks(values: Sequence[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []

    order = sorted(range(n), key=lambda i: float(values[i]), reverse=True)
    ranks = [0.0] * n
    pos = 1
    cursor = 0
    while cursor < n:
        end = cursor + 1
        current = float(values[order[cursor]])
        while end < n and abs(float(values[order[end]]) - current) <= EPS:
            end += 1
        avg_rank = (pos + (pos + (end - cursor) - 1)) / 2.0
        for k in range(cursor, end):
            ranks[order[k]] = avg_rank
        pos += (end - cursor)
        cursor = end
    return ranks


def _rank_advantages(values: Sequence[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    ranks = _average_desc_ranks(values)
    center = (n + 1) / 2.0
    return [float(center - rank) for rank in ranks]


def _char_to_token_pos_via_prefix(tokenizer, full_text: str, char_pos: int) -> int:
    char_pos = max(0, min(char_pos, len(full_text)))
    prefix = full_text[:char_pos]
    return len(tokenizer.encode(prefix, add_special_tokens=False))


def locate_segment_token_spans(
    token_ids: torch.Tensor,  # [1, seq_len]
    prompt_len: int,
    tokenizer,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, Tuple[int, int]], int, int]:
    if token_ids.dim() != 2 or token_ids.shape[0] != 1:
        raise ValueError("token_ids must have shape [1, seq_len]")

    seq_len = int(token_ids.shape[1])
    gen_start = int(prompt_len)
    if attention_mask is not None:
        gen_end = int(attention_mask[0].sum().item())
    else:
        gen_end = seq_len
    gen_end = max(gen_start, min(gen_end, seq_len))

    gen_ids = token_ids[0, gen_start:gen_end]
    gen_text = tokenizer.decode(
        gen_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    out: Dict[str, Tuple[int, int]] = {}

    think_m = THINK_PATTERN.search(gen_text)
    if think_m:
        think_body = think_m.group(1)
        think_body_start = think_m.start(1)
        seg_matches = list(SEGMENT_LABEL_PATTERN.finditer(think_body))
        for i, m in enumerate(seg_matches):
            label = _canonical_label(m.group(1))
            if label is None:
                continue
            char_start = think_body_start + m.end()
            char_end = (
                think_body_start + seg_matches[i + 1].start()
                if i + 1 < len(seg_matches)
                else think_body_start + len(think_body)
            )
            s = _char_to_token_pos_via_prefix(tokenizer, gen_text, char_start)
            e = _char_to_token_pos_via_prefix(tokenizer, gen_text, char_end)
            s = max(0, min(s, gen_ids.shape[0]))
            e = max(s, min(e, gen_ids.shape[0]))
            if e > s:
                out[label] = (gen_start + s, gen_start + e)

    resp_m = RESPONSE_PATTERN.search(gen_text)
    if resp_m:
        char_start, char_end = resp_m.start(1), resp_m.end(1)
        s = _char_to_token_pos_via_prefix(tokenizer, gen_text, char_start)
        e = _char_to_token_pos_via_prefix(tokenizer, gen_text, char_end)
        s = max(0, min(s, gen_ids.shape[0]))
        e = max(s, min(e, gen_ids.shape[0]))
        if e > s:
            out["response"] = (gen_start + s, gen_start + e)

    return out, gen_start, gen_end


def _span_weight_curve(span_len: int, decay_mode: str, anchor_window: int,
                       decay_factor: float, exp_decay_rate: float) -> "torch.Tensor":
    """Distribute an equal credit budget within each span (Appendix F.1).

    Linear and exponential shapes are rescaled to the endpoint schedule's
    total weight S_L. Individual normalized weights may exceed one.
    """
    if span_len <= 0:
        return torch.zeros(0, dtype=torch.float32)
    anchor = min(max(0, int(anchor_window)), span_len)
    budget = anchor + float(decay_factor) * (span_len - anchor)
    positions = torch.arange(span_len, dtype=torch.float32)
    if decay_mode == "linear":
        raw = torch.ones(1, dtype=torch.float32) if span_len == 1 else (
            decay_factor + (1.0 - decay_factor) * positions / (span_len - 1)
        )
    elif decay_mode == "exponential":
        raw = torch.pow(torch.tensor(exp_decay_rate, dtype=torch.float32), (span_len - 1) - positions)
    elif decay_mode in {"hard", "endpoint"}:
        raw = torch.full((span_len,), float(decay_factor), dtype=torch.float32)
        if anchor:
            raw[-anchor:] = 1.0
        return raw
    else:
        raise ValueError(f"Unknown token credit mode: {decay_mode}")
    return raw * (budget / raw.sum().clamp_min(1e-12))


def build_endpoint_anchored_advantages(
    token_ids: torch.Tensor,  # [1, seq_len]
    prompt_len: int,
    tokenizer,
    reward_parts: "RewardComponents",
    segment_advantages: Dict[str, float],
    total_advantage: float,
    cfg: "TrainConfig",
    attention_mask: Optional[torch.Tensor] = None,
    anchor_window: int = 5,
    decay_factor: float = 0.8,
    decay_mode: str = "hard",
    exp_decay_rate: float = 0.7,
) -> Tuple[torch.Tensor, TokenAdvantageStats]:
    seq_len = int(token_ids.shape[1])
    token_advantages = torch.zeros(seq_len, dtype=torch.float32, device=token_ids.device)
    stats = TokenAdvantageStats()

    spans, gen_start, gen_end = locate_segment_token_spans(
        token_ids=token_ids,
        prompt_len=prompt_len,
        tokenizer=tokenizer,
        attention_mask=attention_mask,
    )
    stats.total_gen_tokens = max(0, gen_end - gen_start)
    mapped_mask = torch.zeros(seq_len, dtype=torch.bool, device=token_ids.device)

    seg_adv_visual = float(segment_advantages.get("visual", 0.0))
    seg_adv_emotion = float(segment_advantages.get("emotion", 0.0))
    seg_adv_situation = float(segment_advantages.get("situation", 0.0))
    seg_adv_strategy = float(segment_advantages.get("strategy", 0.0))
    seg_adv_outcome = float(segment_advantages.get("outcome", 0.0))

    # Ablation semantics:
    # - w_o_visual    : remove *training signal* on the "first" (visual) segment tokens.
    # - w_o_emotion   : remove training signal on the "second" (emotion) segment tokens.
    # - w_o_situation : remove training signal on the "third" (situation) segment tokens.
    # - w_o_strategy  : remove training signal on the "fourth" (strategy) segment tokens.
    #
    # Note: We still compute raw scores for logging/analysis, but do not inject the
    # corresponding segment advantages into token-level training objective.
    if cfg.ablation == "w_o_visual":
        seg_adv_visual = 0.0
    if cfg.ablation == "w_o_emotion":
        seg_adv_emotion = 0.0
    if cfg.ablation == "w_o_situation":
        seg_adv_situation = 0.0
    if cfg.ablation == "w_o_strategy":
        seg_adv_strategy = 0.0

    stats.adv_visual = float(seg_adv_visual)
    stats.adv_emotion = float(seg_adv_emotion)
    stats.adv_situation = float(seg_adv_situation)
    stats.adv_strategy = float(seg_adv_strategy)
    stats.adv_outcome = float(seg_adv_outcome)

    if reward_parts.format < 0.5:
        token_advantages[gen_start:gen_end] = float(total_advantage)
        stats.coverage = 0.0
        return token_advantages, stats

    seg_map = {"response": seg_adv_outcome}
    if cfg.ablation != "w_o_visual":
        seg_map["first"] = seg_adv_visual
    if cfg.ablation != "w_o_emotion":
        seg_map["second"] = seg_adv_emotion
    if cfg.ablation != "w_o_situation":
        seg_map["third"] = seg_adv_situation
    if cfg.ablation != "w_o_strategy":
        seg_map["fourth"] = seg_adv_strategy
    for seg_name, seg_adv in seg_map.items():
        if seg_name not in spans:
            continue
        seg_start, seg_end = spans[seg_name]
        seg_start = max(gen_start, min(seg_start, gen_end))
        seg_end = max(seg_start, min(seg_end, gen_end))
        if seg_end <= seg_start:
            continue

        span_len = seg_end - seg_start
        weights = _span_weight_curve(
            span_len=span_len,
            decay_mode=decay_mode,
            anchor_window=anchor_window,
            decay_factor=decay_factor,
            exp_decay_rate=exp_decay_rate,
        ).to(device=token_advantages.device)
        token_advantages[seg_start:seg_end] = weights * float(seg_adv)
        mapped_mask[seg_start:seg_end] = True

    stats.covered_tokens = int(mapped_mask[gen_start:gen_end].sum().item())
    if stats.total_gen_tokens > 0:
        stats.coverage = float(stats.covered_tokens / stats.total_gen_tokens)
    else:
        stats.coverage = 0.0
    return token_advantages, stats


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
    # 大模型需要特定的前缀和后缀来区分谁在说话。这个函数会把你的字典自动翻译成 Qwen 认识的底层字符串格式。
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    tokenizer = processor.tokenizer
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"   # Qwen 是从左往右预测下一个词的（自回归）。如果把空位补在右边（尾部），模型在预测时就会看到一堆乱码，彻底懵掉。

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


def _parse_strategy_score(text: str) -> float:
    if not text:
        return 0.0

    text = text.strip()
    try:
        maybe_json = json.loads(text)
        if isinstance(maybe_json, dict) and "score" in maybe_json:
            val = float(maybe_json["score"])
            if val in (0.0, 0.5, 1.0):
                return val
    except Exception:
        pass

    m = STRATEGY_SCORE_PATTERN.search(text)
    if m:
        try:
            val = float(m.group(1))
            if val in (0.0, 0.5, 1.0):
                return val
        except ValueError:
            pass

    return 0.0


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _normalize_likert_score(value: float) -> float:
    return min(1.0, max(0.0, (float(value) - 1.0) / 4.0))


def _parse_triplet_scores(text: str) -> Dict[str, float]:
    parsed = _extract_json_object(text)
    result = {
        "situation_score": 0.0,
        "strategy_score": 0.0,
        "outcome_score": 0.0,
    }

    if isinstance(parsed, dict):
        for key in result:
            if key not in parsed:
                continue
            try:
                raw_value = float(parsed[key])
            except (TypeError, ValueError):
                continue

            # Full FACES uses discrete 1-5 scores; invalid critic fields score zero.
            if not isinstance(parsed[key], bool) and raw_value.is_integer() and 1.0 <= raw_value <= 5.0:
                result[key] = _normalize_likert_score(raw_value)

    return result


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

        with suppress_stdout_stderr():
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )   # trust_remote_code=True is required for custom model classes such as Qwen.
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
        with suppress_stdout_stderr():
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
            with suppress_stdout_stderr():
                self.processor = AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=True
                )
            if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
                if self.processor.tokenizer.pad_token_id is None and self.processor.tokenizer.eos_token_id is not None:
                    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                self.processor.tokenizer.padding_side = "left"
        except Exception:
            self.processor = None

    def _chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
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
    ) -> Tuple[List[float], List[float], List[float]]:
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
        outcome_totals = [0.0] * n

        for variant in JUDGE_PROMPT_VARIANTS:
            variant_situation = [0.0] * n
            variant_strategy = [0.0] * n
            variant_outcome = [0.0] * n

            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                prompts: List[str] = []
                for i in range(start, end):
                    user_prompt = variant["user_template"].format(
                        context=(contexts[i] or "")[-3000:],
                        candidate_situation=(candidate_situations[i] or "")[:1200],
                        reference_situation=(reference_situations[i] or "")[:1200],
                        candidate_strategy=(candidate_strategies[i] or "")[:1200],
                        reference_strategy=(reference_strategies[i] or "")[:1200],
                        candidate_response=(candidate_responses[i] or "")[:1200],
                        reference_response=(reference_responses[i] or "")[:1200],
                    )
                    prompts.append(self._chat_prompt(variant["system"], user_prompt))

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
                        do_sample=False,
                        remove_invalid_values=True,
                        logits_processor=_SANITIZE_LOGITS,
                        max_new_tokens=self.max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                input_lens = model_inputs["attention_mask"].sum(dim=1).tolist()
                for row in range(gen_ids.shape[0]):
                    new_tokens = gen_ids[row, int(input_lens[row]) :]
                    raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    parsed = _parse_triplet_scores(raw)
                    idx = start + row
                    variant_situation[idx] = parsed["situation_score"]
                    variant_strategy[idx] = parsed["strategy_score"]
                    variant_outcome[idx] = parsed["outcome_score"]

                del model_inputs, gen_ids

            for i in range(n):
                situation_totals[i] += variant_situation[i]
                strategy_totals[i] += variant_strategy[i]
                outcome_totals[i] += variant_outcome[i]

        denom = float(len(JUDGE_PROMPT_VARIANTS))
        return (
            [score / denom for score in situation_totals],
            [score / denom for score in strategy_totals],
            [score / denom for score in outcome_totals],
        )


def _normalize_cosine_to_unit(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


class VisualEvidenceScorer:
    def __init__(self, model_path: str, device: str, threshold: float, top_k: int):
        self.device = device
        self.threshold = threshold   # Similarity threshold.
        self.top_k = max(1, int(top_k))  # Average over the top-k most relevant images.
        with suppress_stdout_stderr():
            self.processor = CLIPProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
        dtype = torch.float16 if "cuda" in device else torch.float32
        with suppress_stdout_stderr():
            self.model = load_pretrained_with_fallback(
                model_class=CLIPModel,
                model_path=model_path,
                model_kwargs={
                    "dtype": dtype,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                },
                model_name="clip_scorer",
            ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        text_cfg = getattr(self.model.config, "text_config", None)
        self.max_text_len = int(getattr(text_cfg, "max_position_embeddings", 77) or 77)
        if self.max_text_len <= 0:
            self.max_text_len = 77

    def score_batch(
        self,
        candidate_visual_texts: Sequence[str],
        images: Sequence[Image.Image],
    ) -> Tuple[List[float], List[float]]:
        if not candidate_visual_texts:
            return [], []
        if not images:
            zeros = [0.0] * len(candidate_visual_texts)
            return zeros, zeros

        with torch.no_grad():
            image_inputs = self.processor(images=list(images), return_tensors="pt")
            image_inputs = _move_to_device(dict(image_inputs), self.device)
            image_features = self.model.get_image_features(**image_inputs)
            # L2 normalization makes dot products equivalent to cosine similarity.
            image_features = F.normalize(image_features, p=2, dim=-1)

            text_inputs = self.processor(
                text=list(candidate_visual_texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_text_len,
            )
            text_inputs = _move_to_device(dict(text_inputs), self.device)
            try:
                text_features = self.model.get_text_features(**text_inputs)
                text_features = F.normalize(text_features, p=2, dim=-1)

                sim = torch.matmul(text_features, image_features.transpose(0, 1))
                k = min(self.top_k, sim.shape[1])
                raw_cos = torch.topk(sim, k=k, dim=1).values.mean(dim=1)
                norm_scores = _normalize_cosine_to_unit(raw_cos)
                binary_scores = (norm_scores >= self.threshold).float()
                return (
                    binary_scores.detach().cpu().tolist(),
                    norm_scores.detach().cpu().tolist(),
                )
            except Exception as exc:
                # Fallback to per-sample scoring so one malformed text does not crash the whole step.
                print(f"[VISUAL] batch text scoring failed, fallback per-sample: {exc}")

            zeros = [0.0] * len(candidate_visual_texts)
            binary_list: List[float] = zeros.copy()
            norm_list: List[float] = zeros.copy()

            for idx, text in enumerate(candidate_visual_texts):
                try:
                    single_text_inputs = self.processor(
                        text=[text],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_text_len,
                    )
                    single_text_inputs = _move_to_device(dict(single_text_inputs), self.device)
                    single_text_features = self.model.get_text_features(**single_text_inputs)
                    single_text_features = F.normalize(single_text_features, p=2, dim=-1)
                    sim = torch.matmul(single_text_features, image_features.transpose(0, 1))
                    k = min(self.top_k, sim.shape[1])
                    raw_cos = torch.topk(sim, k=k, dim=1).values.mean(dim=1)
                    norm_score = float(_normalize_cosine_to_unit(raw_cos)[0].item())
                    binary_score = 1.0 if norm_score >= float(self.threshold) else 0.0
                    norm_list[idx] = norm_score
                    binary_list[idx] = binary_score
                except Exception as single_exc:
                    print(f"[VISUAL] sample text scoring failed idx={idx}: {single_exc}")
                    continue

        return binary_list, norm_list


class EmotionSemanticScorer:
    def __init__(self, model_path: str, device: str):
        self.device = device
        with suppress_stdout_stderr():  # Suppress noisy backend warnings.
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
        dtype = torch.float16 if "cuda" in device else torch.float32
        with suppress_stdout_stderr():
            self.model = load_pretrained_with_fallback(
                model_class=AutoModel,
                model_path=model_path,
                model_kwargs={
                    "dtype": dtype,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                },
                model_name="emotion_scorer",
            ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)

    def _encode(self, texts: Sequence[str]) -> torch.Tensor:
        # Return an empty [0, 1] tensor for empty inputs.
        if not texts:
            return torch.empty((0, 1), dtype=torch.float32, device=self.device)

        model_inputs = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        model_inputs = _move_to_device(dict(model_inputs), self.device)

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
            raise ValueError("candidates and references must have same length")
        cand_emb = self._encode(candidates)
        ref_emb = self._encode(references)
        cos = (cand_emb * ref_emb).sum(dim=-1)
        score = _normalize_cosine_to_unit(cos)
        return score.detach().cpu().tolist()


class TraditionalSurfaceSkillScorer:
    """FIRES-style lightweight reward: format + ROUGE-L + EPITOME skill match."""

    def __init__(self, model_path: str, device: str, skill_decay: float):
        from rouge_score import rouge_scorer

        self.device = device
        self.skill_decay = float(skill_decay)
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        with suppress_stdout_stderr():
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

    def rouge_l(self, candidate_response: str, reference_response: str) -> float:
        candidate_response = (candidate_response or "").strip()
        reference_response = (reference_response or "").strip()
        if not candidate_response or not reference_response:
            return 0.0
        try:
            return float(
                self.rouge.score(reference_response, candidate_response)["rougeL"].fmeasure
            )
        except Exception:
            return 0.0

    def _predict_logits(self, texts: Sequence[str]) -> torch.Tensor:
        model_inputs = self.tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        model_inputs = _move_to_device(dict(model_inputs), self.device)
        with torch.no_grad():
            logits = self.model(**model_inputs).logits
        return logits

    def skill_scores(
        self,
        contexts: Sequence[str],
        candidate_responses: Sequence[str],
        reference_responses: Sequence[str],
    ) -> List[float]:
        if not candidate_responses:
            return []
        candidate_inputs: List[str] = []
        reference_inputs: List[str] = []
        for context, cand, ref in zip(contexts, candidate_responses, reference_responses):
            context_tail = str(context or "")[-512:]
            candidate_inputs.append(context_tail + str(cand or ""))
            reference_inputs.append(context_tail + str(ref or ""))

        try:
            cand_logits = self._predict_logits(candidate_inputs)
            ref_logits = self._predict_logits(reference_inputs)
            mse = F.mse_loss(cand_logits, ref_logits, reduction="none").mean(dim=-1)
            scores = torch.exp(-mse * self.skill_decay).clamp(0.0, 1.0)
            return scores.detach().cpu().tolist()
        except Exception as exc:
            print(f"[TRADITIONAL_REWARD] DIFF-EPITOME scorer failed: {exc}")
            return [0.0] * len(candidate_responses)


@dataclass
class RewardComponents:
    format: float
    visual: float
    emotion: float
    situation: float
    strategy: float
    outcome: float
    total: float
    reason: str


class RewardEngine:
    def __init__(
        self,
        cfg: TrainConfig,
        judge: Optional[OutcomeJudge],
        visual_scorer: Optional[VisualEvidenceScorer],
        emotion_scorer: Optional[EmotionSemanticScorer],
        traditional_scorer: Optional[TraditionalSurfaceSkillScorer] = None,
    ):
        self.cfg = cfg
        self.judge = judge
        self.visual_scorer = visual_scorer
        self.emotion_scorer = emotion_scorer
        self.traditional_scorer = traditional_scorer

    def _score_with_traditional_reward(
        self,
        sample: ConversationSample,
        completions: Sequence[str],
        parts: List[RewardComponents],
        valid_indices: Sequence[int],
    ) -> List[RewardComponents]:
        if self.traditional_scorer is None:
            raise RuntimeError("traditional_scorer is required for w_o_multi_source_reward")

        candidate_responses = [
            extract_response_text(completions[idx]).strip() for idx in valid_indices
        ]
        reference_responses = [
            (sample.ground_truth_response or "").strip() for _ in valid_indices
        ]
        contexts = [sample.user_text for _ in valid_indices]

        rouge_scores = [
            self.traditional_scorer.rouge_l(candidate, reference)
            for candidate, reference in zip(candidate_responses, reference_responses)
        ]
        skill_scores = self.traditional_scorer.skill_scores(
            contexts=contexts,
            candidate_responses=candidate_responses,
            reference_responses=reference_responses,
        )

        for local_idx, idx in enumerate(valid_indices):
            rouge = float(rouge_scores[local_idx])
            skill = float(skill_scores[local_idx])
            total = (
                float(self.cfg.traditional_format_weight) * float(parts[idx].format)
                + float(self.cfg.traditional_rouge_weight) * rouge
                + float(self.cfg.traditional_skill_weight) * skill
            )
            # Store the scalar traditional reward in all semantic slots so the
            # fine-grained credit-assignment machinery can still operate.
            parts[idx].visual = total
            parts[idx].emotion = total
            parts[idx].situation = total
            parts[idx].strategy = total
            parts[idx].outcome = total
            parts[idx].total = total
            parts[idx].reason = f"traditional_reward rouge={rouge:.4f} skill={skill:.4f}"

        return parts

    def score_completions(
        self,
        sample: ConversationSample,
        completions: Sequence[str],
        images: Sequence[Image.Image],
    ) -> List[RewardComponents]:
        parts: List[RewardComponents] = []
        segments_cache: List[Dict[str, str]] = []
        for completion in completions:
            r_f = format_reward(completion)
            if r_f < 0.5:
                parts.append(
                    RewardComponents(
                        format=0.0,
                        visual=0.0,
                        emotion=0.0,
                        situation=0.0,
                        strategy=0.0,
                        outcome=0.0,
                        total=float(self.cfg.format_fail_score),
                        reason="format_check_failed",
                    )
                )
                segments_cache.append({})
                continue

            parts.append(
                RewardComponents(
                    format=1.0,
                    visual=0.0,
                    emotion=0.0,
                    situation=0.0,
                    strategy=0.0,
                    outcome=0.0,
                    total=0.0,
                    reason="ok",
                )
            )
            segments_cache.append(parse_think_segments(completion))

        valid_indices = [i for i, p in enumerate(parts) if p.format > 0.5]
        if not valid_indices:
            return parts

        if self.cfg.ablation == "w_o_multi_source_reward":
            return self._score_with_traditional_reward(
                sample=sample,
                completions=completions,
                parts=parts,
                valid_indices=valid_indices,
            )

        if self.judge is None or self.visual_scorer is None or self.emotion_scorer is None:
            raise RuntimeError("multi-source reward scorers are not initialized")

        # visual: score first segment with CLIP and threshold to 0/1
        visual_indices: List[int] = []
        visual_texts: List[str] = []
        for idx in valid_indices:
            first_seg = segments_cache[idx].get("first", "").strip()
            if not first_seg:
                parts[idx].visual = 0.0
                parts[idx].reason = "missing first segment"
                continue
            visual_indices.append(idx)
            visual_texts.append(first_seg)
        if visual_indices:
            visual_binary, _ = self.visual_scorer.score_batch(visual_texts, images)
            for local_idx, score in enumerate(visual_binary):
                idx = visual_indices[local_idx]
                parts[idx].visual = float(score)

        # emotion: continuous score with MiniLM against GT second segment
        emotion_indices: List[int] = []
        emotion_candidates: List[str] = []
        emotion_refs: List[str] = []
        for idx in valid_indices:
            second_seg = segments_cache[idx].get("second", "").strip()
            gt_second = (sample.ground_truth_second or "").strip()
            if not second_seg or not gt_second:
                parts[idx].emotion = 0.0
                if not second_seg:
                    parts[idx].reason = "missing second segment"
                continue
            emotion_indices.append(idx)
            emotion_candidates.append(second_seg)
            emotion_refs.append(gt_second)
        if emotion_indices:
            emotion_scores = self.emotion_scorer.score_batch(emotion_candidates, emotion_refs)
            for local_idx, score in enumerate(emotion_scores):
                idx = emotion_indices[local_idx]
                parts[idx].emotion = float(score)

        # Qwen judge: evaluate situation / strategy / outcome jointly with three prompt variants.
        candidate_situations = [
            segments_cache[idx].get("third", "").strip() for idx in valid_indices
        ]
        candidate_strategies = [
            segments_cache[idx].get("fourth", "").strip() for idx in valid_indices
        ]
        candidate_responses = [
            extract_response_text(completions[idx]).strip() for idx in valid_indices
        ]
        reference_situations = [
            (sample.ground_truth_third or "").strip() for _ in valid_indices
        ]
        reference_strategies = [
            (sample.ground_truth_fourth or "").strip() for _ in valid_indices
        ]
        reference_responses = [
            (sample.ground_truth_response or "").strip() for _ in valid_indices
        ]

        situation_scores, strategy_scores, outcome_scores = self.judge.score_triplet_batch(
            contexts=[sample.user_text] * len(valid_indices),
            candidate_situations=candidate_situations,
            reference_situations=reference_situations,
            candidate_strategies=candidate_strategies,
            reference_strategies=reference_strategies,
            candidate_responses=candidate_responses,
            reference_responses=reference_responses,
            batch_size=self.cfg.reward_batch_size,
        )

        for local_idx, idx in enumerate(valid_indices):
            parts[idx].outcome = float(outcome_scores[local_idx])

            if candidate_situations[local_idx] and reference_situations[local_idx]:
                parts[idx].situation = float(situation_scores[local_idx])
            else:
                parts[idx].situation = 0.0
                if not candidate_situations[local_idx]:
                    parts[idx].reason = "missing third segment"

            if candidate_strategies[local_idx] and reference_strategies[local_idx]:
                parts[idx].strategy = float(strategy_scores[local_idx])
            else:
                parts[idx].strategy = 0.0
                if not candidate_strategies[local_idx]:
                    parts[idx].reason = "missing forth segment"

        for p in parts:
            p.total = compose_a3_total_reward(p, self.cfg)

        return parts

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
    def __init__(self, checkpoint_root: str, keep_top_k: int, ablation_label: str):
        self.checkpoint_root = Path(checkpoint_root)
        self.keep_top_k = keep_top_k
        self.ablation_label = str(ablation_label or "full")
        self.best_txt = self.checkpoint_root / "best.txt"
        self.best_json = self.checkpoint_root / "best_records.json"
        self.records: List[Dict[str, Any]] = []
        self._ensure_files()
        self._load_existing()

    def _with_defaults(self, record: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(record)
        out.setdefault("ablation", self.ablation_label)
        return out

    def _ensure_files(self) -> None:
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        if not self.best_json.exists():
            self.best_json.write_text("[]\n", encoding="utf-8")
        if not self.best_txt.exists():
            header = (
                "# Top validation points (sorted by val_avg_total desc)\n"
                "rank\tablation\tstep\tepoch\tval_avg_total\tval_avg_format\tval_avg_visual\t"
                "val_avg_emotion\tval_avg_situation\tval_avg_strategy\tval_avg_outcome\tcheckpoint\n"
            )
            self.best_txt.write_text(header, encoding="utf-8")

    def _load_existing(self) -> None:
        if self.best_json.exists():
            try:
                data = json.loads(self.best_json.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.records = [self._with_defaults(rec) for rec in data if isinstance(rec, dict)]
            except Exception:
                self.records = []

    def update(self, record: Dict[str, Any]) -> None:
        self.records.append(self._with_defaults(record))
        self.records.sort(
            key=lambda x: (float(x.get("val_avg_total", 0.0)), -int(x.get("step", 0))),
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
            "# Top validation points (sorted by val_avg_total desc)",
            "rank\tablation\tstep\tepoch\tval_avg_total\tval_avg_format\tval_avg_visual\t"
            "val_avg_emotion\tval_avg_situation\tval_avg_strategy\tval_avg_outcome\tcheckpoint",
        ]
        for i, rec in enumerate(self.records, start=1):
            lines.append(
                f"{i}\t{rec.get('ablation', self.ablation_label)}\t"
                f"{rec.get('step', 0)}\t{rec.get('epoch', 0)}\t"
                f"{float(rec.get('val_avg_total', 0.0)):.6f}\t"
                f"{float(rec.get('val_avg_format', 0.0)):.6f}\t"
                f"{float(rec.get('val_avg_visual', 0.0)):.6f}\t"
                f"{float(rec.get('val_avg_emotion', 0.0)):.6f}\t"
                f"{float(rec.get('val_avg_situation', 0.0)):.6f}\t"
                f"{float(rec.get('val_avg_strategy', 0.0)):.6f}\t"
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
                "ablation": str(trainer_state.get("config", {}).get("ablation", "full")),
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
                f"Ablation: {str(trainer_state.get('config', {}).get('ablation', 'full'))}\n"
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
        np.random.set_state(state["numpy"])       # Restore NumPy random state.
    if "torch" in state:
        torch.set_rng_state(state["torch"])       # Restore PyTorch CPU random state.
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])     # Restore PyTorch CUDA random state.

'''
Extract vision-only keyword arguments before forwarding through the vision encoder.
'''
def _extract_forward_vision_kwargs(inputs: Dict[str, Any]) -> Dict[str, Any]:  # Extract the fields used by the vision encoder.
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


def _default_val_index_file(cfg: TrainConfig) -> Path:
    if cfg.val_index_file:
        return Path(cfg.val_index_file).expanduser()
    val_path = Path(cfg.val_file).expanduser()
    sample_count = max(0, int(cfg.val_max_samples))
    return val_path.with_name(
        f"{val_path.stem}.fixed_{sample_count}_uniform_indices.json"
    )


def _build_uniform_sample_indices(dataset_size: int, sample_count: int) -> List[int]:
    sample_count = min(max(0, int(sample_count)), max(0, int(dataset_size)))
    if sample_count <= 0:
        return []
    if sample_count >= dataset_size:
        return list(range(dataset_size))

    indices: List[int] = []
    seen = set()
    for i in range(sample_count):
        idx = min(dataset_size - 1, int((i + 0.5) * dataset_size / sample_count))
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)

    if len(indices) < sample_count:
        for idx in range(dataset_size):
            if idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)
            if len(indices) >= sample_count:
                break

    return sorted(indices)


def _load_or_create_fixed_val_indices(
    dataset_size: int,
    cfg: TrainConfig,
) -> Tuple[List[int], Path, str]:
    sample_count = min(dataset_size, max(0, int(cfg.val_max_samples)))
    index_file = _default_val_index_file(cfg)
    if sample_count <= 0:
        return list(range(dataset_size)), index_file, "full"

    index_file.parent.mkdir(parents=True, exist_ok=True)
    expected_val_file = str(Path(cfg.val_file).expanduser())

    if index_file.exists():
        try:
            payload = json.loads(index_file.read_text(encoding="utf-8"))
            raw_indices: Sequence[Any]
            metadata_ok = False
            if isinstance(payload, list):
                raw_indices = payload
                metadata_ok = True
            elif isinstance(payload, dict):
                raw_indices = payload.get("indices", [])
                metadata_ok = (
                    int(payload.get("dataset_size", dataset_size)) == dataset_size
                    and int(payload.get("sample_count", sample_count)) == sample_count
                    and str(payload.get("val_file", expected_val_file)) == expected_val_file
                )
            else:
                raw_indices = []

            loaded_indices: List[int] = []
            seen = set()
            for item in raw_indices:
                try:
                    idx = int(item)
                except (TypeError, ValueError):
                    continue
                if idx < 0 or idx >= dataset_size or idx in seen:
                    continue
                seen.add(idx)
                loaded_indices.append(idx)

            if metadata_ok and len(loaded_indices) == sample_count:
                return loaded_indices, index_file, "loaded"
        except Exception:
            pass

    indices = _build_uniform_sample_indices(dataset_size, sample_count)
    payload = {
        "val_file": expected_val_file,
        "dataset_size": dataset_size,
        "sample_count": len(indices),
        "strategy": "uniform",
        "indices": indices,
    }
    index_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return indices, index_file, "created"


# Run validation by generating policy responses and scoring them with the judge.
def run_validation(
    policy_model,
    processor,
    reward_engine: RewardEngine,
    val_dataset: JsonlConversationDataset,
    cfg: TrainConfig,
) -> Dict[str, float]:
    policy_model.eval()
    dataset_size = len(val_dataset)
    if cfg.val_max_samples > 0:
        sampled_indices, index_file, index_state = _load_or_create_fixed_val_indices(
            dataset_size,
            cfg,
        )
        print(
            f"[VAL] {index_state} fixed {len(sampled_indices)}/{dataset_size} "
            f"uniform validation indices -> {index_file}"
        )
    else:
        sampled_indices = list(range(dataset_size))
        print(f"[VAL] using full validation set ({dataset_size}/{dataset_size})")
    val_count = len(sampled_indices)

    total_total = 0.0
    total_format = 0.0
    total_visual = 0.0
    total_emotion = 0.0
    total_situation = 0.0
    total_strategy = 0.0
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
                    logits_processor=_SANITIZE_LOGITS,
                    max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            prompt_len = int(inputs["input_ids"].shape[1])
            generated_tokens = gen_ids[:, prompt_len:]
            completion = processor.tokenizer.decode(
                generated_tokens[0], skip_special_tokens=True
            )
            comp = reward_engine.score_completions(
                sample=sample,
                completions=[completion],
                images=images,
            )[0]

            total_total += float(comp.total)
            total_format += float(comp.format)
            total_visual += float(comp.visual)
            total_emotion += float(comp.emotion)
            total_situation += float(comp.situation)
            total_strategy += float(comp.strategy)
            total_outcome += float(comp.outcome)
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
            "val_avg_total": 0.0,
            "val_avg_format": 0.0,
            "val_avg_visual": 0.0,
            "val_avg_emotion": 0.0,
            "val_avg_situation": 0.0,
            "val_avg_strategy": 0.0,
            "val_avg_outcome": 0.0,
            "val_count": 0,
        }

    return {
        "val_avg_total": total_total / valid_samples,
        "val_avg_format": total_format / valid_samples,
        "val_avg_visual": total_visual / valid_samples,
        "val_avg_emotion": total_emotion / valid_samples,
        "val_avg_situation": total_situation / valid_samples,
        "val_avg_strategy": total_strategy / valid_samples,
        "val_avg_outcome": total_outcome / valid_samples,
        "val_count": valid_samples,
    }


def train_one_update_step(
    policy_model,
    processor,
    reward_engine: RewardEngine,
    train_dataset: JsonlConversationDataset,
    batch_indices: Sequence[int],
    optimizer,
    scheduler,
    cfg: TrainConfig,
) -> Dict[str, float]:
    def _empty_metrics() -> Dict[str, float]:
        return {
            "loss": 0.0,
            "avg_total": 0.0,
            "avg_format": 0.0,
            "avg_visual": 0.0,
            "avg_emotion": 0.0,
            "avg_situation": 0.0,
            "avg_strategy": 0.0,
            "avg_outcome": 0.0,
            "avg_kl": 0.0,
            "seg_adv_visual": 0.0,
            "seg_adv_emotion": 0.0,
            "seg_adv_situation": 0.0,
            "seg_adv_strategy": 0.0,
            "seg_adv_outcome": 0.0,
            "seg_coverage": 0.0,
            "valid_samples": 0,
            "valid_sequences": 0,
            "optimizer_updates": 0,
            "lr": scheduler.get_last_lr()[0],
        }

    if not _cuda_is_healthy(cfg.policy_device):
        print("[TRAIN] CUDA context unhealthy at step start, skipping entire batch.")
        return _empty_metrics()

    policy_model.train()
    optimizer.zero_grad(set_to_none=True)

    batch_total_total = 0.0
    batch_total_format = 0.0
    batch_total_visual = 0.0
    batch_total_emotion = 0.0
    batch_total_situation = 0.0
    batch_total_strategy = 0.0
    batch_total_outcome = 0.0
    batch_total_loss = 0.0
    batch_total_kl = 0.0
    batch_seg_adv_visual = 0.0
    batch_seg_adv_emotion = 0.0
    batch_seg_adv_situation = 0.0
    batch_seg_adv_strategy = 0.0
    batch_seg_adv_outcome = 0.0
    batch_seg_coverage = 0.0
    valid_samples = 0
    valid_sequences = 0
    optimizer_updates = 0
    cuda_poisoned = False
    rollout_cache: List[RolloutSequenceCache] = []

    for idx in batch_indices:
        if cuda_poisoned or not _cuda_is_healthy(cfg.policy_device):
            cuda_poisoned = True
            print(f"[TRAIN] CUDA poisoned, aborting batch at idx={idx}")
            break

        sample = train_dataset[idx]
        images = load_resized_images(sample.image_paths, cfg.max_frames, cfg.max_image_size)
        messages = build_messages(sample, images)

        try:
            inputs, _ = prepare_actor_inputs(processor, messages, images, cfg.policy_device)
        except Exception as exc:
            print(f"[TRAIN] prepare input failed idx={idx}: {exc}")
            if _is_cuda_runtime_error(exc):
                cuda_poisoned = True
                break
            continue

        try:
            policy_model.eval()
            with torch.no_grad():
                gen_ids = policy_model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=50,
                    remove_invalid_values=True,
                    logits_processor=_SANITIZE_LOGITS,
                    max_new_tokens=cfg.max_new_tokens,
                    num_return_sequences=cfg.group_size,   # Generate multiple sampled responses for the same prompt.
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
        except Exception as exc:
            print(f"[TRAIN] generation failed idx={idx}: {exc}")
            policy_model.train()
            if _is_cuda_runtime_error(exc):
                cuda_poisoned = True
                break
            continue

        if gen_ids.dim() == 1:
            gen_ids = gen_ids.unsqueeze(0)

        prompt_len = int(inputs["input_ids"].shape[1])
        generated_only = gen_ids[:, prompt_len:]

        completions: List[str] = []
        for row in range(generated_only.shape[0]):
            completion = processor.tokenizer.decode(generated_only[row], skip_special_tokens=True)
            completions.append(completion)  # completions stores G candidate responses for each prompt.

        reward_parts = reward_engine.score_completions(
            sample=sample,
            completions=completions,
            images=images,
        )
        reward_list = [float(x.total) for x in reward_parts]
        group_max_total_reward = max([0.0] + reward_list)
        total_rank_advs = _rank_advantages(reward_list)
        dim_rank_advs = {
            "visual": _rank_advantages([float(rp.visual) for rp in reward_parts]),
            "emotion": _rank_advantages([float(rp.emotion) for rp in reward_parts]),
            "situation": _rank_advantages([float(rp.situation) for rp in reward_parts]),
            "strategy": _rank_advantages([float(rp.strategy) for rp in reward_parts]),
            "outcome": _rank_advantages([float(rp.outcome) for rp in reward_parts]),
        }

        vision_kwargs = _extract_forward_vision_kwargs(inputs)
        vision_kwargs_cpu = _tensor_tree_to_cpu(vision_kwargs)
        pad_id = processor.tokenizer.pad_token_id

        try:
            for g in range(gen_ids.shape[0]):
                if not _cuda_is_healthy(cfg.policy_device):
                    cuda_poisoned = True
                    print(f"[TRAIN] CUDA poisoned during rollout-cache build idx={idx} group={g}")
                    break

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
                    if _is_cuda_runtime_error(exc):
                        cuda_poisoned = True
                    break

                adv_stats = TokenAdvantageStats()
                if cfg.ablation == "w_o_fine_grained_credit":
                    token_advs_shifted = torch.full_like(
                        token_mask,
                        float(total_rank_advs[g]) if reward_parts[g].format >= 0.5 else 0.0,
                        dtype=torch.float32,
                    )
                else:
                    segment_advantages = {
                        "visual": float(dim_rank_advs["visual"][g]),
                        "emotion": float(dim_rank_advs["emotion"][g]),
                        "situation": float(dim_rank_advs["situation"][g]),
                        "strategy": float(dim_rank_advs["strategy"][g]),
                        "outcome": float(dim_rank_advs["outcome"][g]),
                    }
                    segment_advantages = gate_positive_segment_advantages(
                        segment_advantages=segment_advantages,
                        total_reward=float(reward_parts[g].total),
                        group_max_total_reward=group_max_total_reward,
                        cfg=cfg,
                    )
                    token_advantages, adv_stats = build_endpoint_anchored_advantages(
                        token_ids=seq_ids,
                        prompt_len=prompt_len,
                        tokenizer=processor.tokenizer,
                        reward_parts=reward_parts[g],
                        segment_advantages=segment_advantages,
                        total_advantage=float(total_rank_advs[g]),
                        cfg=cfg,
                        attention_mask=attention_mask,
                        anchor_window=cfg.endpoint_anchor_window,
                        decay_factor=cfg.endpoint_anchor_decay,
                        decay_mode={"endpoint": "hard", "linear": "linear", "exponential": "exponential"}[cfg.token_credit_mode],
                        exp_decay_rate=0.7,
                    )
                    token_advs_shifted = token_advantages[1:].unsqueeze(0)

                rollout_cache.append(
                    RolloutSequenceCache(
                        seq_ids=seq_ids.detach().cpu(),
                        labels=labels.detach().cpu(),
                        attention_mask=attention_mask.detach().cpu(),
                        vision_kwargs=vision_kwargs_cpu,
                        old_logp=old_logp.detach().cpu(),
                        ref_logp=ref_logp.detach().cpu(),
                        token_mask=token_mask.detach().cpu(),
                        token_advantages=token_advs_shifted.detach().cpu(),
                    )
                )
                valid_sequences += 1

                batch_seg_adv_visual += abs(float(adv_stats.adv_visual))
                batch_seg_adv_emotion += abs(float(adv_stats.adv_emotion))
                batch_seg_adv_situation += abs(float(adv_stats.adv_situation))
                batch_seg_adv_strategy += abs(float(adv_stats.adv_strategy))
                batch_seg_adv_outcome += abs(float(adv_stats.adv_outcome))
                batch_seg_coverage += float(adv_stats.coverage)

                del seq_ids, labels, attention_mask, old_out, old_logp, ref_out, ref_logp
                del token_mask, token_advs_shifted
                if cfg.ablation != "w_o_fine_grained_credit":
                    del token_advantages
        finally:
            policy_model.train()

        if cuda_poisoned:
            del inputs, gen_ids, generated_only
            break

        batch_total_total += float(sum(p.total for p in reward_parts) / len(reward_parts))
        batch_total_format += float(sum(p.format for p in reward_parts) / len(reward_parts))
        batch_total_visual += float(sum(p.visual for p in reward_parts) / len(reward_parts))
        batch_total_emotion += float(sum(p.emotion for p in reward_parts) / len(reward_parts))
        batch_total_situation += float(sum(p.situation for p in reward_parts) / len(reward_parts))
        batch_total_strategy += float(sum(p.strategy for p in reward_parts) / len(reward_parts))
        batch_total_outcome += float(sum(p.outcome for p in reward_parts) / len(reward_parts))
        valid_samples += 1

        del inputs, gen_ids, generated_only

    if cuda_poisoned:
        print("[TRAIN] CUDA poisoned this batch - skipping optimizer.step().")
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        return _empty_metrics()

    if valid_samples == 0:
        optimizer.zero_grad(set_to_none=True)
        return _empty_metrics()

    if valid_sequences == 0 or not rollout_cache:
        print("[TRAIN] no valid sequences in this batch, skipping optimizer.step().")
        optimizer.zero_grad(set_to_none=True)
        return _empty_metrics()

    updates_per_rollout = max(1, int(cfg.grpo_updates_per_rollout))
    clip_eps = max(0.0, float(cfg.grpo_clip_epsilon))

    for update_idx in range(updates_per_rollout):
        if cuda_poisoned or not _cuda_is_healthy(cfg.policy_device):
            cuda_poisoned = True
            print(f"[TRAIN] CUDA poisoned before optimizer update {update_idx + 1}/{updates_per_rollout}")
            break

        optimizer.zero_grad(set_to_none=True)
        update_loss_value = 0.0
        update_kl_value = 0.0
        update_valid_sequences = 0

        for seq_idx, rollout in enumerate(rollout_cache):
            if not _cuda_is_healthy(cfg.policy_device):
                cuda_poisoned = True
                print(f"[TRAIN] CUDA poisoned during optimizer update seq={seq_idx}")
                break

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
                if _is_cuda_runtime_error(exc):
                    cuda_poisoned = True
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

        if cuda_poisoned:
            optimizer.zero_grad(set_to_none=True)
            break

        if update_valid_sequences == 0:
            print(f"[TRAIN] optimizer update {update_idx + 1} had no valid sequences, skipping step().")
            optimizer.zero_grad(set_to_none=True)
            continue

        torch.nn.utils.clip_grad_norm_(
            [p for p in policy_model.parameters() if p.requires_grad],
            max_norm=float(cfg.max_grad_norm),
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        optimizer_updates += 1
        batch_total_loss += update_loss_value / update_valid_sequences
        batch_total_kl += update_kl_value / update_valid_sequences

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cuda_poisoned and optimizer_updates == 0:
        return _empty_metrics()

    return {
        "loss": batch_total_loss / max(1, optimizer_updates),
        "avg_total": batch_total_total / valid_samples,
        "avg_format": batch_total_format / valid_samples,
        "avg_visual": batch_total_visual / valid_samples,
        "avg_emotion": batch_total_emotion / valid_samples,
        "avg_situation": batch_total_situation / valid_samples,
        "avg_strategy": batch_total_strategy / valid_samples,
        "avg_outcome": batch_total_outcome / valid_samples,
        "avg_kl": batch_total_kl / max(1, optimizer_updates),
        "seg_adv_visual": batch_seg_adv_visual / max(1, valid_sequences),
        "seg_adv_emotion": batch_seg_adv_emotion / max(1, valid_sequences),
        "seg_adv_situation": batch_seg_adv_situation / max(1, valid_sequences),
        "seg_adv_strategy": batch_seg_adv_strategy / max(1, valid_sequences),
        "seg_adv_outcome": batch_seg_adv_outcome / max(1, valid_sequences),
        "seg_coverage": batch_seg_coverage / max(1, valid_sequences),
        "valid_samples": valid_samples,
        "valid_sequences": valid_sequences,
        "optimizer_updates": optimizer_updates,
        "lr": scheduler.get_last_lr()[0],
    }


def build_policy_model_and_processor(cfg: TrainConfig):
    model_class = resolve_model_class(cfg.base_model_path)
    with suppress_stdout_stderr():
        processor = AutoProcessor.from_pretrained(
            cfg.base_model_path, trust_remote_code=True
        )

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

    with suppress_stdout_stderr():
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
    # Configure LoRA parameters.
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
    with suppress_stdout_stderr():
        model = PeftModel.from_pretrained(
            base_model, str(adapter_dir), is_trainable=True
        )  # Keep LoRA adapters trainable when resuming training.
    return model


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    ckpt_mgr = CheckpointManager(cfg.checkpoint_dir, cfg.base_model_path)
    best_tracker = BestTracker(cfg.checkpoint_dir, cfg.best_k, cfg.ablation)

    print(_c("=" * 80, "1;38;5;45"))
    print(_c(" FACES GRPO Training", "1;38;5;220"))
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

        trainer_state = payload.get("trainer_state", {})
        if "cursor_state" in trainer_state:
            cursor.load_state_dict(trainer_state["cursor_state"])
        global_step = int(trainer_state.get("global_step", 0))
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])
        if "rng_state" in trainer_state:
            restore_rng_state(trainer_state["rng_state"])

        if "best_records" in trainer_state and isinstance(trainer_state["best_records"], list):
            best_tracker.records = [
                best_tracker._with_defaults(rec)
                for rec in trainer_state["best_records"]
                if isinstance(rec, dict)
            ]
            best_tracker._flush()  # Flush updated tracker records.

        print(
            _info_line(
                "[RESUME]",
                f"Recovered step={global_step}, epoch={cursor.current_epoch}, "
                f"position={cursor.position}. Optimizer/scheduler/RNG restored.",
                "1;38;5;226",
            )
        )
    
    judge: Optional[OutcomeJudge] = None
    visual_scorer: Optional[VisualEvidenceScorer] = None
    emotion_scorer: Optional[EmotionSemanticScorer] = None
    traditional_scorer: Optional[TraditionalSurfaceSkillScorer] = None

    if cfg.ablation == "w_o_multi_source_reward":
        print(
            _info_line(
                "[REWARD]",
                "Using traditional reward backend: Format + ROUGE-L + DIFF-EPITOME.",
                "1;38;5;214",
            )
        )
        traditional_scorer = TraditionalSurfaceSkillScorer(
            model_path=cfg.traditional_reward_model_path,
            device=cfg.reward_device,
            skill_decay=cfg.traditional_skill_decay,
        )
    else:
        # Evaluate overall generation quality; strategy uses the same judge.
        judge = OutcomeJudge(
            model_path=cfg.reward_model_path,
            device=cfg.reward_device,
            temperature=cfg.reward_temperature,
            top_p=cfg.reward_top_p,
            max_new_tokens=cfg.reward_max_new_tokens,
        )
        visual_scorer = VisualEvidenceScorer(
            model_path=cfg.visual_model_path,
            device=cfg.visual_device,
            threshold=cfg.visual_threshold,
            top_k=cfg.visual_top_k,
        )
        emotion_scorer = EmotionSemanticScorer(
            model_path=cfg.emotion_model_path,
            device=cfg.emotion_device,
        )
    reward_engine = RewardEngine(
        cfg=cfg,
        judge=judge,
        visual_scorer=visual_scorer,
        emotion_scorer=emotion_scorer,
        traditional_scorer=traditional_scorer,
    )

    last_val_metrics = {
        "val_avg_total": 0.0,
        "val_avg_format": 0.0,
        "val_avg_visual": 0.0,
        "val_avg_emotion": 0.0,
        "val_avg_situation": 0.0,
        "val_avg_strategy": 0.0,
        "val_avg_outcome": 0.0,
        "val_count": 0,
    }
    
    # Save the current step, adapter state, and metrics as a checkpoint.
    if cfg.eval_only:
        eval_metrics = run_validation(
            policy_model=policy_model, processor=processor,
            reward_engine=reward_engine, val_dataset=val_dataset, cfg=cfg,
        )
        print("[EVAL-ONLY-RESULT] " + json.dumps(eval_metrics, ensure_ascii=False, sort_keys=True))
        return

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
        print(
            _info_line(
                "[CKPT]",
                f"ablation={cfg.ablation} checkpoint saved: {ckpt}",
                "1;38;5;220",
            )
        )
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
                reward_engine=reward_engine,
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
                        f"Running validation at step={global_step} (ablation={cfg.ablation}) ...",
                        "1;38;5;117",
                    )
                )
                last_val_metrics = run_validation(
                    policy_model=policy_model,
                    processor=processor,
                    reward_engine=reward_engine,
                    val_dataset=val_dataset,
                    cfg=cfg,
                )
                print(
                    _info_line(
                        "[VAL]",
                        f"ablation={cfg.ablation} step={global_step} avg_total={last_val_metrics['val_avg_total']:.4f} "
                        f"format={last_val_metrics['val_avg_format']:.4f} "
                        f"visual={last_val_metrics['val_avg_visual']:.4f} "
                        f"emotion={last_val_metrics['val_avg_emotion']:.4f} "
                        f"situation={last_val_metrics['val_avg_situation']:.4f} "
                        f"strategy={last_val_metrics['val_avg_strategy']:.4f} "
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
                            "val_avg_total": last_val_metrics["val_avg_total"],
                            "val_avg_format": last_val_metrics["val_avg_format"],
                            "val_avg_visual": last_val_metrics["val_avg_visual"],
                            "val_avg_emotion": last_val_metrics["val_avg_emotion"],
                            "val_avg_situation": last_val_metrics["val_avg_situation"],
                            "val_avg_strategy": last_val_metrics["val_avg_strategy"],
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
        print(
            _info_line(
                "[VAL]",
                f"Running final validation at training end (ablation={cfg.ablation}) ...",
                "1;38;5;117",
            )
        )
        last_val_metrics = run_validation(
            policy_model=policy_model,
            processor=processor,
            reward_engine=reward_engine,
            val_dataset=val_dataset,
            cfg=cfg,
        )
        print(
            _info_line(
                "[VAL]",
                f"ablation={cfg.ablation} final avg_total={last_val_metrics['val_avg_total']:.4f} "
                f"format={last_val_metrics['val_avg_format']:.4f} "
                f"visual={last_val_metrics['val_avg_visual']:.4f} "
                f"emotion={last_val_metrics['val_avg_emotion']:.4f} "
                f"situation={last_val_metrics['val_avg_situation']:.4f} "
                f"strategy={last_val_metrics['val_avg_strategy']:.4f} "
                f"outcome={last_val_metrics['val_avg_outcome']:.4f} "
                f"count={int(last_val_metrics['val_count'])}",
                "1;38;5;117",
            )
        )

    best_tracker.update(
        {
            "ablation": cfg.ablation,
            "step": global_step,
            "epoch": cursor.current_epoch,
            "val_avg_total": last_val_metrics["val_avg_total"],
            "val_avg_format": last_val_metrics["val_avg_format"],
            "val_avg_visual": last_val_metrics["val_avg_visual"],
            "val_avg_emotion": last_val_metrics["val_avg_emotion"],
            "val_avg_situation": last_val_metrics["val_avg_situation"],
            "val_avg_strategy": last_val_metrics["val_avg_strategy"],
            "val_avg_outcome": last_val_metrics["val_avg_outcome"],
            "checkpoint": str(final_ckpt),
        }
    )

    print(_c("=" * 80, "1;38;5;45"))
    print(_c(" Training complete.", "1;38;5;220"))
    print(_info_line("[FINAL]", f"Ablation: {cfg.ablation}", "1;38;5;82"))
    print(_info_line("[FINAL]", f"Final checkpoint: {final_ckpt}", "1;38;5;82"))
    print(
        _info_line(
            "[FINAL]",
            f"Final val avg total: {last_val_metrics['val_avg_total']:.4f}",
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
