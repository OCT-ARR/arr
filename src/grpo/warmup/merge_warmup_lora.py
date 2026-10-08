"""Merge a Warm-up GRPO LoRA checkpoint into the SFT-initialized base model."""

import argparse
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# ============================ Hyperparameters ============================
# Device used for merging:
# - "cpu"      : run fully on CPU, slower but most stable
# - "auto"     : let transformers place modules automatically
# - "cuda:0"   : use one specific GPU
MERGE_DEVICE = "auto"
# =======================================================================


def resolve_model_class():
    """Prefer the Qwen-VL class, then fall back to AutoModelForCausalLM."""
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration
    except ImportError:
        try:
            from transformers import Qwen2VLForConditionalGeneration

            return Qwen2VLForConditionalGeneration
        except ImportError:
            return AutoModelForCausalLM


def parse_step_from_name(name: str) -> str:
    match = re.search(r"checkpoint-step-(\d+)$", name)
    if not match:
        raise ValueError(f"Cannot parse step from directory name: {name}")
    return match.group(1)


def build_output_dir_name(checkpoint_dir: Path) -> str:
    name = checkpoint_dir.name
    match = re.search(r"(?:checkpoint-)?step-(\d+)$", name)
    if match:
        return f"merge_{match.group(1)}"
    tag = re.sub(r"^checkpoint-", "", name)
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("._-") or "merged"
    return f"merge_{tag}"


def find_latest_checkpoint_dir(checkpoint_root: Path) -> Path:
    candidates = sorted(
        [p for p in checkpoint_root.glob("checkpoint-step-*") if p.is_dir()],
        key=lambda p: int(parse_step_from_name(p.name)),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-step-* directory found under {checkpoint_root}")
    return candidates[-1]


def resolve_adapter_path(checkpoint_dir: Path) -> Path:
    adapter_dir = checkpoint_dir / "adapter"
    if (adapter_dir / "adapter_config.json").exists() and (
        adapter_dir / "adapter_model.safetensors"
    ).exists():
        return adapter_dir
    if (checkpoint_dir / "adapter_config.json").exists() and (
        checkpoint_dir / "adapter_model.safetensors"
    ).exists():
        return checkpoint_dir
    raise FileNotFoundError(f"LoRA adapter file not found in: {checkpoint_dir}")


def read_base_model_path(checkpoint_dir: Path, default_base: Path) -> str:
    txt = checkpoint_dir / "base_model_path.txt"
    if txt.exists():
        content = txt.read_text(encoding="utf-8").strip()
        if content:
            return content
    return str(default_base)


def resolve_device_map(device: str):
    dev = device.strip()
    lower = dev.lower()
    if lower == "cpu":
        return "cpu"
    if lower == "auto":
        return "auto"
    if lower.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable, but device={dev} was requested")
        return {"": dev}
    raise ValueError(f"Unsupported device: {device}. Use cpu / auto / cuda:x")


def infer_dtype(device: str):
    lower = device.strip().lower()
    if lower == "cpu":
        return torch.float32
    if lower.startswith("cuda") or lower == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return torch.float32


def main():
    parser = argparse.ArgumentParser(description="Merge a Warm-up GRPO LoRA checkpoint")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step number, for example 245 for checkpoint-step-245",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Explicit checkpoint directory path. Takes priority over --step",
    )
    parser.add_argument(
        "--base-model",
        "--base-model-path",
        "--base_model_path",
        dest="base_model",
        type=str,
        default=None,
        help="Base model path. Defaults to checkpoint/base_model_path.txt, then checkpoints/sft_merged",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=MERGE_DEVICE,
        help='Merge device. Supports "cpu" / "auto" / "cuda:x"',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom merged model output directory. Recommended: checkpoints/warmup_merged",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]

    checkpoint_root = project_root / "checkpoints" / "warmup_grpo"
    output_root = project_root / "checkpoints" / "warmup_merged"
    default_base_model = project_root / "checkpoints" / "sft_merged"

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    elif args.step is not None:
        checkpoint_dir = checkpoint_root / f"checkpoint-step-{args.step}"
    else:
        checkpoint_dir = find_latest_checkpoint_dir(checkpoint_root)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_dir}")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_path = (
        args.base_model
        if args.base_model
        else read_base_model_path(checkpoint_dir, default_base_model)
    )
    adapter_path = resolve_adapter_path(checkpoint_dir)
    model_cls = resolve_model_class()
    device_map = resolve_device_map(args.device)
    dtype = infer_dtype(args.device)

    print("=" * 80)
    print("Starting model merge")
    print(f"Base model:     {base_model_path}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Adapter path:   {adapter_path}")
    print(f"Output dir:     {output_dir}")
    print(f"Device:         {args.device}")
    print("=" * 80)

    print("[1/4] Loading base model...")
    base_model = model_cls.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
    )

    print("[2/4] Loading and merging LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.merge_and_unload()

    print("[3/4] Saving merged full model...")
    model.save_pretrained(str(output_dir))

    print("[4/4] Saving processor/tokenizer...")
    try:
        processor = AutoProcessor.from_pretrained(
            str(checkpoint_dir), trust_remote_code=True
        )
        processor.save_pretrained(str(output_dir))
    except Exception as exc:
        print(f"Processor save failed; falling back to tokenizer: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir), trust_remote_code=True
        )
        tokenizer.save_pretrained(str(output_dir))

    print(f"Merge completed: {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
