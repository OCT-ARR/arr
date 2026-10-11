#!/usr/bin/env python3
"""FIRES scalar-reward GRPO entry point for the matched training-cost benchmark."""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FORMAT = re.compile(r"\s*<think>.*?</think>\s*<response>(.*?)</response>\s*", re.DOTALL)


def scalar_reward(format_valid, rouge_l, diff_epitome):
    """FIRES Section 3.4.3: zero on invalid format; weights 0.1, 1.0, 0.1."""
    if not format_valid:
        return 0.0
    if not math.isfinite(rouge_l) or not math.isfinite(diff_epitome):
        raise ValueError("Non-finite FIRES reward component")
    return 0.1 + rouge_l + 0.1 * float(diff_epitome <= 1.0)


def group_advantages(rewards):
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-8)


def grpo_loss(logp, old_logp, reference_logp, advantage, kl_beta):
    """FIRES Eqs. (2)-(3): completion probability ratio and sequence KL.

    Rollouts come from the old policy. Importance weighting makes the sampled
    nonnegative KL estimator target KL(current || reference), not KL(old || ref).
    """
    ratio = (logp.double() - old_logp.double()).sum().exp()
    reference_delta = (reference_logp.double() - logp.double()).sum()
    kl = ratio * (reference_delta.expm1() - reference_delta)
    return -(ratio * advantage - kl_beta * kl)


def load_epitome_classifier(checkpoint, config):
    """Load the bi-encoder weights used by the cited DIFF-EPITOME scorer.

    Architecture/checkpoint names follow EmpGPT-3/from_epitome/models.py.
    The rationale head and unused normalization parameters do not affect the
    three-class empathy prediction; all parameters used below load strictly.
    """
    import torch
    from transformers import RobertaModel

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.context_encoder = RobertaModel(config)
            self.response_encoder = RobertaModel(config)
            hidden = config.hidden_size
            self.query = torch.nn.Linear(hidden, hidden)
            self.key = torch.nn.Linear(hidden, hidden)
            self.value = torch.nn.Linear(hidden, hidden)
            self.attention_output = torch.nn.Linear(hidden, hidden)
            self.dense = torch.nn.Linear(hidden, hidden)
            self.classifier = torch.nn.Linear(hidden, 3)

        def forward(self, context, response):
            context_hidden = self.context_encoder(**context).last_hidden_state
            response_hidden = self.response_encoder(**response).last_hidden_state
            scores = self.query(response_hidden) @ self.key(context_hidden).transpose(-1, -2)
            attention = (scores / math.sqrt(config.hidden_size)).softmax(dim=-1)
            # Match the cited scorer's single-head cross-attention in eval mode.
            fused = response_hidden + self.attention_output(attention @ self.value(context_hidden))
            return self.classifier(self.dense(fused[:, 0]).relu())

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected an EPITOME classifier state dictionary")
    prefixes = {
        "seeker_encoder.roberta.": "context_encoder.",
        "responder_encoder.roberta.": "response_encoder.",
        "attn.q_linear.": "query.",
        "attn.k_linear.": "key.",
        "attn.v_linear.": "value.",
        "attn.out.": "attention_output.",
        "empathy_classifier.dense.": "dense.",
        "empathy_classifier.out_proj.": "classifier.",
    }
    unused = {"norm.alpha", "norm.bias", "rationale_classifier.weight", "rationale_classifier.bias"}
    converted = {}
    for name, value in state.items():
        if name in unused or name.endswith((".embeddings.position_ids", ".embeddings.token_type_ids")):
            continue
        for source, target in prefixes.items():
            if name.startswith(source):
                converted[target + name[len(source):]] = value
                break
        else:
            raise ValueError("Unexpected EPITOME checkpoint layout; use the cited classifier weights")
    model = Classifier()
    model.load_state_dict(converted, strict=True)
    return model


class FiresRewards:
    def __init__(self, model_path, device, roberta_path="roberta-base"):
        from rouge_score import rouge_scorer
        from transformers import RobertaConfig, RobertaTokenizer
        self.device = device
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        paths = {mechanism: Path(model_path) / f"finetuned_{mechanism}.pth"
                 for mechanism in ("ER", "IP", "EX")}
        if not all(path.is_file() for path in paths.values()):
            raise FileNotFoundError("FIRES requires finetuned_ER.pth, finetuned_IP.pth, and finetuned_EX.pth")
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_path, do_lower_case=True)
        config = RobertaConfig.from_pretrained(roberta_path)
        config._attn_implementation = "eager"
        self.models = {}
        for mechanism, path in paths.items():
            model = load_epitome_classifier(path, config).to(device)
            model.eval()
            model.requires_grad_(False)
            self.models[mechanism] = model

    def score(self, sample, completions):
        import torch
        matches = [FORMAT.fullmatch(text) for text in completions]
        valid = [i for i, match in enumerate(matches) if match is not None]
        rewards = torch.zeros(len(completions), dtype=torch.float32)
        if not valid:
            return rewards
        responses = [matches[i].group(1).strip() for i in valid]
        texts = responses + [sample.ground_truth_response]
        # The cited scorer encodes context and response separately, capped at 64
        # tokens each, and predicts a discrete level for every mechanism.
        context = self.tokenizer([sample.user_text] * len(texts),
                                 return_tensors="pt", truncation=True,
                                 max_length=64, padding=True).to(self.device)
        response = self.tokenizer(texts, return_tensors="pt", truncation=True,
                                  max_length=64, padding=True).to(self.device)
        levels = []
        with torch.no_grad():
            for model in self.models.values():
                logits = model(context, response).float()
                if logits.shape != (len(texts), 3) or not torch.isfinite(logits).all():
                    raise ValueError("Each EPITOME classifier must predict three finite class logits")
                levels.append(logits.argmax(dim=-1))
        predictions = torch.stack(levels, dim=-1).float()
        distances = (predictions[:-1] - predictions[-1]).square().mean(dim=-1).cpu()
        for j, i in enumerate(valid):
            rouge_l = self.rouge.score(sample.ground_truth_response, responses[j])["rougeL"].fmeasure
            rewards[i] = scalar_reward(True, float(rouge_l), float(distances[j]))
        return rewards


def synchronize(torch):
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def train_step(model, processor, dataset, indices, rewards, optimizer, scheduler, cfg, updates):
    import torch
    from experiments.optimization_baselines import common as C
    from experiments.optimization_baselines.objectives import trim_completion
    from src.grpo.faces_grpo import train_faces_grpo as T
    cache = []
    for index in indices:
        sample = dataset[index]
        images = T.load_resized_images(sample.image_paths, cfg.max_frames, cfg.max_image_size)
        inputs, _ = T.prepare_actor_inputs(processor, T.build_messages(sample, images),
                                            images, cfg.policy_device)
        prompt_length = inputs["input_ids"].shape[1]
        model.eval()
        with torch.no_grad():
            ids = model.generate(**inputs, do_sample=True, temperature=cfg.temperature,
                                 top_p=cfg.top_p, top_k=50, remove_invalid_values=True,
                                 logits_processor=T._SANITIZE_LOGITS,
                                 num_return_sequences=cfg.group_size,
                                 max_new_tokens=cfg.max_new_tokens,
                                 pad_token_id=processor.tokenizer.pad_token_id,
                                 eos_token_id=processor.tokenizer.eos_token_id)
        completions = [trim_completion(row[prompt_length:].tolist(),
                                      processor.tokenizer.eos_token_id,
                                      processor.tokenizer.pad_token_id) for row in ids]
        texts = [processor.tokenizer.decode(row, skip_special_tokens=True) for row in completions]
        scores = rewards.score(sample, texts)
        if len(completions) != cfg.group_size or not torch.isfinite(scores).all():
            raise ValueError("Invalid FIRES rollout group")
        advantages = group_advantages(scores)
        for completion, advantage in zip(completions, advantages):
            item = C.sequence(inputs, completion, cfg, prompt_length + cfg.max_new_tokens)
            with torch.no_grad():
                old_logp, _ = C.forward(model, item, cfg.policy_device)
                with model.disable_adapter():
                    reference_logp, _ = C.forward(model, item, cfg.policy_device)
            cache.append((item, old_logp.cpu(), reference_logp.cpu(), float(advantage)))
        del inputs, ids, images

    parameters = [p for p in model.parameters() if p.requires_grad]
    total_loss = 0.0
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        # Dropout is disabled below; train mode keeps gradient checkpointing active.
        model.train()
        for item, old_logp, reference_logp, advantage in cache:
            logp, _ = C.forward(model, item, cfg.policy_device)
            loss = grpo_loss(logp, old_logp.to(cfg.policy_device),
                             reference_logp.to(cfg.policy_device), advantage, cfg.kl_beta)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite FIRES GRPO loss")
            (loss / len(cache)).backward()
            total_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
    return total_loss / (len(cache) * updates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_path", default="checkpoints/sft_merged")
    parser.add_argument("--train_file", default="data/MESC-FT/train.jsonl")
    parser.add_argument("--reward_model_path",
                        default="checkpoints/reward_model/epitome")
    parser.add_argument("--epitome_roberta_path", default="roberta-base",
                        help="RoBERTa-base tokenizer/config directory or public model identifier")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints/training_cost/fires"))
    parser.add_argument("--policy_device", default="cuda:0")
    parser.add_argument("--reward_device", default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--updates_per_rollout", type=int, default=4,
                        help="Optimizer updates per rollout batch; matched benchmark setting")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.batch_size, args.max_new_tokens, args.max_steps, args.updates_per_rollout) <= 0 or args.group_size < 2:
        parser.error("Positive budgets and group_size >= 2 are required")
    if "TRAINING_BENCHMARK_STEPS" in os.environ and int(os.environ["TRAINING_BENCHMARK_STEPS"]) != args.max_steps:
        parser.error("max_steps must match the requested benchmark step count")
    os.chdir(ROOT)
    import torch
    from transformers import get_cosine_schedule_with_warmup
    from src.grpo.faces_grpo import train_faces_grpo as T
    cfg = T.TrainConfig(base_model_path=args.base_model_path, train_file=args.train_file,
                        policy_device=args.policy_device, reward_device=args.reward_device,
                        batch_size=args.batch_size, group_size=args.group_size,
                        max_new_tokens=args.max_new_tokens, seed=args.seed)
    T.set_seed(cfg.seed)
    dataset = T.JsonlConversationDataset(cfg.train_file)
    cursor = T.BatchCursor(len(dataset), cfg.batch_size, cfg.epochs, cfg.seed)
    if cursor.total_steps < args.max_steps:
        raise ValueError("The dataset and epoch budget cannot supply the requested training steps")
    base, processor = T.build_policy_model_and_processor(cfg)
    model = T.create_new_lora_model(base, cfg)
    from experiments.optimization_baselines.common import disable_dropout
    disable_dropout(model)
    rewards = FiresRewards(args.reward_model_path, cfg.reward_device, args.epitome_roberta_path)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                                  betas=(0.9, 0.95))
    total_updates = args.max_steps * args.updates_per_rollout
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                    int(total_updates * cfg.warmup_ratio), total_updates)
    timing_path = os.environ.get("TRAINING_BENCHMARK_TIMINGS_FILE")
    for step in range(1, args.max_steps + 1):
        _, indices = cursor.next_batch()
        synchronize(torch)
        started = time.perf_counter()
        loss = train_step(model, processor, dataset, indices, rewards, optimizer,
                          scheduler, cfg, args.updates_per_rollout)
        synchronize(torch)
        seconds = time.perf_counter() - started
        if timing_path:
            with open(timing_path, "a", encoding="utf-8") as report:
                report.write(json.dumps({"step": step, "wall_seconds": seconds}) + "\n")
        print(f"FIRES step={step}/{args.max_steps} loss={loss:.6f}", flush=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.checkpoint_dir / "adapter")
    processor.save_pretrained(args.checkpoint_dir)


if __name__ == "__main__":
    main()
