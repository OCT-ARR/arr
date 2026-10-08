# -*- coding: utf-8 -*-
"""Generate five scored candidates per prompt for reward aggregation diagnostics.

Uses the canonical FACES reward engine; output is generated locally at runtime."""
import os
import sys
import json
import gc
import time
import argparse

import torch
from PIL import Image

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.grpo.faces_grpo import train_faces_grpo as T


def log(msg):
    print(f"[stage1 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _patched_config(model_path):
    """训练保存的 full_step-215 checkpoint 把 text_config.rope_scaling 写成了 null,
    导致 Qwen2.5-VL 注意力层里 self.rope_scaling["mrope_section"] 触发
    'NoneType' object is not subscriptable。这里从基座模型 (merge_245) 恢复该字段。
    如果 checkpoint 本身正常, 此函数什么都不改。"""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    base_path = str(PROJECT_ROOT / "checkpoints" / "warmup_merged")
    rope = None
    try:
        base_cfg = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
        base_tc = getattr(base_cfg, "text_config", None)
        rope = getattr(base_tc, "rope_scaling", None) if base_tc is not None else None
        if rope is None:
            rope = getattr(base_cfg, "rope_scaling", None)
    except Exception as e:
        log(f"  (warn) 读基座 config 失败, 用 Qwen2.5-VL 默认 mrope: {e}")
    if rope is None:
        rope = {"mrope_section": [16, 24, 24], "rope_theta": 1000000.0,
                "rope_type": "default", "type": "default"}
    tc = getattr(cfg, "text_config", None)
    if tc is not None and getattr(tc, "rope_scaling", None) is None:
        tc.rope_scaling = rope
        log(f"  patched text_config.rope_scaling <- {rope}")
    if getattr(cfg, "rope_scaling", None) is None:
        cfg.rope_scaling = rope
    return cfg


def load_policy(model_path, device):
    """加载 policy 模型 (被评估的 GRPO 模型)。"""
    from transformers import AutoProcessor
    log(f"loading policy model: {model_path} -> {device}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model_class = T.resolve_model_class(model_path)
    model_kwargs = {
        "config": _patched_config(model_path),
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": {"": device},
    }
    if getattr(T, "HAS_FLASH_ATTN", False):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = T.load_pretrained_with_fallback(
        model_class=model_class,
        model_path=model_path,
        model_kwargs=model_kwargs,
        model_name="policy",
    )
    model.eval()
    model.requires_grad_(False)
    if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor


def generate_k(model, processor, sample, images, k, device, cfg, greedy_first=True):
    """对单个 sample 生成 K 个 completion。
    第 1 个用 greedy (确定性, 对应评估时的主回复), 其余用采样得到 group 多样性。
    复用训练脚本的 prepare_actor_inputs (process_vision_info + 左 padding) 与 generate 参数,
    保证与训练/验证时的生成路径一致 —— 否则 Qwen-VL 图像占位符对不上会生成空串。
    """
    messages = T.build_messages(sample, images)
    inputs, _ = T.prepare_actor_inputs(processor, messages, images, device)
    input_len = int(inputs["input_ids"].shape[1])

    sanitize = getattr(T, "_SANITIZE_LOGITS", None)

    completions = []
    for j in range(k):
        do_sample = not (greedy_first and j == 0)
        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        if sanitize is not None:
            gen_kwargs["logits_processor"] = sanitize
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)
        else:
            gen_kwargs.update(do_sample=False)
        try:
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            txt = processor.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        except Exception as e:
            log(f"    generate failed (sample idx, j={j}): {e}")
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            txt = ""
        completions.append(txt)
    return completions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_model", default=str(PROJECT_ROOT / "checkpoints" / "faces_main"))
    ap.add_argument("--test_data", default=str(PROJECT_ROOT / "data" / "MESC-FT" / "test.jsonl"))
    ap.add_argument("--k", type=int, default=5, help="每个 prompt 采样的 completion 数 (group size)")
    ap.add_argument("--limit", type=int, default=369, help="只跑前 N 条 (0=全部)")
    ap.add_argument("--policy_device", default="cuda:0")
    ap.add_argument("--reward_device", default="cuda:1")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "reward_diagnostics" / "scored_candidates.jsonl"))
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--deadline_sec", type=int, default=0, help=">0 时到点即安全停止, 已完成的样本照常落盘")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.chdir(PROJECT_ROOT)
    T.set_seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    out_path = args.out
    done_ids = set()
    # 断点续跑: 已写过的 idx 跳过
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["idx"])
                except Exception:
                    pass
        log(f"resume: {len(done_ids)} samples already scored, will skip them")

    # ---- 配置: 复用训练默认 TrainConfig, 只覆盖设备 / 数据 ----
    cfg = T.TrainConfig()
    cfg.policy_device = args.policy_device
    cfg.reward_device = args.reward_device
    cfg.max_new_tokens = args.max_new_tokens
    # visual/emotion 打分器放到 reward 卡上 (训练默认在 cpu, 这里用 GPU 更快)
    visual_device = args.reward_device
    emotion_device = args.reward_device

    # ---- 数据 ----
    log(f"loading test data: {args.test_data}")
    dataset = T.JsonlConversationDataset(args.test_data)
    samples = dataset.samples
    if args.limit > len(samples):
        raise ValueError(f"Requested {args.limit} prompts but dataset has {len(samples)}")
    if args.limit > 0:
        samples = samples[: args.limit]
    log(f"total samples: {len(samples)} (k={args.k} -> {len(samples) * args.k} completions)")

    # ---- 打分器 (与训练一致) ----
    log("initializing reward scorers (CLIP + MiniLM + Qwen3-VL judge + traditional)...")
    judge = T.OutcomeJudge(
        model_path=cfg.reward_model_path,
        device=cfg.reward_device,
        temperature=cfg.reward_temperature,
        top_p=cfg.reward_top_p,
        max_new_tokens=cfg.reward_max_new_tokens,
    )
    visual_scorer = T.VisualEvidenceScorer(
        model_path=cfg.visual_model_path,
        device=visual_device,
        threshold=cfg.visual_threshold,
        top_k=cfg.visual_top_k,
    )
    emotion_scorer = T.EmotionSemanticScorer(
        model_path=cfg.emotion_model_path,
        device=emotion_device,
    )
    # 独立信号 (不依赖 Qwen judge): FIRES 风格 ROUGE-L + EPITOME skill
    traditional_scorer = T.TraditionalSurfaceSkillScorer(
        model_path=cfg.traditional_reward_model_path,
        device=visual_device,
        skill_decay=cfg.traditional_skill_decay,
    )
    engine = T.RewardEngine(
        cfg=cfg,
        judge=judge,
        visual_scorer=visual_scorer,
        emotion_scorer=emotion_scorer,
        traditional_scorer=traditional_scorer,
    )

    # ---- policy ----
    policy, processor = load_policy(args.policy_model, cfg.policy_device)

    fout = open(out_path, "a", encoding="utf-8")
    n_written = 0
    for si, sample in enumerate(samples):
        idx = si
        if idx in done_ids:
            continue
        if args.deadline_sec > 0 and (time.time() - t_start) > args.deadline_sec:
            log(f"deadline hit ({args.deadline_sec}s), stopping safely at sample {si}")
            break

        images = T.load_resized_images(sample.image_paths, cfg.max_frames, cfg.max_image_size)
        completions = generate_k(policy, processor, sample, images, args.k, cfg.policy_device, cfg)

        # 复用训练打分链路: 一次性给 K 个 completion 打六维分
        try:
            parts = engine.score_completions(sample, completions, images)
        except Exception as e:
            log(f"  score_completions failed at sample {si}: {e}")
            continue

        # 独立信号 (逐 completion): ROUGE-L + EPITOME skill, 不经过 Qwen judge
        ref_resp = (sample.ground_truth_response or "").strip()
        cand_resps = [T.extract_response_text(c).strip() for c in completions]
        try:
            indep_rouge = [traditional_scorer.rouge_l(c, ref_resp) for c in cand_resps]
        except Exception:
            indep_rouge = [0.0] * len(completions)
        try:
            indep_skill = traditional_scorer.skill_scores(
                contexts=[sample.user_text] * len(completions),
                candidate_responses=cand_resps,
                reference_responses=[ref_resp] * len(completions),
            )
        except Exception:
            indep_skill = [0.0] * len(completions)

        indep_minilm = emotion_scorer.score_batch(cand_resps, [ref_resp] * len(cand_resps))

        rec = {
            "idx": idx,
            "prompt": sample.user_text[-2000:],
            "n_images": len(images),
            "reference": ref_resp,
            "completions": [
                {
                    "text": completions[j],
                    "response": cand_resps[j],
                    "format": parts[j].format,
                    "visual": parts[j].visual,
                    "emotion": parts[j].emotion,
                    "situation": parts[j].situation,
                    "strategy": parts[j].strategy,
                    "outcome": parts[j].outcome,
                    "total_gated": parts[j].total,  # 训练真实门控总分 (参照)
                    "indep_rouge_l": float(indep_rouge[j]),
                    "indep_epitome_skill": float(indep_skill[j]),
                    "indep_minilm_sim": float(indep_minilm[j]),
                }
                for j in range(len(completions))
            ],
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        n_written += 1
        for img in images:
            try:
                img.close()
            except Exception:
                pass
        if (si + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (n_written) / max(1e-6, elapsed)
            log(f"  progress {si + 1}/{len(samples)}  written={n_written}  {rate:.2f} samp/s")
            gc.collect()
            torch.cuda.empty_cache()

    fout.close()
    log(f"DONE. wrote {n_written} new samples to {out_path} (total elapsed {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
