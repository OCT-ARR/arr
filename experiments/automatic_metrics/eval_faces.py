# Launch examples:
#   CUDA_VISIBLE_DEVICES=0 python3 experiments/automatic_metrics/eval_faces.py
#   CUDA_VISIBLE_DEVICES=0,1 python3 experiments/automatic_metrics/eval_faces.py --reward_backend critic --critic_device cuda:1

import os
import json
import re
import gc
import argparse

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from prettytable import PrettyTable
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ================= Configuration =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PATHS = {
    "GRPO_MODEL": os.path.join(PROJECT_ROOT, "checkpoints", "faces_main"),
    "REWARD_MODEL": os.path.join(
        PROJECT_ROOT, "checkpoints", "reward_model", "roberta_epitome_multitask", "final"
    ),
    "TEST_DATA": os.path.join(PROJECT_ROOT, "data", "MESC-FT", "test.jsonl"),
}

NUM_GPUS = torch.cuda.device_count()
MAX_NEW_TOKENS = 256
MAX_IMAGES = 8

# Critic evaluation configuration
DEFAULT_REWARD_BACKEND = "critic"  # critic | legacy
CRITIC_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
CRITIC_DEVICE = "cuda:1"
CRITIC_MAX_NEW_TOKENS = 256
CRITIC_MAX_IMAGES = 8
FORMAT_FAIL_SCORE = 0.0
FACES_COGNITIVE_WEIGHT = 1.0 / 3.0
DEFAULT_FBERT_MODEL_TYPE = "roberta-large"
DEFAULT_FBERT_NUM_LAYERS = 24
# ===========================================


# ================= NLTK configuration =================
import nltk

LOCAL_NLTK_PATH = os.path.join(PROJECT_ROOT, "nltk_data")
nltk.data.path.insert(0, LOCAL_NLTK_PATH)
nltk.data.path.insert(1, os.path.expanduser("~/nltk_data"))
print(f" NLTK data path: {LOCAL_NLTK_PATH}")

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer


# qwen_vl_utils (optional)
try:
    from qwen_vl_utils import process_vision_info

    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    print(" qwen_vl_utils is not installed; using the fallback image-processing path")


# Automatic model-class selection
try:
    from transformers import Qwen2_5_VLForConditionalGeneration

    ModelClass = Qwen2_5_VLForConditionalGeneration
except ImportError:
    try:
        from transformers import Qwen2VLForConditionalGeneration

        ModelClass = Qwen2VLForConditionalGeneration
    except ImportError:
        ModelClass = AutoModel


def clean_response(text):
    """Clean model output and extract the final response."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"<response>(.*?)</response>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if "System:" in text:
        text = text.split("System:")[-1]
    return text.strip()


def check_format(text):
    pattern = r"<think>.*?</think>\s*<response>.*?</response>"
    match = re.search(pattern, text or "", re.DOTALL | re.IGNORECASE)
    return 1.0 if match else 0.0


def extract_response_text(text):
    match = re.search(r"<response>(.*?)</response>", text or "", re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_think_text(text):
    match = re.search(r"<think>(.*?)</think>", text or "", re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def load_data(path):
    """Load test data."""
    dataset = []
    print(f" Loading data from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                user_text, gt_text, gt_full, image_paths = "", "", "", []
                for turn in item["conversations"]:
                    if turn["from"] == "user":
                        user_text = turn["value"]
                        if "images" in turn:
                            image_paths = turn["images"]
                    elif turn["from"] == "assistant":
                        gt_full = turn["value"]
                        gt_text = gt_full
                        match = re.search(
                            r"<response>(.*?)</response>",
                            gt_text,
                            re.DOTALL | re.IGNORECASE,
                        )
                        if match:
                            gt_text = match.group(1).strip()
                if user_text and gt_text:
                    dataset.append(
                        {
                            "idx": idx,
                            "system": item.get("system", ""),
                            "prompt": user_text,
                            "images": image_paths,
                            "ground_truth": gt_text,
                            "ground_truth_full": gt_full,
                        }
                    )
            except Exception:
                continue
    print(f" Loaded {len(dataset)} samples.")
    return dataset


def resolve_critic_model_path(path_or_repo: str) -> str:
    """Prefer a local critic checkpoint when the default repo id is cached on disk."""
    if os.path.isdir(path_or_repo):
        return path_or_repo

    default_repo = "Qwen/Qwen3-VL-8B-Instruct"
    local_candidates = [
        os.path.join(
            PROJECT_ROOT,
            "models",
            "modelscope_cache",
            "Qwen",
            "Qwen3-VL-8B-Instruct",
        ),
    ]

    if path_or_repo == default_repo:
        for candidate in local_candidates:
            if os.path.isdir(candidate):
                print(f"   Using local critic model: {candidate}")
                return candidate

    return path_or_repo


def validate_image(img_path, max_size=1024):
    """Validate and preprocess an image."""
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if w < 10 or h < 10 or w > 10000 or h > 10000:
            return None
        if w > max_size or h > max_size:
            ratio = min(max_size / w, max_size / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        return img
    except Exception:
        return None


def safe_single_inference(model, processor, item, device, max_new_tokens):
    """Run safe inference for one sample."""
    try:
        msgs = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": item["system"] or "You are a helpful assistant.",
                    }
                ],
            },
            {"role": "user", "content": []},
        ]

        images = []
        if item["images"]:
            for img_path in item["images"][:MAX_IMAGES]:
                img = validate_image(img_path)
                if img is not None:
                    msgs[1]["content"].append({"type": "image", "image": img})
                    images.append(img)

        prompt_text = item["prompt"].encode("utf-8", errors="ignore").decode("utf-8")
        msgs[1]["content"].append({"type": "text", "text": prompt_text})

        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
                if hasattr(processor, "tokenizer")
                else 0,
            )

        input_len = inputs.input_ids.shape[1]
        output = processor.decode(gen_ids[0][input_len:], skip_special_tokens=True)

        for img in images:
            try:
                img.close()
            except Exception:
                pass

        return output

    except Exception as e:
        if "CUDA" in str(e):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return None


def worker_inference(gpu_id, data_chunk, model_path, max_new_tokens, result_queue):
    """Single-GPU worker process."""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    print(f" GPU {gpu_id}: Loading GRPO model (bf16)...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = ModelClass.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": gpu_id},
        trust_remote_code=True,
    ).eval()

    print(f" GPU {gpu_id}: GRPO model loaded, processing {len(data_chunk)} samples...")

    results = []
    failed = 0

    pbar = tqdm(data_chunk, desc=f"GPU {gpu_id}", position=gpu_id, leave=True)
    for item in pbar:
        output = safe_single_inference(model, processor, item, device, max_new_tokens)

        if output is None:
            item_no_img = item.copy()
            item_no_img["images"] = []
            output = safe_single_inference(
                model, processor, item_no_img, device, max_new_tokens
            )
            if output is None:
                failed += 1
                output = ""

        results.append(
            {
                "idx": item["idx"],
                "prompt": item["prompt"],
                "images": item["images"],
                "ground_truth": item["ground_truth"],
                "ground_truth_full": item.get("ground_truth_full", item["ground_truth"]),
                "hypothesis": output,
            }
        )
        pbar.set_postfix({"failed": failed})

    del model
    torch.cuda.empty_cache()

    print(f" GPU {gpu_id}: Completed. Failed: {failed}/{len(data_chunk)}")
    result_queue.put(results)


def parallel_inference(dataset, model_path, num_gpus, max_new_tokens):
    """Run parallel multi-GPU inference."""
    print(f"\n Starting parallel inference on {num_gpus} GPUs...")

    chunks = [[] for _ in range(num_gpus)]
    for i, item in enumerate(dataset):
        chunks[i % num_gpus].append(item)

    for i, chunk in enumerate(chunks):
        print(f"   GPU {i}: {len(chunk)} samples")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    result_queue = mp.Queue()
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_inference,
            args=(gpu_id, chunks[gpu_id], model_path, max_new_tokens, result_queue),
        )
        p.start()
        processes.append(p)

    all_results = []
    for _ in range(num_gpus):
        results = result_queue.get()
        all_results.extend(results)

    for p in processes:
        p.join()

    all_results.sort(key=lambda x: x["idx"])
    prompts = [r["prompt"] for r in all_results]
    refs = [r["ground_truth"] for r in all_results]
    ref_fulls = [r["ground_truth_full"] for r in all_results]
    hyps = [r["hypothesis"] for r in all_results]
    images = [r["images"] for r in all_results]
    return prompts, refs, ref_fulls, hyps, images


def single_gpu_inference(dataset, model_path, max_new_tokens):
    """Run single-GPU inference."""
    device = "cuda:0"
    print(f"\n Single GPU inference on {device} (bf16)...")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = ModelClass.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    prompts, refs, ref_fulls, hyps, images = [], [], [], [], []
    for item in tqdm(dataset, desc="Inference"):
        output = safe_single_inference(model, processor, item, device, max_new_tokens)
        if output is None:
            item_no_img = item.copy()
            item_no_img["images"] = []
            output = safe_single_inference(
                model, processor, item_no_img, device, max_new_tokens
            )
            if output is None:
                output = ""

        prompts.append(item["prompt"])
        refs.append(item["ground_truth"])
        ref_fulls.append(item.get("ground_truth_full", item["ground_truth"]))
        hyps.append(output)
        images.append(item["images"])

    del model
    torch.cuda.empty_cache()
    return prompts, refs, ref_fulls, hyps, images


def safe_tokenize(text):
    """Safely tokenize text; fall back to regex if NLTK fails."""
    if not text or not text.strip():
        return []
    try:
        return word_tokenize(text.lower())
    except Exception:
        return re.findall(r"\b\w+\b", text.lower())


def resolve_fbert_model_type(model_type: str) -> str:
    if os.path.isdir(model_type):
        return model_type
    local_candidates = [
        os.path.join(PROJECT_ROOT, "experiments", "significance", "models", model_type),
        os.path.join(PROJECT_ROOT, "models", model_type),
    ]
    for candidate in local_candidates:
        if os.path.isdir(candidate):
            print(f"   Using local FBERT model: {candidate}")
            return candidate
    return model_type


def compute_fbert_score(refs, hyps, model_type, num_layers, batch_size):
    """Compute BERTScore-F1 as FBERT, matching the paper's automatic metrics."""
    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise RuntimeError(
            "bert-score is required for FBERT. Install requirements.txt or run with --without_fbert."
        ) from exc

    print("   Computing FBERT / BERTScore-F1...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    resolved_model_type = resolve_fbert_model_type(model_type)
    kwargs = {
        "model_type": resolved_model_type,
        "batch_size": batch_size,
        "device": device,
        "verbose": True,
        "idf": False,
        "rescale_with_baseline": False,
    }
    if num_layers is not None and int(num_layers) > 0:
        kwargs["num_layers"] = int(num_layers)
    _, _, f1 = bert_score(hyps, refs, **kwargs)
    return float(f1.mean().item() * 100.0) if len(f1) else 0.0


def extract_json_block(text):
    if not text:
        return None
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def clip01(x):
    return max(0.0, min(1.0, float(x)))


def compose_faces_critic_total(visual, emotion, situation, strategy, outcome, format_score):
    if format_score < 0.5:
        return FORMAT_FAIL_SCORE
    cognitive = FACES_COGNITIVE_WEIGHT * (visual + emotion + situation)
    return cognitive * outcome * strategy


def build_critic_eval_messages(prompt, model_think, model_response, ref, ref_full, pil_images):
    reference_think = extract_think_text(ref_full)
    critic_instruction = (
        "You are a strict multimodal emotional-support dialogue evaluator. "
        "Score the given dialogue context, model chain of thought, model response, "
        "reference chain of thought, and reference response. "
        "Output JSON only, with all required fields: "
        "visual_score, emotion_score, situation_score, strategy_score, outcome_score. "
        "All scores must be in [0, 1]. Do not output explanatory text."
    )

    critic_user_text = (
        f"[CONTEXT]\n{prompt[-1800:]}\n\n"
        f"[MODEL_THINK]\n{model_think}\n\n"
        f"[MODEL_RESPONSE]\n{model_response}\n\n"
        f"[REFERENCE_THINK]\n{reference_think or 'Not provided.'}\n\n"
        f"[REFERENCE_RESPONSE]\n{ref}\n\n"
        "Scoring criteria:\n"
        "1) visual_score: whether the visual descriptions in the reasoning are supported by image evidence.\n"
        "2) emotion_score: whether emotion reasoning is consistent with the context, visual evidence, and reference reasoning when provided.\n"
        "3) situation_score: whether situation reasoning is grounded in the context and semantically aligned with reference reasoning when provided.\n"
        "4) strategy_score: whether the support strategy fits the user's needs and aligns with reference reasoning when provided.\n"
        "5) outcome_score: helpfulness, empathy, safety, and relevance of the final response compared with the reference response.\n"
        "Output JSON only."
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": critic_instruction}]},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in pil_images[:CRITIC_MAX_IMAGES]],
                {"type": "text", "text": critic_user_text},
            ],
        },
    ]
    return messages


def compute_critic_metrics(prompts, refs, ref_fulls, raw_hyps, image_paths_batch, critic_model_path, critic_device):
    print("   Computing Critic-based metrics...")

    if not torch.cuda.is_available() and "cuda" in critic_device:
        print("   CUDA is unavailable; skipping critic metrics")
        return {}

    try:
        critic_model_path = resolve_critic_model_path(critic_model_path)
        local_only = os.path.isdir(critic_model_path)

        critic_processor = AutoProcessor.from_pretrained(
            critic_model_path,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        dtype = torch.bfloat16 if "cuda" in critic_device else torch.float32
        critic_model = AutoModelForCausalLM.from_pretrained(
            critic_model_path,
            torch_dtype=dtype,
            device_map={"": critic_device},
            trust_remote_code=True,
            local_files_only=local_only,
        ).eval()
    except Exception as e:
        print(f"   Critic model loading failed; skipping critic metrics: {e}")
        return {}

    format_list = []
    visual_list, emotion_list, situation_list, strategy_list = [], [], [], []
    outcome_list = []
    total_list = []

    for prompt, ref, ref_full, raw_hyp, image_paths in tqdm(
        zip(prompts, refs, ref_fulls, raw_hyps, image_paths_batch),
        total=len(raw_hyps),
        desc="  Critic Metrics",
    ):
        r_F = check_format(raw_hyp)
        if r_F == 0.0:
            format_list.append(0.0)
            visual_list.append(0.0)
            emotion_list.append(0.0)
            situation_list.append(0.0)
            strategy_list.append(0.0)
            outcome_list.append(0.0)
            total_list.append(FORMAT_FAIL_SCORE)
            continue

        model_response = extract_response_text(raw_hyp)
        model_think = extract_think_text(raw_hyp)
        if not model_response:
            model_response = clean_response(raw_hyp)

        pil_images = []
        for path in image_paths[:CRITIC_MAX_IMAGES]:
            img = validate_image(path)
            if img is not None:
                pil_images.append(img)

        messages = build_critic_eval_messages(
            prompt,
            model_think,
            model_response,
            ref,
            ref_full,
            pil_images,
        )

        try:
            text_prompt = critic_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if HAS_QWEN_VL_UTILS:
                image_inputs, video_inputs = process_vision_info(messages)
                critic_inputs = critic_processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                critic_inputs = critic_processor(
                    text=[text_prompt],
                    images=pil_images if pil_images else None,
                    padding=True,
                    return_tensors="pt",
                )

            critic_inputs = critic_inputs.to(critic_device)

            with torch.no_grad():
                out_ids = critic_model.generate(
                    **critic_inputs,
                    max_new_tokens=CRITIC_MAX_NEW_TOKENS,
                    do_sample=False,
                )

            gen_ids = out_ids[:, critic_inputs.input_ids.shape[1] :]
            out_text = critic_processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
            data = extract_json_block(out_text) or {}

            visual = clip01(data.get("visual_score", 0.0))
            emotion = clip01(data.get("emotion_score", 0.0))
            situation = clip01(data.get("situation_score", 0.0))
            strategy = clip01(data.get("strategy_score", 0.0))
            outcome = clip01(data.get("outcome_score", 0.0))
            total = compose_faces_critic_total(
                visual=visual,
                emotion=emotion,
                situation=situation,
                strategy=strategy,
                outcome=outcome,
                format_score=r_F,
            )

            format_list.append(r_F)
            visual_list.append(visual)
            emotion_list.append(emotion)
            situation_list.append(situation)
            strategy_list.append(strategy)
            outcome_list.append(outcome)
            total_list.append(total)
        except Exception:
            format_list.append(r_F)
            visual_list.append(0.0)
            emotion_list.append(0.0)
            situation_list.append(0.0)
            strategy_list.append(0.0)
            outcome_list.append(0.0)
            total_list.append(FORMAT_FAIL_SCORE)
            continue
        finally:
            for img in pil_images:
                try:
                    img.close()
                except Exception:
                    pass

    del critic_model
    gc.collect()
    torch.cuda.empty_cache()

    if not total_list:
        return {}

    return {
        "Critic-Format": sum(format_list) / len(format_list),
        "Critic-Visual": sum(visual_list) / len(visual_list),
        "Critic-Emotion": sum(emotion_list) / len(emotion_list),
        "Critic-Situation": sum(situation_list) / len(situation_list),
        "Critic-Strategy": sum(strategy_list) / len(strategy_list),
        "Critic-Outcome": sum(outcome_list) / len(outcome_list),
        "Critic-Total": sum(total_list) / len(total_list),
    }


def calc_metrics(prompts, refs, hyps, args):
    """Compute the original automatic metrics."""
    print("\n  Calculating Metrics...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    reward_tokenizer = AutoTokenizer.from_pretrained(PATHS["REWARD_MODEL"])
    reward_model = (
        AutoModelForSequenceClassification.from_pretrained(
            PATHS["REWARD_MODEL"], num_labels=3
        )
        .to(device)
        .eval()
    )

    cc = SmoothingFunction()
    bleu2_list, rouge_list, meteor_list = [], [], []

    print("   Computing BLEU/ROUGE/METEOR...")
    for r, h in tqdm(zip(refs, hyps), total=len(refs), desc="  NLP Metrics"):
        try:
            rt = safe_tokenize(r)
            ht = safe_tokenize(h)
            if not ht:
                ht = [""]
            if not rt:
                rt = [""]

            bleu2_list.append(
                sentence_bleu(
                    [rt], ht, weights=(0.5, 0.5), smoothing_function=cc.method1
                )
            )
            rouge_list.append(rouge.score(r, h)["rougeL"].fmeasure)
            try:
                meteor_list.append(meteor_score([rt], ht))
            except Exception:
                pass
        except Exception:
            pass

    total_grams, unique_grams = 0, set()
    for h in hyps:
        tokens = safe_tokenize(h)
        if len(tokens) >= 3:
            grams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
            total_grams += len(grams)
            unique_grams.update(grams)

    print("   Computing DIFF-EPITOME...")
    mse_list = []
    metric_batch_size = 16
    with torch.no_grad():
        for i in tqdm(range(0, len(hyps), metric_batch_size), desc="  Diff-Epitome"):
            p_b = prompts[i : i + metric_batch_size]
            h_b = hyps[i : i + metric_batch_size]
            r_b = refs[i : i + metric_batch_size]
            txt_gen = [p[-512:] + " " + h for p, h in zip(p_b, h_b)]
            txt_ref = [p[-512:] + " " + r for p, r in zip(p_b, r_b)]
            inp_g = reward_tokenizer(
                txt_gen,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(device)
            inp_r = reward_tokenizer(
                txt_ref,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(device)
            out_g = reward_model(**inp_g).logits
            out_r = reward_model(**inp_r).logits
            for vg, vr in zip(out_g, out_r):
                mse_list.append(F.mse_loss(vg, vr).item())

    del reward_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fbert_score = None
    if not args.without_fbert:
        fbert_score = compute_fbert_score(
            refs,
            hyps,
            args.fbert_model_type,
            args.fbert_num_layers,
            args.fbert_batch_size,
        )

    results = {
        "BLEU-2": sum(bleu2_list) / len(bleu2_list) * 100 if bleu2_list else 0,
        "ROUGE-L": sum(rouge_list) / len(rouge_list) * 100 if rouge_list else 0,
        "METEOR": sum(meteor_list) / len(meteor_list) * 100 if meteor_list else 0,
        "FBERT": fbert_score,
        "Distinct-3": len(unique_grams) / total_grams * 100 if total_grams else 0,
        "DIFF-EPITOME": sum(mse_list) / len(mse_list) if mse_list else 0,
    }
    if fbert_score is None:
        results.pop("FBERT")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward_backend",
        type=str,
        default=DEFAULT_REWARD_BACKEND,
        choices=["critic", "legacy"],
        help="Whether to compute critic-based reward metrics.",
    )
    parser.add_argument(
        "--critic_model_path",
        type=str,
        default=CRITIC_MODEL_PATH,
        help="Critic model path used in critic metrics.",
    )
    parser.add_argument(
        "--critic_device",
        type=str,
        default=CRITIC_DEVICE,
        help="Critic device, e.g. cuda:1",
    )
    parser.add_argument(
        "--without_fbert",
        action="store_true",
        help="Skip FBERT / BERTScore-F1 for quick local smoke tests.",
    )
    parser.add_argument(
        "--fbert_model_type",
        type=str,
        default=DEFAULT_FBERT_MODEL_TYPE,
        help="BERTScore model id or local path.",
    )
    parser.add_argument(
        "--fbert_num_layers",
        type=int,
        default=DEFAULT_FBERT_NUM_LAYERS,
        help="BERTScore layer count; 24 uses the final roberta-large layer.",
    )
    parser.add_argument(
        "--fbert_batch_size",
        type=int,
        default=8,
        help="Batch size for FBERT / BERTScore-F1.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" GRPO Model Evaluation (bf16 Full Precision)")
    print(f" Model: {PATHS['GRPO_MODEL']}")
    print(f" Reward Eval Backend: {args.reward_backend}")
    print("=" * 60)

    print(f"  Detected {NUM_GPUS} GPUs")

    dataset = load_data(PATHS["TEST_DATA"])

    if NUM_GPUS >= 2:
        prompts, refs, ref_fulls, hyps, images_batch = parallel_inference(
            dataset, PATHS["GRPO_MODEL"], NUM_GPUS, MAX_NEW_TOKENS
        )
    else:
        prompts, refs, ref_fulls, hyps, images_batch = single_gpu_inference(
            dataset, PATHS["GRPO_MODEL"], MAX_NEW_TOKENS
        )

    valid_indices = [i for i, h in enumerate(hyps) if h.strip()]
    if len(valid_indices) < len(hyps):
        print(f" Using {len(valid_indices)}/{len(hyps)} valid samples for metrics")
        prompts = [prompts[i] for i in valid_indices]
        refs = [refs[i] for i in valid_indices]
        ref_fulls = [ref_fulls[i] for i in valid_indices]
        hyps = [hyps[i] for i in valid_indices]
        images_batch = [images_batch[i] for i in valid_indices]

    cleaned_hyps = [clean_response(h) for h in hyps]
    res = calc_metrics(prompts, refs, cleaned_hyps, args)

    if args.reward_backend == "critic":
        critic_res = compute_critic_metrics(
            prompts,
            refs,
            ref_fulls,
            hyps,
            images_batch,
            args.critic_model_path,
            args.critic_device,
        )
        res.update(critic_res)

    table = PrettyTable()
    table.field_names = ["Metric", "GRPO Result"]
    for k, v in res.items():
        if "EPITOME" in k:
            val = f"{v:.4f} (MSE)"
        else:
            val = f"{v:.4f}"
        table.add_row([k, val])

    print("\n" + "=" * 60)
    print(" GRPO Evaluation Results")
    print("=" * 60)
    print(table)
    print("=" * 60)


if __name__ == "__main__":
    main()
