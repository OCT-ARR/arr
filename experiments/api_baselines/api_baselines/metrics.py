from __future__ import annotations

import re
from pathlib import Path

import nltk
from nltk.corpus import wordnet as nltk_wordnet
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer


def configure_local_nltk_data(repo_root: Path | None = None) -> None:
    """Prefer local NLTK assets so metrics never depend on live downloads."""
    candidates: list[Path] = []
    if repo_root is not None:
        candidates.append(repo_root / "nltk_data")
    metrics_path = Path(__file__).resolve()
    candidates.extend(
        [
            metrics_path.parents[2] / "nltk_data",
            Path("./nltk_data"),
        ]
    )
    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in nltk.data.path:
                nltk.data.path.insert(0, path_str)


def ensure_meteor_resources() -> None:
    configure_local_nltk_data()
    missing = []
    for resource in ["corpora/wordnet"]:
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.data.find(f"{resource}.zip")
            except LookupError:
                missing.append(resource)
    if missing:
        raise RuntimeError(
            "METEOR requires local NLTK resource(s): "
            + ", ".join(missing)
            + ". Put them under ./nltk_data "
            + "or another nltk.data.path entry."
        )


def safe_tokenize(text: str) -> list[str]:
    configure_local_nltk_data()
    if not text or not text.strip():
        return []
    try:
        return word_tokenize(text.lower())
    except Exception:
        return re.findall(r"\b\w+\b", text.lower())


def compute_basic_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    ensure_meteor_resources()
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoother = SmoothingFunction()
    bleu2_scores: list[float] = []
    rouge_scores: list[float] = []
    meteor_scores: list[float] = []

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = safe_tokenize(ref) or [""]
        hyp_tokens = safe_tokenize(hyp) or [""]
        try:
            bleu2_scores.append(
                sentence_bleu(
                    [ref_tokens],
                    hyp_tokens,
                    weights=(0.5, 0.5),
                    smoothing_function=smoother.method1,
                )
            )
        except Exception:
            bleu2_scores.append(0.0)

        try:
            rouge_scores.append(rouge.score(ref, hyp)["rougeL"].fmeasure)
        except Exception:
            rouge_scores.append(0.0)

        meteor_scores.append(meteor_score([ref_tokens], hyp_tokens, wordnet=nltk_wordnet))

    total_grams = 0
    unique_grams: set[tuple[str, str, str]] = set()
    for hyp in hypotheses:
        tokens = safe_tokenize(hyp)
        if len(tokens) < 3:
            continue
        grams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
        total_grams += len(grams)
        unique_grams.update(grams)

    return {
        "B-2": 100 * sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0.0,
        "R-L": 100 * sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
        "M.": 100 * sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0,
        "Distinct-3": 100 * len(unique_grams) / total_grams if total_grams else 0.0,
    }


def compute_fbert(
    references: list[str],
    hypotheses: list[str],
    repo_root: Path,
    batch_size: int = 8,
    model_type: str = "roberta-large",
    num_layers: int = 24,
) -> float:
    """Compute BERTScore-F1 as FBERT, matching the paper-facing metrics."""
    import torch

    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise RuntimeError("bert-score is required for FBERT.") from exc

    resolved_model_type = model_type
    model_path = Path(model_type).expanduser()
    local_candidates = [
        model_path,
        repo_root / "experiments" / "significance" / "models" / model_type,
        repo_root / "models" / model_type,
    ]
    for candidate in local_candidates:
        if candidate.is_dir():
            resolved_model_type = str(candidate)
            break

    kwargs = {
        "model_type": resolved_model_type,
        "batch_size": batch_size,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "verbose": True,
        "idf": False,
        "rescale_with_baseline": False,
    }
    if num_layers > 0:
        kwargs["num_layers"] = num_layers
    _, _, f1 = bert_score(hypotheses, references, **kwargs)
    return float(f1.mean().item() * 100.0) if len(f1) else 0.0


def compute_diff_epitome(
    prompts: list[str],
    references: list[str],
    hypotheses: list[str],
    repo_root: Path,
    batch_size: int = 16,
    reward_model_path: str | Path = "checkpoints/reward_model/roberta_epitome_multitask/final",
) -> float:
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    reward_model_path = Path(reward_model_path)
    if not reward_model_path.is_absolute():
        reward_model_path = repo_root / reward_model_path
    if not reward_model_path.exists():
        raise FileNotFoundError(f"Reward model not found: {reward_model_path}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(reward_model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(reward_model_path),
        num_labels=3,
    ).to(device).eval()

    values: list[float] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(hypotheses), batch_size), desc="DIFF-EPITOME"):
            p_batch = prompts[i : i + batch_size]
            h_batch = hypotheses[i : i + batch_size]
            r_batch = references[i : i + batch_size]
            gen_text = [p[-512:] + " " + h for p, h in zip(p_batch, h_batch)]
            ref_text = [p[-512:] + " " + r for p, r in zip(p_batch, r_batch)]
            gen_inputs = tokenizer(
                gen_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(device)
            ref_inputs = tokenizer(
                ref_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(device)
            gen_logits = model(**gen_inputs).logits
            ref_logits = model(**ref_inputs).logits
            for gen_vec, ref_vec in zip(gen_logits, ref_logits):
                values.append(F.mse_loss(gen_vec, ref_vec).item())

    return sum(values) / len(values) if values else 0.0
