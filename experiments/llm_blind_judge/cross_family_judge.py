#!/usr/bin/env python3
"""Cross-family API judging with three prompts, bidirectional swaps, cap-15, and paired statistics.
Defaults to GLM-4.6V-Flash on the full selected split; accepts other opponent prediction files."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MT_ROOT = PROJECT_ROOT / "experiments" / "api_baselines"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MT_ROOT))
from api_baselines.env import load_dotenv
from api_baselines.clients import build_client, APIClientError
from experiments.llm_blind_judge import stickerconv_eval as base
from experiments.llm_blind_judge import pairwise_controlled_eval as controlled

LEFT_NAME = "faces_main"
RIGHT_NAME = "grpo_baseline"

JUDGE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gpt4o": {
        "name": "gpt_4o_judge",
        "provider": "openai_compatible_chat",
        "model": "gpt-4o",
        "api_key_env": "COMPAT_API_KEY",
        "base_url_env": "COMPAT_BASE_URL",
        "timeout_sec": 180,
    },
    "zhipu": {
        "name": "zhipu_glm_judge",
        "provider": "openai_compatible_chat",
        "model": "glm-4.6v-flash",
        "api_key_env": "ZHIPU_API_KEY",
        "base_url_env": "ZHIPU_BASE_URL",
        "timeout_sec": 180,
        # Disable thinking so the response budget is reserved for the judgment.
        "extra_payload": {"thinking": {"type": "disabled"}},
        "max_tokens": 768,
    },
}
JUDGE_GENERATION = {"temperature": 0.0, "max_tokens": 256, "max_images": 8}

DATASETS = {
    key: {"display_name": ds.display_name, "test_file": ds.test_file,
          "left_generation_file": ds.left_generation_file,
          "right_generation_file": ds.right_generation_file}
    for key, ds in controlled.DATASETS.items()
}

HERE = Path(__file__).resolve().parent
OUT_ROOT = PROJECT_ROOT / "outputs" / "cross_family_judge"


def count_words(text: Any) -> int:
    raw = str(text or "").strip()
    return 0 if not raw else len(raw.split())


def truncate_words(text: Any, limit: int) -> str:
    raw = str(text or "").strip()
    if limit <= 0 or not raw:
        return ""
    return " ".join(raw.split()[:limit]).strip()


def stable_seed(base_seed: int, *parts: str) -> int:
    payload = "::".join(parts).encode("utf-8")
    return int(base_seed + (zlib.crc32(payload) % 100000))


def bootstrap_ci(values: Sequence[float], n_boot: int, seed: int) -> Tuple[Optional[float], Optional[float]]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def exact_two_sided_sign_test(left_wins: int, right_wins: int) -> float:
    decisive = left_wins + right_wins
    if decisive == 0:
        return 1.0
    tail = min(left_wins, right_wins)
    lower_prob = sum(math.comb(decisive, i) for i in range(tail + 1)) / (2 ** decisive)
    return float(min(1.0, 2.0 * lower_prob))


def _aggregate_pairwise_variants(
    variant_records: Sequence[Dict[str, Any]], left_name: str, right_name: str,
) -> Tuple[List[str], str]:
    """Merge the two candidate orders before voting over prompt variants (Table 15)."""
    votes: List[str] = []
    for record in variant_records:
        winner1 = record["round1"]["mapped_winner"]
        winner2 = record["round2"]["mapped_winner"]
        if winner1 == winner2:
            winner = winner1
        elif winner1 == "TIE":
            winner = winner2
        elif winner2 == "TIE":
            winner = winner1
        else:
            winner = "TIE"
        record["final_variant_winner"] = winner
        votes.append(winner)

    left_votes, right_votes = votes.count(left_name), votes.count(right_name)
    winner_name = left_name if left_votes > right_votes else (
        right_name if right_votes > left_votes else "TIE"
    )
    return votes, winner_name


class APIPairwiseJudge:
    """Same PAIRWISE_PROMPTS / rules / parsing as the 32B judge, API backend."""

    def __init__(self, judge_config: Dict[str, Any], max_retries: int = 3, retry_wait_sec: float = 3.0):
        self.client = build_client(judge_config)
        self.max_retries = max_retries
        self.retry_wait_sec = retry_wait_sec
        self.generation = {**JUDGE_GENERATION, "max_tokens": judge_config.get("max_tokens", JUDGE_GENERATION["max_tokens"])}

    def _judge_once(self, system_prompt: str, user_prompt: str, image_paths: Sequence[str]) -> Tuple[Optional[Dict[str, Any]], str]:
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                result = self.client.generate(
                    prompt=full_prompt,
                    image_paths=list(image_paths),
                    generation=self.generation,
                )
                return base.extract_first_json_object(result.text), result.text
            except APIClientError as exc:
                last_err = exc
                time.sleep(self.retry_wait_sec * (attempt + 1))
        return None, f"[ERROR after {self.max_retries} retries] {last_err}"

    def score_pairwise(
        self,
        sample: "base.EvalSample",
        left_name: str,
        left_pred: "base.ModelPrediction",
        right_name: str,
        right_pred: "base.ModelPrediction",
    ) -> Dict[str, Any]:
        variant_records: List[Dict[str, Any]] = []
        include_think = bool(left_pred.think.strip() or right_pred.think.strip())

        def judge_round(prompt, a_name, a_pred, b_name, b_pred):
            user_prompt = base.judge_prompt_template(prompt["user"], include_think).format(
                conversation=sample.conversation,
                think_a=base.format_think_for_judge(a_pred),
                response_a=base.format_response_for_judge(a_pred),
                think_b=base.format_think_for_judge(b_pred),
                response_b=base.format_response_for_judge(b_pred),
            )
            user_prompt = base.append_pairwise_judge_rules(user_prompt, include_think=include_think)
            parsed, raw_text = self._judge_once(prompt["system"], user_prompt, sample.image_paths)
            raw_winner = None if parsed is None else parsed.get("winner")
            winner_label = "TIE" if parsed is None else base.parse_winner(raw_winner)
            valid = parsed is not None and str(raw_winner or "").strip().upper() in {"A", "B", "TIE"}
            mapped_winner = {"A": a_name, "B": b_name}.get(winner_label, "TIE")
            return {
                "candidate_a_name": a_name, "candidate_b_name": b_name,
                "parsed": parsed, "raw_output": raw_text,
                "winner_label": winner_label, "mapped_winner": mapped_winner,
                "valid": valid,
            }

        for prompt in base.PAIRWISE_PROMPTS:
            round1 = judge_round(prompt, left_name, left_pred, right_name, right_pred)
            round2 = judge_round(prompt, right_name, right_pred, left_name, left_pred)
            variant_records.append({"round1": round1, "round2": round2})

        votes, winner_name = _aggregate_pairwise_variants(variant_records, left_name, right_name)

        total_rounds = len(variant_records) * 2
        invalid_rounds = sum(
            1 for rec in variant_records for rd in (rec["round1"], rec["round2"]) if not rd["valid"]
        )
        return {
            "sample_id": sample.sample_id,
            "left_name": left_name,
            "right_name": right_name,
            "votes": votes,
            "winner_name": winner_name,
            "valid_round_count": total_rounds - invalid_rounds,
            "invalid_round_count": invalid_rounds,
            "variants": variant_records,
        }


def load_predictions(generation_file, model_name, samples):
    return controlled.load_predictions(base, generation_file, model_name, samples)


def apply_length_control(pred: "base.ModelPrediction", mode: str, max_words: int) -> "base.ModelPrediction":
    if mode == "full_context":
        return pred
    if mode == "response_only":
        return base.ModelPrediction(
            sample_id=pred.sample_id, model_name=pred.model_name, raw_output=pred.raw_output,
            think="", response=pred.response, format_valid=pred.format_valid,
            think_missing=True, response_missing=pred.response_missing,
        )
    if mode == "length_controlled":
        target = min(max_words, count_words(pred.response))
        return base.ModelPrediction(
            sample_id=pred.sample_id, model_name=pred.model_name, raw_output=pred.raw_output,
            think="", response=truncate_words(pred.response, target), format_valid=pred.format_valid,
            think_missing=True, response_missing=pred.response_missing,
        )
    raise ValueError(f"Unknown mode: {mode}")


def load_cache(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.is_file():
        return {}
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Reaggregate cached raw judgments so resumed runs use the same protocol.
            row["votes"], row["winner_name"] = _aggregate_pairwise_variants(
                row["variants"], row["left_name"], row["right_name"]
            )
            out[int(row["sample_id"])] = row
    return out


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    global LEFT_NAME, RIGHT_NAME
    parser = argparse.ArgumentParser(description="Cross-family blind pairwise evaluation.")
    parser.add_argument("--judge", choices=sorted(JUDGE_CONFIGS), default="zhipu",
                         help="External judge backend; zhipu selects GLM-4.6V-Flash.")
    parser.add_argument("--judge_model", default=None,
                         help="override the model name for the chosen judge, e.g. --judge zhipu --judge_model glm-4.5v")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="mesc")
    parser.add_argument("--mode", choices=["full_context", "response_only", "length_controlled"], default="length_controlled")
    parser.add_argument("--max_words", type=int, default=15)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="print prompt/plan for 1 sample, no API calls")
    parser.add_argument("--left_name", default=LEFT_NAME)
    parser.add_argument("--right_name", default=RIGHT_NAME)
    parser.add_argument("--left_generation_file", type=Path)
    parser.add_argument("--right_generation_file", type=Path)
    parser.add_argument("--output_root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    LEFT_NAME, RIGHT_NAME = args.left_name, args.right_name
    if args.sample_limit < 0 or args.max_words <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("Invalid sample limit, word cap, or bootstrap count")

    load_dotenv(MT_ROOT / ".env")

    judge_config = dict(JUDGE_CONFIGS[args.judge])
    if args.judge_model:
        judge_config["model"] = args.judge_model

    ds = dict(DATASETS[args.dataset])
    if args.left_generation_file:
        ds["left_generation_file"] = args.left_generation_file
    if args.right_generation_file:
        ds["right_generation_file"] = args.right_generation_file
    samples = base.load_test_samples(ds["test_file"], args.sample_limit)
    if not samples:
        raise RuntimeError(f"No samples loaded from {ds['test_file']}")
    left_preds_raw = load_predictions(ds["left_generation_file"], LEFT_NAME, samples)
    right_preds_raw = load_predictions(ds["right_generation_file"], RIGHT_NAME, samples)

    left_preds = [apply_length_control(p, args.mode, args.max_words) for p in left_preds_raw]
    right_preds = [apply_length_control(p, args.mode, args.max_words) for p in right_preds_raw]

    mode_tag = f"{args.mode}_cap{args.max_words}" if args.mode == "length_controlled" else args.mode
    out_dir = args.output_root / f"{args.judge}_{args.dataset}_{mode_tag}" / f"{LEFT_NAME}_vs_{RIGHT_NAME}"
    output_path = out_dir / "pairwise.jsonl"
    summary_path = out_dir / "summary.json"

    if args.dry_run:
        sample, lp, rp = samples[0], left_preds[0], right_preds[0]
        prompt = base.PAIRWISE_PROMPTS[0]
        include_think = bool(lp.think.strip() or rp.think.strip())
        user_prompt = base.append_pairwise_judge_rules(
            base.judge_prompt_template(prompt["user"], include_think).format(
                conversation=sample.conversation,
                think_a=base.format_think_for_judge(lp), response_a=base.format_response_for_judge(lp),
                think_b=base.format_think_for_judge(rp), response_b=base.format_response_for_judge(rp),
            ),
            include_think=include_think,
        )
        print(f"[DRY] dataset={args.dataset} mode={mode_tag} n_samples={len(samples)} "
              f"n_images_first_sample={len(sample.image_paths)}")
        print("----- system -----")
        print(prompt["system"])
        print("----- user (variant 0, round1) -----")
        print(user_prompt[:2000])
        return

    judge = APIPairwiseJudge(judge_config)
    cache = load_cache(output_path)
    todo_idx = [i for i, s in enumerate(samples) if s.sample_id not in cache]
    print(f"[JUDGE] judge={args.judge}({judge_config['model']}) {args.dataset}/{mode_tag}: "
          f"{len(cache)} cached, {len(todo_idx)} to run (6 API calls/sample: 3 prompts x 2 directions)")

    for i in todo_idx:
        sample, lp, rp = samples[i], left_preds[i], right_preds[i]
        t0 = time.time()
        row = judge.score_pairwise(sample, LEFT_NAME, lp, RIGHT_NAME, rp)
        append_jsonl(output_path, row)
        cache[sample.sample_id] = row
        print(f"[JUDGE] sample_id={sample.sample_id} winner={row['winner_name']} "
              f"invalid={row['invalid_round_count']} ({time.time() - t0:.1f}s)", flush=True)

    records = [cache[s.sample_id] for s in samples]
    # Persist reaggregated cache entries so the JSONL and summary agree.
    base.write_jsonl(output_path, records)
    summ = base.summarize_pairwise(records, LEFT_NAME, RIGHT_NAME)
    left_indicator = [1.0 if r["winner_name"] == LEFT_NAME else 0.0 for r in records]
    decisive_indicator = [
        1.0 if r["winner_name"] == LEFT_NAME else 0.0 for r in records if r["winner_name"] in {LEFT_NAME, RIGHT_NAME}
    ]
    left_ci = bootstrap_ci(left_indicator, args.bootstrap_samples, stable_seed(args.seed, args.dataset, mode_tag, "left"))
    decisive_ci = bootstrap_ci(decisive_indicator, args.bootstrap_samples, stable_seed(args.seed, args.dataset, mode_tag, "decisive"))
    decisive_n = summ["left_wins"] + summ["right_wins"]

    summary = {
        **summ,
        "dataset": ds["display_name"],
        "mode": mode_tag,
        "judge_name": args.judge,
        "judge_model": judge_config["model"],
        "judge_backend": judge_config["provider"],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_ci95_left_win_rate": list(left_ci) if left_ci[0] is not None else None,
        "bootstrap_ci95_decisive_left_share": list(decisive_ci) if decisive_ci[0] is not None else None,
        "decisive_n": decisive_n,
        "decisive_left_share": (summ["left_wins"] / decisive_n) if decisive_n else None,
        "sign_test_p_value": exact_two_sided_sign_test(summ["left_wins"], summ["right_wins"]),
        "output_path": str(output_path),
    }
    write_json(summary_path, summary)
    print(f"[DONE] {args.dataset}/{mode_tag}: L={summ['left_wins']} R={summ['right_wins']} T={summ['ties']} "
          f"decisive_left_share={summary['decisive_left_share']} p={summary['sign_test_p_value']:.4g}")
    print(f"[DONE] summary -> {summary_path}")


if __name__ == "__main__":
    main()
