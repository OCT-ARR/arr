# 🌐 API baseline reproduction

Generate final responses from six directly prompted multimodal models, then
run the shared automatic metrics. Commands below run from the repository root.

| Model alias | Public model identifier | Local variables |
| --- | --- | --- |
| `gpt_4o` | `gpt-4o` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |
| `claude_haiku_4_5` | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` |
| `gemini_2_5_flash_lite` | `gemini-2.5-flash-lite` | `GEMINI_API_KEY`, `GEMINI_BASE_URL` |
| `glm_4_1v_thinking_flashx` | `glm-4.1v-thinking-flashx` | `COMPAT_API_KEY`, `COMPAT_BASE_URL` |
| `doubao_1_6_vision` | `doubao-seed-1-6-vision-250815` | `COMPAT_API_KEY`, `COMPAT_BASE_URL` |
| `kimi_k2_5` | `kimi-k2.5` | `COMPAT_API_KEY`, `COMPAT_BASE_URL` |

## ⚙️ Setup

```bash
pip install -r experiments/api_baselines/requirements.txt
cp experiments/api_baselines/.env.example experiments/api_baselines/.env
```

Fill the variables for the providers you will use in the local `.env` file,
or export them in your shell. All endpoint fields and credentials in the
template are blank. Base URLs should include the provider's API version path;
the client appends `/chat/completions`, `/messages`, or
`/models/<model>:generateContent` as appropriate. `DASHSCOPE_ENDPOINT`, if used,
is a complete native generation URL.

[configs/models.json](configs/models.json) selects the client protocol and the
environment variable names. To use another OpenAI-compatible provider, adapt
[models_openai_compatible_example.json](configs/models_openai_compatible_example.json)
and pass it with `--models-config configs/models_openai_compatible_example.json`.
Set the public model identifier accepted by your provider. Separate providers
can use separate environment variable names via `api_key_env` and `base_url_env`.

## 📌 Generate responses

Unpack the datasets and prepare their visual assets first. Dataset aliases are
`mesc`, `meld_esc300`, `stickerconv`, and `avamerg`.

```bash
python experiments/api_baselines/run_generation.py --datasets mesc --models all
python experiments/api_baselines/run_generation.py --datasets meld_esc300 --models all
python experiments/api_baselines/run_generation.py --datasets stickerconv,avamerg --models all
```

The prompt requests final responses only. Generation appends one record per
sample under `experiments/api_baselines/outputs/raw/<dataset>/<model>.jsonl`
and skips successful sample IDs on rerun. Use `--limit 10` for a smaller run
or `--dry-run` to print prompts without sending requests.

## 📊 Automatic metrics

```bash
python experiments/api_baselines/run_metrics.py --datasets all --models all
```

BERTScore and DIFF-EPITOME are enabled by default. Prepare RoBERTa-large and
the [EPITOME support-skill model](../../src/MiniLM_reward_model/README.md)
locally. BERTScore uses layer 24 without IDF or baseline rescaling. Optional
flags `--without-fbert` and `--without-diff-epitome` omit those metrics.

The same metric implementation is exposed at
[experiments/automatic_metrics/run_metrics.py](../automatic_metrics/run_metrics.py)
for locally generated candidates.

## 🌐 Cross-family judging

[GLM judging](../llm_blind_judge/cross_family_judge.py) uses `ZHIPU_API_KEY`
and `ZHIPU_BASE_URL` from the same local environment file. The optional GPT-4o
judge uses `COMPAT_API_KEY` and `COMPAT_BASE_URL`.

```bash
python experiments/llm_blind_judge/cross_family_judge.py --dataset mesc
```

[Repository overview](../../README.md)
