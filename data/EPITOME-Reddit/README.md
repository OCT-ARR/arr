# 📚 EPITOME-Reddit Support-Skill Data

🧠 This folder contains the three EPITOME/Reddit support-skill CSV files used to
train the optional DIFF-EPITOME reward model.

## 📁 Files

| File | Target skill |
| --- | --- |
| `emotional-reactions-reddit.csv` | Emotional reaction |
| `interpretations-reddit.csv` | Interpretation |
| `explorations-reddit.csv` | Exploration |

These files are not used during SFT. They are used only for the optional
support-skill reward model in `src/MiniLM_reward_model/`, which can be used by the
traditional reward ablation in the GRPO stage.

This is not MiniLM training data. MiniLM is a separate public
sentence-transformer downloaded to:

```text
models/paraphrase-multilingual-MiniLM-L12-v2/
```

Each CSV has the following core columns:

```text
sp_id,rp_id,seeker_post,response_post,level,rationales
```

Train the optional reward model with:

```bash
python src/MiniLM_reward_model/train_epitome_multitask.py \
  --data_dir data/EPITOME-Reddit \
  --output_dir checkpoints/reward_model/roberta_epitome_multitask
```

[Repository overview](../../README.md)
