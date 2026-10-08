# 🌐 API helpers

Shared modules used by [generation](../run_generation.py) and
[automatic metrics](../run_metrics.py).

| Module | Purpose |
| --- | --- |
| [clients.py](clients.py) | OpenAI-compatible, Anthropic, Gemini, and DashScope clients with environment-based configuration |
| [data.py](data.py) | Dataset loading and final-response extraction |
| [env.py](env.py) | Local environment-file loading |
| [io_utils.py](io_utils.py) | JSON and JSONL input/output |
| [metrics.py](metrics.py) | Text metrics, BERTScore, and DIFF-EPITOME |
| [prompting.py](prompting.py) | Generation prompt construction |

[Repository overview](../../../README.md)
