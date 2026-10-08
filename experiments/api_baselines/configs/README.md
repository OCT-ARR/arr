# ⚙️ API configuration

| File | Purpose |
| --- | --- |
| [datasets.json](datasets.json) | Dataset aliases and relative split paths |
| [models.json](models.json) | Public model identifiers, client protocols, and environment variable names |
| [models_openai_compatible_example.json](models_openai_compatible_example.json) | OpenAI-compatible model template |
| [prompt.json](prompt.json) | Final-response-only generation prompt |

Set API keys and endpoint addresses in the local environment or
`experiments/api_baselines/.env`, using the blank
[template](../.env.example). Model entries select these variables through
`api_key_env` and `base_url_env`.

[Repository overview](../../../README.md)
