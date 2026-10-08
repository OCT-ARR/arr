from __future__ import annotations

from .data import Sample


def build_generation_prompt(sample: Sample, prompt_config: dict) -> str:
    instruction = prompt_config["system_instruction"].strip()
    user_template = prompt_config["user_template"].strip()
    return user_template.format(
        system_instruction=instruction,
        context=sample.context,
    )

