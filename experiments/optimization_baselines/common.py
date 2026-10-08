"""Shared multimodal inputs and FACES rewards for the reconstructed baselines."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config(path, method=None):
    data = json.loads(Path(path).read_text())
    if method is not None and method not in data['methods']:
        raise ValueError(f'Unknown method: {method}')
    return data


def train_config(data):
    from src.grpo.faces_grpo import train_faces_grpo as T
    allowed = {f.name for f in fields(T.TrainConfig)}
    unknown = set(data['shared']) - allowed
    if unknown:
        raise ValueError(f'Unknown shared settings: {sorted(unknown)}')
    return T.TrainConfig(**data['shared'])


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_hash(sample):
    from dataclasses import asdict
    return hashlib.sha256(json.dumps(asdict(sample), sort_keys=True).encode()).hexdigest()


def reward_engine(cfg):
    from src.grpo.faces_grpo import train_faces_grpo as T
    return T.RewardEngine(
        cfg,
        T.OutcomeJudge(cfg.reward_model_path, cfg.reward_device, cfg.reward_temperature,
                       cfg.reward_top_p, cfg.reward_max_new_tokens),
        T.VisualEvidenceScorer(cfg.visual_model_path, cfg.visual_device,
                               cfg.visual_threshold, cfg.visual_top_k),
        T.EmotionSemanticScorer(cfg.emotion_model_path, cfg.emotion_device))


def prompt_inputs(processor, sample, cfg, max_sequence_length):
    from src.grpo.faces_grpo import train_faces_grpo as T
    images = T.load_resized_images(sample.image_paths, cfg.max_frames, cfg.max_image_size)
    if sample.image_paths and not images:
        raise FileNotFoundError('No referenced images could be loaded')
    inputs, _ = T.prepare_actor_inputs(processor, T.build_messages(sample, images),
                                       images, cfg.policy_device)
    if inputs['input_ids'].shape[1] >= max_sequence_length:
        raise ValueError('Prompt including visual tokens exceeds max_sequence_length; '
                         'adjust image budget or explicitly override the context limit')
    return inputs, images


def sequence(inputs, completion, cfg, max_sequence_length):
    import torch
    from src.grpo.faces_grpo import train_faces_grpo as T
    prompt_length = inputs['input_ids'].shape[1]
    if not completion or prompt_length + len(completion) > max_sequence_length:
        raise ValueError('Completion is empty or exceeds the configured context limit')
    ids = torch.cat([inputs['input_ids'], torch.tensor([completion],
                    device=cfg.policy_device, dtype=torch.long)], dim=1)
    # Do not infer padding from token IDs: pad_token_id may equal eos_token_id.
    attention = torch.cat([inputs['attention_mask'], torch.ones(
        (1, len(completion)), dtype=inputs['attention_mask'].dtype,
        device=cfg.policy_device)], dim=1)
    kwargs = dict(input_ids=ids, attention_mask=attention, use_cache=False)
    kwargs.update(T._extract_forward_vision_kwargs(inputs))
    return {'kwargs': T._tensor_tree_to_cpu(kwargs), 'prompt_length': prompt_length}


def forward(model, item, device, value_head=None):
    import torch
    from src.grpo.faces_grpo import train_faces_grpo as T
    kwargs = T._tensor_tree_to_device(item['kwargs'], device)
    output = model(**kwargs, output_hidden_states=value_head is not None)
    start = item['prompt_length'] - 1
    logits = output.logits[:, start:-1, :].float()
    labels = kwargs['input_ids'][:, start + 1:]
    logp = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(0).squeeze(-1)
    values = None
    if value_head is not None:
        hidden = output.hidden_states[-1][:, start:-1, :].float()
        values = value_head(hidden).squeeze(0).squeeze(-1)
    return logp, values


def generate(model, processor, inputs, cfg, max_sequence_length, count=None):
    import torch
    from experiments.optimization_baselines.objectives import trim_completion
    available = max_sequence_length - inputs['input_ids'].shape[1]
    model.eval()
    with torch.no_grad():
        ids = model.generate(**inputs, do_sample=True, temperature=cfg.temperature,
            top_p=cfg.top_p, top_k=0, num_return_sequences=count or cfg.group_size,
            max_new_tokens=min(cfg.max_new_tokens, available),
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id)
    offset = inputs['input_ids'].shape[1]
    return [trim_completion(row[offset:].tolist(), processor.tokenizer.eos_token_id,
                            processor.tokenizer.pad_token_id) for row in ids]


def disable_dropout(model):
    import torch
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
