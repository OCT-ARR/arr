"""Preference and policy objectives, independent of model or dataset loading."""
import torch
import torch.nn.functional as F


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta):
    """Inputs are sums of completion-token log probabilities, including EOS."""
    margin = beta * ((chosen - rejected) - (reference_chosen - reference_rejected))
    return -F.logsigmoid(margin).mean()


def simpo_loss(chosen, rejected, beta, gamma):
    """Inputs are means of completion-token log probabilities, including EOS."""
    return -F.logsigmoid(beta * (chosen - rejected) - gamma).mean()


def leave_one_out(rewards):
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError('RLOO requires at least two completions of the same prompt')
    return rewards - (rewards.sum() - rewards) / (rewards.numel() - 1)


def generalized_advantages(rewards, values, gamma=1.0, lam=0.95):
    """One unpadded completion; EOS or the configured length cap terminates it."""
    if rewards.ndim != 1 or rewards.shape != values.shape or not rewards.numel():
        raise ValueError('Expected equally sized, nonempty reward/value vectors')
    advantages = torch.zeros_like(rewards)
    running = rewards.new_zeros(())
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    return advantages, advantages + values


def ppo_loss(logp, old_logp, advantages, values, old_values, returns,
             clip=0.2, value_clip=0.2, value_coefficient=0.5):
    ratio = (logp - old_logp).exp()
    policy = -torch.minimum(ratio * advantages,
                            ratio.clamp(1 - clip, 1 + clip) * advantages).mean()
    clipped_values = old_values + (values - old_values).clamp(-value_clip, value_clip)
    value = 0.5 * torch.maximum((values - returns).square(),
                                 (clipped_values - returns).square()).mean()
    return policy + value_coefficient * value


def trim_completion(ids, eos_ids, pad_id):
    """Keep the first EOS even when padding uses the same token ID."""
    eos = set(eos_ids if isinstance(eos_ids, (list, tuple, set)) else [eos_ids])
    result = []
    for token in ids:
        token = int(token)
        if token in eos:
            result.append(token)
            break
        if pad_id is not None and token == pad_id:
            break
        result.append(token)
    if not result:
        raise ValueError('Empty completion')
    return result
