# utils.py
# Part of EmoNAVI-Kit: smooth helpers for emotional optimizers

import torch


def smooth_tanh(x: float, scale: float = 8.0) -> float:
    """
    Apply scaled tanh and normalize to [0, 1].
    Used for scalar modulation (e.g., emo_scalar).
    """
    return 0.5 * (torch.tanh(torch.tensor(x * scale)) + 1.0)


def clamp(x: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Clamp a scalar value within [min_val, max_val].
    """
    return max(min(x, max_val), min_val)


def rescale(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """
    Linearly rescale a value x from one range to another.
    For example: loss_delta ∈ [0, 0.1] → emo_scalar ∈ [0.0, 1.0]
    """
    ratio = (x - in_min) / (in_max - in_min + 1e-8)
    return out_min + (out_max - out_min) * clamp(ratio, 0.0, 1.0)


def soft_clip(tensor: torch.Tensor, threshold: float = 1.0, alpha: float = 0.1) -> torch.Tensor:
    """
    Smoothly clips large tensor values toward threshold using interpolation.
    Less harsh than hard clamp.
    """
    return torch.where(
        tensor.abs() > threshold,
        threshold + (tensor.abs() - threshold) * alpha,
        tensor
    ) * tensor.sign()