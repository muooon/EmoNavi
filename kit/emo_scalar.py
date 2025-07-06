# emo_scalar.py
# Part of EmoNAVI-Kit: emotional learning components

import torch
import math


def compute_emo_scalar(short_ema: float, long_ema: float, scaling_factor: float = 8.0) -> float:
    """
    Compute a smoothed emotional scalar (emo_scalar) from short- and long-term EMA values.

    Args:
        short_ema (float): Short-term exponential moving average of loss.
        long_ema (float): Long-term exponential moving average of loss.
        scaling_factor (float): Factor to control tanh sensitivity. Higher = sharper transitions.

    Returns:
        float: A scalar between 0 and 1 representing the optimizer's perceived urgency.
    """
    diff = short_ema - long_ema
    scaled = scaling_factor * diff
    emo_scalar = 0.5 * (torch.tanh(torch.tensor(scaled)) + 1.0)  # normalize to [0, 1]
    return emo_scalar.item()