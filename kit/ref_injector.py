# ref_injector.py
# Part of EmoNAVI-Kit: emotional learning components

import torch
from torch.nn import Parameter


def soft_inject(param: Parameter, target: torch.Tensor, emo_scalar: float):
    """
    Gently injects a portion of the target value into the parameter, scaled by emo_scalar.

    Args:
        param (Parameter): The model parameter to be updated (in-place).
        target (Tensor): The "shadow" or reference value to blend into the parameter.
        emo_scalar (float): A scalar in [0,1] determining the injection intensity.

    Returns:
        None
    """
    with torch.no_grad():
        param.data.mul_(1.0 - emo_scalar).add_(emo_scalar * target)