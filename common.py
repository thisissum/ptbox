import os
import random
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
import pandas as pd
#yess
def seed_everything(seed=512):
    """
    Set seed for random, numpy, hash, torch, torch.cuda, torch.backends.cudnn
    :param seed: int
    :return: None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = seed

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def init_logger(log_path=None, log_file_level=logging.NOTSET):
    """
    Return logger instance.
    Example:
        >>> logger = init_logger(log_path)
        >>> logger.info("Your Info")
    :param log_path: str or pathlib.Path
    :param log_file_level: logger set level
    :return: logging.Logger
    """
    if isinstance(log_path, Path):
        log_path = str(Path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="[%(asctime)s] [%(pathname)s] [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if isinstance(log_path, str):
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)
    return logger


def to_device(inputs, device='cuda:0'):
    """
    Move inputs to target device
    Example:
        >>> inputs_on_device = to_device(inputs, device=target_device)
    :param inputs: List[inputs] or Tuple[inputs] or torch.Tensor or torch.nn.Module or dict[inputs]
    :param device: str or torch.device
    :return: flattened list or tuple or dict, with content on target device.
    """
    if isinstance(inputs, [torch.Tensor, torch.nn.Module]):
        return inputs.to(device)
    elif isinstance(inputs, list):
        return [to_device(input) for input in inputs]
    elif isinstance(inputs, tuple):
        return tuple(to_device(input, device) for input in inputs)
    elif isinstance(inputs, dict):
        return {key: to_device(val) for key, val in inputs.items()}
    elif isinstance(inputs, tuple) and hasattr(inputs, "_fields"):
        return inputs.__class__(*(to_device(item, device) for item in inputs))
    else:
        return inputs