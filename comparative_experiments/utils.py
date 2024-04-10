import logging
import os
import random
import numpy as np
import torch

def setup_random_state(seed: int, if_torch: bool = False) -> None:
    """
    Set the random state for reproducibility

    Parameters:
        seed (int): The seed for the random state
        if_torch (bool): If True, set the random state for PyTorch

    Returns:
        None
    """

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if not if_torch:
        return
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def setup_logging(level=logging.INFO, log_format='%(asctime)s - %(levelname)s - %(message)s', date_format='%Y-%m-%d %H:%M:%S'):
    """
    Sets up the logging configuration.

    Parameters:
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        log_format (str): Format for the logging messages.
        date_format (str): Format for the date/time part of the logging messages.

    Returns:
        None
    """
    logging.basicConfig(level=level, format=log_format, datefmt=date_format)

def get_device(device: str, cuda_index: int = 0) -> torch.device:
    """
    Sets up the device for the PyTorch model.

    Parameters:
        device (str): Device to use ('cpu' or 'cuda').
        cuda_index (int): Index of the CUDA device to use.

    Returns:
        torch.device: The device to use.
    """
    if device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda', cuda_index)
    return torch.device('cpu')