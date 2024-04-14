import logging
import os
import random

import numpy as np
import torch


def setup_random_state(seed: int, if_torch: bool = False) -> None:
    """Set the random state for reproducibility

    Parameters
    ----------
    seed : int
        The seed for the random state
    if_torch : bool
        If True, set the random state for PyTorch
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


def setup_logging(level=logging.INFO, log_format='%(asctime)s - %(levelname)s - %(message)s',
                  date_format='%Y-%m-%d %H:%M:%S', logger_name=None):
    """Sets up the logging configuration.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).
    log_format : str
        Format for the logging messages.
    date_format : str
        Format for the date/time part of the logging messages.
    logger_name : str
        Optional name of the specific logger to configure. If None, configures the root logger.
    """

    if logger_name:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            logger.removeHandler(handler)
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
        logger.addHandler(handler)
    else:
        logging.basicConfig(level=level, format=log_format, datefmt=date_format)


def get_device(device_type: str, cuda_index: int = 0) -> torch.device:
    """Sets up the device for the PyTorch model.

    Parameters
    ----------
    device_type : str
        Device to use ('cpu' or 'cuda').
    cuda_index : int
        Index of the CUDA device to use.

    Returns
    -------
    get_device : torch.device
        The device to use.
    """

    if device_type == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda', cuda_index)
    return torch.device('cpu')


def compile_results_from_logs(log_filename: str) -> Dict[str, Dict[str, float]]:
    """Compiles the results of experiments from a log file.

    Parameters
    ----------
    log_filename : str
        The path to the log file containing the experiment results.

    Returns
    -------
    results : Dict[str, Dict[str, float]]
        A dictionary where each key is an experiment name and each value is another dictionary mapping
        metric names to their computed values for that experiment.
    """

    results = {}
    with open(log_filename, 'r') as log_file:
        for line in log_file:
            if 'Experiment:' in line:
                parts = line.split('Results: ')
                experiment_name = parts[0].split('Experiment: ')[1].strip()
                result_dict = eval(parts[1].strip())
                results[experiment_name] = result_dict
    return results
