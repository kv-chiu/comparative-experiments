import logging
import os
import random
from typing import Dict, Tuple

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


def create_sequences(
        data: np.ndarray,
        x_columns: List[int],
        x_seq_length: int,
        y_columns: List[int],
        y_seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for training a sequence model.

    Parameters
    ----------
    data : np.ndarray
        The data to create sequences from.
    x_columns : List[int]
        The columns to use for the input sequences.
    x_seq_length : int
        The length of the input sequences.
    y_columns : List[int]
        The columns to use for the output sequences.
    y_seq_length : int
        The length of the output sequences.

    Returns
    -------
    X : np.ndarray
        The input sequences.
    y : np.ndarray
        The output sequences.

    Examples
    --------
    >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    >>> x_columns = [0, 1]
    >>> x_seq_length = 2
    >>> y_columns = [2]
    >>> y_seq_length = 1
    >>> X, y = create_sequences(data, x_columns, x_seq_length, y_columns, y_seq_length)
    >>> X.shape
    (3, 2, 2)
    >>> y.shape
    (3, 1, 1)
    >>> X[0]
    array([[1, 2],
           [4, 5]])
    >>> y[0]
    array([[9]])
    >>> X[1]
    array([[4, 5],
           [7, 8]])
    >>> y[1]
    array([[12]])
    >>> X[2]
    array([[7, 8],
           [10, 11]])
    >>> y[2]
    array([[15]])
    """

    X, y = [], []
    for i in range(len(data) - (x_seq_length + y_seq_length) + 1):
        X.append(data[i: i + x_seq_length, x_columns])
        y.append(data[i + x_seq_length: i + x_seq_length + y_seq_length, y_columns])
    return np.array(X), np.array(y)
