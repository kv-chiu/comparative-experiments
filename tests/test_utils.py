import logging
import os
import random
from io import StringIO

import numpy as np
import pytest
import torch

from comparative_experiments.utils import setup_logging, setup_random_state, get_device, create_sequences


def test_setup_logging():
    log_stream = StringIO()
    logger_name = 'test_logger'
    log_format = '%(levelname)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    setup_logging(level=logging.DEBUG, log_format=log_format, date_format=date_format, logger_name=logger_name)
    logger = logging.getLogger(logger_name)
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter(fmt=log_format))
    logger.addHandler(handler)

    logger.debug("Test debug message")
    logger.info("Test info message")

    log_contents = log_stream.getvalue()
    assert "DEBUG: Test debug message" in log_contents, "Debug log not found or incorrect"
    assert "INFO: Test info message" in log_contents, "Info log not found or incorrect"

    # Cleanup
    logger.removeHandler(handler)
    handler.close()


def test_setup_random_state():
    seed = 3407
    setup_random_state(seed, if_torch=False)
    assert np.random.randint(0, 1000) == np.random.RandomState(seed).randint(0,
                                                                             1000), "Numpy random state not set correctly"
    assert random.randint(0, 1000) == random.Random(seed).randint(0, 1000), "Python random state not set correctly"
    assert os.environ['PYTHONHASHSEED'] == f'{seed}', "PYTHONHASHSEED not set correctly"


def test_setup_random_state_torch():
    seed = 3407
    setup_random_state(seed, if_torch=True)
    assert torch.randint(0, 1000, (1,)).item() == torch.randint(0, 1000, (1,), generator=torch.Generator().manual_seed(
        seed)).item(), "Torch random state not set correctly"


def test_get_device():
    device = get_device('cpu')
    assert device.type == 'cpu', "CPU device not set correctly"

    # If CUDA is available, this test checks if the CUDA device is set correctly
    if torch.cuda.is_available():
        device = get_device('cuda')
        assert device.type == 'cuda', "CUDA device not set correctly"
    else:
        pytest.skip("CUDA is not available, skipping CUDA device test")


def test_create_sequences():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    x_columns = [0, 1]
    x_seq_length = 2
    y_columns = [2]
    y_seq_length = 1
    X, y = create_sequences(data, x_columns, x_seq_length, y_columns, y_seq_length)
    assert X.shape == (3, 2, 2), "X shape is incorrect"
    assert y.shape == (3, 1, 1), "y shape is incorrect"
    assert np.array_equal(X[0], np.array([[1, 2], [4, 5]])), "X data is incorrect"
    assert np.array_equal(y[0], np.array([[9]])), "y data is incorrect"
    assert np.array_equal(X[1], np.array([[4, 5], [7, 8]])), "X data is incorrect"
    assert np.array_equal(y[1], np.array([[12]])), "y data is incorrect"
    assert np.array_equal(X[2], np.array([[7, 8], [10, 11]])), "X data is incorrect"
    assert np.array_equal(y[2], np.array([[15]])), "y data is incorrect"
