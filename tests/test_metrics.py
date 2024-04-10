import numpy as np
from comparative_experiments.metrics import mse, rmse, mae, r2  # Adjust the import path as necessary

def test_mse():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    expected = np.mean((y_true - y_pred) ** 2)
    assert mse(y_true, y_pred) == expected, "MSE calculation is incorrect."

def test_rmse():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
    assert rmse(y_true, y_pred) == expected, "RMSE calculation is incorrect."

def test_mae():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    expected = np.mean(np.abs(y_true - y_pred))
    assert mae(y_true, y_pred) == expected, "MAE calculation is incorrect."

def test_r2():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    expected = 1 - (ss_res / ss_tot)
    assert r2(y_true, y_pred) == expected, "R2 calculation is incorrect."
