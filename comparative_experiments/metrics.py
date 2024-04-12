import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Mean Squared Error (MSE) between true and predicted values.
    
    Parameters
    ----------
    y_true : np.ndarray
        The true values.
    y_pred : np.ndarray
        The predicted values.
    
    Returns
    -------
    mse : float
        The Mean Squared Error.
    """

    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Root Mean Squared Error (RMSE) between true and predicted values.
    
    Parameters
    ----------
    y_true : np.ndarray
        The true values.
    y_pred : np.ndarray
        The predicted values.
    
    Returns
    -------
    rmse : float
        The Root Mean Squared Error.
    """

    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Mean Absolute Error (MAE) between true and predicted values.
    
    Parameters
    ----------
    y_true : np.ndarray
        The true values.
    y_pred : np.ndarray
        The predicted values.
    
    Returns
    -------
    mae : float
        The Mean Absolute Error.
    """

    return np.mean(np.abs(y_true - y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the R-squared (R2) value between true and predicted values.
    
    Parameters
    ----------
    y_true : np.ndarray
        The true values.
    y_pred : np.ndarray
        The predicted values.
    
    Returns
    -------
    r2 : float
        The R-squared value, indicating the proportion of the variance in the dependent variable that is predictable from the independent variables.
    """

    total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
    unexplained_variance = np.sum((y_true - y_pred) ** 2)
    return 1 - (unexplained_variance / total_variance)
