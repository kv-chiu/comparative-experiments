import logging
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class GeneralDataset(Dataset):
    """
    General dataset class for PyTorch.

    Parameters
    ----------
    X : torch.Tensor
        Input data.
    y : torch.Tensor
        Labels or ground truth data.
    transform : Callable
        Transformation function.

    Attributes
    ----------
    X : torch.Tensor
        Input data.
    y : torch.Tensor
        Labels or ground truth data.
    transform : Callable
        Transformation function.

    Methods
    ----------
    __len__()
        Returns the length of the dataset.
    __getitem__(idx)
        Returns a sample from the dataset.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor, transform: Callable = None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'X': self.X[idx], 'y': self.y[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample


def create_dataloader(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
        transform: Callable = None,
        device: torch.device | str = 'cpu'
) -> DataLoader:
    """Create a PyTorch DataLoader from input data.

    Parameters
    ----------
    X : np.ndarray
        Input data.
    y : np.ndarray
        Labels or ground truth data.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle the data.
    transform : Callable
        Transformation function.
    device : torch.device or str
        Device to use ('cpu' or 'cuda').

    Returns
    -------
    dataloader : DataLoader
        DataLoader for the input data.
    """

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    dataset = GeneralDataset(X_tensor, y_tensor, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def evaluate_model(
        model: torch.nn.Module,
        dataloader: DataLoader,
        loss_fn: Callable,
        device: str | torch.device = 'cpu'
) -> float:
    """Evaluate the model on the given data.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    dataloader : DataLoader
        DataLoader for the data.
    loss_fn : Callable
        Loss function.
    device : str or torch.device
        Device to use ('cpu' or 'cuda').

    Returns
    -------
    avg_loss : float
        Average loss over the data.
    """

    model.to(device)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch['X'].to(device), batch['y'].to(device)
            predictions = model(X)
            total_loss += loss_fn(predictions, y).item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_model(
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        epochs: int = 5,
        eval_interval: int = 1,
        device: str | torch.device = 'cpu'
) -> None:
    """Train the model with controlled evaluation frequency.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be trained.
    train_loader : DataLoader
        DataLoader for training data.
    val_loader : DataLoader
        DataLoader for validation data.
    loss_fn : Callable
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer to use.
    epochs : int
        Number of epochs to train.
    eval_interval : int
        Interval at which to evaluate the model.
    device : str or torch.device
        Device to use ('cpu' or 'cuda').
    """

    model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            X, y = batch['X'].to(device), batch['y'].to(device)
            optimizer.zero_grad()
            predictions = model(X)
            loss = loss_fn(predictions, y)
            loss.backward()
            optimizer.step()

        # Logging training loss at the last batch for simplicity
        logging.info(f'Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item()}')

        # Perform evaluation at the specified interval
        if (epoch + 1) % eval_interval == 0:
            val_loss = evaluate_model(model, val_loader, loss_fn, device)
            logging.info(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}')


def infer(model: torch.nn.Module, test_loader: DataLoader, device: str | torch.device = 'cpu') -> np.ndarray:
    """Perform inference using the trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    test_loader : DataLoader
        DataLoader for test data.
    device : str or torch.device
        Device to use ('cpu' or 'cuda').

    Returns
    -------
    infer_result : np.ndarray
        Predictions from the model.
    """

    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            batch_predictions = model(X)
            predictions.extend(batch_predictions.cpu().numpy())
    return np.array(predictions)


def load_model_from_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str | torch.device = 'cpu') -> torch.nn.Module:
    """Load model from a saved checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        Model to load the checkpoint into.
    checkpoint_path : str
        Path to the checkpoint file.
    device : str or torch.device
        Device to use ('cpu' or 'cuda').

    Returns
    -------
    model : torch.nn.Module
        Model loaded from the checkpoint.
    """

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def save_pytorch_model(model: torch.nn.Module, path: str) -> None:
    """Save a PyTorch model to a file.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save.
    path : str
        Path to save the model to.
    """

    torch.save(model.state_dict(), path)
