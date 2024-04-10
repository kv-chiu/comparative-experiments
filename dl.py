import logging
from typing import Callable

from .utils import setup_logging, setup_random_state, get_device

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Assuming setup_logging, setup_random_state, get_device are defined in utils.py and properly imported here
# from .utils import setup_logging, setup_random_state, get_device

# Setup logging and device as per your setup
setup_logging()
device = get_device('cuda', 0)
setup_random_state(3407, if_torch=True)

class GeneralDataset(Dataset):
    """
    General dataset class for PyTorch.
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

def create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True, transform: Callable = None) -> DataLoader:
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = GeneralDataset(X, y, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def evaluate_model(model, dataloader, loss_fn, device='cpu') -> float:
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

def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=5, eval_interval=1, device='cpu') -> None:
    """
    Train the model with controlled evaluation frequency.

    Parameters:
        model: The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        loss_fn: Loss function.
        optimizer: Optimizer.
        epochs (int): Number of epochs to train for.
        eval_interval (int): Interval (in epochs) at which to perform evaluation.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        None
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

def infer(model, X, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(X.to(device))
    return predictions
