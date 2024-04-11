from comparative_experiments.dl import GeneralDataset, create_dataloader, evaluate_model, train_model, evaluate_model, train_model, infer
from comparative_experiments.utils import setup_logging, setup_random_state, get_device

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def test_general_dataset():
    X = torch.tensor([1, 2, 3], dtype=torch.float32)
    y = torch.tensor([2, 4, 6], dtype=torch.float32)
    dataset = GeneralDataset(X, y)
    assert len(dataset) == 3, "Dataset length was not set correctly."
    sample = dataset[1]
    assert sample['X'] == 2, "Sample X value was not set correctly."
    assert sample['y'] == 4, "Sample y value was not set correctly."

def test_create_dataloader():
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    batch_size = 2
    dataloader = create_dataloader(X, y, batch_size)
    assert isinstance(dataloader, DataLoader), "Dataloader was not created correctly."
    for batch in dataloader:
        assert batch['X'].shape[0] == batch_size, "Batch X shape was not set correctly."
        assert batch['y'].shape[0] == batch_size, "Batch y shape was not set correctly."

def test_evaluate_model():
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.linear = nn.Linear(2, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = DummyModel()
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    batch_size = 2
    dataloader = create_dataloader(X, y, batch_size)
    loss_fn = nn.MSELoss()
    loss = evaluate_model(model, dataloader, loss_fn)
    assert isinstance(loss, float), "Evaluation loss was not returned correctly."

def test_train_model():
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.linear = nn.Linear(2, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = DummyModel()
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    batch_size = 2
    train_loader = create_dataloader(X, y, batch_size)
    val_loader = create_dataloader(X, y, batch_size)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_model(model, train_loader, val_loader, loss_fn, optimizer)

def test_infer():
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.linear = nn.Linear(2, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = DummyModel()
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    batch_size = 2
    dataloader = create_dataloader(X, y, batch_size)
    predictions = infer(model, dataloader)
    assert predictions.shape[0] == X.shape[0], "Inference predictions shape was not set correctly."
    assert predictions.shape[1] == 1, "Inference predictions shape was not set correctly."
    assert isinstance(predictions, np.ndarray), "Inference predictions type was not set correctly."

def test_general_dataset_with_cuda():
    device = get_device('cuda', 0)
    X = torch.tensor([1, 2, 3], dtype=torch.float32).to(device)
    y = torch.tensor([2, 4, 6], dtype=torch.float32).to(device)
    dataset = GeneralDataset(X, y)
    assert len(dataset) == 3, "Dataset length was not set correctly."
    sample = dataset[1]
    assert sample['X'] == 2, "Sample X value was not set correctly."
    assert sample['y'] == 4, "Sample y value was not set correctly."

def test_create_dataloader_with_cuda():
    device = get_device('cuda', 0)
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    batch_size = 2
    dataloader = create_dataloader(X, y, batch_size, device=device)
    assert isinstance(dataloader, DataLoader), "Dataloader was not created correctly."
    for batch in dataloader:
        assert batch['X'].shape[0] == batch_size, "Batch X shape was not set correctly."
        assert batch['y'].shape[0] == batch_size, "Batch y shape was not set correctly."

def test_evaluate_model_with_cuda():
    device = get_device('cuda', 0)
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.linear = nn.Linear(2, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = DummyModel().to(device)
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    batch_size = 2
    dataloader = create_dataloader(X, y, batch_size, device=device)
    loss_fn = nn.MSELoss()
    loss = evaluate_model(model, dataloader, loss_fn, device=device)
    assert isinstance(loss, float), "Evaluation loss was not returned correctly."

def test_train_model_with_cuda():
    device = get_device('cuda', 0)
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.linear = nn.Linear(2, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = DummyModel().to(device)
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    batch_size = 2
    train_loader = create_dataloader(X, y, batch_size, device=device)
    val_loader = create_dataloader(X, y, batch_size, device=device)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_model(model, train_loader, val_loader, loss_fn, optimizer, device=device)

def test_infer_with_cuda():
    device = get_device('cuda', 0)
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.linear = nn.Linear(2, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = DummyModel().to(device)
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    batch_size = 2
    dataloader = create_dataloader(X, y, batch_size, device=device)
    predictions = infer(model, dataloader, device=device)
    assert predictions.shape[0] == X.shape[0], "Inference predictions shape was not set correctly."
    assert predictions.shape[1] == 1, "Inference predictions shape was not set correctly."
    assert isinstance(predictions, np.ndarray), "Inference predictions type was not set correctly."
