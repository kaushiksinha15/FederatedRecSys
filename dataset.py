import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SequentialRecDataset(Dataset):
    """
    Mock dataset for sequential recommendation.
    Generates random sequences of item IDs and targets.
    """
    def __init__(self, num_samples=1000, seq_len=10, num_items=5000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_items = num_items
        
        # Randomly generate sequences of interactions (item IDs)
        self.sequences = torch.randint(1, num_items, (num_samples, seq_len))
        # The target is the next item the user will interact with
        self.targets = torch.randint(1, num_items, (num_samples,))
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def get_client_dataloader(client_id, batch_size=32, num_samples=500):
    """
    Returns a dataloader for a specific client.
    We introduce some artificial bias based on client_id to simulate different behaviors.
    """
    # Simulate different behavior clusters by skewing the items
    # Clients 1 & 2 prefer items 1-1000, Clients 3 & 4 prefer items 1000-2000
    base_item = 1 if client_id <= 2 else 1000
    
    dataset = SequentialRecDataset(num_samples=num_samples, seq_len=10, num_items=2000)
    
    # Skew the dataset artificially to create "clusters" of behavior
    dataset.sequences = (dataset.sequences % 1000) + base_item
    dataset.targets = (dataset.targets % 1000) + base_item
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
