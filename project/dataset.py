"""
Class for a generic dataset.
"""
import numpy as np
import os
import torch
from torch.utils.data import Dataset

# melange includes
from src.logger import Logger


class Dataset(Dataset):
    """
    datatype class.  
    """
    def __init__(self,
        name:   str,
    ):
        self.name = name
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
    