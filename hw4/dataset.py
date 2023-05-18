# ========================================================
#             Media and Cognition
#             Homework 4  Sequence Modeling
#             dataset.py - dataset definition
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2023
# ========================================================

import torch
from torch.utils.data import Dataset
import numpy as np
import os

class LMDataset(Dataset):
    def __init__(self, data_dir, split, max_seq_len):
        super().__init__()
        self.data = np.memmap(os.path.join(data_dir, '%s.bin'%split), dtype=np.uint16, mode='r')
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.data) - self.max_seq_len - 1
    
    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index:index+self.max_seq_len].astype(np.int64))
        y = torch.from_numpy(self.data[index+1:index+1+self.max_seq_len].astype(np.int64))
        return x, y