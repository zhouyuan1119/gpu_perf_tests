""" A fake dataset class that returns random tensors """
import torch
from torch.utils.data import Dataset

class FakeDataset(Dataset):
    def __init__(self, shape, length, dtype, range=None, num_classes=10):
        self.shape_ = shape
        self.length_ = length
        self.dtype_ = dtype
        full_shape = (length,) + shape
        if (dtype == torch.float32) or (dtype == torch.float16) or (dtype == torch.float64):
            assert range is None, 'Float with range not supported!'
            self.data_ = torch.randn(*full_shape, dtype=dtype)
        elif dtype == torch.int64:
            assert range is not None, 'Range must be specified when using int!'
            self.data_ = torch.randint(range[0], range[1], full_shape, dtype=dtype)
        else:
            raise NotImplementedError
        self.labels_ = torch.randint(0, num_classes, (self.length_,), dtype=torch.int64)
    
    def __len__(self):
        return self.length_
    
    def __getitem__(self, idx):
        return self.data_[idx], self.labels_[idx]