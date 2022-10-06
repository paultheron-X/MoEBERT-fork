import torch
import torch.nn as nn
import torch.nn.functional as F

class DSelectKGate(nn.Module):
    def __init__(self, k, dim=1):
        super().__init__()
        self.k = k
        self.dim = dim

    def forward(self, x):
        x = x.sort(dim=self.dim)[0]
        return x[:, -self.k:]