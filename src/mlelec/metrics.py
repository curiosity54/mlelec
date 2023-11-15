# Evaluation metrics
import torch
from typing import List, Optional


def L2_loss(pred: torch.tensor, target: torch.tensor):
    """L2 loss function"""
    return torch.sum((pred - target) ** 2)


def Eigval_loss(
    pred: torch.tensor, target: torch.tensor, overlap: Optional[torch.tensor] = None
):
    """Loss function for eigenvalues"""
    return torch.sum((pred - target) ** 2)
