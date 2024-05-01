# Evaluation metrics
import numpy as np
import torch
from typing import List, Optional, Union
from metatensor import TensorMap


def L2_loss(
    pred: Union[torch.tensor, TensorMap], target: Union[torch.tensor, TensorMap]
):
    """L2 loss function"""
    if isinstance(pred, torch.Tensor):
        assert isinstance(target, torch.Tensor)
        assert (
            pred.shape == target.shape
        ), "Prediction and target must have the same shape"
        # target = target.to(pred)
        return torch.sum((pred - target) ** 2)
    elif isinstance(pred, TensorMap):
        assert isinstance(
            target, TensorMap
        ), "Target must be a TensorMap if prediction is a TensorMap"
        loss = 0
        for key, block in pred.items():
            targetblock = target.block(key)
            assert (
                block.samples == targetblock.samples
            ), "Prediction and target must have the same samples"
            loss += torch.sum((block.values - targetblock.values) ** 2)
        return loss
    elif isinstance(pred, list):
        if any(isinstance(t, np.ndarray) for t in target):
            target = [torch.from_numpy(t) for t in target]
        if not all(isinstance(t, torch.Tensor) for t in target + pred):
            raise ValueError("All targets and predictions must be tensors.")
        loss_fn = torch.nn.functional.mse_loss
        loss = [loss_fn(targ, predic) for targ, predic in zip(target, pred)]
        return torch.mean(torch.stack(loss))
            


def Eigval_loss(
    pred: torch.tensor, target: torch.tensor, overlap: Optional[torch.tensor] = None
):
    """Loss function for eigenvalues"""
    return torch.sum((pred - target) ** 2)


def Custom_loss():
    pass
