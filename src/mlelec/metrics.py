# Evaluation metrics
import torch
from typing import List, Optional, Union
from metatensor import TensorMap
from mlelec.utils.pbc_utils import blocks_to_matrix
from mlelec.data.dataset import PySCFPeriodicDataset


def L2_kspace_loss(pred: Union[TensorMap], 
                   target: Union[TensorMap],
                   dataset: PySCFPeriodicDataset):
    
    assert isinstance(target, TensorMap), "Target must be a TensorMap"
    assert isinstance(pred, TensorMap), "Prediction must be a TensorMap"

    loss = 0
    pred_real = blocks_to_matrix(pred, dataset)
    target_real = blocks_to_matrix(target, dataset)
    pred_kspace = dataset.compute_matrices_kspace(pred_real)
    target_kspace = dataset.compute_matrices_kspace(target_real)
    for ifr in range(len(target_kspace)):
        loss += torch.sum((pred_kspace[ifr] - target_kspace[ifr]) * torch.conj(pred_kspace[ifr] - target_kspace[ifr]))
    assert torch.norm(loss-loss.real) <1e-10
    return loss.real

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


def Eigval_loss(
    pred: torch.tensor, target: torch.tensor, overlap: Optional[torch.tensor] = None
):
    """Loss function for eigenvalues"""
    return torch.sum((pred - target) ** 2)


def Custom_loss():
    pass
