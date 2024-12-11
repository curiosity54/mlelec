# Evaluation metrics
from typing import List, Optional, Union

import numpy as np
import torch
from metatensor import TensorMap

from mlelec.utils.property_utils import compute_batch_polarisability


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
        return torch.sum(torch.stack(loss))


def Eigval_loss(
    pred: torch.tensor, target: torch.tensor, overlap: Optional[torch.tensor] = None
):
    """Loss function for eigenvalues"""
    return torch.sum((pred - target) ** 2)


def Custom_loss():
    pass


def mse_per_atom(frames, pred, target):
    norm_loss = []
    for i in range(len(pred)):
        norm_loss.append(
            (torch.linalg.norm(pred[i] - target[i])) ** 2
            / (frames[i].get_global_number_of_atoms() ** 2)
        )
    return torch.mean(torch.stack(norm_loss))


def mse_total(frames, pred, target):
    norm_loss = []
    for i in range(len(pred)):
        norm_loss.append((torch.linalg.norm(pred[i] - target[i])) ** 2)
    return torch.mean(torch.stack(norm_loss))


def loss_fn_combined(
    ml_data,
    pred_focks,
    orthogonal,
    mfs,
    indices,
    loss_fn,
    frames,
    eigval,
    dipole,
    polar,
    var_eigval,
    var_dipole,
    var_polar,
    weight_eigval=1.0,
    weight_dipole=1.0,
    weight_polar=1.0,
):

    pred_dipole, pred_polar, pred_eigval = compute_batch_polarisability(
        ml_data, pred_focks, indices, mfs, orthogonal
    )

    loss_polar = loss_fn(frames, pred_polar, polar) / var_polar
    loss_dipole = loss_fn(frames, pred_dipole, dipole) / var_dipole
    loss_eigval = loss_fn(frames, pred_eigval, eigval) / var_eigval

    # weighted sum of the various loss contributions
    return (
        weight_eigval * loss_eigval
        + weight_dipole * loss_dipole
        + weight_polar * loss_polar,
        loss_eigval,
        loss_dipole,
        loss_polar,
    )
