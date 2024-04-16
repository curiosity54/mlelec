# Evaluation metrics
import torch
from typing import List, Optional, Union
from metatensor import TensorMap
from mlelec.utils.pbc_utils import blocks_to_matrix, inverse_fourier_transform
from mlelec.data.dataset import PySCFPeriodicDataset
from mlelec.utils.symmetry import ClebschGordanReal
import warnings
import numpy as np

def L2_kspace_loss(pred: Union[TensorMap], 
                   target: Union[TensorMap, list],
                   dataset: PySCFPeriodicDataset,
                   cg: Optional[ClebschGordanReal] = None,
                   kpts: Union[List, torch.Tensor] =  [[0.,0.,0.]],
                   norm = None,
                   desired_ifr = None):
                   
    """L2 loss function for k-space matrices computed at given 'kpts' 
    kpts: list(list of kpts) per frame 
    target: list of Hks per frame
    ASSUMPTION: len(target) = len(kpts) and target[ifr] has the associated kpts in kpt[ifr].  
    Not implemented for ttarget = TensorMap
    """
    
    assert isinstance(target, TensorMap) or isinstance(target, list), "Target must be a TensorMap or a list"
    assert isinstance(pred, TensorMap), "Prediction must be a TensorMap"

    loss = 0
    pred_real = blocks_to_matrix(pred, dataset, cg = cg)
    # print(len(pred_real), pred_real[0][0,0,0].shape)
    if isinstance(target, list):
        target_kspace = target
    else:
        raise NotImplementedError("Target must be a list of k-space matrices")
        warnings.warn("Target is a TensorMap. Computing k-space matrices for target")
        target_real = blocks_to_matrix(target, dataset, cg = cg)
        target_kspace = dataset.compute_matrices_kspace(target_real)


    if desired_ifr is not None:
        # for ifr in range(len(target_kspace)):
            pred_kspace = []
            
            # for k in kpts[ifr]:
            #     # a = inverse_fourier_transform(torch.stack(list(pred_real[ifr].values())), torch.from_numpy(np.array(list(pred_real[ifr].keys()), dtype=np.float64)), k)
            #     # print('a', a.shape)
            #     pred_kspace.append(inverse_fourier_transform(torch.stack(list(pred_real[desired_ifr].values())), 
            #                                                  torch.from_numpy(np.array(list(pred_real[desired_ifr].keys()), dtype=np.float64)), 
            #                                                  k))
            pred_H = torch.stack(list(pred_real[desired_ifr].values()))
            T = torch.from_numpy(np.array(list(pred_real[desired_ifr].keys()), dtype = np.float64))
            pred_kspace = inverse_fourier_transform(pred_H, T_list = T, k = kpts)
                # print('pred_kspace', pred_kspace[-1].shape)
            # pred_kspace = torch.stack(pred_kspace)
            # assert pred_kspace.shape == target_kspace[ifr].shape
            # print(pred_kspace.shape, target_kspace[ifr].shape)  
            loss += torch.sum([(pred_kspace[ifr] - target_kspace[ifr]) * torch.conj(pred_kspace[ifr] - target_kspace[ifr]) \
                               for ifr in range(len(target_kspace))])
    else:
        for ifr in range(len(target)):
            pred_H = torch.stack(list(pred_real[ifr].values()))
            T = torch.from_numpy(np.array(list(pred_real[ifr].keys()), dtype = np.float64)).to(pred_H)
            if norm is None:
                norm = 1/np.sqrt(kpts[ifr].shape[0])
            pred_kspace = inverse_fourier_transform(pred_H, T_list = T, k = kpts[ifr], norm = norm) #1/np.sqrt(T.shape[0]))

            # assert pred_kspace.shape == target_kspace[ifr].shape
            loss += torch.sum((pred_kspace - target_kspace[ifr]) * torch.conj(pred_kspace - target_kspace[ifr]))
            
    assert torch.norm(loss-loss.real) < 1e-10
    return loss.real

def L2_loss(pred: Union[torch.tensor, TensorMap], target: Union[torch.tensor, TensorMap], loss_per_block = False):
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
        losses = []
        for key, block in pred.items():
            targetblock = target.block(key)
            assert (
                block.samples == targetblock.samples
            ), "Prediction and target must have the same samples"
            losses.append(torch.sum((block.values - targetblock.values) ** 2))
        if loss_per_block:
            return losses, sum(losses)
        else:
            return sum(losses)


def Eigval_loss(
    pred: torch.tensor, target: torch.tensor, overlap: Optional[torch.tensor] = None
):
    """Loss function for eigenvalues"""
    return torch.sum((pred - target) ** 2)


def Custom_loss():
    pass
