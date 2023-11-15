# sets up a linear model
import torch
import torch.nn as nn
from mlelec.data.dataset import MLDataset
from typing import List, Dict


def _norm_layer(x):
    """
    x: torch.tensor of shape (nstr, ncomponents, nfeatures)

    returns: torch.tensor of shape (nstr, nfeatures) i.e. compute norm of features
    """
    assert (
        len(x.shape) == 3
    ), "Input tensor must be of shape (nstr, ncomponents, nfeatures)"
    norm = torch.einsum("imq,imq->iq", x, x)
    return norm


class MLP(nn.Module):
    def __init__(
        self,
        nlayers: int,
        nin: int,
        nout: int,
        nhidden: int,
        activation: callable = None,
        norm: callable = None,
        bias: bool = False,
        device=None,
    ):
        self.mlp = [nn.Linear(nin, nhidden, bias=bias)]
        for _ in range(nlayers - 1):
            self.mlp.append(nn.Linear(nhidden, nhidden, bias=bias))
            if activation is not None:
                self.mlp.append(activation)
            if norm is not None:
                self.mlp.append(norm)
        self.mlp.append(nn.Linear(nhidden, nout, bias=bias))
        self.mlp = nn.Sequential(*self.mlp)
        self.mlp.to(device)

    def forward(self, x):
        return self.mlp(x)


class TargetModel(nn.Module):
    def __init__(self, target, dataset):
        super().__init__()
        self.models = self._submodels(target)

    def _submodels(self, target):
        # return 1 model if no subtargets present, else spawn
        # multiple models
        pass


class BlockModel(nn.Module):
    def __init__(self):
        super().__init__()
