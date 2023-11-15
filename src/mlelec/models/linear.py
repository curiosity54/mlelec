# sets up a linear model
import torch
import torch.nn as nn
from mlelec.data.dataset import MLDataset
from typing import List, Dict 

class MLP(torch.nn):
    def __init__(self, nlayers, nin, nout, nhidden, activation:callable, norm:callable):
        self.mlp=[torch.nn.Linear(nin, nhidden, bias=False)]
        for _ in range(nlayers-1):
            self.mlp.append(Linear(nhidden, nhidden, bias=False))


class TargetModel(torch.nn):
    def __init__(self, target, dataset):
        super().__init__()
        self.models = self._submodels(target)
        

    def _submodels(self, target):
        #return 1 model if no subtargets present, else spawn
        # multiple models 

class BlockModel(torch.nn):
    def __init__(self):
        super().__init__()


        

