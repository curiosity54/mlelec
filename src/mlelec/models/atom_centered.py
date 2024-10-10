# sets up a linear model
import torch
import torch.nn as nn
from mlelec.data.dataset import MLDataset
from typing import List, Dict, Optional, Union
import metatensor.torch as mts
from metatensor.torch import Labels, TensorMap, TensorBlock
import mlelec.metrics as mlmetrics
from mlelec.utils.metatensor_utils import labels_where
import numpy as np
import warnings
from mlelec.models.linear_integrated import NormLayer, E3LayerNorm, EquivariantNonLinearity, MLP

class atomic_predictor(torch.nn.Module):
    def __init__(self,set_bias=False, norm=False, **kwargs):
         super().__init__()
         self.input_size = kwargs["input_size"]
         self.intermediate_size = kwargs.get("intermediate_size",16)
         self.hidden_size = kwargs.get("nhidden", 16)
         self.device = kwargs.get("device", "cpu")
         self.apply_norm = norm 
         bias = set_bias
         #self.input_layer = torch.nn.Linear(
         #    in_features=self.input_size, out_features=self.intermediate_size)

         #self.hidden_layer = torch.nn.Linear(
         #        in_features=self.intermediate_size, out_features=self.hidden_size
         #    )

         #self.out_layer = torch.nn.Linear(
         #        in_features=self.hidden_size, out_features=1
         #    )

         #self.model = torch.nn.Sequential( self.input_layer,

         #                                 # self.hidden_layer,

         #                                 self.out_layer
         #    )
         self.model = MLP( nin= self.input_size, #feat.values.shape[-1],
                    nout=1,
                    nhidden=kwargs.get("nhidden", 16),
                    nlayers=kwargs.get("nlayers", 2),
                    bias=bias,
                    activation=kwargs.get("activation", None),
                    apply_layer_norm = self.apply_norm,)

         self.to(self.device)

    def property_per_structure(self, x, samples):

         structure_map, new_samples, _ = StructureMap(
             samples["structure"], "cpu"
         )
         self.structure_sum =  torch.zeros((len(new_samples), np.prod(x.shape[1:])), device=x.device)
         self.structure_sum.index_add_(0, structure_map, x)

         return self.structure_sum

    def forward(self, x, samples):
         pred = self.model(x)
         pred = self.property_per_structure(pred[...,0], samples)
         return pred

def StructureMap(samples_structure, device="cpu"):
    if isinstance(samples_structure, torch.Tensor):
        samples_structure = samples_structure.numpy()
    unique_structures, unique_structures_idx = np.unique(
        samples_structure, return_index=True
    )
    new_samples = samples_structure[unique_structures_idx]
    # we need a list keeping track of where each atomic contribution goes
    # (e.g. if structure ids are [3,3,3,1,1,1,6,6,6] that will be stored as
    # the unique structures [3, 1, 6], structure_map will be
    # [0,0,0,1,1,1,2,2,2]
    replace_rule = dict(zip(unique_structures, range(len(unique_structures))))
    structure_map = torch.tensor(
        [replace_rule[i] for i in samples_structure],
        dtype=torch.long,
        device=device,
    )
    return structure_map, new_samples, replace_rule

def get_single_sample_idx(i, sample_array):
    #return first instance of structure i
    try:
        return np.where(sample_array[:,0]==i)[0][0]
    except:
        warnings.warn('sample not found')
        return None

def iterate_minibatches(inputs, outputs, batch_size, sample_array):
    nstruct = sample_array[-1][0]
    # print("number of structures found ", nstruct)
    for index in range(0, nstruct , batch_size):
        start = get_single_sample_idx(index,sample_array)
        stop = get_single_sample_idx(index+batch_size,sample_array)
#         print(index, start, stop)
        yield inputs[start : stop], outputs[index : index + batch_size], index

