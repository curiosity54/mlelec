# sets up a linear model
import torch
import torch.nn as nn
from mlelec.data.dataset import MLDataset
from typing import List, Dict, Optional
from mlelec.utils.twocenter_utils import map_targetkeys_to_featkeys, _to_uncoupled_basis
from metatensor import Labels, TensorMap, TensorBlock


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


class BlockModel(nn.Module):
    "other custom models"

    def __init__(self):
        super().__init__()
        pass


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
        super().__init__()
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


class LinearTargetModel(nn.Module):
    def __init__(
        self,
        dataset: MLDataset,
        features: Optional[TensorMap] = None,
        device=None,
        metrics: "l2loss" = None,
        **kwargs,
    ):
        self.dataset = dataset
        self.metrics = metrics
        self.loss_fn = getattr(mlelec.metrics, metrics)

        # FIXME: generalize to other targets
        super().__init__()
        if features is None:
            from mlelec.features.acdc import (
                single_center_features,
                pair_features,
                twocenter_hermitian_features,
            )

            hypers = kwargs.get("hypers", None)
            if hypers is None:
                print("Computing features with default hypers")
                hypers = {
                    "cutoff": 2.0,
                    "max_radial": 2,
                    "max_angular": 1,
                    "atomic_gaussian_width": 0.2,
                    "center_atom_weight": 1,
                    "radial_basis": {"Gto": {}},
                    "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
                }
            single = single_center_features(dataset.structures, hypers, 2)

            pairs = pair_features(
                dataset.structures,
                hypers,
                order_nu=2,
                feature_names=single[0].properties.names,
            )
            self.features = twocenter_hermitian_features(single, pairs)

    def _submodels(self, target, **kwargs):
        self.submodels = {}
        # return 1 model if no subtargets present, else spawn
        # multiple models
        if len(self.dataset.target.block_keys) == 1:
            self.models = MLP(
                nlayers=1,
                nin=self.features.values.shape[-1],
                nout=1,
                nhidden=10,
                bias=False,
            )
        else:
            for k in self.dataset.target.block_keys:
                feat = map_targetkeys_to_featkeys(self.features, k)
                # print(feat,tuple(k))
                self.submodels[tuple(k)] = MLP(
                    nlayers=kwargs["nlayers"],
                    nin=feat.values.shape[-1],
                    nout=kwargs["nout"],
                    nhidden=kwargs["nhidden"],
                    bias=kwargs["bias"],
                )
            self.models = torch.nn.ModuleList(self.submodels)

    def forward(self):
        pred_blocks = []
        for targ_key, submodel in self.submodels.items():
            try:
                feat = map_targetkeys_to_featkeys(self.features, targ_key)
                nsamples, ncomp, nprops = feat.values.shape
                feat = feat.reshape(-1, feat.shape[-1])
            except ValueError:
                print("Key not found in features - skipped")

            pred = submodel(feat.values)
            pred_block = TensorBlock(
                values=pred.reshape((nsamples, ncomp, 1)),
                samples=feat.samples,
                components=feat.components,
                properties=self.predict_properties,
            )
            pred_blocks.append(pred_block)
        pred_tmap = TensorMap(self.dataset.target.block_keys, pred_blocks)

        self.reconstructed = _to_uncoupled_basis(pred_tmap)

        loss = self.metrics(self.reconstructed, self.dataset.target.tensor)
        return loss
