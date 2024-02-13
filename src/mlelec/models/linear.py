# sets up a linear model
import torch
import torch.nn as nn
from mlelec.data.dataset import MLDataset
from typing import List, Dict, Optional, Union
from mlelec.utils.twocenter_utils import (
    map_targetkeys_to_featkeys,
    _to_uncoupled_basis,
    _to_matrix,
)
from metatensor import Labels, TensorMap, TensorBlock
import mlelec.metrics as mlmetrics
from mlelec.utils.metatensor_utils import labels_where
import numpy as np


def norm_over_components(x):
    x = x.swapaxes(1, 2)
    return torch.einsum("ifm, mfi-> if", x, x.T) + 1e-7


def clamped_norm(x, clamp: float):
    # this potentially causes instability/nan's
    # return x.norm(p=2, dim=-1, keepdim=True).clamp(min=clamp)
    x = x.swapaxes(1, 2)
    return torch.einsum("ifm, mfi-> if", x, x.T) + 1e-7
    # return torch.sum(x @ x.T + 1e-7)


def rescale(x, norm, new_norm):
    # return x / norm * new_norm
    return x / (norm + 1e-7) * new_norm


class NormLayer(nn.Module):
    def __init__(self, nonlinearity: callable = None, device=None):
        super().__init__()
        self.device = device
        self.nonlinearity = nonlinearity
        self.NORM_CLAMP = 2**-12

    def forward(self, x):
        norm = norm_over_components(x)  # , self.NORM_CLAMP)
        _norm = nn.LayerNorm(x.shape[-1], device=self.device)
        if self.nonlinearity is not None:
            # Transform the norms only
            new_norm = self.nonlinearity(_norm(norm.squeeze(-1))).unsqueeze(-1)
        else:
            print(norm.shape, norm.device, norm.squeeze(-1).shape, self.device)
            new_norm = _norm(norm).unsqueeze(-1)
            print(new_norm, new_norm.device)

        norm_x = rescale(x, norm, new_norm)
        return norm


# def _norm_layer(x, nonlinearity: callable = None):
#     """
#     x: torch.tensor of shape (nstr, ncomponents, nfeatures)

#     returns: torch.tensor of shape (nstr, nfeatures) i.e. compute norm of features
#     """
#     NORM_CLAMP = 2**-12
#     norm = clamped_norm(x, NORM_CLAMP)
#     group_norm = nn.GroupNorm(num_groups=1, num_channels=x.shape[-1])
#     if nonlinearity is not None:
#         # Transform the norms only
#         norm = nonlinearity(group_norm(norm.squeeze(-1))).unsqueeze(-1)

#     # assert (
#     #         len(x.shape) == 3
#     #     ), "Input tensor must be of shape (nstr, ncomponents, nfeatures)"
#     #     norm = torch.einsum("imq,imq->iq", x, x)
#     # norm = torch.linalg.norm(_symm)
#     return norm


class BlockModel(nn.Module):
    "other custom models"
    ## Could be the MLP as below or other custom model
    # for now - we just use MLP in the model

    def __init__(self):
        super().__init__()
        pass


class MLP(nn.Module):
    def __init__(
        self,
        nlayers: int,
        nin: int,
        nhidden: int,
        nout: int = 1,
        activation: Union[str, callable] = None,
        norm: bool = False,
        bias: bool = False,
        device=None,
    ):
        super().__init__()
        self.mlp = [
            # nn.LayerNorm(nin, bias=False, elementwise_affine=False), # DONT DO THIS
            nn.Linear(nin, nhidden, bias=bias),
        ]
        if norm:
            norm_layer = NormLayer(nonlinearity=activation, device=device)
        for _ in range(nlayers - 1):
            self.mlp.append(nn.Linear(nhidden, nhidden, bias=bias))
            if activation is not None:
                self.mlp.append(activation)
            if norm:
                self.mlp.append(norm_layer)
        self.mlp.append(nn.Linear(nhidden, nout, bias=bias))
        self.mlp = nn.Sequential(*self.mlp)
        self.mlp.to(device)

    def forward(self, x):
        return self.mlp(x)


class LinearTargetModel(nn.Module):
    def __init__(
        self,
        dataset: MLDataset,
        device=None,
        **kwargs,
    ):
        self.dataset = dataset
        self.device = device

        # FIXME: generalize to other targets
        super().__init__()
        self._submodels(self.dataset.features, **kwargs)
        self.dummy_property = self.dataset.target.blocks[0].properties

    def _submodels(self, features, **kwargs):
        self.submodels = {}
        # return 1 model if no subtargets present, else spawn
        # multiple models
        if len(self.dataset.target.block_keys) == 1:
            self.model = MLP(
                nlayers=1,
                nin=features.values.shape[-1],
                nout=1,
                nhidden=10,
                bias=False,
                device=self.device,
            )
        else:
            for k in self.dataset.target.block_keys:
                feat = map_targetkeys_to_featkeys(features, k)
                # print(feat,tuple(k))
                # print(feat.values.device, self.device)
                self.submodels[str(tuple(k))] = MLP(
                    nin=feat.values.shape[-1],
                    device=self.device,
                    # nout=1,
                    **kwargs,
                )

                # print(k, self.submodels[str(tuple(k))])
            self.model = torch.nn.ModuleDict(self.submodels)
        self.model.to(self.device)

    def forward(
        self,
        features: TensorMap = None,
        return_type: str = "tensor",
        batch_indices=None,
        **kwargs,
    ):
        return_choice = [
            "coupled_blocks",
            "uncoupled_blocks",
            "tensor",
            "loss",
        ]
        assert (
            return_type.lower() in return_choice
        ), f"return_type must be one of {return_choice}"
        if return_type == "loss":
            loss_fn = kwargs.get("loss_fn", None)
            if loss_fn is None:
                loss_fn = mlmetrics.L2_loss
            elif isinstance(loss_fn, str):
                try:
                    loss_fn = getattr(mlmetrics, loss_fn.capitalize())
                except:
                    raise NotImplementedError(
                        f"Selected loss function {loss_fn} not implemented"
                    )

        pred_blocks = []
        for i, (targ_key, submodel) in enumerate(self.submodels.items()):
            # try:
            feat = map_targetkeys_to_featkeys(
                features, self.dataset.target.block_keys[i]
            )
            nsamples, ncomp, nprops = feat.values.shape
            featval = feat.values
            # featval = feat.values.reshape(-1, feat.values.shape[-1])
            # except ValueError:
            # print("Key not found in features - skipped")

            pred = submodel(featval)
            pred_block = TensorBlock(
                values=pred.reshape((nsamples, ncomp, 1)),
                samples=feat.samples,
                components=feat.components,
                properties=self.dummy_property,
            )
            pred_blocks.append(pred_block)
        pred_tmap = TensorMap(self.dataset.target.block_keys, pred_blocks)
        self.reconstructed_uncoupled = _to_uncoupled_basis(pred_tmap)
        if batch_indices is None:
            # identify the frames in the batch
            batch_indices = list(
                set(
                    [
                        tuple(set(b.samples["structure"].tolist()))
                        for (k, b) in pred_tmap.items()
                    ]
                )
            )
            print(batch_indices)
            # fr_idx = [list(i)[0] for i in fr_idx]
            # print(batch_indices)
            batch_indices = list(batch_indices[0])
            # print(batch_indices)
        batch_frames = [self.dataset.target.frames[i] for i in batch_indices]
        # batch_frames = self.dataset.target.frames
        self.reconstructed_tensor = _to_matrix(
            self.reconstructed_uncoupled,
            batch_frames,
            self.dataset.target.orbitals,
            device=self.device,
        )
        if return_type == "coupled_blocks":
            return pred_tmap
        elif return_type == "uncoupled_blocks":
            return self.reconstructed_uncoupled
        elif return_type == "tensor":
            return self.reconstructed_tensor
        elif return_type == "loss":
            loss = loss_fn(self.reconstructed_tensor, self.dataset.target.tensor)

        return loss


class LinearModelPeriodic(nn.Module):
    def __init__(
        self,
        twocfeat,
        target_blocks,
        frames,
        orbitals,
        device=None,
        cell_shifts=None,
        **kwargs,
    ):
        super().__init__()
        self.feats = twocfeat
        self.target_blocks = target_blocks
        self.target_blocks = target_blocks
        self.frames = frames
        self.orbitals = orbitals
        self.cell_shifts = cell_shifts
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device
        self.dummy_property = self.target_blocks[0].properties
        self._submodels(**kwargs)
        print(self.cell_shifts, len(self.cell_shifts))

    def _submodels(self, **kwargs):
        self.blockmodels = {}

        for k in self.target_blocks.keys:
            blockval = torch.linalg.norm(self.target_blocks[k].values)
            if blockval > 1e-10:
                feat = map_targetkeys_to_featkeys(self.feats, k)
                self.blockmodels[str(tuple(k))] = MLP(
                    nin=feat.values.shape[-1],
                    nout=1,
                    nhidden=kwargs.get("nhidden", 10),
                    nlayers=kwargs.get("nlayers", 2),
                )
        self.model = torch.nn.ModuleDict(self.blockmodels)
        self.model.to(self.device)

    def forward(self, return_matrix=False):
        self.recon = {}

        pred_blocks = []

        for k, block in self.target_blocks.items():
            # print(k)
            blockval = torch.linalg.norm(block.values)

            if blockval > 1e-10:
                sample_names = block.samples.names
                feat = map_targetkeys_to_featkeys(self.feats, k)
                featnorm = torch.linalg.norm(feat.values)
                nsamples, ncomp, nprops = block.values.shape
                # nsamples, ncomp, nprops = feat.values.shape
                # _,sidx = labels_where(feat.samples, Labels(sample_names, values = np.asarray(block.samples.values).reshape(-1,len(sample_names))), return_idx=True)
                assert np.all(block.samples.values == feat.samples.values[:, :6]), (
                    k,
                    block.samples.values.shape,
                    feat.samples.values.shape,
                )
                pred = self.blockmodels[str(tuple(k))](feat.values / featnorm)
                # print(pred.shape, nsamples)

                pred_blocks.append(
                    TensorBlock(
                        values=pred.reshape((nsamples, ncomp, 1)),
                        samples=block.samples,
                        components=block.components,
                        properties=self.dummy_property,
                    )
                )
            else:
                pred_blocks.append(block.copy())
        pred_tmap = TensorMap(self.target_blocks.keys, pred_blocks)
        self.recon_blocks = self.model_return(pred_tmap, return_matrix=return_matrix)
        return self.recon_blocks
        # _to_matrix(_to_uncoupled_basis(pred_sum_dict[s]), frames = self.frames, orbitals=self.orbitals)

    def model_return(self, target: TensorMap, return_matrix=False):
        recon_blocks = {}

        for translation in self.cell_shifts:
            blocks = []
            for key, block in target.items():
                _, i = labels_where(
                    block.samples,
                    Labels(
                        ["cell_shift_a", "cell_shift_b", "cell_shift_c"],
                        values=np.asarray(
                            [translation[0], translation[1], translation[2]]
                        ).reshape(1, -1),
                    ),
                    return_idx=True,
                )
                blocks.append(
                    TensorBlock(
                        samples=Labels(
                            target.sample_names[:-3],
                            values=np.asarray(block.samples.values[i])[:, :-3],
                        ),
                        values=block.values[i],
                        components=block.components,
                        properties=block.properties,
                    )
                )
            tmap = TensorMap(target.keys, blocks)
            recon_blocks[tuple(translation)] = tmap

        if return_matrix:
            rmat = {}
            for s in self.cell_shifts[:]:
                rmat[tuple(s)] = _to_matrix(
                    _to_uncoupled_basis(recon_blocks[tuple(s)]),
                    frames=self.frames,
                    orbitals=self.orbitals,
                    NH=True,
                )  # DONT FORGET NH=True
            return rmat
        return recon_blocks

    def fit_ridge_analytical(self, return_matrix = False) -> None:
        from sklearn.linear_model import RidgeCV
        self.recon = {}

        pred_blocks = []
        ridges = []

        for k, block in self.target_blocks.items():
            # print(k)
            blockval = torch.linalg.norm(block.values)

            if blockval > 1e-10:
                if k['L'] == 0:
                    bias = True
                else: 
                    bias = False
                sample_names = block.samples.names
                feat = map_targetkeys_to_featkeys(self.feats, k)
                featnorm = torch.linalg.norm(feat.values)
                nsamples, ncomp, nprops = block.values.shape
                # nsamples, ncomp, nprops = feat.values.shape
                # _,sidx = labels_where(feat.samples, Labels(sample_names, values = np.asarray(block.samples.values).reshape(-1,len(sample_names))), return_idx=True)
                assert np.all(block.samples.values == feat.samples.values[:, :6]), (
                    k,
                    block.samples.values.shape,
                    feat.samples.values.shape,
                )
                
                x = feat.values.reshape(feat.values.shape[0]*feat.values.shape[1], -1).cpu().numpy()
                y = block.values.reshape(block.values.shape[0]*block.values.shape[1], -1).cpu().numpy()

                ridge = RidgeCV(alphas = np.logspace(-21, -1, 40), fit_intercept = bias).fit(x, y)
                # print(pred.shape, nsamples)
                pred = ridge.predict(x)
                ridges.append(ridge)

                pred_blocks.append(
                    TensorBlock(
                        values=pred.reshape((nsamples, ncomp, 1)),
                        samples=block.samples,
                        components=block.components,
                        properties=self.dummy_property,
                    )
                )
            else:
                pred_blocks.append(block.copy())
        pred_tmap = TensorMap(self.target_blocks.keys, pred_blocks)
        self.recon_blocks = self.model_return(pred_tmap, return_matrix=return_matrix)
        
        return self.recon_blocks, ridges
