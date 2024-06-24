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


class EquivariantNonLinearity(nn.Module):
    def __init__(self, nonlinearity: callable = None, epsilon=1e-6, device=None):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.epsilon = epsilon

    def forward(self, x):
        assert len(x.shape) == 3
        x_inv = x.clone()
        norm = torch.sqrt(torch.sum(x_inv**2) + epsilon)
        if x.shape[1] > 1:
            # create an inv
            x_inv = torch.einsum("imf,imf->if", x_inv, x_inv)
        silu = torch.nn.SiLU()
        x_inv = torch.nn.SiLU()(x_inv)
        x_inv = x_inv.reshape(x.shape[0], x.shape[2])
        # should probably norm x here
        out = torch.einsum("if, imf->imf", x_inv, x)
        normout = torch.sqrt(torch.sum(out**2) + epsilon)
        out = out * norm / normout
        return out
        # return torch.einsum("if, imf->imf", x_inv, x)


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
        if nlayers == 1:
            self.mlp = nn.Linear(nin, nout, bias=bias)
            self.mlp.to(device)
            return

        self.mlp = [
            # nn.LayerNorm(nin, bias=False, elementwise_affine=False), # DONT DO THIS
            nn.Linear(nin, nhidden, bias=bias),
        ]
        if norm:
            norm_layer = NormLayer(nonlinearity=activation, device=device)
        for _ in range(nlayers - 2):
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
        device="cpu",
        **kwargs,
    ):
        self.dataset = dataset
        self.device = device

        # FIXME: generalize to other targets
        super().__init__()
        self._submodels(
            self.dataset.features, set_bias=kwargs.get("bias", False), **kwargs
        )
        self.dummy_property = self.dataset.target.blocks[0].properties

    def _submodels(self, features, set_bias=False, **kwargs):
        self.submodels = {}
        # return 1 model if no subtargets present, else spawn
        # multiple models
        if len(self.dataset.target.block_keys) == 1:
            self.model = MLP(
                nlayers=kwargs.get("nlayers", 2),
                nin=features.values.shape[-1],
                nout=1,
                nhidden=kwargs.get("nhidden", 10),
                bias=set_bias,
                device=self.device,
            )
        else:
            for k in self.dataset.target.block_keys:
                bias = False
                feat = map_targetkeys_to_featkeys(features, k)
                # print(feat,tuple(k))
                # print(feat.values.device, self.device)
                if k["L"] == 0 and set_bias:
                    bias = True
                self.submodels[str(tuple(k))] = MLP(
                    nin=feat.values.shape[-1],
                    nout=1,
                    nhidden=kwargs.get("nhidden", 10),
                    nlayers=kwargs.get("nlayers", 2),
                    bias=bias,
                )

                # print(k, self.submodels[str(tuple(k))])
            self.model = torch.nn.ModuleDict(self.submodels)
        self.model.to(self.device)

    def forward(
        self,
        features: TensorMap = None,
        return_type: str = "tensor",
        batch_indices=None,
        ridge_fit: bool = False,
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
            if ridge_fit:
                weights = self.ridge_weights.get(targ_key)
            else:
                weights = None

            if weights is None:
                feat = map_targetkeys_to_featkeys(
                    features, self.dataset.target.block_keys[i]
                )
                nsamples, ncomp, nprops = feat.values.shape
                featval = feat.values

                pred = submodel(featval)
                pred_block = TensorBlock(
                    values=pred.reshape((nsamples, ncomp, 1)),
                    samples=feat.samples,
                    components=feat.components,
                    properties=self.dummy_property,
                )
                pred_blocks.append(pred_block)

            else:
                # print("in the weights != None loop now")
                for w_layer in submodel.children():
                    assert w_layer.weight.data.shape == weights.shape
                    w_layer.weight.data = torch.from_numpy(weights).to(self.device)
    
                feat = map_targetkeys_to_featkeys(
                    features, self.dataset.target.block_keys[i]
                )
                nsamples, ncomp, nprops = feat.values.shape
                featval = feat.values
                # if nsamples != 0:
                pred = submodel(featval)
                pred_block = TensorBlock(
                    values=pred.reshape((nsamples, ncomp, 1)),
                    samples=feat.samples,
                    components=feat.components,
                    properties=self.dummy_property,
                )
                pred_blocks.append(pred_block)
        
        pred_tmap = TensorMap(self.dataset.target.block_keys, pred_blocks)

        if return_type == "coupled_blocks":
            return pred_tmap
        elif return_type == "uncoupled_blocks":
            self.reconstructed_uncoupled = _to_uncoupled_basis(
                pred_tmap, device=self.device
            )
            return self.reconstructed_uncoupled
        elif return_type == "tensor":
            self.reconstructed_uncoupled = _to_uncoupled_basis(
                pred_tmap, device=self.device
            )
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
                # print(batch_indices)
                # fr_idx = [list(i)[0] for i in fr_idx]
                # print(batch_indices)
                batch_indices = list(batch_indices[0])
                # print(batch_indices)
            if isinstance(batch_indices, torch.Tensor):
                sort_idx = np.argsort(batch_indices.numpy())
            else:
                sort_idx = np.argsort(batch_indices)
            batch_frames = [self.dataset.target.frames[i] for i in np.array(batch_indices)[sort_idx]]
            # batch_frames = [self.dataset.target.frames[i] for i in  batch_indices]
            # batch_frames = self.dataset.target.frames
            self.reconstructed_tensor = _to_matrix(
                self.reconstructed_uncoupled,
                batch_frames,
                orbitals=self.dataset.target.orbitals,
                device=self.device,
            )
            # return self.reconstructed_tensor
            inv_idx = {i: f for i, f in zip(sort_idx, range(len(batch_indices)))}
            
            return torch.stack([self.reconstructed_tensor[inv_idx[i]] for i in range(len(batch_indices))])
        elif return_type == "loss":
            loss = loss_fn(self.reconstructed_tensor, self.dataset.target.tensor)
        else:
            raise NotImplementedError("return_type not implemented")
        return loss

    def fit_ridge_analytical(self, alpha, cv, set_bias=False) -> None:
        from sklearn.linear_model import RidgeCV

        # set_bias will set bias=True for the invariant model
        self.recon = {}

        pred_blocks = []
        self.ridges = {}
        # kernels = []
        for k, block in self.dataset.target_train.items():
            # print(k)
            blockval = torch.linalg.norm(block.values)
            bias = False
            if True:  # blockval > 1e-10:
                if k["L"] == 0 and set_bias:
                    bias = True
                sample_names = block.samples.names
                feat = map_targetkeys_to_featkeys(self.dataset.feat_train, k)

                # featnorm = torch.linalg.norm(feat.values)
                targetnorm = torch.linalg.norm(block.values)
                nsamples, ncomp, nprops = block.values.shape
                # nsamples, ncomp, nprops = feat.values.shape
                # _,sidx = labels_where(feat.samples, Labels(sample_names, values = np.asarray(block.samples.values).reshape(-1,len(sample_names))), return_idx=True)
                assert np.all(block.samples.values == feat.samples.values), (
                    k,
                    block.samples.values.shape,
                    feat.samples.values.shape,
                )

                x = (
                    (
                        feat.values.reshape(
                            (feat.values.shape[0] * feat.values.shape[1], -1)
                        )
                        / 1
                    )
                    .cpu()
                    .numpy()
                )
                y = (
                    (
                        block.values.reshape(
                            block.values.shape[0] * block.values.shape[1], -1
                        )
                        / 1
                    )
                    .cpu()
                    .numpy()
                )
                # ridge = KernelRidge(alpha =[1e-5,1e-1, 1])# np.logspace(-15,-1,40))
                # ridge = ridge.fit(x,y)
                ridge = RidgeCV(
                    alphas=alpha, cv=cv, fit_intercept=bias
                ).fit(x, y)
                # print(ridge.intercept_, np.mean(ridge.coef_), ridge.alpha_)
                # print(pred.shape, nsamples)
                pred = ridge.predict(x)
                # if k['L']==0:
                #     print('SCORE', ridge.score(x,y) )
                # self.ridges.append(ridge)
                self.ridges[tuple(k)] = ridge
                
                pred_blocks.append(
                    TensorBlock(
                        values=torch.from_numpy(pred.reshape((nsamples, ncomp, 1)))
                        .to(self.device)
                        .to(torch.float32),
                        samples=block.samples,
                        components=block.components,
                        properties=self.dummy_property,
                    )
                )
            else:
                pred_blocks.append(
                    TensorBlock(
                        values=block.values.to(torch.float32),
                        samples=block.samples,
                        components=block.components,
                        properties=block.properties,
                    )
                )

                # block.copy())
        self.ridge_weights = {}  # Store weights after fit
        for idx_key, ridge in self.ridges.items():
            self.ridge_weights[str(idx_key)] = ridge.coef_
        # for i, ridge in enumerate(self.ridges):
        #     self.ridge_weights[str(tuple(self.dataset.target.block_keys[i]))] = ridge.coef_
        pred_tmap = TensorMap(self.dataset.target_train.keys, pred_blocks)
        self.recon_blocks = pred_tmap  # return_matrix=return_matrix)

        return self.recon_blocks, self.ridges

    def predict_ridge_analytical(self, test_target, test_features, return_matrix=False) -> None:

        # set_bias will set bias=True for the invariant model
        self.recon_val = {}

        pred_blocks_test = []
        # kernels = []
        for imdl, tkey in enumerate(test_target.keys):
            target = test_target.block(tkey)
            nsamples, ncomp, nprops = target.values.shape

            feat = map_targetkeys_to_featkeys(test_features, tkey)
            x = (
                (
                    feat.values.reshape(
                        (feat.values.shape[0] * feat.values.shape[1], -1)
                    )
                    / 1
                )
                .cpu()
                .numpy()
            )
            pred = self.ridges[imdl].predict(x)
            pred_blocks_test.append(
                TensorBlock(
                    values=torch.from_numpy(pred.reshape((nsamples, ncomp, 1)))
                    .to(self.device)
                    .to(torch.float32),
                    samples=target.samples,
                    components=target.components,
                    properties=self.dummy_property,
                )
            )

        pred_tmap_test = TensorMap(test_target.keys, pred_blocks_test)
        self.recon_blocks_test = pred_tmap_test
        return self.recon_blocks_test


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
        self.frames = frames
        self.orbitals = orbitals
        self.cell_shifts = cell_shifts
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.dummy_property = self.target_blocks[0].properties
        self._submodels(set_bias=kwargs.get("bias", False), **kwargs)
        print(self.cell_shifts, len(self.cell_shifts))

    def _submodels(self, set_bias=False, **kwargs):
        self.blockmodels = {}
        for k in self.target_blocks.keys:
            bias = False
            if k["L"] == 0 and set_bias:
                bias = True
            blockval = torch.linalg.norm(self.target_blocks[k].values)
            if True:  # blockval > 1e-10:
                feat = map_targetkeys_to_featkeys(self.feats, k)
                self.blockmodels[str(tuple(k))] = MLP(
                    nin=feat.values.shape[-1],
                    nout=1,
                    nhidden=kwargs.get("nhidden", 10),
                    nlayers=kwargs.get("nlayers", 2),
                    bias=bias,
                )
        self.model = torch.nn.ModuleDict(self.blockmodels)
        print(self.device)
        self.model.to(self.device)

    def forward(self, return_matrix=False):
        self.recon = {}

        pred_blocks = []

        for k, block in self.target_blocks.items():
            # print(k)
            blockval = torch.linalg.norm(block.values)
            if True:
                # if blockval > 1e-10:
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
                pred = self.blockmodels[str(tuple(k))](feat.values)
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
        if not return_matrix:
            return target
        recon_blocks = {}

        for translation in self.cell_shifts:
            blocks = []
            for key, block in target.items():
                # TODO: replace labels_where
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
                    _to_uncoupled_basis(recon_blocks[tuple(s)], device=self.device),
                    frames=self.frames,
                    orbitals=self.orbitals,
                    NH=True,
                    device=self.device,
                )  # DONT FORGET NH=True
            return rmat
        return recon_blocks

    def fit_ridge_analytical(self, return_matrix=False, set_bias=False) -> None:
        from sklearn.linear_model import RidgeCV
        from sklearn.kernel_ridge import KernelRidge

        # set_bias will set bias=True for the invariant model
        self.recon = {}

        pred_blocks = []
        ridges = []
        kernels = []
        for k, block in self.target_blocks.items():
            # print(k)
            blockval = torch.linalg.norm(block.values)
            bias = False
            if True:  # blockval > 1e-10:
                if k["L"] == 0 and set_bias:
                    bias = True
                sample_names = block.samples.names
                feat = map_targetkeys_to_featkeys(self.feats, k)
                featkey = map_targetkeys_to_featkeys(self.feats, k, return_key=True)

                featnorm = torch.linalg.norm(feat.values)
                targetnorm = torch.linalg.norm(block.values)
                nsamples, ncomp, nprops = block.values.shape
                # nsamples, ncomp, nprops = feat.values.shape
                # _,sidx = labels_where(feat.samples, Labels(sample_names, values = np.asarray(block.samples.values).reshape(-1,len(sample_names))), return_idx=True)
                assert np.all(block.samples.values == feat.samples.values[:, :6]), (
                    k,
                    block.samples.values.shape,
                    feat.samples.values.shape,
                )

                x = (
                    (
                        feat.values.reshape(
                            (feat.values.shape[0] * feat.values.shape[1], -1)
                        )
                        / 1
                    )
                    .cpu()
                    .numpy()
                )
                kernel = x @ x.T
                kernels.append(kernel)
                y = (
                    (
                        block.values.reshape(
                            block.values.shape[0] * block.values.shape[1], -1
                        )
                        / 1
                    )
                    .cpu()
                    .numpy()
                )
                # ridge = KernelRidge(alpha =[1e-5,1e-1, 1])# np.logspace(-15,-1,40))
                # ridge = ridge.fit(x,y)
                ridge = RidgeCV(
                    alphas=np.logspace(-15, -1, 40), fit_intercept=bias
                ).fit(x, y)
                # print(ridge.intercept_, np.mean(ridge.coef_), ridge.alpha_)
                # print(pred.shape, nsamples)
                pred = ridge.predict(x)
                # if k['L']==0:
                #     print('SCORE', ridge.score(x,y) )
                ridges.append(ridge)

                pred_blocks.append(
                    TensorBlock(
                        values=torch.from_numpy(pred.reshape((nsamples, ncomp, 1)))
                        .to(self.device)
                        .to(torch.float32),
                        samples=block.samples,
                        components=block.components,
                        properties=self.dummy_property,
                    )
                )
            else:
                pred_blocks.append(
                    TensorBlock(
                        values=block.values.to(torch.float32),
                        samples=block.samples,
                        components=block.components,
                        properties=block.properties,
                    )
                )

                # block.copy())

        pred_tmap = TensorMap(self.target_blocks.keys, pred_blocks)
        self.recon_blocks = self.model_return(pred_tmap, return_matrix=return_matrix)

        return self.recon_blocks, ridges, kernels

    def regularization_loss(self, regularization):
        return (
            regularization
            * torch.sum(self.layer.weight.T @ self.layer.weight)
            / 1  # len(self.feats.samples) # normalize by number of samples
        )
