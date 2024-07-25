import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union
from mlelec.utils.twocenter_utils import map_targetkeys_to_featkeys_integrated
import metatensor.torch as mts
from metatensor.torch import Labels, TensorMap, TensorBlock
from metatensor.torch.learn import ModuleMap
import numpy as np
import warnings
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV


class EquivariantNonLinearity(nn.Module):
    def __init__(self, nonlinearity: callable = None, epsilon=1e-6, norm=True, layersize=None, device=None):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.epsilon = epsilon
        self.device = device or 'cpu'
        self.nn = [self.nonlinearity]
        if norm:
            self.nn.append(nn.LayerNorm(layersize, device=self.device))
        self.nn = nn.Sequential(*self.nn)

    def forward(self, x):
        assert len(x.shape) == 3
        x_inv = torch.einsum("imf,imf->if", x, x)
        x_inv = torch.sqrt(x_inv + self.epsilon)
        x_inv = self.nn(x_inv)
        out = torch.einsum("if, imf->imf", x_inv, x)
        return out


class MLP(nn.Module):
    def __init__(
        self,
        nlayers: int,
        nin: int,
        nhidden: Union[int, list],
        nout: int = 1,
        activation: Union[str, callable] = None,
        bias: bool = False,
        device=None,
        apply_layer_norm=False,
    ):
        super().__init__()

        self.mlp = []
        if nlayers == 0:
            # Purely linear model
            self.mlp.append(nn.Linear(nin, nout, bias=bias))
        else:
            if not isinstance(nhidden, list):
                nhidden = [nhidden] * nlayers
            else:
                assert len(nhidden) == nlayers, "len(nhidden) must be equal to nlayers"

            # Input layer
            self.mlp.append(nn.Linear(nin, nhidden[0], bias=bias))

            # Hidden layers
            last_n = nhidden[0]
            for n in nhidden[1:]:
                self.mlp.extend(self.middle_layer(last_n, n, activation, bias, device, apply_layer_norm))
                last_n = n

            # Output layer
            self.mlp.extend(self.middle_layer(last_n, nout, activation, bias, device, apply_layer_norm))

        self.mlp = nn.Sequential(*self.mlp)
        self.mlp.to(device)

    def middle_layer(self, n_in, n_out, activation=None, bias=False, device='cpu', apply_layer_norm=False):
        layers = [nn.Linear(n_in, n_out, bias=bias)]
        # self.initialize_layer(layers[-1])
        if activation:
            if isinstance(activation, str):
                activation = getattr(nn, activation)()
            elif not isinstance(activation, torch.nn.Module):
                raise ValueError('activation must be a string or a torch.nn.Module instance')
            layers.insert(0, EquivariantNonLinearity(nonlinearity=activation, device=device, norm=apply_layer_norm, layersize=n_in))
        return layers
    
    def initialize_layer(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class EquivariantNonlinearModel(nn.Module):
    def __init__(
        self,
        mldata,
        nhidden: Union[int, list],
        nlayers: int,
        activation: Union[str, callable] = 'SiLU',
        apply_norm: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.feats = mldata.features
        self.target_blocks = mldata.model_metadata
        self.frames = mldata.structures
        self.orbitals = mldata.model_basis
        self.apply_norm = apply_norm
        self.device = mldata.device
        self.dummy_property = self.target_blocks[0].properties
        self._initialize_submodels(set_bias=kwargs.get("bias", False), nhidden=nhidden, nlayers=nlayers, activation=activation)
        self.ridges = None

    # def _initialize_submodels(self, set_bias=False, nhidden=16, nlayers=2, activation=None, **kwargs):
    #     self.blockmodels = nn.ModuleDict()
    #     self.block_properties = {}

    #     for k, b in self.target_blocks.items():
    #         block_key = str(tuple(k))
    #         nprop = b.values.shape[-1]
    #         self.block_properties[tuple(k.values.tolist())] = b.properties

    #         feat = map_targetkeys_to_featkeys_integrated(self.feats, k)
    #         bias = k["L"] == 0 and set_bias

    #         self.blockmodels[block_key] = MLP(
    #             nin=feat.values.shape[-1],
    #             nout=nprop,
    #             nhidden=nhidden,
    #             nlayers=nlayers,
    #             bias=bias,
    #             activation=activation,
    #             apply_layer_norm=self.apply_norm,
    #         )

    #     self.model = self.blockmodels.to(self.device)

    # def forward(self, features, target_blocks=None, return_matrix=False):
    #     if target_blocks is None:
    #         target_blocks = self.target_blocks
    #         warnings.warn('Using training target_blocks; provide test target_blocks for inference.')

    #     pred_blocks = []

    #     for k, block in target_blocks.items():
    #         block_key = str(tuple(k))
    #         feat = map_targetkeys_to_featkeys_integrated(features, k)
    #         pred = self.blockmodels[block_key](feat.values)
    #         pred_blocks.append(
    #             TensorBlock(
    #                 values=pred,
    #                 samples=feat.samples,
    #                 components=feat.components,
    #                 properties=self.block_properties[tuple(k.values.tolist())],
    #             )
    #         )

    #     pred_tmap = TensorMap(target_blocks.keys, pred_blocks)
    #     return self.model_return(pred_tmap, return_matrix=return_matrix)
    
    def _initialize_submodels(self, set_bias=False, nhidden=16, nlayers=2, activation=None, **kwargs):
        self.block_properties = {}

        modules = []
        out_properties = []
        for k, b in self.target_blocks.items():
            nprop = b.values.shape[-1]
            # self.block_properties[tuple(k.values.tolist())] = b.properties
            out_properties.append(b.properties)

            feat = map_targetkeys_to_featkeys_integrated(self.feats, k)
            bias = k["L"] == 0 and set_bias

            modules.append(MLP(
                nin=feat.values.shape[-1],
                nout=nprop,
                nhidden=nhidden,
                nlayers=nlayers,
                bias=bias,
                activation=activation,
                apply_layer_norm=self.apply_norm,
            ))

        self.model = ModuleMap(self.target_blocks.keys, modules, out_properties).to(self.device)

    def forward(self, features, target_blocks=None, return_matrix=False):
        if target_blocks is None:
            keys = self.model.in_keys
            warnings.warn('Using training target_blocks; provide test target_blocks for inference.')
        else:
            keys = target_blocks.keys

        feat_blocks = []
        for k in keys:
            feat_blocks.append(map_targetkeys_to_featkeys_integrated(features, k))
        feat_map = mts.TensorMap(keys, feat_blocks)
        pred = self.model.forward(feat_map)

        return pred

    def fit_ridge_analytical(
        self,
        return_matrix=False,
        set_bias=False,
        kernel_ridge=False,
        alphas=None,
        alpha=5e-9,
        cv=3,
    ) -> None:
        is_complex = False
        if alphas is None:
            alphas = np.logspace(-18, 1, 35)

        self.recon = {}
        pred_blocks = []
        self.ridges = []

        for k, block in self.target_blocks.items():
            bias = False
            if k["L"] == 0 and set_bias:
                bias = True

            feat = map_targetkeys_to_featkeys_integrated(self.feats, k)
            nsamples, ncomp, _ = block.values.shape
            feat = _match_feature_and_target_samples(block, feat, return_idx=True)
            assert torch.all(block.samples.values == feat.samples.values[:, :]), (_match_feature_and_target_samples(block, feat))

            if feat.values.is_complex():
                is_complex = True
                x_real = feat.values.real
                x_imag = feat.values.imag
                y_real = block.values.real
                y_imag = block.values.imag
                x = x_real.reshape((x_real.shape[0] * x_real.shape[1], -1)).cpu().numpy()
                y = y_real.reshape((y_real.shape[0] * y_real.shape[1], -1)).cpu().numpy()
                x2 = x_imag.reshape((x_imag.shape[0] * x_imag.shape[1], -1)).cpu().numpy()
                y2 = y_imag.reshape((y_imag.shape[0] * y_imag.shape[1], -1)).cpu().numpy()
            else:
                x = feat.values.reshape((feat.values.shape[0] * feat.values.shape[1], -1)).cpu().numpy()
                y = block.values.reshape(block.values.shape[0] * block.values.shape[1], -1).cpu().numpy()

            if kernel_ridge:
                ridge = KernelRidge(alpha=alpha).fit(x, y)
                if nsamples > 2:
                    gscv = GridSearchCV(ridge, dict(alpha=alphas), cv=cv).fit(x, y)
                    alpha = gscv.best_params_["alpha"]
                ridge = KernelRidge(alpha=alpha).fit(x, y)
            else:
                ridge = RidgeCV(alphas=alphas, fit_intercept=bias).fit(x, y)
                if is_complex:
                    ridge_c = RidgeCV(alphas=alphas, fit_intercept=bias).fit(x2, y2)

            pred = ridge.predict(x)
            self.ridges.append(ridge)
            if is_complex: 
                pred2 = ridge_c.predict(x2)
                self.ridges.append(ridge_c)
                pred_real = pred
                pred_imag = pred2
                pred = pred_real + 1j * pred_imag

            pred_blocks.append(
                TensorBlock(
                    values=torch.from_numpy(pred.reshape((nsamples, ncomp, 1))).to(self.device),
                    samples=block.samples,
                    components=block.components,
                    properties=self.dummy_property,
                )
            )

        pred_tmap = TensorMap(self.target_blocks.keys, pred_blocks)
        self.recon_blocks = self.model_return(pred_tmap, return_matrix=return_matrix)
        return self.recon_blocks, self.ridges

    def predict_ridge_analytical(self, ridges=None, hfeat=None, target_blocks=None):
        if self.ridges is None:
            assert ridges is not None, 'Ridges must be fitted first'
            self.ridges = ridges
        if hfeat is None:
            hfeat = self.feats
            warnings.warn('Using train hfeat, otherwise provide test hfeat')
        if target_blocks is None:
            target_blocks = self.target_blocks
            warnings.warn('Using train target_blocks, otherwise provide test target_blocks')

        pred_blocks = []
        for imdl, (key, tkey) in enumerate(zip(self.ridges, target_blocks.keys)):
            feat = map_targetkeys_to_featkeys_integrated(hfeat, tkey)
            nsamples, ncomp, _ = feat.values.shape
            x = feat.values.reshape((feat.values.shape[0] * feat.values.shape[1], -1)).cpu().numpy()
            pred = self.ridges[imdl].predict(x)
            pred_blocks.append(
                TensorBlock(
                    values=torch.from_numpy(pred.reshape((nsamples, ncomp, 1)))
                    .to(self.device),
                    samples=feat.samples,
                    components=feat.components,
                    properties=self.dummy_property,
                )
            )
        return TensorMap(target_blocks.keys, pred_blocks)

    def regularization_loss(self, regularization):
        return (
            regularization
            * torch.sum(self.layer.weight.T @ self.layer.weight)
            / 1  # normalize by number of samples
        )
    
    def model_return(self, target: torch.ScriptObject, return_matrix=False):
        if not return_matrix:
            return target
        else:
            raise NotImplementedError


def _match_feature_and_target_samples(target_block, feat_block, return_idx=False):
    intersection, idx1, idx2 = feat_block.samples.intersection_and_mapping(target_block.samples)
    if not return_idx:
        idx1 = torch.where(idx1 == -1)[0]
        idx2 = torch.where(idx2 == -1)[0]
        if np.prod(idx1.shape) > 0 and np.prod(idx2.shape) == 0:
            return feat_block.samples.values[idx1]
        elif np.prod(idx2.shape) > 0 and np.prod(idx1.shape) == 0:
            return target_block.samples.values[idx2]
        else:
            return feat_block.samples.values[idx1], target_block.samples.values[idx2]
    else:
        idx1 = torch.where(idx1 != -1)
        idx2 = torch.where(idx2 != -1)
        assert len(idx1) == len(idx2)
        return TensorBlock(values=feat_block.values[idx1],
                           samples=intersection,
                           properties=feat_block.properties,
                           components=feat_block.components)
