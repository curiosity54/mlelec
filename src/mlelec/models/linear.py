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
import warnings


def norm_over_components(x):
    x = x.swapaxes(1, 2)
    return torch.einsum("ifm, mfi-> if", x, x.T) + 1e-7


def clamped_norm(x, clamp: float):
    # this potentially causes instability/nan's
    # return x.norm(p=2, dim=-1, keepdim=True).clamp(min=clamp)
    # x = x.swapaxes(1, 2)
    return torch.einsum("imf, imf-> if", x, x) + 1e-7
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
    def __init__(self, nonlinearity: callable = None, epsilon=1e-6, norm = True, layersize = None, device=None):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.epsilon = epsilon
        if device is None:
            device = 'cpu'
        self.device = device
        
        if norm:
            self.nn = [
                nn.LayerNorm(layersize, device = self.device),
                self.nonlinearity,
                nn.LayerNorm(layersize, device = self.device)
            ]
        else:
            self.nn = [self.nonlinearity]

        self.nn = nn.Sequential(*self.nn)
        
    def forward(self, x):

        assert len(x.shape) == 3

        x_inv = torch.einsum("imf,imf->if", x, x)#.flatten()
        x_inv = self.nn(x_inv)
        out = torch.einsum("if, imf->imf", x_inv, x) #/norm

        return out


def _norm_layer(x, norm_clamp=2e-12,  nonlinearity: callable = None):
    """
    x: torch.tensor of shape (nstr, ncomponents, nfeatures)

    returns: torch.tensor of shape (nstr, nfeatures) i.e. compute norm of features
    """
    norm = clamped_norm(x, norm_clamp)
    group_norm = nn.GroupNorm(num_groups=1, num_channels=x.shape[-1])
    if nonlinearity is not None:
        # Transform the norms only
        norm = nonlinearity(group_norm(norm.squeeze(-1))).unsqueeze(-1)

    # assert (
    #         len(x.shape) == 3
    #     ), "Input tensor must be of shape (nstr, ncomponents, nfeatures)"
    #     norm = torch.einsum("imq,imq->iq", x, x)
    # norm = torch.linalg.norm(_symm)
    return norm


class BlockModel(nn.Module):
    "other custom models"
    ## Could be the MLP as below or other custom model
    # for now - we just use MLP in the model

    def __init__(self):
        super().__init__()
        pass

# from mlelec.models.nn import EquiLayerNorm
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
        apply_layer_norm = False, 
        # activation_with_linear = False,
    ):
        super().__init__()

        if nlayers == 0:
            # Purely linear model
            self.mlp = [nn.Linear(nin, nout, bias=bias)]

        else:

            if not isinstance(nhidden, list):
                nhidden = [nhidden] * nlayers
            else:
                assert len(nhidden) == nlayers, "len(nhidden) must be equal to nlayers"

            # Input layer
            self.mlp = [nn.Linear(nin, nhidden[0], bias = bias)]
            
            # Hidden layers
            last_n = nhidden[0]
            for n in nhidden[1:]:
                self.mlp.extend(self.middle_layer(last_n, n, activation, bias, device, apply_layer_norm))
                last_n = n
            
            # Output layer
            # self.mlp.append(nn.Linear(nhidden[-1], nout, bias=bias))
            self.mlp.extend(self.middle_layer(last_n, nout, activation, bias, device, apply_layer_norm))
                            
        self.mlp = nn.Sequential(*self.mlp)
        self.mlp.to(device)

    def middle_layer(self, 
                     n_in, 
                     n_out, 
                     activation = None, 
                     bias = False, 
                     device = 'cpu',
                     apply_layer_norm = False,
                     ):
        
        if activation is None:
            return [nn.Linear(n_in, n_out, bias = bias)]
        
        else:
            if isinstance(activation, str):
                activation = getattr(nn, activation)()
            elif not issubclass(activation, torch.nn.Module):
                raise ValueError('activation must be a string or a torch.nn.Module instance')
                
            return [EquivariantNonLinearity(nonlinearity = activation, 
                                            device = device, 
                                            norm = apply_layer_norm, 
                                            layersize = n_in),
                    nn.Linear(n_in, n_out, bias = bias)]

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
        apply_layer_norm = False
        if len(self.dataset.target.block_keys) == 1:
            self.model = MLP(
                nlayers=kwargs.get("nlayers", 2),
                nin=features.values.shape[-1],
                nout=1,
                nhidden=kwargs.get("nhidden", 10),
                bias=set_bias,
                device=self.device,
                apply_layer_norm = False,
            )
        else:
            for k in self.dataset.target.block_keys:
                bias = False
                feat = map_targetkeys_to_featkeys(features, k)
                # print(feat,tuple(k))
                # print(feat.values.device, self.device)
                if k["L"] == 0 and set_bias:
                    bias = True
                    apply_layer_norm = False
                self.submodels[str(tuple(k))] = MLP(
                    nin=feat.values.shape[-1],
                    nout=1,
                    nhidden=kwargs.get("nhidden", 10),
                    nlayers=kwargs.get("nlayers", 2),
                    bias=bias,
                    apply_layer_norm = apply_layer_norm,
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
        if return_type == "coupled_blocks":
            return pred_tmap
        elif return_type == "uncoupled_blocks":
            self.reconstructed_uncoupled = _to_uncoupled_basis(
                pred_tmap, device=self.device
            )
            return self.reconstructed_uncoupled
        elif return_type == "tensor":
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
            batch_frames = [self.dataset.target.frames[i] for i in batch_indices]
            # batch_frames = self.dataset.target.frames
            self.reconstructed_tensor = _to_matrix(
                self.reconstructed_uncoupled,
                batch_frames,
                self.dataset.target.orbitals,
                device=self.device,
            )
            return self.reconstructed_tensor
        elif return_type == "loss":
            loss = loss_fn(self.reconstructed_tensor, self.dataset.target.tensor)
        else:
            raise NotImplementedError("return_type not implemented")
        return loss

    def fit_ridge_analytical(self, set_bias=False) -> None:
        from sklearn.linear_model import RidgeCV

        # set_bias will set bias=True for the invariant model
        self.recon = {}

        pred_blocks = []
        self.ridges = []
        # kernels = []
        for k, block in self.dataset.target_train.items():
            blockval = torch.linalg.norm(block.values)
            bias = False
            if True:  # blockval > 1e-10:
                if k["L"] == 0 and set_bias:
                    bias = True
                sample_names = block.samples.names
                feat = map_targetkeys_to_featkeys(self.dataset.feat_train, k)

                targetnorm = torch.linalg.norm(block.values)
                nsamples, ncomp, nprops = block.values.shape
                # nsamples, ncomp, nprops = feat.values.shape
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
                    alphas=np.logspace(-15, -1, 40), fit_intercept=bias
                ).fit(x, y)
                # print(ridge.intercept_, np.mean(ridge.coef_), ridge.alpha_)
                pred = ridge.predict(x)
                self.ridges.append(ridge)

                pred_blocks.append(
                    TensorBlock(
                        values=torch.from_numpy(pred.reshape((nsamples, ncomp, 1))).to(
                            self.device
                        ),
                        samples=block.samples,
                        components=block.components,
                        properties=self.dummy_property,
                    )
                )
            else:
                pred_blocks.append(
                    TensorBlock(
                        values=block.values,
                        samples=block.samples,
                        components=block.components,
                        properties=block.properties,
                    )
                )

                # block.copy())

        pred_tmap = TensorMap(self.dataset.target.blocks.keys, pred_blocks)
        self.recon_blocks = pred_tmap  # return_matrix=return_matrix)

        return self.recon_blocks, self.ridges

    def predict_ridge_analytical(self, return_matrix=False) -> None:
        from sklearn.linear_model import RidgeCV

        # set_bias will set bias=True for the invariant model
        self.recon_val = {}

        pred_blocks_val = []
        # kernels = []
        for imdl, tkey in enumerate(self.dataset.target_val.keys):
            target = self.dataset.target_val.block(tkey)
            nsamples, ncomp, nprops = target.values.shape

            feat = map_targetkeys_to_featkeys(self.dataset.feat_val, tkey)
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
            pred_blocks_val.append(
                TensorBlock(
                    values=torch.from_numpy(pred.reshape((nsamples, ncomp, 1))).to(
                        self.device
                    ),
                    samples=target.samples,
                    components=target.components,
                    properties=self.dummy_property,
                )
            )

        pred_tmap_val = TensorMap(self.dataset.target.blocks.keys, pred_blocks_val)
        self.recon_blocks_val = pred_tmap_val
        return self.recon_blocks_val

class LinearModelPeriodic(nn.Module):
    def __init__(
        self,
        twocfeat,
        target_blocks,
        frames,
        orbitals,
        device=None,
        apply_norm = False,
        **kwargs,
    ):
        super().__init__()
        self.feats = twocfeat
        self.target_blocks = target_blocks
        self.frames = frames
        self.orbitals = orbitals
        self.apply_norm = apply_norm
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.dummy_property = self.target_blocks[0].properties
        self._submodels(set_bias=kwargs.get("bias", False), **kwargs)

    def _submodels(self, set_bias=False, **kwargs):
        self.blockmodels = {}
        for k in self.target_blocks.keys:
            bias = False
            if k["L"] == 0 and set_bias:
                bias = True
            blockval = torch.linalg.norm(self.target_blocks[k].values)
            # if  blockval > 1e-10:
            if True:  # <<<<<<<<<<<<<<<<<<<<<

                feat = map_targetkeys_to_featkeys(self.feats, k)
                self.blockmodels[str(tuple(k))] = MLP(
                    nin=feat.values.shape[-1],
                    nout=1,
                    nhidden=kwargs.get("nhidden", 16),
                    nlayers=kwargs.get("nlayers", 2),
                    bias=bias,
                    activation=kwargs.get("activation", None),
                    # activation_with_linear=kwargs.get("activation_with_linear", False),
                    apply_layer_norm=self.apply_norm,
                )
        self.model = torch.nn.ModuleDict(self.blockmodels)
        self.model.to(self.device)

    def forward(self, return_matrix=False):
        self.recon = {}

        pred_blocks = []

        for k, block in self.target_blocks.items():
            # print(k)
            blockval = torch.linalg.norm(block.values)
            if True:
                # if blockval > 1e-10:
                # sample_names = block.samples.names
                feat = map_targetkeys_to_featkeys(self.feats, k)
                # feat = _match_feature_and_target_samples(block, map_targetkeys_to_featkeys(self.feats, k), return_idx=True)

                # nsamples, ncomp, nprops = block.values.shape
                # nsamples, ncomp, nprops = feat.values.shape
                pred = self.blockmodels[str(tuple(k))](feat.values)

                pred_blocks.append(
                    TensorBlock(
                        values=pred,#.reshape((nsamples, ncomp, 1)),
                        samples=feat.samples,
                        components= block.components,
                        properties=self.dummy_property,
                    )
                )
            else:
                pred_blocks.append(block.copy())
        pred_tmap = TensorMap(self.target_blocks.keys, pred_blocks)
        self.recon_blocks = self.model_return(pred_tmap, return_matrix=return_matrix)
        return self.recon_blocks
        # _to_matrix(_to_uncoupled_basis(pred_sum_dict[s]), frames = self.frames, orbitals=self.orbitals)

    def predict(self, features, target_blocks, return_matrix=False):
        pred_blocks = []
        for k, block in target_blocks.items():
            # print(k)
            blockval = torch.linalg.norm(block.values)
            if True:
                # if blockval > 1e-10:
                sample_names = block.samples.names
                feat = map_targetkeys_to_featkeys(features, k)
                # feat = _match_feature_and_target_samples(block, map_targetkeys_to_featkeys(features, k), return_idx=True) # FIXME: return_idx does the opposite of its name?

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
                raise NotImplementedError
                # pred_blocks.append(block.copy())
        pred_tmap = TensorMap(target_blocks.keys, pred_blocks)
        recon_blocks = self.model_return(pred_tmap, return_matrix=return_matrix)
        return recon_blocks

    def model_return(self, target: TensorMap, return_matrix=False):
        if not return_matrix:
            return target
        else:
            raise NotImplementedError
        recon_blocks = {}

        for translation in self.cell_shifts:  # TODOD <<<< FIX
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

    def fit_ridge_analytical(
        self,
        return_matrix=False,
        set_bias=False,
        kernel_ridge=False,
        alphas=None,
        alpha=5e-9,
        cv=3,
    ) -> None:
        if alphas is None:
            alphas = np.logspace(-18, 1, 35)
        from sklearn.linear_model import RidgeCV
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import GridSearchCV

        # set_bias will set bias=True for the invariant model
        self.recon = {}

        pred_blocks = []
        self.ridges = []
        for k, block in self.target_blocks.items():
            blockval = torch.linalg.norm(block.values)
            bias = False
            if True:  # blockval > 1e-10:
                if k["L"] == 0 and set_bias:
                    bias = True
                feat = map_targetkeys_to_featkeys(self.feats, k)
                nsamples, ncomp, _ = block.values.shape

                # print(_match_feature_and_target_samples(block, feat))
                feat = _match_feature_and_target_samples(block, feat, return_idx=True)
                assert np.all(block.samples.values == feat.samples.values[:, :6]), (_match_feature_and_target_samples(block, feat))

                x = ((feat.values.reshape((feat.values.shape[0] * feat.values.shape[1], -1))).cpu().numpy())
                y = ((block.values.reshape(block.values.shape[0] * block.values.shape[1], -1)).cpu().numpy())

                if kernel_ridge:
                    # warnings.warn("Using KernelRidge")
                    ridge = KernelRidge(alpha=alpha).fit(x, y)
                    if nsamples > 2:
                        gscv = GridSearchCV(ridge, dict(alpha=alphas), cv=cv).fit(x, y)
                        alpha = gscv.best_params_["alpha"]
                    else:
                        alpha = alpha
                    ridge = KernelRidge(alpha=alpha).fit(x, y)
                else:
                    # warnings.warn("Using RidgeCV")
                    ridge = RidgeCV(alphas=alphas, fit_intercept=bias).fit(x, y)
                # print(ridge.intercept_, np.mean(ridge.coef_), ridge.alpha_)

                pred = ridge.predict(x)
                self.ridges.append(ridge)

                pred_blocks.append(
                    TensorBlock(
                        values=torch.from_numpy(pred.reshape((nsamples, ncomp, 1))).to(
                            self.device
                        ),
                        samples=block.samples,
                        components=block.components,
                        properties=self.dummy_property,
                    )
                )
            else:
                pred_blocks.append(
                    TensorBlock(
                        values=block.values,
                        samples=block.samples,
                        components=block.components,
                        properties=block.properties,
                    )
                )

                # block.copy())

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
            # k = Labels( targetkeynames, values =np.array(eval(key)).reshape(1,-1))
            # nsamples, ncomp, nprops = target.values.shape
            
            feat = map_targetkeys_to_featkeys(hfeat, tkey)
            nsamples, ncomp, _ = feat.values.shape
            x = ((feat.values.reshape((feat.values.shape[0] * feat.values.shape[1], -1))/1).cpu().numpy())
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
            / 1  # len(self.feats.samples) # normalize by number of samples
        )


def _match_feature_and_target_samples(target_block, feat_block, return_idx = False):
    intersection, idx1, idx2 = feat_block.samples.intersection_and_mapping(target_block.samples)
    
    if not return_idx:
        idx1 = np.where(idx1 == -1)
        idx2 = np.where(idx2 == -1)
        if np.prod(np.shape(idx1)) > 0 and np.prod(np.shape(idx2)) == 0:
            return feat_block.samples.values[idx1]
        elif np.prod(np.shape(idx2)) > 0 and np.prod(np.shape(idx1)) == 0:
            return target_block.samples.values[idx2]
        else:
            return feat_block.samples.values[idx1], target_block.samples.values[idx2]
    else:
        idx1 = np.where(idx1 != -1)
        idx2 = np.where(idx2 != -1)
        assert len(idx1) == len(idx2)
        return TensorBlock(values = feat_block.values[idx1],
                           samples = intersection,
                           properties = feat_block.properties,
                           components = feat_block.components)
        
