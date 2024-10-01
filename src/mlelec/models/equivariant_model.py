import warnings
from typing import List, Union

import metatensor.torch as mts
import numpy as np
import torch
import torch.nn as nn
from metatensor.torch import TensorBlock, TensorMap
from metatensor.torch.learn import ModuleMap
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV

from mlelec.utils.twocenter_utils import map_targetkeys_to_featkeys_integrated


class EquivariantNonLinearity(nn.Module):
    """
    Applies a non-linear activation followed by optional Layer Normalization.
    The non-linear activation is applied to the inverse square root of the sum
    of squares of the input tensor's feature dimension.

    Args:
        nonlinearity (callable): The non-linear activation function to apply.
        epsilon (float): A small value to prevent division by zero in sqrt.
        norm (bool): Whether to apply Layer Normalization.
        layersize (int): The size of the layer (number of features).
        device (str): The device to use for the layer.
    """

    def __init__(
        self,
        nonlinearity: callable = None,
        epsilon=1e-6,
        norm=True,
        layersize=None,
        device=None,
    ):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.epsilon = epsilon
        self.device = device or "cpu"

        # Use LayerNorm if norm is True
        self.norm = nn.LayerNorm([layersize], device=self.device) if norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3, (
            "Input tensor must have 3 dimensions"
            " (batch_size, num_samples, num_features)."
        )

        # Compute the inverse square root of the sum of squares of the input tensor's
        # feature dimension
        x_inv = torch.sqrt(torch.einsum("imf,imf->if", x, x) + self.epsilon)

        # Apply the nonlinearity and optional normalization
        if self.nonlinearity:
            x_inv = self.nonlinearity(x_inv)
        if self.norm:
            x_inv = self.norm(x_inv)

        # Scale the original tensor by the transformed inverse tensor
        return torch.einsum("if,imf->imf", x_inv, x)


class simpleMLP(nn.Module):
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
        self.device = device or "cpu"

        if nlayers == 0:
            # Single linear layer if no hidden layers
            self.mlp = nn.Linear(nin, nout, bias=bias).to(self.device)
        else:
            # Ensure nhidden is a list of appropriate length
            if isinstance(nhidden, int):
                nhidden = [nhidden] * nlayers

            # Build the network layers
            layers = []
            for i in range(nlayers):
                in_dim = nin if i == 0 else nhidden[i - 1]
                out_dim = nhidden[i]

                layers.append(nn.Linear(in_dim, out_dim, bias=bias))

                if activation:
                    nonlinearity = (
                        getattr(nn, activation)()
                        if isinstance(activation, str)
                        else activation
                    )
                    layers.append(
                        EquivariantNonLinearity(
                            nonlinearity=nonlinearity,
                            layersize=out_dim,
                            device=self.device,
                            norm=apply_layer_norm,
                        )
                    )

            # Final output layer
            layers.append(nn.Linear(nhidden[-1], nout, bias=bias))

            # Convert list of layers to a Sequential model
            self.mlp = nn.Sequential(*layers).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with optional Layer Normalization and
    Equivariant Nonlinearity.

    Args:
        nlayers (int): Number of hidden layers.
        nin (int): Number of input features.
        nhidden (Union[int, list]): Number of hidden units in each layer.
        nout (int): Number of output features.
        activation (Union[str, callable]): Activation function.
        bias (bool): Whether to include bias in linear layers.
        device (str): Device to run the model on.
        apply_layer_norm (bool): Whether to apply layer normalization.
    """

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

        self.device = device or "cpu"

        if nlayers == 0:
            # Purely linear model
            self.mlp = nn.Sequential(nn.Linear(nin, nout, bias=bias))
            self._initialize_layer(self.mlp[0])
        else:
            if not isinstance(nhidden, list):
                nhidden = [nhidden] * nlayers
            else:
                assert len(nhidden) == nlayers, "len(nhidden) must be equal to nlayers"

            layers = [self._create_layer(nin, nhidden[0], None, bias, apply_layer_norm)]
            for in_dim, out_dim in zip(nhidden[:-1], nhidden[1:]):
                layers.append(
                    self._create_layer(
                        in_dim, out_dim, activation, bias, apply_layer_norm
                    )
                )
            layers.append(
                self._create_layer(
                    nhidden[-1], nout, activation, bias, apply_layer_norm
                )
            )

            self.mlp = nn.Sequential(
                *[layer for sublist in layers for layer in sublist]
            )

        self.mlp.to(self.device)

    def _create_layer(
        self,
        n_in: int,
        n_out: int,
        activation: Union[str, callable],
        bias: bool,
        apply_layer_norm: bool,
    ) -> List[nn.Module]:
        """
        Create a single layer with optional activation and normalization.

        Args:
            n_in (int): Number of input features.
            n_out (int): Number of output features.
            activation (Union[str, callable]): Activation function.
            bias (bool): Whether to include bias in linear layers.
            apply_layer_norm (bool): Whether to apply layer normalization.

        Returns:
            List[nn.Module]: List of layers including Linear, activation,
            and normalization.
        """
        layers = [nn.Linear(n_in, n_out, bias=bias)]
        # self._initialize_layer(layers[-1])

        if activation:
            if isinstance(activation, str):
                activation = getattr(nn, activation)()
            elif not isinstance(activation, torch.nn.Module):
                raise ValueError(
                    "activation must be a string or a torch.nn.Module instance"
                )
            layers.insert(
                0,
                EquivariantNonLinearity(
                    nonlinearity=activation,
                    layersize=n_in,
                    device=self.device,
                    norm=apply_layer_norm,
                ),
            )

        return layers

    def _initialize_layer(self, layer: nn.Module):
        """
        Initialize the parameters of a layer.

        Args:
            layer (nn.Module): The layer to initialize.
        """
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.mlp(x)


def compute_ncore(basis_dict):
    # Same function as in QMDataset. TODO: avoid code duplication
    ncore = {}
    for s in basis_dict:
        basis = np.array(basis_dict[s])
        nmin = np.min(basis[:, 0])
        ncore[s] = 0
        for n in np.arange(nmin):
            for l_ in range(n):
                ncore[s] += 2 * (2 * l_ + 1)
        llist = set(basis[np.argwhere(basis[:, 0] == nmin)][:, 0, 1])
        llist_nmin = set(range(max(llist) + 1))
        l_diff = llist_nmin - llist
        for l_ in l_diff:
            ncore[s] += 2 * (2 * l_ + 1)
    return ncore


class EquivariantModel(nn.Module):
    """
    A model for equivariant nonlinear transformations with support for ridge regression.

    Args:
        mldata (MLDataset): The dataset containing features, target blocks and metadata.
        nhidden (Union[int, list]): Number of hidden units in each layer.
        nlayers (int): Number of hidden layers.
        activation (Union[str, callable]): Activation function.
        apply_norm (bool): Whether to apply layer normalization.
        **kwargs: Additional arguments for model configuration.
    """

    def __init__(
        self,
        mldata,
        nhidden: Union[int, list],
        nlayers: int,
        activation: Union[str, callable] = "SiLU",
        apply_norm: bool = True,
        set_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.feats = mldata.features
        self.target_blocks = mldata.model_metadata
        self.frames = mldata.structures
        self.orbitals = mldata.model_basis
        self.basis_name = mldata.model_basis_name
        self.ncore = compute_ncore(self.orbitals)
        self.is_molecule = mldata.qmdata.is_molecule
        self.apply_norm = apply_norm
        self.device = mldata.device
        self.dummy_property = self.target_blocks[0].properties
        self.set_bias = set_bias
        self._initialize_submodels(
            set_bias=self.set_bias,
            nhidden=nhidden,
            nlayers=nlayers,
            activation=activation,
        )
        # self.ridges = None

    def _initialize_submodels(self, nhidden=16, nlayers=2, activation=None, **kwargs):
        """
        Initialize submodels for each target block.

        Args:
            set_bias (bool): Whether to set bias for L=0 blocks.
            nhidden (int): Number of hidden units in each layer.
            nlayers (int): Number of hidden layers.
            activation (Union[str, callable]): Activation function.
            **kwargs: Additional arguments for model configuration.
        """
        modules = []
        out_properties = []
        for k, b in self.target_blocks.items():
            nprop = b.values.shape[-1]
            out_properties.append(b.properties)

            feat = map_targetkeys_to_featkeys_integrated(self.feats, k)
            bias = k["L"] == 0 and self.set_bias

            modules.append(
                simpleMLP(
                    nin=feat.values.shape[-1],
                    nout=nprop,
                    nhidden=nhidden,
                    nlayers=nlayers,
                    bias=bias,
                    activation=activation,
                    apply_layer_norm=self.apply_norm,
                )
            )

        self.model = ModuleMap(self.target_blocks.keys, modules, out_properties).to(
            self.device
        )

    def forward(self, features, target_blocks=None, return_matrix=False):
        """
        Forward pass for the model.

        Args:
            features (TensorMap): Input features.
            target_blocks (Optional[TensorMap]): Target blocks for inference.
            return_matrix (bool): Whether to return the prediction as a matrix.

        Returns:
            TensorMap: Predicted tensor map.
        """
        if target_blocks is None:
            keys = self.model.in_keys
            warnings.warn(
                "Using training target_blocks; provide test target_blocks for inference"
            )
        else:
            keys = target_blocks.keys

        feat_blocks = []
        for k in keys:
            feat_blocks.append(map_targetkeys_to_featkeys_integrated(features, k))
        feat_map = mts.TensorMap(keys, feat_blocks)
        pred = self.model.forward(feat_map)

        if return_matrix:
            raise NotImplementedError("`return_matrix not implemented yet")

        return pred

    def fit_ridge_analytical(
        self,
        target_blocks,
        return_matrix=False,
        set_bias=None,
        kernel_ridge=False,
        alphas=None,
        alpha=5e-9,
        cv=3,
    ) -> None:
        """
        Fit the model using ridge regression.

        Args:
            return_matrix (bool): Whether to return the prediction as a matrix.
            set_bias (bool): Whether to set bias for L=0 blocks.
            kernel_ridge (bool): Whether to use kernel ridge regression.
            alphas (Optional[np.ndarray]): Array of alpha values for ridge regression.
            alpha (float): Regularization strength for ridge regression.
            cv (int): Number of cross-validation folds.

        Returns:
            Tuple[TensorMap, List[RidgeCV]]: Fitted tensor map and list of ridge models.
        """
        is_complex = False
        if alphas is None:
            alphas = np.logspace(-18, 1, 35)

        # self.recon = {}
        pred_blocks = []
        self.ridges = {}
        set_bias = set_bias if set_bias is not None else self.set_bias

        for k, block in target_blocks.items():
            bias = False
            if k["L"] == 0 and set_bias:
                bias = True

            feat = map_targetkeys_to_featkeys_integrated(self.feats, k)
            nsamples, ncomp, nprop = block.values.shape
            feat = _match_feature_and_target_samples(block, feat, return_idx=True)
            assert torch.all(
                block.samples.values == feat.samples.values[:, :]
            ), _match_feature_and_target_samples(block, feat)

            if feat.values.is_complex():
                is_complex = True
                x_real = feat.values.real
                x_imag = feat.values.imag
                y_real = block.values.real
                y_imag = block.values.imag
                x = (
                    x_real.reshape((x_real.shape[0] * x_real.shape[1], -1))
                    .cpu()
                    .numpy()
                )
                y = (
                    y_real.reshape((y_real.shape[0] * y_real.shape[1], -1))
                    .cpu()
                    .numpy()
                )
                x2 = (
                    x_imag.reshape((x_imag.shape[0] * x_imag.shape[1], -1))
                    .cpu()
                    .numpy()
                )
                y2 = (
                    y_imag.reshape((y_imag.shape[0] * y_imag.shape[1], -1))
                    .cpu()
                    .numpy()
                )
            else:
                x = (
                    feat.values.reshape(
                        (feat.values.shape[0] * feat.values.shape[1], -1)
                    )
                    .cpu()
                    .numpy()
                )
                y = (
                    block.values.reshape(
                        block.values.shape[0] * block.values.shape[1], -1
                    )
                    .cpu()
                    .numpy()
                )

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
            self.ridges[tuple(k.values.tolist())] = ridge

            if is_complex:
                pred2 = ridge_c.predict(x2)
                self.ridges.append(ridge_c)
                pred_real = pred
                pred_imag = pred2
                pred = pred_real + 1j * pred_imag

            pred_blocks.append(
                TensorBlock(
                    values=torch.from_numpy(pred.reshape((nsamples, ncomp, nprop))).to(
                        self.device
                    ),
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
            )

        pred_tmap = mts.sort(TensorMap(target_blocks.keys, pred_blocks))
        recon_blocks = self.model_return(pred_tmap, return_matrix=return_matrix)
        return recon_blocks  # , self.ridges

    def predict_ridge_analytical(self, ridges=None, hfeat=None, target_blocks=None):
        """
        Predict using the fitted ridge regression models.

        Args:
            ridges (Optional[List[RidgeCV]]): List of fitted ridge models.
            hfeat (Optional[TensorMap]): Features for prediction.
            target_blocks (Optional[TensorMap]): Target blocks for prediction.

        Returns:
            TensorMap: Predicted tensor map.
        """
        if self.ridges is None:
            assert ridges is not None, "Ridges must be fitted first"
            self.ridges = ridges
        if hfeat is None:
            hfeat = self.feats
            warnings.warn("Using train hfeat, otherwise provide test hfeat")
        if target_blocks is None:
            target_blocks = self.target_blocks
            warnings.warn(
                "Using train target_blocks, otherwise provide test target_blocks"
            )

        pred_blocks = []
        for imdl, (key, tkey) in enumerate(
            zip(self.ridges.values(), target_blocks.keys)
        ):
            feat = map_targetkeys_to_featkeys_integrated(hfeat, tkey)
            nsamples, ncomp, _ = feat.values.shape
            x = (
                feat.values.reshape((feat.values.shape[0] * feat.values.shape[1], -1))
                .cpu()
                .numpy()
            )
            pred = self.ridges[imdl].predict(x)
            pred_blocks.append(
                TensorBlock(
                    values=torch.from_numpy(pred.reshape((nsamples, ncomp, 1))).to(
                        self.device
                    ),
                    samples=feat.samples,
                    components=feat.components,
                    properties=self.dummy_property,
                )
            )
        return mts.sort(TensorMap(target_blocks.keys, pred_blocks))

    def init_with_ridge_weights(self, target_blocks=None, **kwargs):

        set_bias = kwargs.get("set_bias", self.set_bias)
        kernel_ridge = kwargs.get("kernel_ridge", False)
        alphas = kwargs.get("alphas", np.logspace(-10, 0, 10))

        if target_blocks is None:
            assert hasattr(self, "ridges"), "You must provide `target_blocks`"
        else:
            _ = self.fit_ridge_analytical(
                target_blocks,
                kernel_ridge=kernel_ridge,
                set_bias=set_bias,
                alphas=alphas,
            )
        for k in self.model.in_keys:
            mlp = self.model.get_module(k).mlp
            ridge = self.ridges[tuple(k.values.tolist())]
            ridge_weights = ridge.coef_
            ridge_bias = ridge.intercept_
            with torch.no_grad():
                mlp.weight.copy_(torch.from_numpy(ridge_weights))
                if mlp.bias is not None:
                    mlp.bias.copy_(torch.tensor(ridge_bias))
                else:
                    assert ridge_bias == 0
        return

    def regularization_loss(self, regularization: float) -> torch.Tensor:
        """
        Compute the regularization loss.

        Args:
            regularization (float): Regularization strength.

        Returns:
            torch.Tensor: Regularization loss.
        """
        return (
            regularization
            * torch.sum(self.layer.weight.T @ self.layer.weight)
            / 1  # normalize by number of samples
        )

    def model_return(self, target: torch.ScriptObject, return_matrix=False):
        """
        Return the model output.

        Args:
            target (torch.ScriptObject): The target object.
            return_matrix (bool): Whether to return the prediction as a matrix.

        Returns:
            torch.ScriptObject: Model output.
        """
        if not return_matrix:
            return target
        else:
            raise NotImplementedError


def _match_feature_and_target_samples(target_block, feat_block, return_idx=False):
    """
    Match feature and target samples.

    Args:
        target_block (TensorBlock): The target block.
        feat_block (TensorBlock): The feature block.
        return_idx (bool): Whether to return the indices of the matching samples.

    Returns:
        Union[TensorBlock, Tuple[torch.Tensor, torch.Tensor]]:
            The matched feature block or a tuple of indices.
    """
    intersection, idx1, idx2 = feat_block.samples.intersection_and_mapping(
        target_block.samples
    )
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
        return TensorBlock(
            values=feat_block.values[idx1],
            samples=intersection,
            properties=feat_block.properties,
            components=feat_block.components,
        )
