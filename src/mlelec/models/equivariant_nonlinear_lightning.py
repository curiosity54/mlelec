# equivariant_nonlinear_lightning.py

import lightning as pl
import torch
import numpy as np
from mlelec.models.equivariant_nonlinear_model import EquivariantNonlinearModel
from mlelec.data.derived_properties import compute_eigenvalues, compute_atom_resolved_density
from mlelec.utils.pbc_utils import blocks_to_matrix

from typing import Union
from abc import ABC, abstractmethod

# BaseLoss and Custom Loss definitions
class BaseLoss(ABC):
    @abstractmethod
    def compute(self, predictions, targets, **kwargs):
        pass

class MSELoss(BaseLoss):
    def compute(self, predictions, targets, **kwargs):
        """L2 loss function"""
        if isinstance(predictions, torch.Tensor):
            assert isinstance(targets, torch.Tensor)
            assert (
                predictions.shape == targets.shape
            ), "Prediction and targets must have the same shape"
            return torch.norm(predictions - targets) ** 2
        
        elif isinstance(predictions, torch.ScriptObject):
            if predictions._type().name() == "TensorMap":
                assert isinstance(targets, torch.ScriptObject) and targets._type().name() == "TensorMap", "Target must be a TensorMap if prediction is a TensorMap"
            losses = []
            for key, block in predictions.items():
                targetblock = targets.block(key)
                assert (
                    block.samples == targetblock.samples
                ), "Prediction and target must have the same samples"
                losses.append(torch.norm(block.values - targetblock.values)**2)

        elif isinstance(predictions, dict):
            assert isinstance(targets, dict), "Target must be a dictionary"
            losses = []
            for key, p in predictions.items():
                t = targets[key]
                losses.append(torch.norm(p - t)**2)

        elif isinstance(predictions, list):

            losses = []
            for p, t in zip(predictions, targets):
                losses.append(torch.norm(p - t)**2)

       
        return sum(losses)
    
class RMSE(BaseLoss):
    def compute(self, predictions, targets, **kwargs):
        """L2 loss function"""
        if isinstance(predictions, torch.Tensor):
            assert isinstance(targets, torch.Tensor)
            assert (
                predictions.shape == targets.shape
            ), "Prediction and targets must have the same shape"
            # return torch.norm(predictions - targets) ** 2
            diff = predictions - targets
            return np.sqrt(torch.mean(diff*diff.conj()).detach())
        
        elif isinstance(predictions, torch.ScriptObject):
            if predictions._type().name() == "TensorMap":
                assert isinstance(targets, torch.ScriptObject) and targets._type().name() == "TensorMap", "Target must be a TensorMap if prediction is a TensorMap"
            se = []
            for key, block in predictions.items():
                targetblock = targets.block(key)
                assert (
                    block.samples == targetblock.samples
                ), "Prediction and target must have the same samples"
                diff = block.values - targetblock.values
                se.append((diff*diff.conj()).flatten())

        elif isinstance(predictions, dict):
            assert isinstance(targets, dict), "Target must be a dictionary"
            se = []
            for key, p in predictions.items():
                t = targets[key]
                diff = p - t
                se.append((diff*diff.conj()).flatten())

        elif isinstance(predictions, list):

            se = []
            for p, t in zip(predictions, targets):
                diff = p - t
                se.append((diff*diff.conj()).flatten())


        return np.sqrt(torch.mean(torch.cat(se)).detach())

class CustomDerivedLoss(BaseLoss):
    def compute(self, predictions, targets, **kwargs):
        derived_predictions = kwargs.get('derived_predictions')
        derived_diff = predictions - derived_predictions
        return torch.mean(torch.square(derived_diff))

# EquivariantNonlinearLightningModel definition
class LitEquivariantNonlinearModel(pl.LightningModule):
    def __init__(
        self,
        mldata,
        nhidden: Union[int, list],
        nlayers: int,
        activation: Union[str, callable] = 'SiLU',
        apply_norm: bool = True,
        learning_rate: float = 1e-3,
        loss_fn: BaseLoss = MSELoss(),
        is_indirect: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.model = EquivariantNonlinearModel(
            mldata=mldata,
            nhidden=nhidden,
            nlayers=nlayers,
            activation=activation,
            apply_norm=apply_norm,
            **kwargs
        )
        self.model = self.model.double()
        self.metadata = mldata.model_metadata
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.derived_pred_kwargs = kwargs
        self.qmdata = mldata.qmdata
        self.is_molecule = mldata.qmdata.is_molecule
        self.is_indirect = is_indirect
        self.save_hyperparameters({
            'nhidden': nhidden,
            'nlayers': nlayers,
            'activation': activation,
            'apply_norm': apply_norm,
            'learning_rate': learning_rate,
            'loss_fn': type(loss_fn).__name__,
            'is_indirect': is_indirect,
            **kwargs  # Add other necessary arguments if they are pickleable
        })

    def forward(self, features, target_blocks=None, return_matrix=False):
        return self.model(features, target_blocks, return_matrix)

    def training_step(self, batch, batch_idx):
        features = batch.features
        targets = batch.fock_blocks
        predictions = self.forward(features, self.metadata)
        derived_predictions = self.compute_derived_predictions(predictions, batch, **self.derived_pred_kwargs)
        loss = 0
        for k, p in derived_predictions.items():
            t = batch._asdict()[k]
            loss = loss + self.loss_fn.compute(p, t)
        # loss = self.loss_fn.compute(predictions, targets, derived_predictions=derived_predictions)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features = batch.features
        targets = batch.fock_blocks
        predictions = self.forward(features, self.metadata)
        derived_predictions = self.compute_derived_predictions(predictions, batch, **self.derived_pred_kwargs)
        derived_metrics = {}

        loss = 0
        for k, p in derived_predictions.items():
            t = batch._asdict()[k]
            loss = loss + self.loss_fn.compute(p, t)
            derived_metrics[f'rmse_{k}'] = RMSE().compute(p, t)
        # loss = self.loss_fn.compute(predictions, targets, derived_predictions=derived_predictions)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric_name, metric_value in derived_metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        features = batch.features
        targets = batch.fock_blocks
        predictions = self.forward(features, self.metadata)
        derived_predictions = self.compute_derived_predictions(predictions, batch, **self.derived_pred_kwargs)
        derived_metrics = {}

        loss = 0
        for k, p in derived_predictions.items():
            t = batch._asdict()[k]
            loss = loss + self.loss_fn.compute(p, t)
            derived_metrics[f'rmse_{k}'] = RMSE().compute(p, t)


        # loss = self.loss_fn.compute(predictions, targets, derived_predictions=derived_predictions)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric_name, metric_value in derived_metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def compute_derived_predictions(self, predictions, batch, **kwargs):
        # Compute derived predictions based on keyword arguments.
        
        basis = self.model.orbitals
        frames = self.model.frames
        ncore = self.model.ncore

        HT = blocks_to_matrix(predictions, basis, frames, device = self.device, detach = False, check_hermiticity=False)
        # TODO: The next line needs to be handled inside blocks_to_matrix!
        if self.is_molecule:
            H = [HT[i][0,0,0] for i in batch.sample_id]
            S = batch.overlap_realspace
        else:
            HT = [HT[i] for i in batch.sample_id]
            # Bloch sums. TODO: Not very nice to use QMDataset methods here?
            H = self.qmdata.bloch_sum(HT, is_tensor=True)
            S = batch.overlap_kspace

        to_return = {}
        target_atom_resolved_density = kwargs.get('atom_resolved_density', False)
        target_eigenvalues = kwargs.get("eigenvalues", False)
        if target_atom_resolved_density or target_eigenvalues:
            eigsys = compute_eigenvalues(H, S, return_eigenvectors=target_atom_resolved_density)
            if target_atom_resolved_density:
                eigenvalues, eigenvectors = eigsys
                atom_resolved_density, _ = compute_atom_resolved_density(eigenvectors, frames, basis, ncore)
                to_return['atom_resolved_density'] = atom_resolved_density
            else:
                eigenvalues = eigsys
            if target_eigenvalues:
                to_return['eigenvalues'] = eigenvalues
       
        return to_return

####
# Maybe move to separate file

import lightning as pl
from metatensor.learn import DataLoader
from mlelec.data.mldataset import MLDataset

class MLDatasetDataModule(pl.LightningDataModule):
    def __init__(self, mldata: MLDataset, batch_size=32, shuffle=False):
        super().__init__()
        self.collate_fn = mldata.group_and_join
        self.train_dataset = mldata.train_dataset
        self.val_dataset = mldata.val_dataset
        self.test_dataset = mldata.test_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
