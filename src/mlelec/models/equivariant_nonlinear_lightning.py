# equivariant_nonlinear_lightning.py

import lightning as pl
import torch
from mlelec.models.equivariant_nonlinear_model import EquivariantNonlinearModel
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
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.derived_pred_kwargs = kwargs
        self.save_hyperparameters({
            'nhidden': nhidden,
            'nlayers': nlayers,
            'activation': activation,
            'apply_norm': apply_norm,
            'learning_rate': learning_rate,
            'loss_fn': type(loss_fn).__name__,
            **kwargs  # Add other necessary arguments if they are pickleable
        })

    def forward(self, features, target_blocks=None, return_matrix=False):
        return self.model(features, target_blocks, return_matrix)

    def training_step(self, batch, batch_idx):
        features = batch.features
        targets = batch.fock_blocks
        predictions = self.forward(features, targets)
        derived_predictions = self.compute_derived_predictions(predictions, **self.derived_pred_kwargs)
        loss = self.loss_fn.compute(predictions, targets, derived_predictions=derived_predictions)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features = batch.features
        targets = batch.fock_blocks
        predictions = self.forward(features, targets)
        derived_predictions = self.compute_derived_predictions(predictions, **self.derived_pred_kwargs)
        loss = self.loss_fn.compute(predictions, targets, derived_predictions=derived_predictions)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        features = batch.features
        targets = batch.fock_blocks
        predictions = self.forward(features, targets)
        derived_predictions = self.compute_derived_predictions(predictions, **self.derived_pred_kwargs)
        loss = self.loss_fn.compute(predictions, targets, derived_predictions=derived_predictions)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def compute_derived_predictions(self, predictions, **kwargs):
        # Compute derived predictions based on keyword arguments.
        if kwargs.get('mean_derived', False):
            return torch.mean(predictions, dim=-1, keepdim=True)
        # Add more custom derived prediction logic here
        return predictions


import lightning as pl
from metatensor.learn import DataLoader
from mlelec.data.mldataset import MLDataset

class MLDatasetDataModule(pl.LightningDataModule):
    def __init__(self, mldata: MLDataset, batch_size=32):
        super().__init__()
        self.collate_fn = mldata.group_and_join
        self.train_dataset = mldata.train_dataset
        self.val_dataset = mldata.val_dataset
        self.test_dataset = mldata.test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
