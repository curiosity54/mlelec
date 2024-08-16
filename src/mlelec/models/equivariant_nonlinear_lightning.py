# equivariant_nonlinear_lightning.py

import lightning as pl

import numpy as np

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mlelec.models.equivariant_nonlinear_model import EquivariantNonlinearModel
from mlelec.data.derived_properties import compute_eigenvalues, compute_atom_resolved_density, compute_dipoles
from mlelec.utils.pbc_utils import blocks_to_matrix

from typing import Union
from abc import ABC, abstractmethod

# BaseLoss and Custom Loss definitions
class BaseLoss(ABC):
    @abstractmethod
    def compute(self, predictions, targets, **kwargs):
        pass

def compute_difference(tensor1, tensor2):
    """
    Compute the difference between two tensors, masking the larger one to match the size of the smaller one.
    
    Args:
        tensor1 (torch.Tensor): The first tensor (smaller or same size).
        tensor2 (torch.Tensor): The second tensor (larger or same size).
    
    Returns:
        torch.Tensor: The difference tensor.
    """
    # Get the shape of the smaller tensor
    small_shape = tensor1.shape
    # Slice the larger tensor to match the smaller tensor's shape
    slices = tuple(slice(0, dim) for dim in small_shape)
    # Compute the difference
    tensor_diff = tensor1 - tensor2[slices]
    return tensor_diff

def adaptive_weighting_scheme(loss_values, previous_epoch_losses):
    """
    Function to compute adaptive weights based on the current loss values.
    
    Args:
        loss_values (torch.Tensor): A tensor containing the different loss contributions.
        previous_epoch_losses (torch.Tensor): A tensor containing the previous epoch's loss contributions.

    Returns:
        torch.Tensor: A tensor containing the adaptive weights, summing to 1.
    """
    beta = 0.1

    # Debugging: Print the current and previous losses
    # print("Current Losses: ", loss_values)
    # print("Previous Epoch Losses: ", previous_epoch_losses)

    if previous_epoch_losses is None:
        # Handle initial case where there are no previous losses
        weights = torch.ones_like(loss_values) / loss_values.shape[0]
    else:
        s = torch.clip(torch.exp(beta * (loss_values - previous_epoch_losses)), 0, 1e8).detach()
        norm = s.sum()
        weights = s / norm
    # print('weights:',weights)
    return weights

class MSELoss(BaseLoss):
    def compute(self, predictions, targets, **kwargs):
        """L2 loss function"""
        if isinstance(predictions, torch.Tensor):
            assert isinstance(targets, torch.Tensor)
            # assert (
            #     predictions.shape == targets.shape
            # ), "Prediction and targets must have the same shape"
            diff = compute_difference(predictions, targets)
            return torch.norm(diff) ** 2
        
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
                diff = compute_difference(p, t)
                losses.append(torch.norm(diff)**2)
       
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
            diff = compute_difference(predictions, targets)
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
                diff = compute_difference(p, t)
                se.append((diff*diff.conj()).flatten())

        elif isinstance(predictions, list):

            se = []
            for p, t in zip(predictions, targets):
                diff = compute_difference(p, t)
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
        optimizer = None,
        learning_rate: float = 1e-3,
        lr_scheduler_patience: int=10,
        lr_scheduler_factor: float=0.1,
        lr_scheduler_min_lr: float=1e-6,
        loss_fn: BaseLoss = MSELoss(),
        is_indirect: bool = False,
        adaptive_loss_weights: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.automatic_optimization = False
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
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_min_lr = lr_scheduler_min_lr
        self.loss_fn = loss_fn
        
        # Buffers to accumulate losses for the current and previous epochs for adaptive loss weighting
        self.adaptive_loss_weights = adaptive_loss_weights
        self.previous_epoch_losses = None  
        self.current_epoch_losses = []   
        self.current_weights = torch.tensor([1.0])   

        self.derived_pred_kwargs = kwargs
        self.qmdata = mldata.qmdata
        self.is_molecule = mldata.qmdata.is_molecule
        self.is_indirect = is_indirect
        self.optimizer = optimizer
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

    def configure_optimizers(self):
        if self.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  
            scheduler = {
                    'scheduler': ReduceLROnPlateau(optimizer, patience=self.lr_scheduler_patience, factor=self.lr_scheduler_factor, min_lr=self.lr_scheduler_min_lr),
                    'monitor': 'train_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            return [optimizer], [scheduler]
        elif self.optimizer.lower() == 'lbfgs':
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.learning_rate, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
            return [optimizer]  # Return only the optimizer since no scheduler is defined here

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()  # Retrieve the optimizer object

        def closure():
            # Zero the gradients
            optimizer.zero_grad()

            features = batch.features
            targets = batch.fock_blocks
            # predictions = self.forward(features, targets)
            predictions = self.forward(features, self.metadata)
            derived_predictions = self.compute_derived_predictions(predictions, batch, **self.derived_pred_kwargs)

            loss = self.compute_weighted_loss(derived_predictions, batch)

            # Perform backward pass
            loss.backward()
            # self.manual_backward(loss)
            return loss

        if self.optimizer.lower() == 'lbfgs':
            # For LBFGS, pass the closure to the optimizer's step function
            loss = optimizer.step(closure)
        else:
            # For other optimizers, manually call the closure and perform the step
            loss = closure()
            optimizer.step()

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch to store the losses for adaptive weighting.
        """
        # Aggregate losses across all batches in the epoch
        if self.current_epoch_losses:
            epoch_losses = torch.stack(self.current_epoch_losses).sum(dim=0)

        # Update the current weights using the losses from the previous epoch
        if self.adaptive_loss_weights:
            self.current_weights = adaptive_weighting_scheme(epoch_losses, self.previous_epoch_losses)

        self.previous_epoch_losses = epoch_losses.clone()

    def validation_step(self, batch, batch_idx):
        features = batch.features
        targets = batch.fock_blocks
        predictions = self.forward(features, self.metadata)
        derived_predictions = self.compute_derived_predictions(predictions, batch, **self.derived_pred_kwargs)
        derived_metrics = {}

        loss, derived_metrics = self.compute_weighted_loss(derived_predictions, batch, compute_metrics = True)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric_name, metric_value in derived_metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        features = batch.features
        targets = batch.fock_blocks
        predictions = self.forward(features, self.metadata)
        derived_predictions = self.compute_derived_predictions(predictions, batch, **self.derived_pred_kwargs)
       
        loss, derived_metrics = self.compute_weighted_loss(derived_predictions, batch, compute_metrics = True)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric_name, metric_value in derived_metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def compute_weighted_loss(self, derived_predictions, batch, compute_metrics=False):
        """
        Compute the weighted loss using the adaptive weighting scheme.
        
        Args:
            derived_predictions (dict): Dictionary of predicted values.
            batch (Batch): The current batch of data.
        
        Returns:
            torch.Tensor: The weighted sum of losses.
        """
        # print(derived_predictions, )
        loss_contributions = []
        if compute_metrics:
            derived_metrics = {}

        for k, p in derived_predictions.items():
            t = batch._asdict()[k]
            # print(t, 't')
            loss_contributions.append(self.loss_fn.compute(p, t))
            if compute_metrics:
                derived_metrics[f'rmse_{k}'] = RMSE().compute(p, t)

        # Convert loss contributions to a tensor

        # print(loss_contributions, 'lc') 

        loss_contributions = torch.stack(loss_contributions)

        # Compute the weighted sum of losses
        weighted_loss = torch.sum(self.current_weights * loss_contributions)
        
        # Accumulate the losses for the current epoch
        self.current_epoch_losses.append(loss_contributions.detach())

        if compute_metrics:
            return weighted_loss, derived_metrics
        else:
            return weighted_loss

    def compute_derived_predictions(self, predictions, batch, **kwargs):
        # Compute derived predictions based on keyword arguments.
        
        basis = self.model.orbitals
        basis_name = self.model.basis_name
        frames = self.model.frames
        ncore = self.model.ncore
        batch_frames = [frames[i] for i in batch.sample_id]
        frames_dict = {A: f for A, f in zip(batch.sample_id, batch_frames)}

        HT = blocks_to_matrix(predictions, basis, frames_dict, device = self.device, detach = False, check_hermiticity=False)
        # TODO: The next line needs to be handled inside blocks_to_matrix!
        if self.is_molecule:
            H = [h[0,0,0] for h in HT]
            S = batch.overlap_realspace
        else:
            # Bloch sums. TODO: Not very nice to use QMDataset methods here?
            H = self.qmdata.bloch_sum(HT, is_tensor=True)
            # H = [H[i] for i in batch.sample_id]
            S = batch.overlap_kspace
            
        to_return = {}
        target_properties = kwargs.get("target_properties", [])
        for property in target_properties:
            eigenvalues, eigenvectors = compute_eigenvalues(H, S, return_eigenvectors=True)
            if property == 'eigenvalues':
                to_return['eigenvalues'] = eigenvalues
            elif property.lower() == 'atom_resolved_density' or property.lower() == 'ard':
                atom_resolved_density, _ = compute_atom_resolved_density(eigenvectors, batch_frames, basis, ncore)
                to_return['atom_resolved_density'] = atom_resolved_density
            elif property.lower() == 'dipoles':
                dipoles = compute_dipoles(H, S, frames=batch_frames, basis_name=basis_name, basis=basis, unfix=True, requires_grad=True)
                to_return['dipoles'] = dipoles
            elif property.lower() == 'polarizability' or property.lower() == 'polarisability':
                raise NotImplementedError('polarizability not implemented yet')
            else:
                raise NotImplementedError(f'{property} not implemented yet')
    
        return to_return

####
# Maybe move to separate file

import lightning as pl
from mlelec.data.mldataset import MLDataset
import metatensor.torch as mts

class MLDatasetDataModule(pl.LightningDataModule):
    def __init__(self, mldata: MLDataset, batch_size=32, shuffle=False, num_workers=0):
        super().__init__()
        self.collate_fn = mldata.group_and_join
        self.train_dataset = mldata.train_dataset
        self.val_dataset = mldata.val_dataset
        self.test_dataset = mldata.test_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def train_dataloader(self):
        return mts.learn.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return mts.learn.DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return mts.learn.DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)
