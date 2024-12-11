import os

import torch
from tqdm import tqdm
from mlelec.metrics import loss_fn_combined


class Trainer:
    def __init__(self, model, optimizer, scheduler, device):
        """
        Initialize the trainer class.

        Args:
            model: PyTorch model to train and validate.
            optimizer: Optimizer for training.
            loss_fn: Loss function to use.
            device: Device to use ('cuda' or 'cpu').
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def train_step(
        self,
        dataloader,
        ml_data,
        all_mfs,
        loss_fn,
        ref_eva,
        ref_dipole,
        ref_polar,
        var_eva,
        var_dipole,
        var_polar,
        weight_eva,
        weight_dipole,
        weight_polar,
        ORTHOGONAL,
    ):

        self.model.train()  # Set model to training mode
        train_loss = 0
        train_loss_eva = 0
        train_loss_polar = 0
        train_loss_dipole = 0

        for data in dataloader:
            self.optimizer.zero_grad()
            idx = data["idx"]

            # Forward pass
            pred = self.model(
                data["input"],
                return_type="tensor",
                batch_indices=[i.item() for i in idx],
            )
            train_polar_ref = ref_polar[[i.item() for i in idx]]
            train_dip_ref = ref_dipole[[i.item() for i in idx]]
            train_eva_ref = [ref_eva[j][: pred[i].shape[0]] for i, j in enumerate(idx)]

            loss, loss_eva, loss_dipole, loss_polar = loss_fn_combined(
                ml_data,
                pred,
                ORTHOGONAL,
                all_mfs,
                idx,
                loss_fn,
                data["frames"],
                train_eva_ref,
                train_dip_ref,
                train_polar_ref,
                var_eva,
                var_dipole,
                var_polar,
                weight_eva,
                weight_dipole,
                weight_polar,
            )

            train_loss += loss.item()
            train_loss_eva += loss_eva.item()
            train_loss_polar += loss_polar.item()
            train_loss_dipole += loss_dipole.item()

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

        avg_train_loss = train_loss / len(dataloader)
        avg_train_loss_eva = train_loss_eva / len(dataloader)
        avg_train_loss_polar = train_loss_polar / len(dataloader)
        avg_train_loss_dipole = train_loss_dipole / len(dataloader)

        # losses.append(avg_train_loss)
        # losses_eva.append(avg_train_loss_eva)
        # losses_polar.append(avg_train_loss_polar)
        # losses_dipole.append(avg_train_loss_dipole)

        lr = self.optimizer.param_groups[0]["lr"]

        return {
            "lr": lr,
            "train_loss": avg_train_loss,
            "train_loss_eva": avg_train_loss_eva,
            "train_loss_polar": avg_train_loss_polar,
            "train_loss_dipole": avg_train_loss_dipole,
        }

    def validation_step(
        self,
        dataloader,
        ml_data,
        all_mfs,
        loss_fn,
        ref_eva,
        ref_dipole,
        ref_polar,
        var_eva,
        var_dipole,
        var_polar,
        weight_eva,
        weight_dipole,
        weight_polar,
        ORTHOGONAL,
    ):

        self.model.eval()  # Set model to training mode
        val_loss = 0
        val_loss_eva = 0
        val_loss_polar = 0
        val_loss_dipole = 0

        for data in dataloader:
            self.optimizer.zero_grad()
            idx = data["idx"]

            # Forward pass
            pred = self.model(
                data["input"],
                return_type="tensor",
                batch_indices=[i.item() for i in idx],
            )
            val_polar_ref = ref_polar[[i.item() for i in idx]]
            val_dip_ref = ref_dipole[[i.item() for i in idx]]
            val_eva_ref = [ref_eva[j][: pred[i].shape[0]] for i, j in enumerate(idx)]

            vloss, vloss_eva, vloss_dipole, vloss_polar = loss_fn_combined(
                ml_data,
                pred,
                ORTHOGONAL,
                all_mfs,
                idx,
                loss_fn,
                data["frames"],
                val_eva_ref,
                val_dip_ref,
                val_polar_ref,
                var_eva,
                var_dipole,
                var_polar,
                weight_eva,
                weight_dipole,
                weight_polar,
            )

            val_loss += vloss.item()
            val_loss_eva += vloss_eva.item()
            val_loss_polar += vloss_polar.item()
            val_loss_dipole += vloss_dipole.item()

        avg_val_loss = val_loss / len(dataloader)
        avg_val_loss_eva = val_loss_eva / len(dataloader)
        avg_val_loss_polar = val_loss_polar / len(dataloader)
        avg_val_loss_dipole = val_loss_dipole / len(dataloader)

        # losses.append(avg_train_loss)
        # losses_eva.append(avg_train_loss_eva)
        # losses_polar.append(avg_train_loss_polar)
        # losses_dipole.append(avg_train_loss_dipole)

        return {
            "val_loss": avg_val_loss,
            "val_loss_eva": avg_val_loss_eva,
            "val_loss_polar": avg_val_loss_polar,
            "val_loss_dipole": avg_val_loss_dipole,
        }

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        patience,
        save_path,
        verbose,
        dump,
        **kwargs,
    ):

        history = []
        best_val_loss = float("inf")
        epochs_no_improve = 0

        iterator = tqdm(range(epochs), ncols=120)
        for epoch in iterator:

            train_metrics = self.train_step(train_loader, **kwargs)

            val_metrics = self.validation_step(val_loader, **kwargs)

            epoch_metrics = {"epoch": epoch, **train_metrics, **val_metrics}

            history.append(epoch_metrics)

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                epochs_no_improve = 0

            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs.")

            # Early stopping
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

            if epoch % verbose == 0:
                iterator.set_postfix(
                    {
                        "train_loss": train_metrics["train_loss"],
                        "Val_loss": val_metrics["val_loss"],
                        "lr": train_metrics["lr"],
                    }
                )

            # Save the model every n epochs
            if epoch % dump == 0:
                checkpoint_path = os.path.join(save_path, f"model_epoch{epoch}.pt")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        return history
