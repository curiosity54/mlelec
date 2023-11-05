# Overall train script - assembles required pipeline - trains/loads and saves model
import sys
import os
import torch
import datetime
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils import data

# from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim  # Adam, AdamW, SGD
import hickle
from typing import Optional
import mlelec.metrics as metrics


# from data.dataset import Dataset

# ---data---
# attempt to read data ( structures, targets)
# if not found, calculate them with pyscf
# generate a dataset object

# ---model---
# ---features--- (if necessary)
# attempt to read features if required for prescribed model
# calculate them if necessary

# instantiate model
# Check for a checkpoint

# Train


# save model


class Trainer:
    def __init__(
        self,
        model,
        dataset_split: tuple[
            float, float, float
        ],  # tuple: (train_data, val_data, test_data)
        mol_name: str,
        train_batch_size=64,
        learning_rate=1e-4,
        max_epochs: int = 100000,
        log_interval: int = 100,
        save_interval: Optional[int] = None,
        save_path="./train_results",
        checkpoint: str = "best",  # last, best
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        start_from_checkpoint: bool = False,
        optimizer: Optional[str] = None,
        optimizer_kwargs: Optional[dict] = None,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.mol_name = mol_name
        self.batch_size = train_batch_size
        self.lr = learning_rate
        self.log_interval = log_interval
        if save_interval is None:
            self.save_interval = self.log_interval
        self.max_epochs = max_epochs
        self.save_path = save_path
        self.train_data, self.val_data, self.test_data = dataset_split
        self.data_map = {
            "train": self.train_data,
            "validation": self.val_data,
            "test": self.test_data,
        }
        self.checkpoint = checkpoint

        timezone = datetime.timezone(dt.timedelta(hours=2))
        self.save_path = Path(
            save_path + "/" + datetime.datetime.now(timezone).strftime("%Y%m%d-%H%M%S")
        )
        # self.writer = SummaryWriter(self.save_path + "/" + "_trn")

        self.best_val_loss = 10**10
        if optimizer is not None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = getattr(optim, optimizer)(
                self.model.parameters(), lr=self.lr, **optimizer_kwargs
            )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=100, factor=0.1
        )
        self.epoch = 0

        if start_from_checkpoint:
            try:
                self.load(step=self.checkpoint)
                print("Checkpoint loaded")
            except:
                print(" No checkpoint matches requested checkpoint")

    def save(self, step, best=False):
        _dict = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        # Save the model
        if self.save_all_checkpoints:
            torch.save(_dict, str(self.save_path / f"model-{step}.pt"))
        torch.save(_dict, str(self.save_path / "model-last.pt"))

        # Save the best model if that is the case
        if best:
            torch.save(_dict, str(self.results_folder / "model-best.pt"))

        # Save the arguments
        hickle.dump(self.args, self.save_path + "args.pickle")

    def load(self, step="last"):
        data_dict = torch.load(str(self.save_path / f"model-{step}.pt"))

        self.step = data_dict["step"]
        self.best_val_loss = data_dict["best_val_loss"]
        self.model.load_state_dict(data_dict["model"])
        self.optimizer.load_state_dict(data_dict["opt"])
        self.scheduler.load_state_dict(data_dict["scheduler"])

    def loss(self, metric: str, partition: str = "validation"):
        loss_fn = getattr(metrics, metric)
        with torch.no_grad():
            data = next(self.data_map[partition])[0].to(self.device)
            loss = self.model(data, loss_fn)  # TODO - make sure model returns loss
            self.writer.add_scalar(f"Loss {partition}", loss.item(), self.epoch)
            print(f"Loss {partition} \t {loss.item()}")
        return loss

    def train(self, metric: str):
        loss_fn = getattr(metrics, metric)
        self.model.train()
        with tqdm(start=self.epoch, total=self.max_epochs) as pbar:
            # early_stopping_counter = 0
            while self.epoch < self.max_epochs:
                for batch_idx, data in enumerate(self.train_data):
                    data = data.to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.model(data, loss_fn)
                    loss.backward()
                    self.optimizer.step()
                    # self.writer.add_scalar("Loss train", loss.item(), self.epoch)
                    if batch_idx % self.log_interval == 0:
                        print(
                            f"Train Epoch: {self.epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}"
                        )
                        # self.writer.add_scalar("Loss train", loss.item(), self.epoch)
                    self.epoch += 1
                    pbar.update(1)
                    if self.epoch % self.save_interval == 0:
                        loss_val = self.loss(metric, self.val_data, "validation")
                        new_best = loss_val < self.best_val_loss
                        self.save(self.epoch, best=new_best)
