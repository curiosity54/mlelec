# train utility - assembles required pipeline - trains/loads and saves model
import sys
import os
import torch
import datetime
from tqdm.auto import tqdm
from torch.utils import data

# from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim  # Adam, AdamW, SGD
import hickle
from typing import Optional
import mlelec.metrics as metrics


class ModelTrainer:
    def __init__(
        self,
        model,
        dataset_split: tuple,  # tuple: (train_data, val_data, test_data)
        mol_name: str,
        train_batch_size=64,
        learning_rate=1e-4,
        max_epochs: int = 100000,
        log_interval: int = 100,
        save_path="./train_results",
        checkpoint: str = "best",  # last, best or number to begin
        save_all_checkpoints: bool = False,
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
        self.max_epochs = max_epochs
        self.save_path = save_path
        self.train_data, self.val_data, self.test_data = dataset_split
        self.data_map = {
            "train": self.train_data,
            "validation": self.val_data,
            "test": self.test_data,
        }
        self.checkpoint = checkpoint
        self.save_all_checkpoints = save_all_checkpoints
        self.save_path = os.path.join(
            save_path, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
            self.optimizer, patience=100, factor=0.1
        )
        self.epoch = 0

        if start_from_checkpoint:
            try:
                self.load(step=self.checkpoint)
                print("Checkpoint loaded")
            except:
                print(" No checkpoint matches requested checkpoint")

    def save(self, step=0, current_best=False):
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
        if current_best:
            torch.save(_dict, str(self.save_path / "model-best.pt"))

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

    def train(self, metric: str, early_stop_criteria=50):
        loss_fn = getattr(metrics, metric)
        self.model.train()
        with tqdm(start=self.epoch, total=self.max_epochs) as pbar:
            early_stop_count = 0
            while self.epoch < self.max_epochs:
                train_loss = 0
                for batch_idx, data in enumerate(self.train_data):
                    # data = data.to(self.device)
                    self.optimizer.zero_grad()
                    pred = self.model(
                        data["input"], return_type="tensor", batch_indices=data["idx"]
                    )
                    loss = loss_fn(pred, data["output"])
                    loss.backward()
                    self.optimizer.step()
                    # self.writer.add_scalar("Loss train", loss.item(), self.epoch)
                    self.epoch += 1
                    train_loss += loss.item()
                    pbar.update(1)
                if self.epoch % self.log_interval == 0:
                    print(f"Train Epoch: {self.epoch} , Loss: {train_loss:.6f}")
                    # self.writer.add_scalar("Loss train", loss.item(), self.epoch)
                    val_loss = self.validate(loss_fn)
                    new_best = val_loss < self.best_val_loss
                    self.save(self.epoch, current_best=new_best)
                    if new_best:
                        self.best_val_loss = val_loss
                        early_stop_count = 0
                    else:
                        early_stop_count += 1
                    if early_stop_count > early_stop_criteria:
                        print(f"Early stopping at epoch {self.epoch}")
                        print(f"Epoch {self.epoch}, train loss {train_loss}")

                        print(f"Epoch {self.epoch} val loss {val_loss}")
                        # Save last best model
                        break

    def validate(self, loss_fn):
        self.model.train(False)
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.val_data):
                pred = self.model(
                    data["input"], return_type="tensor", batch_indices=data["idx"]
                )
                vloss = loss_fn(pred, data["output"])
                val_loss += vloss.item()

        print(f"Val Epoch: {self.epoch} , Loss: {val_loss:.6f}")
        # self.writer.add_scalar("Loss validation", loss.item(), self.epoch)
        self.model.train(True)
        return val_loss

    def test(self):
        pass
