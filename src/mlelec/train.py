# Overall train script - assembles required pipeline - trains/loads and saves model
import sys
import os
import torch 
import datetime
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import cpu_count
from torch.optim import Adam
import pickle

# from data.dataset import Dataset
class Trainer(): 
    def __init__(
        self,
        model,
        dataset_split:tuple[float, float, float],  # tuple: (train_data, val_data, test_data)
        mol_name:str,
        train_batch_size=64,
        learning_rate=1e-4,
        max_epochs:int=100000,
        log_interval:int=100,
        save_path="./train_results",
        checkpoint:str="best",  # last, best
        device:str="cuda" if torch.cuda.is_available() else "cpu",
        start_from_checkpoint:bool=False,
    ):
        self.device = device
        self.model = model
        self.mol_name = mol_name
        self.batch_size = train_batch_size
        self.lr = learning_rate
        self.log_interval = log_interval
        self.max_epochs = max_epochs
        self.save_path = save_path
        self.train_data, self.val_data, self.test_data = dataset_split
        self.checkpoint = checkpoint
        
        timezone = datetime.timezone(dt.timedelta(hours=2)) 
        self.save_path = Path(save_path + "/" +datetime.datetime.now(timezone).strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(self.save_path + "/" + "_trn")

        self.best_val_loss = 10**10
        if start_from_checkpoint:
            try:
                self.load()
                print("Checkpoint loaded")
            except:
                print(" No available checkpoint")

    def save(self, ):
        cd 
    def load(self, )
 
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
