from typing import Dict, List, Optional

import ase
import torch
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from ase.io import read
import hickle


# Dataset class  - to load and pass around structures, targets and
# required auxillary data wherever necessary
class precomputer_molecules(Enum):
    water = "water"
    ethane = "ethane"


class MoleculeDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        mol_name: str,
        frame_slice: str = ":",
        target: str = "fock",
    ):
        self.path = data_path
        self.structures = None
        self.mol_name = mol_name
        self.frame_slice = frame_slice
        self.target = target
        self.load_structures()
        self.load_target()

    def load_structures(self):
        try:
            print("Loading structures")
            self.structures = read(
                self.path + "/{}.xyz".format(self.mol_name), index=self.frame_slice
            )
        except:
            raise FileNotFoundError("No structures found at the given path")

    def load_target(self):
        if self.target == "fock":
            # also load overlap
            pass


class MLDataset(MoleculeDataset, Dataset):
    def __init__(
        self,
        structures: List[ase.Atoms],
        target: torch.tensor,
        aux_data: Optional[Dict] = None,
    ):
        super().__init__()
        # aux_data could be basis, overlaps for H-learning, Lattice translations etc.
        self.structures = structures
        self.target = target
        self.nstructs = len(structures)
        self.natoms_list = [len(frame) for frame in structures]
        self.species = set([f.number for f in self.structures])

        if aux_data is not None:
            self.aux_data = {}
            for k in aux_data:
                self.aux_data[k] = aux_data[k]

        # self.features = None
        # self.feat_keys = None

    def _shuffle(self, random_seed: int = None):
        if random_seed is None:
            self.rng = torch.default_generator
        else:
            self.rng = torch.Generator().manual_seed(random_seed)

    def _split(
        self, train_frac: float, val_frac: float, test_frac: Optional[float] = None
    ):
        if test_frac is None:
            test_frac = 1 - (train_frac + val_frac)
            assert test_frac > 0
        assert train_frac + val_frac + test_frac == 1
        self.train, self.val, self.test = torch.utils.data.random_split(
            range(self.nstructs), [train_frac, val_frac, test_frac], generator=self.rng
        )
        # self.train = DataLoader(self.train, batch_size=self.batch_size)

    # def set_features(self, feat: TensorMap):
    #     self.features = feat
    #     self.feat_keys = self.features.keys()
    def __len__(self):
        return self.nstructs
