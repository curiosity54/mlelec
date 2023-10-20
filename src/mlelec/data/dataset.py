from typing import Dict, List, Optional

import ase
import torch
from torch.utils.data import Dataset, DataLoader


# Dataset class  - to load and pass around structures, targets and
# required auxillary data wherever necessary
class MLDataset(Dataset):
    def __init__(
        self,
        structures: List[ase.Atoms],
        target: torch.tensor,
        aux_data: Optional[Dict] = None,
    ):
        # aux_data could be overlaps for H-learning, Lattice translations etc.
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
