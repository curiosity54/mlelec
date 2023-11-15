from typing import Dict, List, Optional, Union

import ase
import torch
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from ase.io import read
import hickle


# Dataset class  - to load and pass around structures, targets and
# required auxillary data wherever necessary
class precomputed_molecules(Enum):
    water_1000 = "../../examples/data/water_1000"
    ethane = "../../examples/data/ethane"


# No model/feature info here
class MoleculeDataset(Dataset):
    def __init__(
        self,
        frame_path: Optional[str] = None,
        mol_name: Union[precomputed_molecules, str] = "water_1000",
        frame_slice: str = ":",
        target: List[str] = ["fock"],  # TODO: list of targets
        use_precomputed: bool = True,
        aux_data: Optional[List] = None,
        data_path: Optional[str] = None,
        aux_path: Optional[str] = None,
    ):
        # aux_data could be basis, overlaps for H-learning, Lattice translations etc.
        self.path = frame_path
        self.structures = None
        self.mol_name = mol_name.lower()
        self.use_precomputed = use_precomputed
        self.frame_slice = frame_slice
        self.target_names = target
        self.target = {t: [] for t in self.target_names}
        if mol_name in precomputed_molecules.__members__ and self.use_precomputed:
            self.path = precomputed_molecules[mol_name].value
        assert self.path is not None, "Path to data not provided"
        self.data_path = self.path
        self.aux_path = self.path
        # allow overwrite of data and aux path if necessary
        if data_path is not None:
            self.data_path = data_path
        if aux_path is not None:
            self.aux_path = aux_path

        self.load_structures()
        self.load_target()
        self.aux_data_names = aux_data
        if "fock" in target:
            if self.aux_data_names is None:
                self.aux_data_names = []
            self.aux_data_names.append("overlap")
        if self.aux_data_names is not None:
            self.aux_data = {t: [] for t in self.aux_data_names}
            self.load_aux_data()

    def load_structures(self):
        # mol_name = self.mol_name.lower()
        # if mol_name in precomputed_molecules.__members__ and self.use_precomputed:
        #     self.path = precomputed_molecules[mol_name].value
        try:
            print("Loading structures")
            # print(self.path + "/{}.xyz".format(mol_name))
            self.structures = read(
                self.path + "/{}.xyz".format(self.mol_name), index=self.frame_slice
            )
        except:
            raise FileNotFoundError("No structures found at the given path")

    def load_target(self):
        try:
            for t in self.target_names:
                print(self.data_path + "/{}.hickle".format(t))
                self.target[t] = hickle.load(self.data_path + "/{}.hickle".format(t))
        except:
            raise FileNotFoundError("Required target not found at the given path")
            # TODO: generate data instead?

    def load_aux_data(self):
        try:
            for t in self.aux_data_names:
                self.aux_data[t] = hickle.load(self.aux_path + "/{}.hickle".format(t))
        except:
            raise FileNotFoundError("Auxillary data not found at the given path")


class MLDataset(Dataset):
    def __init__(
        self,
        molecule_data: MoleculeDataset,
    ):
        super().__init__()
        self.structures = molecule_data.structures
        self.target = molecule_data.target
        self.nstructs = len(self.structures)
        self.natoms_list = [len(frame) for frame in self.structures]
        self.species = set([tuple(f.numbers) for f in self.structures])

        self.aux_data = molecule_data.aux_data
        self.rng = None

    def _shuffle(self, random_seed: int = None):
        if random_seed is None:
            self.rng = torch.default_generator
        else:
            self.rng = torch.Generator().manual_seed(random_seed)

    def _split_indices(
        self, train_frac: float, val_frac: float, test_frac: Optional[float] = None
    ):
        if test_frac is None:
            test_frac = 1 - (train_frac + val_frac)
            assert test_frac > 0
        assert train_frac + val_frac + test_frac == 1

        if self.rng is not None:
            self.train, self.val, self.test = torch.utils.data.random_split(
                range(self.nstructs),
                [train_frac, val_frac, test_frac],
                generator=self.rng,
            )
        else:  # sequential split  #FIXME
            self.train = torch.utils.data.Subset(
                range(self.nstructs), range(int(train_frac * self.nstructs))
            )
            self.val = torch.utils.data.subset(
                range(self.nstructs),
                range(
                    int(train_frac * self.nstructs),
                    int((train_frac + val_frac) * self.nstructs),
                ),
            )
            self.test = torch.utils.data.subset(
                range(self.nstructs),
                range(int((train_frac + val_frac) * self.nstructs), self.nstructs),
            )

        # self.train = DataLoader(self.train, batch_size=self.batch_size)

    def __len__(self):
        return self.nstructs
