from typing import Dict, List, Optional, Union

import ase
import torch
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from ase.io import read
import hickle
from mlelec.targets import ModelTargets
from metatensor import TensorMap, Labels
import metatensor.operations as operations
import numpy as np


# Dataset class  - to load and pass around structures, targets and
# required auxillary data wherever necessary
class precomputed_molecules(Enum):
    water_1000 = "examples/data/water_1000"
    ethane = "examples/data/ethane"


# No model/feature info here
class MoleculeDataset(Dataset):
    def __init__(
        self,
        frame_path: Optional[str] = None,
        mol_name: Union[precomputed_molecules, str] = "water_1000",
        frame_slice: slice = slice(None),
        target: List[str] = ["fock"],  # TODO: list of targets
        use_precomputed: bool = True,
        aux: Optional[List] = None,
        data_path: Optional[str] = None,
        aux_path: Optional[str] = None,
        frames: Optional[List[ase.Atoms]] = None,
        target_data: Optional[dict] = None,
        aux_data: Optional[dict] = None,
        device: str = "cpu",
    ):
        # aux_data could be basis, overlaps for H-learning, Lattice translations etc.
        self.device = device
        self.path = frame_path
        self.structures = None
        self.mol_name = mol_name.lower()
        self.use_precomputed = use_precomputed
        self.frame_slice = frame_slice
        self.target_names = target

        self.target = {t: [] for t in self.target_names}
        if mol_name in precomputed_molecules.__members__ and self.use_precomputed:
            self.path = precomputed_molecules[mol_name].value
        if frames is None:
            assert self.path is not None, "Path to data not provided"
        self.load_structures(frames=frames)

        if target_data is None:
            self.data_path = self.path
            self.aux_path = self.path
            # allow overwrite of data and aux path if necessary
            if data_path is not None:
                self.data_path = data_path
            if aux_path is not None:
                self.aux_path = aux_path

        self.load_target(target_data=target_data)
        self.aux_data_names = aux
        if "fock" in target:
            if self.aux_data_names is None:
                self.aux_data_names = ["overlap"]
            elif "overlap" not in self.aux_data_names:
                self.aux_data_names.append("overlap")

        if self.aux_data_names is not None:
            self.aux_data = {t: [] for t in self.aux_data_names}
            self.load_aux_data(aux_data=aux_data)

    def load_structures(self, frames: Optional[List[ase.Atoms]] = None):
        if frames is not None:
            self.structures = frames
            return
        try:
            print("Loading structures")
            # print(self.path + "/{}.xyz".format(mol_name))
            self.structures = read(
                self.path + "/{}.xyz".format(self.mol_name), index=self.frame_slice
            )
        except:
            raise FileNotFoundError("No structures found at the given path")

    def load_target(self, target_data: Optional[dict] = None):
        # TODO: ensure that all keys of self.target names are filled even if they are not provided in target_data
        if target_data is not None:
            for t in self.target_names:
                self.target[t] = target_data[t].to(device=self.device)

        else:
            try:
                for t in self.target_names:
                    print(self.data_path + "/{}.hickle".format(t))
                    self.target[t] = hickle.load(
                        self.data_path + "/{}.hickle".format(t)
                    )[self.frame_slice].to(device=self.device)
                    # os.join(self.aux_path, "{}.hickle".format(t))
            except Exception as e:
                print(e)
                # raise FileNotFoundError("Required target not found at the given path")
                # TODO: generate data instead?

    def load_aux_data(self, aux_data: Optional[dict] = None):
        if aux_data is not None:
            for t in self.aux_data_names:
                if torch.is_tensor(aux_data[t]):
                    self.aux_data[t] = aux_data[t][self.frame_slice].to(
                        device=self.device
                    )
                else:
                    self.aux_data[t] = aux_data[t]

        else:
            try:
                for t in self.aux_data_names:
                    self.aux_data[t] = hickle.load(
                        self.aux_path + "/{}.hickle".format(t)
                    )
                    # os.join(self.aux_path, "{}.hickle".format(t))
                    if torch.is_tensor(self.aux_data[t]):
                        self.aux_data[t] = self.aux_data[t][self.frame_slice].to(
                            device=self.device
                        )
            except Exception as e:
                print(e)
                # raise FileNotFoundError("Auxillary data not found at the given path")


class MLDataset(Dataset):
    def __init__(self, molecule_data: MoleculeDataset, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.structures = molecule_data.structures
        self.target = molecule_data.target
        print(self.target, next(iter(self.target.values())))
        self.target_class = ModelTargets(molecule_data.target_names[0])
        self.target = self.target_class.instantiate(
            next(iter(molecule_data.target.values())),  # FIXME
            frames=self.structures,
            orbitals=molecule_data.aux_data.get("orbitals", None),
            device=device,
        )

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

    def _get_subset(self, y, indices: torch.tensor):
        assert isinstance(y, TensorMap)
        # for k, b in y.items():
        #     b = b.values.to(device=self.device)
        return operations.slice(
            y,
            axis="samples",
            labels=Labels(
                names=["structure"], values=np.asarray(indices).reshape(-1, 1)
            ),
        )

    def _split_indices(
        self, train_frac: float, val_frac: float, test_frac: Optional[float] = None
    ):
        if self.rng is not None:
            indices = torch.randperm(self.nstructs, generator=self.rng)
        else:
            indices = torch.arange(self.nstructs)
        if test_frac is None:
            test_frac = 1 - (train_frac + val_frac)
            assert test_frac > 0
        assert train_frac + val_frac + test_frac == 1

        self.train_idx = indices[: int(train_frac * self.nstructs)]
        self.val_idx = indices[
            int(train_frac * self.nstructs) : int(
                (train_frac + val_frac) * self.nstructs
            )
        ]
        self.test_idx = indices[int((train_frac + val_frac) * self.nstructs) :]
        assert len(self.test_idx) > 0
        self.target_train = self._get_subset(self.target.blocks, self.train_idx)
        self.target_val = self._get_subset(self.target.blocks, self.val_idx)
        self.target_test = self._get_subset(self.target.blocks, self.test_idx)

        # if self.rng is not None:
        #     torch.shuffle(indices, generator=self.rng)
        #     self.train, self.val, self.test = torch.utils.data.random_split(
        #         range(self.nstructs + 1),
        #         [train_frac, val_frac, test_frac],
        #         generator=self.rng,
        #     )
        #     assert len(self.test) > 0

        # else:  # sequential split  #FIXME
        #     self.train = torch.utils.data.Subset(
        #         range(self.nstructs), range(int(train_frac * self.nstructs))
        #     )
        #     self.val = torch.utils.data.Subset(
        #         range(self.nstructs),
        #         range(
        #             int(train_frac * self.nstructs),
        #             int((train_frac + val_frac) * self.nstructs),
        #         ),
        #     )
        #     self.test = torch.utils.data.Subset(
        #         range(self.nstructs),
        #         range(int((train_frac + val_frac) * self.nstructs), self.nstructs),
        #     )

        # self.train = DataLoader(self.train, batch_size=self.batch_size)

    def __len__(self):
        return self.nstructs

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.structures[idx], self.target[idx]
