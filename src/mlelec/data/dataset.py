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
import os
import metatensor
import warnings
import torch.utils.data as data


# Dataset class  - to load and pass around structures, targets and
# required auxillary data wherever necessary
class precomputed_molecules(Enum):
    water_1000 = "examples/data/water_1000"
    water_rotated = "examples/data/water_rotated"
    ethane = "examples/data/ethane"


# No model/feature info here
class MoleculeDataset(Dataset):
    def __init__(
        self,
        path: Optional[str] = None,
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
        orbs: str = "sto-3g",
    ):
        # aux_data could be basis, overlaps for H-learning, Lattice translations etc.
        self.device = device
        self.path = path
        self.structures = None
        self.mol_name = mol_name.lower()
        self.use_precomputed = use_precomputed
        self.frame_slice = frame_slice
        self.target_names = target

        self.target = {t: [] for t in self.target_names}
        if mol_name in precomputed_molecules.__members__ and self.use_precomputed:
            self.path = precomputed_molecules[mol_name].value
        if target_data is None:
            self.data_path = os.path.join(self.path, orbs)
            self.aux_path = os.path.join(self.path, orbs)
            # allow overwrite of data and aux path if necessary
            if data_path is not None:
                self.data_path = data_path
            if aux_path is not None:
                self.aux_path = aux_path

        if frames is None:
            if self.path is None and self.data_path is not None:
                try:
                    self.path = self.data_path
                    self.load_structures()
                except:
                    raise ValueError(
                        "No structures found at DATA path, either specify frame_path or ensure strures present at data_path"
                    )
            # assert self.path is not None, "Path to data not provided"
        self.load_structures(frames=frames)
        self.pbc = False
        for f in self.structures:
            # print(f.pbc, f.cell)
            if f.pbc.any():
                if not f.cell.any():
                    # "PBC found but no cell vectors found"
                    f.pbc = False
                self.pbc = True
            if self.pbc:
                if not f.cell.any():
                    f.cell = [100, 100, 100]  # TODO this better

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
    def __init__(
        self,
        molecule_data: MoleculeDataset,
        device: str = "cpu",
        model_type: Optional[str] = "acdc",
        features: Optional[TensorMap] = None,
        **kwargs,
    ):
        super().__init__()
        self.molecule_data = molecule_data
        self.device = device
        self.structures = molecule_data.structures
        self.target = molecule_data.target
        # print(self.target, next(iter(self.target.values())))
        # sets the first target as the primary target - # FIXME
        self.target_class = ModelTargets(molecule_data.target_names[0])
        self.target = self.target_class.instantiate(
            next(iter(molecule_data.target.values())),
            frames=self.structures,
            orbitals=molecule_data.aux_data.get("orbitals", None),
            device=device,
            **kwargs,
        )

        self.nstructs = len(self.structures)
        self.natoms_list = [len(frame) for frame in self.structures]
        self.species = set([tuple(f.numbers) for f in self.structures])

        self.aux_data = molecule_data.aux_data
        self.rng = None
        self.model_type = model_type  # flag to know if we are using acdc features or want to cycle hrough positons
        if self.model_type == "acdc":
            self.features = features

    def _shuffle(self, random_seed: int = None):
        if random_seed is None:
            self.rng = torch.default_generator
        else:
            self.rng = torch.Generator().manual_seed(random_seed)

    def _get_subset(self, y: TensorMap, indices: torch.tensor):
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
        assert np.isclose(train_frac + val_frac + test_frac, 1, rtol=1e-6, atol=1e-5), (
            train_frac + val_frac + test_frac
        )

        self.train_idx = indices[: int(train_frac * self.nstructs)].sort()[0]
        self.val_idx = indices[
            int(train_frac * self.nstructs) : int(
                (train_frac + val_frac) * self.nstructs
            )
        ].sort()[0]
        self.test_idx = indices[int((train_frac + val_frac) * self.nstructs) :].sort()[
            0
        ]
        assert (
            len(self.test_idx)
            > 0  # and len(self.val_idx) > 0 and len(self.train_idx) > 0
        ), "Split indices not generated properly"
        self.target_train = self._get_subset(self.target.blocks, self.train_idx)
        self.target_val = self._get_subset(self.target.blocks, self.val_idx)
        self.target_test = self._get_subset(self.target.blocks, self.test_idx)

        self.train_frames = [self.structures[i] for i in self.train_idx]
        self.val_frames = [self.structures[i] for i in self.val_idx]
        self.test_frames = [self.structures[i] for i in self.test_idx]
        # update self.structures to reflect shuffling
        self.structures_original = self.structures.copy()
        self.structures = [self.structures_original[i] for i in indices]

    def _set_features(self, features: TensorMap):
        self.features = features
        if not hasattr(self, "train_idx"):
            warnings.warn("No train/val/test split found, deafult split used")
            self._split_indices(train_frac=0.7, val_frac=0.2, test_frac=0.1)
        self.feature_names = features.keys.values
        self.feat_train = self._get_subset(self.features, self.train_idx)
        self.feat_val = self._get_subset(self.features, self.val_idx)
        self.feat_test = self._get_subset(self.features, self.test_idx)

    def _set_model_return(self, model_return: str = "blocks"):
        ## Helper function to set output in __get_item__ for model training
        assert model_return in [
            "blocks",
            "tensor",
        ], "model_target must be one of [blocks, tensor]"
        self.model_return = model_return

    def __len__(self):
        return self.nstructs

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            # idx = [i.item() for i in idx]
            idx = idx.tolist()
        if not self.model_type == "acdc":
            return self.structures[idx], self.target.tensor[idx]
        else:
            assert (
                self.features is not None
            ), "Features not set, call _set_features() first"
            x = operations.slice(
                self.features,
                axis="samples",
                labels=Labels(
                    names=["structure"], values=np.asarray([idx]).reshape(-1, 1)
                ),
            )
            if self.model_return == "blocks":
                y = operations.slice(
                    self.target.blocks,
                    axis="samples",
                    labels=Labels(
                        names=["structure"], values=np.asarray([idx]).reshape(-1, 1)
                    ),
                )
            else:
                idx = [i.item() for i in idx]
                y = self.target.tensor[idx]
            # x = metatensor.to(x, "torch")
            # y = metatensor.to(y, "torch")

            return x, y, idx

    def collate_fn(self, batch):
        x = batch[0][0]
        y = batch[0][1]
        idx = batch[0][2]
        return {"input": x, "output": y, "idx": idx}


def get_dataloader(
    ml_data: MLDataset,
    collate_fn: callable = None,
    batch_size: int = 4,
    drop_last: bool = False,
    selection: Optional[str] = "all",
    model_return: Optional[str] = None,
):
    assert selection in [
        "all",
        "train",
        "val",
        "test",
    ], "selection must be one of [all, train, val, test]"
    if collate_fn is None:
        collate_fn = ml_data.collate_fn

    assert model_return in [
        "blocks",
        "tensor",
    ], "model_target must be one of [blocks, tensor]"
    ml_data._set_model_return(model_return)
    train_sampler = data.sampler.SubsetRandomSampler(ml_data.train_idx)
    train_sampler = data.sampler.BatchSampler(
        train_sampler, batch_size=batch_size, drop_last=drop_last
    )

    val_sampler = data.sampler.SubsetRandomSampler(ml_data.val_idx)
    val_sampler = data.sampler.BatchSampler(
        val_sampler, batch_size=batch_size, drop_last=drop_last
    )

    test_sampler = data.sampler.SubsetRandomSampler(ml_data.test_idx)
    test_sampler = data.sampler.BatchSampler(
        test_sampler, batch_size=batch_size, drop_last=drop_last
    )
    train_loader = data.DataLoader(
        ml_data,
        sampler=train_sampler,
        collate_fn=collate_fn,
        shuffle=False,
    )
    val_loader = data.DataLoader(
        ml_data,
        sampler=val_sampler,
        collate_fn=collate_fn,
        shuffle=False,
    )

    test_loader = data.DataLoader(
        ml_data,
        sampler=test_sampler,
        collate_fn=collate_fn,
        shuffle=False,
    )
    if selection.lower() == "all":
        return train_loader, val_loader, test_loader
    elif selection.lower() == "train":
        return train_loader
    elif selection.lower() == "val":
        return val_loader
    elif selection.lower() == "test":
        return test_loader
