import copy
import os
import warnings
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Union

import ase
import hickle
import metatensor
import metatensor.operations as operations
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from ase.io import read
from metatensor import Labels, TensorMap, TensorBlock, load
from mlelec.utils.twocenter_utils import map_targetkeys_to_featkeys
from mlelec.targets import ModelTargets


class precomputed_molecules(Enum):
    """
    Enumeration representing precomputed molecules.

    This enumeration provides paths to precomputed molecular data for
    various molecules. Each member represents a specific molecule with
    its corresponding path to the precomputed data.

    Attributes:
        water_1000: Path to precomputed data for 1000 water molecules.
        water_rotated: Path to precomputed data for rotated configurations of a water molecule.
        ethane: Path to precomputed data for ethane.
        qm7: Path to precomputed data for QM7 dataset.
    """

    water_1000 = "examples/data/water_1000"
    water_rotated = "examples/data/water_rotated"
    ethane = "examples/data/ethane"
    qm7 = "examples/data/qm7"
    qm9 = "examples/data/qm9"
    acs = "examples/data/acs"


class MoleculeDataset(Dataset):
    """
    Dataset class for molecular data.

    This class provides a dataset for molecular data. It loads molecular
    structures, targets and auxiliary data from precomputed data or generates 
    data from the .xyz files at the provided paths using a `PySCF` calculator. 
    For Hamiltonian learning, at the moment the targets can be Fock matrices 
    and dipole moments. The auxiliary data can be overlaps and orbitals.

    Args:
        path: Path to the data directory.
        mol_name: Name of the molecule to load(chosen from the`precomputed_molecules`).
        frame_slice: Slice object to select a subset of frames.
        target: List of target names.
        lb_target: List of large basis target names.
        use_precomputed: Flag to use precomputed data.
        aux: List of auxiliary data names.
        data_path: Path to the data directory.
        aux_path: Path to the auxiliary data directory.
        frames: List of ASE Atoms objects.
        target_data: Dictionary of target data.
        aux_data: Dictionary of auxiliary data.
        device: Device to load the data on.
        basis: Basis set for the data.

    Returns:
        MoleculeDataset: A Dataset object for molecular data.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        mol_name: Union[precomputed_molecules, str] = "water_1000",
        frame_slice: slice = slice(None),
        target: List[str] = ["fock"], 
        lb_target: Optional[List] = None, 
        use_precomputed: bool = True,
        aux: Optional[List] = None,
        lb_aux: Optional[List] = None,
        data_path: Optional[str] = None,
        aux_path: Optional[str] = None,
        frames: Optional[List[ase.Atoms]] = None,
        target_data: Optional[dict] = None,
        aux_data: Optional[dict] = None,
        lb_aux_data: Optional[dict] = None,
        device: str = "cpu",
        basis: str = "sto-3g",
        large_basis: str = "def2-tzvp"
    ):
        # aux_data could be basis, overlaps for H-learning, Lattice translations etc.
        self.device = device
        self.path = path
        self.structures = None
        self.mol_name = mol_name.lower()
        self.use_precomputed = use_precomputed
        self.frame_slice = frame_slice
        self.target_names = target
        self.basis = basis
        if lb_target is not None:
            self.lb_target_names = lb_target

        self.target = {t: [] for t in self.target_names}
        if lb_target is not None:
            self.lb_target = {t: [] for t in self.lb_target_names}
        if mol_name in precomputed_molecules.__members__ and self.use_precomputed:
            self.path = precomputed_molecules[mol_name].value
        if target_data is None:
            self.data_path = os.path.join(self.path, basis)
            self.aux_path = os.path.join(self.path, basis)
            if data_path is not None:
                self.data_path = data_path
            if aux_path is not None:
                self.aux_path = aux_path
        if lb_target is not None:
            self.lb_data_path = os.path.join(self.path, large_basis)
            self.lb_aux_path = os.path.join(self.path, large_basis)
            # allow overwrite of data and aux path if necessary
            

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
            if f.pbc.any():
                if not f.cell.any():
                    f.pbc = False
                self.pbc = True
            if self.pbc:
                if not f.cell.any():
                    f.cell = [100, 100, 100]  # TODO this better

        self.load_target(target_data=target_data)
        if lb_target is not None:
            self.load_lb_target()
        self.aux_data_names = aux
        if lb_aux is not None:
            self.lb_aux_data_names = lb_aux
        if "fock" in target:
            if self.aux_data_names is None:
                self.aux_data_names = ["overlap", "orbitals"]
            if lb_aux_data is not None and self.lb_aux_data_names is None:
                self.lb_aux_data_names = ["overlap", "orbitals"]
            elif "overlap" not in self.aux_data_names:
                self.aux_data_names.append("overlap")
            elif lb_aux_data is not None and "overlap" not in self.lb_aux_data_names:
                self.lb_aux_data_names.append("overlap")

        if self.aux_data_names is not None:
            self.aux_data = {t: [] for t in self.aux_data_names}
            self.load_aux_data(aux_data=aux_data)
        if lb_aux is not None and self.lb_aux_data_names is not None:
            self.lb_aux_data = {t: [] for t in self.lb_aux_data_names}
            self.load_lb_aux_data(lb_aux_data=lb_aux_data)

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
            raise FileNotFoundError(
                "No structures found at {}".format(self.path + f"/{self.mol_name}.xyz")
            )

    def load_target(self, target_data: Optional[dict] = None):
        if target_data is not None:
            for t in self.target_names:
                self.target[t] = target_data[t].to(device=self.device)

        else:
            for t in self.target_names:
                file_path = os.path.join(self.data_path, f"{t}.hickle")
                print(file_path)
                data = hickle.load(file_path)[self.frame_slice]
                
                if isinstance(data, np.ndarray) and data.dtype == object:
                    # If data is a numpy array with dtype object, convert each element
                    self.target[t] = [torch.from_numpy(item).to(device=self.device) 
                                      if isinstance(item, np.ndarray) else item.to(device=self.device) for item in data]
                elif isinstance(data, np.ndarray):
                    # If data is a regular numpy array, convert the entire array
                    self.target[t] = torch.from_numpy(data).to(device=self.device)
                elif isinstance(data, list):
                    # If data is a list, ensure it contains numpy arrays or tensors
                    for i, item in enumerate(data):
                        if isinstance(item, np.ndarray):
                            data[i] = torch.from_numpy(item).to(device=self.device)
                        elif isinstance(item, torch.Tensor):
                            data[i] = item.to(device=self.device)
                        else:
                            raise TypeError(f"Unsupported data type for target '{t}' at index {i}: {type(item)}")
                    self.target[t] = data
                else:
                    raise TypeError(f"Unsupported data type for target '{t}': {type(data)}")

                # Ensure all items in the target list are tensors if it's a list
                if isinstance(self.target[t], list):
                    assert all(isinstance(x, torch.Tensor) for x in self.target[t]), \
                        f"Not all items in target '{t}' are tensors after conversion."

                
    def load_lb_target(self):
        for t in self.lb_target_names:
            file_path = os.path.join(self.lb_data_path, f"{t}.hickle")
            print(file_path)
            data = hickle.load(file_path)[self.frame_slice]

            if isinstance(data, np.ndarray) and data.dtype == object:
                # If data is a numpy array with dtype object, convert each element
                self.lb_target[t] = [torch.from_numpy(item).to(device=self.device) 
                                     if isinstance(item, np.ndarray) else item.to(device=self.device) for item in data]
            elif isinstance(data, np.ndarray):
                # If data is a regular numpy array, convert the entire array
                self.lb_target[t] = torch.from_numpy(data).to(device=self.device)
            elif isinstance(data, list):
                # If data is a list, ensure it contains numpy arrays or tensors
                for i, item in enumerate(data):
                    if isinstance(item, np.ndarray):
                        data[i] = torch.from_numpy(item).to(device=self.device)
                    elif isinstance(item, torch.Tensor):
                        data[i] = item.to(device=self.device)
                    else:
                        raise TypeError(f"Unsupported data type for lb_target '{t}' at index {i}: {type(item)}")
                self.lb_target[t] = data
            else:
                raise TypeError(f"Unsupported data type for lb_target '{t}': {type(data)}")

            # Ensure all items in the lb_target list are tensors if it's a list
            if isinstance(self.lb_target[t], list):
                assert all(isinstance(x, torch.Tensor) for x in self.lb_target[t]), \
                    f"Not all items in lb_target '{t}' are tensors after conversion."


    def load_aux_data(self, aux_data: Optional[dict] = None):
        if aux_data is not None:
            for t in self.aux_data_names:
                if torch.is_tensor(aux_data[t]):
                    self.aux_data[t] = aux_data[t][self.frame_slice]
                else:
                    self.aux_data[t] = aux_data[t]

        else:
            try:
                for t in self.aux_data_names:
                    self.aux_data[t] = hickle.load(
                        self.aux_path + "/{}.hickle".format(t)
                    )
                    if isinstance(self.aux_data[t], np.ndarray):
                        self.aux_data[t] = self.aux_data[t].tolist()
                    # os.join(self.aux_path, "{}.hickle".format(t))
                    if torch.is_tensor(self.aux_data[t]):
                        self.aux_data[t] = self.aux_data[t][self.frame_slice].to(
                            device=self.device
                        )
                    elif isinstance(self.aux_data[t], list):
                        self.aux_data[t] = self.aux_data[t][self.frame_slice]
                        for i in range(len(self.aux_data[t])):
                            if isinstance(self.aux_data[t][i], np.ndarray):
                                self.aux_data[t][i] = torch.from_numpy(self.aux_data[t][i]).to(device = self.device)
                            elif isinstance(self.aux_data[t][i], torch.Tensor):
                                self.aux_data[t][i] = self.aux_data[t][i].to(device = self.device)
                            # assert isinstance(self.aux_data[t][i], torch.Tensor)
            except Exception as e:
                print(e)
                # raise FileNotFoundError("Auxillary data not found at the given path")
        if "overlap" in self.aux_data_names and "density" in self.target_names:
            self.target_names.append("elec_population")
            self.target["elec_population"] = torch.sum(
                torch.einsum(
                    "bij, bji ->bi", self.target["density"], self.aux_data["overlap"]
                ),
                axis=1,
            )

    def load_lb_aux_data(self, lb_aux_data: Optional[dict] = None):
        if lb_aux_data is not None:
            for t in self.lb_aux_data_names:
                if torch.is_tensor(lb_aux_data[t]):
                    self.lb_aux_data[t] = lb_aux_data[t][self.frame_slice]
                else:
                    self.lb_aux_data[t] = lb_aux_data[t]
        else:
            for t in self.lb_aux_data_names:
                self.lb_aux_data[t] = hickle.load(
                    self.lb_aux_path + "/{}.hickle".format(t)
                )
                if isinstance(self.lb_aux_data[t], np.ndarray):
                    self.lb_aux_data[t] = self.lb_aux_data[t].tolist()
                    
                if torch.is_tensor(self.lb_aux_data[t]):
                    self.lb_aux_data[t] = self.lb_aux_data[t][self.frame_slice].to(
                        device=self.device
                    )
                elif isinstance(self.lb_aux_data[t], list):
                    self.lb_aux_data[t] = self.lb_aux_data[t][self.frame_slice]
                    for i in range(len(self.lb_aux_data[t])):
                        if isinstance(self.lb_aux_data[t][i], np.ndarray):
                            self.lb_aux_data[t][i] = torch.from_numpy(self.lb_aux_data[t][i]).to(device = self.device)
                        elif isinstance(self.lb_aux_data[t][i], torch.Tensor):
                            self.lb_aux_data[t][i] = self.lb_aux_data[t][i].to(device = self.device)
                        # assert isinstance(self.lb_aux_data[t][i], torch.Tensor)

    def shuffle(self, indices: torch.tensor):
        self.structures = [self.structures[i] for i in indices]
        for t in self.target_names:
            self.target[t] = self.target[t][indices]
        for t in self.aux_data_names:
            try:
                self.aux_data[t] = self.aux_data[t][indices]
            except:
                warnings.warn("Aux data {} skipped shuffling ".format(t))
                continue


class MLDataset(Dataset):
    """
    Dataset class for machine learning data.

    Contains all the data required for machine learning tasks, such as
    input features, target data, auxillary data and also model type and
    training strategies. The class provides methods to shuffle and split
    the data into training, validation and test sets and to generate
    dataloaders for these sets. The class also provides methods to set
    the input features and the model return type.

    Args:
        molecule_data: MoleculeDataset object containing molecular data.
        device: Device to load the data on.
        model_type: Type of model to use.
        features: Input features.
        shuffle: Flag to shuffle the data.
        shuffle_seed: Seed for shuffling.
        kwargs: Additional keyword arguments.

    Returns:
        MLDataset: A Dataset object for machine learning data.

    """

    def __init__(
        self,
        molecule_data: MoleculeDataset,
        device: str = "cpu",
        model_type: Optional[str] = "acdc",
        features: Optional[TensorMap] = None,
        shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
        orthogonal: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.nstructs = len(molecule_data.structures)
        self.rng = None
        if shuffle:
            self._shuffle(shuffle_seed)
        else:
            self.indices = self.indices = torch.arange(self.nstructs)
        self.molecule_data = copy.deepcopy(molecule_data)

        self.structures = self.molecule_data.structures
        self.target = self.molecule_data.target

        self.target_class = ModelTargets(self.molecule_data.target_names[0])
        self.target = self.target_class.instantiate(
            tensor=next(iter(self.molecule_data.target.values())),
            overlap=self.molecule_data.aux_data['overlap'],
            frames=self.structures,
            orbitals=self.molecule_data.aux_data.get("orbitals", None),
            orthogonal=orthogonal,
            device=device,
            **kwargs,
        )

        self.molecule_data.target_blocks = copy.deepcopy(self.target.blocks)
        self.natoms_list = [len(frame) for frame in self.structures]
        self.species = set([tuple(f.numbers) for f in self.structures])

        self.aux_data = self.molecule_data.aux_data
        self.rng = None
        self.model_type = model_type
        if self.model_type == "acdc":
            self.features = features

        self.train_frac = kwargs.get("train_frac", 0.7)
        self.val_frac = kwargs.get("val_frac", 0.2)

    def _shuffle(self, random_seed: int = None):
        if random_seed is None:
            self.rng = torch.default_generator
        else:
            self.rng = torch.Generator().manual_seed(random_seed)

        self.indices = torch.randperm(self.nstructs, generator=self.rng)

    def _get_subset(self, y: TensorMap, indices: torch.tensor):
        indices = indices.cpu().numpy()
        assert isinstance(y, TensorMap)

        return operations.slice(
            y,
            axis="samples",
            labels=Labels(
                names=["structure"], values=np.asarray(indices).reshape(-1, 1)
            ),
        )

    def _split_indices(
        self,
        train_frac: float = None,
        val_frac: float = None,
        test_frac: Optional[float] = None,
    ):

        if train_frac is not None:
            self.train_frac = train_frac
        if val_frac is not None:
            self.val_frac = val_frac
        if test_frac is None:
            test_frac = 1 - (self.train_frac + self.val_frac)
            self.test_frac = test_frac
            assert self.test_frac > 0
        else:
            try:
                self.test_frac = test_frac
                assert np.isclose(
                    self.train_frac + self.val_frac + self.test_frac,
                    1,
                    rtol=1e-6,
                    atol=1e-5,
                ), (
                    self.train_frac + self.val_frac + self.test_frac,
                    "Split fractions do not add up to 1",
                )
            except:
                self.test_frac = 1 - (self.train_frac + self.val_frac)
                assert self.test_frac > 0

        self.train_idx = self.indices[: int(self.train_frac * self.nstructs)].sort()[0]
        self.val_idx = self.indices[
            int(self.train_frac * self.nstructs) : int(
                (self.train_frac + self.val_frac) * self.nstructs
            )
        ].sort()[0]
        self.test_idx = self.indices[
            int((self.train_frac + self.val_frac) * self.nstructs) :
        ].sort()[0]
        if self.test_frac > 0:
            assert len(self.test_idx) > 0, "Split indices not generated properly"
        self.target_train = self._get_subset(self.target.blocks, self.train_idx)
        self.target_val = self._get_subset(self.target.blocks, self.val_idx)
        self.target_test = self._get_subset(self.target.blocks, self.test_idx)

        self.train_frames = [self.structures[i] for i in self.train_idx]
        self.val_frames = [self.structures[i] for i in self.val_idx]
        self.test_frames = [self.structures[i] for i in self.test_idx]

    def _set_features(self, features: TensorMap):
        self.features = features
        if not hasattr(self, "train_idx"):
            warnings.warn("No train/val/test split found, deafult split used")
            self._split_indices(train_frac=0.7, val_frac=0.2, test_frac=0.1)
        self.feature_names = features.keys.values
        self.feat_train = self._get_subset(self.features, self.train_idx)
        self.feat_val = self._get_subset(self.features, self.val_idx)
        self.feat_test = self._get_subset(self.features, self.test_idx)
        self._match_feature_and_target_samples()
        
    def _load_features(self, filename: str):
        self.features = load(filename)
        self.feature_names = self.features.keys.values
        self.feat_train = self._get_subset(self.features, self.train_idx)
        self.feat_val = self._get_subset(self.features, self.val_idx)
        self.feat_test = self._get_subset(self.features, self.test_idx)

    def _match_feature_and_target_samples(self):
        new_blocks = []
        for k, target_block in self.molecule_data.target_blocks.items():
            feat_block = map_targetkeys_to_featkeys(self.features, k)
            intersection, _, idx = feat_block.samples.intersection_and_mapping(target_block.samples)
            idx = np.where(idx != -1)

            new_blocks.append(
                TensorBlock(values = target_block.values[idx],
                            samples = intersection,
                            properties = target_block.properties,
                            components = target_block.components)
            )
        
        self.target._set_blocks(TensorMap(self.molecule_data.target_blocks.keys, new_blocks))
        self.target_train = self._get_subset(self.target.blocks, self.train_idx)
        self.target_val = self._get_subset(self.target.blocks, self.val_idx)
        self.target_test = self._get_subset(self.target.blocks, self.test_idx)

    def _set_model_return(self, model_return: str = "blocks"):
        # Helper function to set output in __get_item__ for model training
        assert model_return in [
            "blocks",
            "tensor",
        ], "model_target must be one of [blocks, tensor]"
        self.model_return = model_return

    def __len__(self):
        return self.nstructs

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        frames = [self.structures[i] for i in idx]
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
                y = [self.target.tensor[i] for i in idx]

            return x, y, idx, frames

    def collate_fn(self, batch):
        x = batch[0][0]
        y = batch[0][1]
        idx = batch[0][2]
        frames = batch[0][3]
        return {"input": x, "output": y, "idx": idx, "frames": frames}


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


from mlelec.data.pyscf_calculator import (
    _map_transidx_to_relative_translation,
    get_scell_phase,
    kpoint_to_translations,
    map_mic_translations,
    map_supercell_to_relativetrans,
    translation_vectors_for_kmesh,
    translations_to_kpoint,
)


class PeriodicDataset(Dataset):
    # TODO: make format compatible with MolecularDataset
    def __init__(
        self,
        frames,
        frame_slice: slice = slice(None),
        kgrid: Union[List[int], List[List[int]]] = [1, 1, 1],
        matrices_kpoint: Union[torch.tensor, np.ndarray] = None,
        matrices_translation: Union[Dict, torch.tensor, np.ndarray] = None,
        target: List[str] = ["real_translation"],
        aux: List[str] = ["real_overlap"],
        use_precomputed: bool = True,
        device="cuda",
        orbs: str = "sto-3g",
        desired_shifts: List = None,
    ):
        self.structures = frames
        self.frame_slice = frame_slice
        self.nstructs = len(frames)
        self.kmesh = kgrid
        self.kgrid_is_list = False
        if isinstance(kgrid[0], list):
            self.kgrid_is_list = True
            assert (
                len(self.kmesh) == self.nstructs
            ), "If kgrid is a list, it must have the same length as the number of structures"
        else:
            self.kmesh = [
                kgrid for _ in range(self.nstructs)
            ]  # currently easiest to do

        self.device = device
        self.basis = orbs
        self.use_precomputed = use_precomputed
        if not use_precomputed:
            raise NotImplementedError("You must use precomputed data for now.")

        self.target_names = target
        self.aux_names = aux
        self.desired_shifts_sup = (
            []
        )  # track the negative counterparts of desired_shifts as well
        self.cells = []
        self.phase_matrices = []
        self.supercells = []
        self.all_relative_shifts = []  # allowed shifts of kmesh

        for ifr, structure in enumerate(self.structures):
            # if self.kgrid_is_list:
            # cell, scell, phase = get_scell_phase(structure, self.kmesh[i])
            # else:
            cell, scell, phase = get_scell_phase(structure, self.kmesh[ifr])

            self.cells.append(cell)
            self.supercells.append(scell)
            self.phase_matrices.append(phase)
            self.all_relative_shifts.append(
                translation_vectors_for_kmesh(
                    cell, self.kmesh[ifr], return_rel=True
                ).tolist()
            )
        self.supercell_matrices = None
        if desired_shifts is not None:
            self.desired_shifts = desired_shifts
        else:
            self.desired_shifts = np.unique(np.vstack(self.all_relative_shifts), axis=0)

        for s in self.desired_shifts:
            self.desired_shifts_sup.append(s)  # make this tuple(s)?
            self.desired_shifts_sup.append([-s[0], -s[1], -s[2]])
        self.desired_shifts_sup = np.unique(self.desired_shifts_sup, axis=0)
        # FIXME - works only for a uniform kgrid across structures
        # ----MIC---- (Uncomment) TODO
        # self.desired_shifts_sup = map_mic_translations(
        #     self.desired_shifts_sup, self.kmesh[0]
        # )  ##<<<<<<

        # self.desired_shifts = []
        # for L in self.desired_shifts_sup:
        #     lL = list(L)
        #     lmL = list(-1 * np.array(lL))
        #     if not (lL in self.desired_shifts):
        #         if not (lmL in self.desired_shifts):
        #             self.desired_shifts.append(lL)

        # ------------------------------
        if matrices_translation is not None:
            self.matrices_translation = {
                tuple(t): [] for t in list(self.desired_shifts_sup)
            }
            self.weights_translation = (
                []
            )  # {tuple(t): [] for t in list(self.desired_shifts)}
            self.phase_diff_translation = (
                []
            )  # {tuple(t): [] for t in list(self.desired_shifts)}

            self.matrices_translation = defaultdict(list)  # {}
            if not isinstance(matrices_translation[0], dict):
                # assume we are given the supercell matrices
                self.supercell_matrices = matrices_translation.copy()
                # convert to dict of translations
                matrices_translation = []
                for ifr in range(self.nstructs):
                    # TODO: we'd need to track frames in which desired shifts not found
                    (
                        translated_mat_dict,
                        weight,
                        phase_diff,
                    ) = map_supercell_to_relativetrans(
                        self.supercell_matrices[ifr],
                        phase=self.phase_matrices[ifr],
                        cell=self.cells[ifr],
                        kmesh=self.kmesh[ifr],
                    )
                    matrices_translation.append(translated_mat_dict)
                    self.weights_translation.append(weight)
                    self.phase_diff_translation.append(phase_diff)

                    # for i, t in enumerate(self.desired_shifts_sup):
                    #     # for ifr in range(self.nstructs):
                    #     # idx_t = self.all_relative_shifts[ifr].index(list(t))
                    #     # print(tuple(t), idx_t, self.matrices_translation.keys())#matrices_translation[ifr][idx_t])
                    #     self.matrices_translation[tuple(t)].append(
                    #         translated_mat_dict[tuple(t)]
                    #     )
                self.matrices_translation = {
                    key: np.asarray(
                        [dictionary[key] for dictionary in matrices_translation]
                    )
                    for key in matrices_translation[0].keys()
                }
                # TODO: tracks the keys from the first frame for translations - need to change when working with different kgrids
            else:
                # assume we are given dicts with translatiojns as keys
                self.matrices_translation = matrices_translation  # NOT TESTED FIXME
            # self.matrices_kpoint = self.get_kpoint_target()
        else:
            assert (
                matrices_kpoint is not None
            ), "Must provide either matrices_kpoint or matrices_translation"

        if matrices_kpoint is not None:
            self.matrices_kpoint = matrices_kpoint
            matrices_translation = self.get_translation_target(matrices_kpoint)
            ## FIXME : this will not work when we use a nonunifrom kgrid <<<<<<<<
            self.matrices_translation = {
                key: [] for key in set().union(*matrices_translation)
            }
            [
                self.matrices_translation[k].append(matrices_translation[ifr][k])
                for ifr in range(self.nstructs)
                for k in matrices_translation[ifr].keys()
            ]
            for k in self.matrices_translation.keys():
                self.matrices_translation[k] = np.stack(
                    self.matrices_translation[k]
                ).real

        self.target = {t: [] for t in self.target_names}
        for t in self.target_names:
            # keep only desired shifts
            if t == "real_translation":
                self.target[t] = self.matrices_translation
            elif t == "kpoint":
                self.target[t] = self.matrices_kpoint

    def get_kpoint_target(self):
        kmatrix = []
        for ifr in range(self.nstructs):
            kmatrix.append(
                translations_to_kpoint(
                    self.matrices_translation[ifr],
                    self.phase_diff_translation[ifr],
                    self.weights_translation[ifr],
                )
            )
        return kmatrix

    def get_translation_target(self, matrices_kpoint):
        target = []
        self.translation_idx_map = []
        if not hasattr(self, "phase_diff_translation"):
            self.phase_diff_translation = []
            for ifr in range(self.nstructs):
                assert (
                    matrices_kpoint[ifr].shape[0] == self.phase_matrices[ifr].shape[0]
                ), "Number of kpoints and phase matrices must be the same"
                Nk = self.phase_matrices[ifr].shape[0]
                nao = matrices_kpoint[ifr].shape[-1]
                idx_map = _map_transidx_to_relative_translation(
                    np.zeros((Nk, nao, Nk, nao)),
                    R_rel=np.asarray(self.all_relative_shifts[ifr]),
                )
                # print(self.all_relative_shifts[ifr], idx_map)
                self.translation_idx_map.append(idx_map)
                frame_phase = {}
                for i, (key, val) in enumerate(idx_map.items()):
                    M, N = val[0][1], val[0][2]
                    frame_phase[key] = np.array(
                        self.phase_matrices[ifr][N] / self.phase_matrices[ifr][M]
                    )
                self.phase_diff_translation.append(frame_phase)

        for ifr in range(self.nstructs):
            sc = kpoint_to_translations(
                matrices_kpoint[ifr],
                self.phase_diff_translation[ifr],
                idx_map=self.translation_idx_map[ifr],
                return_supercell=False,
            )
            # convert this to dict over trnaslations and to gamma point if required
            subset_translations = {tuple(t): sc[tuple(t)] for t in self.desired_shifts}
            target.append(subset_translations)
        return target

    def _run_tests(self):
        self.check_translation_hermiticity()
        self.check_fourier()
        self.check_block_()

    def check_translation_hermiticity(self):
        """Check H(T) = H(-T)^T"""
        pass
        # for i, structure in enumerate(self.structures):
        #     check_translation_hermiticity(structure, self.matrices_translation[i])

    def check_fourier(self):
        pass

    def check_block_(self):
        # mat -> blocks _> couple -> uncouple -> mat
        pass

    def discard_nonhermiticity(self, target="", retain="upper"):
        """For each T, create a hermitian target with the upper triangle reflected across the diagonal
        retain = "upper" or "lower"
        target :str to specify which matrices to discard nonhermiticity from
        """
        retain = retain.lower()
        retain_upper = retain == "upper"
        from mlelec.utils.symmetry import _reflect_hermitian

        for i, mat in enumerate(self.targets[target]):
            assert (
                len(mat.shape) == 2
            ), "matrix to discard non-hermiticity from must be a 2D matrix"

            self.target[target] = _reflect_hermitian(mat, retain_upper=retain_upper)

    def __len__(self):
        return self.nstructs


class PySCFPeriodicDataset(Dataset):
    # TODO: make format compatible with MolecularDataset
    def __init__(
        self,
        frames,
        frame_slice: slice = slice(None),
        kgrid: Union[List[int], List[List[int]]] = [1, 1, 1],
        matrices_kpoint: Union[torch.tensor, np.ndarray] = None,
        matrices_translation: Union[Dict, torch.tensor, np.ndarray] = None,
        overlap_kpoint: Union[torch.tensor, np.ndarray] = None,
        overlap_translation: Union[Dict, torch.tensor, np.ndarray] = None,
        target: List[str] = ["real_translation"],
        aux: List[str] = ["real_overlap"],
        use_precomputed: bool = True,
        device="cuda",
        orbs_name: str = "sto-3g",
        orbs: List = None,
        desired_shifts: List = None,
        nao: int = None,
    ):
        self.structures = frames
        self.frame_slice = frame_slice
        self.nstructs = len(frames)
        self.kmesh = kgrid
        self.kgrid_is_list = False
        if isinstance(kgrid[0], list):
            self.kgrid_is_list = True
            assert (
                len(self.kmesh) == self.nstructs
            ), "If kgrid is a list, it must have the same length as the number of structures"
        else:
            self.kmesh = [
                kgrid for _ in range(self.nstructs)
            ]  # currently easiest to do

        self.device = device
        self.basis = orbs  # actual orbitals
        self.basis_name = orbs_name
        self.use_precomputed = use_precomputed
        if not use_precomputed:
            raise NotImplementedError("You must use precomputed data for now.")

        self.target_names = target
        self.aux_names = aux
        self.desired_shifts_sup = (
            []
        )  # track the negative counterparts of desired_shifts as well
        self.cells = []
        self.phase_matrices = []
        self.supercells = []
        self.all_relative_shifts = []  # allowed shifts of kmesh

        for ifr, structure in enumerate(self.structures):
            # if self.kgrid_is_list:
            # cell, scell, phase = get_scell_phase(structure, self.kmesh[i])
            # else:
            cell, scell, phase = get_scell_phase(
                structure, self.kmesh[ifr], basis=self.basis_name
            )
            self.cells.append(cell)

            self.phase_matrices.append(torch.from_numpy(phase).to(self.device))
            self.all_relative_shifts.append(
                translation_vectors_for_kmesh(
                    cell, self.kmesh[ifr], return_rel=True
                ).tolist()
            )

        if desired_shifts is not None:
            self.desired_shifts = desired_shifts
        else:

            self.desired_shifts = np.unique(np.vstack(self.all_relative_shifts), axis=0)

        for s in self.desired_shifts:
            self.desired_shifts_sup.append(s)  # make this tuple(s)?
            self.desired_shifts_sup.append([-s[0], -s[1], -s[2]])
        self.desired_shifts_sup = np.unique(self.desired_shifts_sup, axis=0)

        assert matrices_kpoint is not None
        self.matrices_kpoint = torch.from_numpy(matrices_kpoint).to(self.device)

        matrices_translation = self.kpts_to_translation_target(
            self.matrices_kpoint, nao
        )
        ## FIXME : this will not work when we use a nonunifrom kgrid <<<<<<<<
        self.matrices_translation = {key: [] for key in matrices_translation[0]}
        [
            self.matrices_translation[tuple(k)].append(
                matrices_translation[ifr][tuple(k)]
            )
            for ifr in range(self.nstructs)
            for k in self.desired_shifts  # matrices_translation[ifr].keys() TOCHECK
        ]

        for k in self.desired_shifts:  # self.matrices_translation.keys(): TOCHECK
            self.matrices_translation[tuple(k)] = torch.stack(
                self.matrices_translation[tuple(k)]
            )
        # DO the same for the overlap
        if overlap_kpoint is not None:
            self.overlap_kpoint = torch.from_numpy(overlap_kpoint).to(self.device)
            overlap_translation = self.kpts_to_translation_target(
                self.overlap_kpoint, nao
            )
            ## FIXME : this will not work when we use a nonunifrom kgrid <<<<<<<<
            self.overlap_translation = {key: [] for key in overlap_translation[0]}
            [
                self.overlap_translation[tuple(k)].append(
                    overlap_translation[ifr][tuple(k)]
                )
                for ifr in range(self.nstructs)
                for k in self.desired_shifts  # matrices_translation[ifr].keys() TOCHECK
            ]

            for k in self.desired_shifts:  # self.matrices_translation.keys(): TOCHECK
                self.overlap_translation[tuple(k)] = torch.stack(
                    self.overlap_translation[tuple(k)]
                )

        self.target = {t: [] for t in self.target_names}
        for t in self.target_names:
            # keep only desired shifts
            if t == "real_translation":
                self.target[t] = self.matrices_translation
            elif t == "kpoint":
                self.target[t] = self.matrices_kpoint

    def get_kpoint_target(self, translated_matrices):
        """function to convert translated matrices to kpoint target with the phase matrices consistent with this dataset. Useful for combining ML predictions of translated matrices to kpoint matrices"""
        kmatrix = []
        for ifr in range(self.nstructs):
            framekmatrix = torch.zeros_like(
                torch.tensor(self.matrices_kpoint[0]), dtype=torch.complex128
            ).to(self.device)
            # ,self.cells[ifr].nao, self.cells[ifr].nao), dtype=np.complex128)
            for kpt in range(np.prod(self.kmesh[ifr])):
                for i, t in enumerate(translated_matrices.keys()):
                    # for i in range(len(translated_matrices)):
                    framekmatrix[kpt] += (
                        translated_matrices[t][ifr] * self.phase_matrices[ifr][i][kpt]
                    )
            kmatrix.append(framekmatrix)
        return kmatrix

    def kpts_to_translation_target(self, matrices_kpoint, nao=None):
        target = []
        for ifr in range(self.nstructs):
            rel_translations = translation_vectors_for_kmesh(
                self.cells[ifr], self.kmesh[ifr], wrap_around=False, return_rel=True
            )

            rel_translations = [tuple(x) for x in rel_translations]
            if nao is None:
                nao = self.cells[ifr].nao
            translated_matrices = torch.zeros(
                (
                    len(self.phase_matrices[ifr]),
                    nao,
                    nao,
                ),
                dtype=torch.complex128,
            ).to(self.device)
            for i in range(len(self.phase_matrices[ifr])):
                for kpt in range(np.prod(self.kmesh[ifr])):
                    translated_matrices[i] += (
                        matrices_kpoint[ifr][kpt]
                        * self.phase_matrices[ifr][i][kpt].conj()
                    )

            # translated_matrices = translated_matrices
            assert torch.isclose(
                torch.linalg.norm(translated_matrices - translated_matrices.real),
                torch.tensor(0.0).to(translated_matrices.real),
            ), (
                "error to real",
                torch.linalg.norm(translated_matrices - translated_matrices.real),
            )

            translated_matrices = {
                k: v for k, v in zip(rel_translations, translated_matrices.real)
            }
            target.append(translated_matrices)
        return target

    def check_block_(self):
        # mat -> blocks _> couple -> uncouple -> mat
        pass

    def __len__(self):
        return self.nstructs
