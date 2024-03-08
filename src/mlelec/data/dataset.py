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
import copy
from collections import defaultdict


# Dataset class  - to load and pass around structures, targets and
# required auxillary data wherever necessary
class precomputed_molecules(Enum):  # RENAME to precomputed_structures?
    water_1000 = "examples/data/water_1000"
    water_rotated = "examples/data/water_rotated"
    ethane = "examples/data/ethane"
    pbc_c2_rotated = "examples/data/pbc/c2_rotated"


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
        self.basis = orbs

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
                self.aux_data_names = ["overlap", "orbitals"]
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
            raise FileNotFoundError(
                "No structures found at {}".format(self.path + f"/{self.mol_name}.xyz")
            )

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
                print("Generating data")
                from mlelec.data.pyscf_calculator import calculator

                calc = calculator(
                    path=self.path,
                    mol_name=self.mol_name,
                    frame_slice=":",
                    target=self.target_names,
                )
                calc.calculate(basis_set=self.basis, verbose=1)
                calc.save_results()
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
        if "overlap" in self.aux_data_names and "density" in self.target_names:
            self.target_names.append("elec_population")
            self.target["elec_population"] = torch.sum(
                torch.einsum(
                    "bij, bji ->bi", self.target["density"], self.aux_data["overlap"]
                ),
                axis=1,
            )
            # This, for each frame is the Trace(overlap @ Density matrix) = number of electrons

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
    def __init__(
        self,
        molecule_data: MoleculeDataset,
        device: str = "cpu",
        model_type: Optional[str] = "acdc",
        features: Optional[TensorMap] = None,
        shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
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
        # self.molecule_data.shuffle(self.indices)
        # self.molecule_data = molecule_data

        self.structures = self.molecule_data.structures
        self.target = self.molecule_data.target
        # print(self.target, next(iter(self.target.values())))
        # sets the first target as the primary target - # FIXME
        self.target_class = ModelTargets(self.molecule_data.target_names[0])
        self.target = self.target_class.instantiate(
            next(iter(self.molecule_data.target.values())),
            frames=self.structures,
            orbitals=self.molecule_data.aux_data.get("orbitals", None),
            device=device,
            **kwargs,
        )

        self.natoms_list = [len(frame) for frame in self.structures]
        self.species = set([tuple(f.numbers) for f in self.structures])

        self.aux_data = self.molecule_data.aux_data
        self.rng = None
        self.model_type = model_type  # flag to know if we are using acdc features or want to cycle hrough positons
        if self.model_type == "acdc":
            self.features = features

        self.train_frac = kwargs.get("train_frac", 0.7)
        self.val_frac = kwargs.get("val_frac", 0.2)

    def _shuffle(self, random_seed: int = None):
        if random_seed is None:
            self.rng = torch.default_generator
        else:
            self.rng = torch.Generator().manual_seed(random_seed)

        self.indices = torch.randperm(
            self.nstructs, generator=self.rng
        )  # .to(self.device)

        # update self.structures to reflect shuffling
        # self.structures_original = self.structures.copy()
        # self.structures = [self.structures_original[i] for i in self.indices]
        # self.target.shuffle(self.indices)
        # self.molecule_data.shuffle(self.indices)

    def _get_subset(self, y: TensorMap, indices: torch.tensor):
        indices = indices.cpu().numpy()
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
        self,
        train_frac: float = None,
        val_frac: float = None,
        test_frac: Optional[float] = None,
    ):
        # TODO: handle this smarter

        # overwrite self train/val/test indices
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

        # self.train_idx = torch.tensor(list(range(int(train_frac * self.nstructs))))
        # self.val_idx = torch.tensor(list(
        #     range(int(train_frac * self.nstructs), int((train_frac + val_frac) * self.nstructs))
        # ))
        # self.test_idx = torch.tensor(list(range(int((train_frac + val_frac) * self.nstructs), self.nstructs)))
        # assert (
        #     len(self.test_idx)
        #     > 0  # and len(self.val_idx) > 0 and len(self.train_idx) > 0
        # ), "Split indices not generated properly"

        self.train_idx = self.indices[: int(self.train_frac * self.nstructs)].sort()[0]
        self.val_idx = self.indices[
            int(self.train_frac * self.nstructs) : int(
                (self.train_frac + self.val_frac) * self.nstructs
            )
        ].sort()[0]
        self.test_idx = self.indices[
            int((self.train_frac + self.val_frac) * self.nstructs) :
        ].sort()[0]
        if self.test_frac>0:
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


from mlelec.data.pyscf_calculator import kpoint_to_translations, translations_to_kpoint
from mlelec.data.pyscf_calculator import (
    get_scell_phase,
    map_supercell_to_relativetrans,
    translation_vectors_for_kmesh,
    _map_transidx_to_relative_translation,
    map_mic_translations,
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
    from mlelec.utils.pbc_utils import fourier_transform, inverse_fourier_transform, get_T_from_pair
    # TODO: make format compatible with MolecularDataset
    def __init__(
        self,
        frames,
        frame_slice: slice = slice(None),
        kmesh: Union[List[int], List[List[int]]] = [1, 1, 1],
        fock_kspace: Union[List, torch.tensor, np.ndarray] = None,
        fock_realspace: Union[Dict, torch.tensor, np.ndarray] = None,
        overlap_kspace: Union[List, torch.tensor, np.ndarray] = None,
        overlap_realspace: Union[Dict, torch.tensor, np.ndarray] = None,
        # aux: List[str] = ["real_overlap"],
        use_precomputed: bool = True, 
        device="cuda",
        orbs_name: str = "sto-3g",
        orbs: List = None,
    ):
        self.structures = frames
        self.frame_slice = frame_slice
        self.nstructs = len(frames)
        self.kmesh = kmesh
        self.kmesh_is_list = False
        if isinstance(kmesh[0], list):
            self.kmesh_is_list = True
            assert (
                len(self.kmesh) == self.nstructs
            ), "If kmesh is a list, it must have the same length as the number of structures"
        else:
            self.kmesh = [kmesh for _ in range(self.nstructs)]  # currently easiest to do

        self.device = device
        self.basis = orbs  # actual orbitals
        self.basis_name = orbs_name
        self.use_precomputed = use_precomputed
        if not use_precomputed:
            raise NotImplementedError("You must use precomputed data for now.")

        # self.target_names = target
        # self.aux_names = aux
        
        self.cells = []
        self.phase_matrices = []
        self.supercells = []

        for ifr, structure in enumerate(self.structures):
            cell, scell, phase = get_scell_phase(
                structure, self.kmesh[ifr], basis=self.basis_name
            )
            self.cells.append(cell)

        self._translation_counter = self.compute_translation_counter()
        self._translation_dict = self.compute_translation_dict()

        if fock_kspace is not None:
            self.set_fock_kspace(fock_kspace)
        if overlap_kspace is not None:
            self.set_overlap_kspace(overlap_kspace)

        if fock_realspace is None:
            assert self.fock_kspace is not None, "Either the real space or reciprocal space Fock matrices must be provided."
            self.fock_realspace, self._fock_realspace_negative_translations = self.compute_matrices_realspace(self.fock_kspace)
            self.realspace_translations = [list(m.keys()) for m in self.fock_realspace]
        else:
            raise NotImplementedError("For now only reciprocal space matrices are allowed.")
            # self.fock_realspace = self.set_matrices_realspace(fock_realspace)

        if overlap_realspace is None and self.overlap_kspace is not None:
            self.overlap_realspace = self.compute_matrices_realspace(self.overlap_kspace)
        else:
            raise NotImplementedError("For now only reciprocal space matrices are allowed.")



        
    



                
    def compute_translation_counter(self):
        from itertools import product
        from mlelec.utils.pbc_utils import get_T_from_pair
        counter_T = []
        for ifr, (frame, kmesh) in enumerate(zip(self.structures, self.kmesh)):
            counter_T.append({})
            natm = frame.get_global_number_of_atoms()
            supercell = frame.repeat(kmesh)
            shifts = [list(p) for p in product(range(kmesh[0]), range(kmesh[1]), range(kmesh[2]))]
            for dummy_T in shifts:
                for i in range(frame.get_global_number_of_atoms()):
                    for j in range(frame.get_global_number_of_atoms()):
                        _, _, mic_T = get_T_from_pair(frame, supercell, i, j, dummy_T, kmesh)
                        mic_T = tuple(mic_T)
                        if mic_T not in counter_T[ifr]:
                            counter_T[ifr][mic_T] = np.zeros((natm, natm))
                        counter_T[ifr][mic_T][i,j] += 1
        return counter_T
    
    def compute_translation_dict(self):
        T_dict = []

        for ifr, counter in enumerate(self._translation_counter):
            natm = self.structures[ifr].get_global_number_of_atoms()
            T_dict.append({})
            full_T_list = list(counter.keys())
            i0 = 0
            i_skip = 0
            for i in range(len(full_T_list)):
                if i < i0 + i_skip:
                    continue
                
                summa = 0
                counter_list = []
                T_list = []
                i_skip = 0
                while summa != natm**2:
                    counter_list.append(counter[full_T_list[i+i_skip]])
                    T_list.append(full_T_list[i+i_skip])
                    summa = np.sum(counter_list)
                    i_skip += 1
                i0 = i
                T_dict[ifr][full_T_list[i]] = T_list
        return T_dict

    # def get_kpoint_target(self, translated_matrices):
    #     """function to convert translated matrices to kpoint target with the phase matrices consistent with this dataset. Useful for combining ML predictions of translated matrices to kpoint matrices"""
    #     kmatrix = []
    #     for ifr in range(self.nstructs):
    #         framekmatrix = torch.zeros_like(
    #             torch.tensor(self.fock_kspace[0]), dtype=torch.complex128
    #         ).to(self.device)
    #         # ,self.cells[ifr].nao, self.cells[ifr].nao), dtype=np.complex128)
    #         for kpt in range(np.prod(self.kmesh[ifr])):
    #             for i, t in enumerate(translated_matrices.keys()):
    #                 # for i in range(len(translated_matrices)):
    #                 framekmatrix[kpt] += (
    #                     translated_matrices[t][ifr] * self.phase_matrices[ifr][i][kpt]
    #                 )
    #         kmatrix.append(framekmatrix)
    #     return kmatrix

 
    # def check_block_(self):
    #     # mat -> blocks _> couple -> uncouple -> mat
    #     pass

    def _set_matrices_realspace(self, matrices_realspace):
        _matrices_realspace = []
        for m in matrices_realspace:
            _matrices_realspace.append({})
            for k in m:
                if isinstance(m[k], torch.Tensor):
                    _matrices_realspace[-1][k] = m[k]
                elif isinstance(m[k], np.ndarray):
                    _matrices_realspace[-1][k] = torch.from_numpy(m[k])
                elif isinstance(m[k], list):
                    _matrices_realspace[-1][k] = torch.tensor(m[k])
                else:
                    raise ValueError('matrices_realspace should be one among torch.tensor, numpy.ndarray, or list')
        
        return _matrices_realspace
   
    def _set_matrices_kspace(self, matrices_kspace):
        if isinstance(matrices_kspace, list):
            if isinstance(matrices_kspace[0], np.ndarray):
                _matrices_kspace = [torch.from_numpy(m).to(self.device) for m in matrices_kspace]
            elif isinstance(matrices_kspace[0], torch.Tensor):
                _matrices_kspace = [m.to(self.device) for m in matrices_kspace]
            elif isinstance(matrices_kspace[0], list):
                _matrices_kspace = [torch.tensor(m).to(self.device) for m in matrices_kspace]
            else:
                raise TypeError("matrices_kspace should be a list [torch.Tensor, np.ndarray, or lists]")
        elif isinstance(matrices_kspace, np.ndarray):
            assert matrices_kspace.shape[0] == len(self.structures), "You must provide matrices_kspace for each structure" 
            _matrices_kspace = [torch.from_numpy(m).to(self.device) for m in matrices_kspace]
        elif isinstance(matrices_kspace, torch.Tensor):
            assert matrices_kspace.shape[0] == len(self.structures), "You must provide matrices_kspace for each structure" 
            _matrices_kspace = [m.to(self.device) for m in matrices_kspace]
        else:
            raise TypeError("matrices_kspace should be either a list [torch.Tensor, np.ndarray, or lists], a np.ndarray, or torch.Tensor.")
        
        return _matrices_kspace
    
    def set_fock_kspace(self, fock_kspace):
        self.fock_kspace = self._set_matrices_kspace(fock_kspace)

    def set_overlap_kspace(self, overlap_kspace):
        self.overlap_kspace = self._set_matrices_kspace(overlap_kspace)
    
    def compute_matrices_realspace(self, matrices_kspace):
        from mlelec.utils.pbc_utils import fourier_transform

        H_T_plus = []
        H_T_minus = []

        for ifr, (kmesh, H_k) in enumerate(zip(self.kmesh, matrices_kspace)):


            H_T_plus.append({})
            H_T_minus.append({})
            
            kpts = self.cells[ifr].get_scaled_kpts(self.cells[ifr].make_kpts(kmesh))
            natm = self.structures[ifr].get_global_number_of_atoms()
            nao = self.cells[ifr].nao // natm # FIXME: in general this is wrong
            
            for T_dummy in self._translation_dict[ifr]:

                H_T_plus[ifr][T_dummy]  = np.zeros((natm*nao, natm*nao), dtype = np.complex128)
                H_T_minus[ifr][T_dummy] = np.zeros((natm*nao, natm*nao), dtype = np.complex128)
                
                for T in self._translation_dict[ifr][T_dummy]:
                    pairs = np.where(self._translation_counter[ifr][T])
                    for i, j in zip(*pairs):
                    
                        idx_i = slice(nao*i, nao*(i+1))
                        idx_j = slice(nao*j, nao*(j+1))

                        H_T_plus[ifr][T_dummy][idx_i, idx_j]  = fourier_transform(H_k, kpts, T)[idx_i, idx_j]
                        H_T_minus[ifr][T_dummy][idx_i, idx_j] = fourier_transform(H_k, kpts, -np.array(T))[idx_i, idx_j]
                        
            for mic_T in H_T_plus[ifr]:
                assert np.allclose(H_T_plus[ifr][mic_T], H_T_plus[ifr][mic_T].real), np.allclose(H_T_plus[ifr][mic_T], H_T_plus[ifr][mic_T].real)
                H_T_plus[ifr][mic_T] = torch.from_numpy(H_T_plus[ifr][mic_T].real)
                assert np.allclose(H_T_minus[ifr][mic_T], H_T_minus[ifr][mic_T].real), np.allclose(H_T_minus[ifr][mic_T], H_T_minus[ifr][mic_T].real)
                H_T_minus[ifr][mic_T] = torch.from_numpy(H_T_minus[ifr][mic_T].real)

        return H_T_plus, H_T_minus

    def compute_matrices_kspace(self, matrices_realspace):
        from mlelec.utils.pbc_utils import inverse_fourier_transform
        matrices_kspace = []
        for ifr, H in enumerate(matrices_realspace):
            kpts = self.cells[ifr].get_scaled_kpts(self.cells[ifr].make_kpts(self.kmesh[ifr]))
            matrices_kspace.append([])
            for k in kpts:
                matrices_kspace[ifr].append(inverse_fourier_transform(np.array(list(H.values())), np.array(list(H.keys())), k))
            matrices_kspace[ifr] = torch.from_numpy(np.array(matrices_kspace[ifr]))
        return matrices_kspace
    
    def __len__(self):
        return self.nstructs
    

# Tests
    
def check_fourier_duality(matrices_kspace, matrices_realspace, kpoints, tol = 1e-10):
    from mlelec.utils.pbc_utils import inverse_fourier_transform
    reconstructed_matrices_kspace = []
    for ifr, H in enumerate(matrices_realspace):
        reconstructed_matrices_kspace.append([])
        kpts = kpoints[ifr]
        for k in kpts:
            reconstructed_matrices_kspace[ifr].append(inverse_fourier_transform(np.array(list(H.values())), np.array(list(H.keys())), k))
        reconstructed_matrices_kspace[ifr] = torch.from_numpy(np.array(reconstructed_matrices_kspace[ifr]))
        assert reconstructed_matrices_kspace[ifr].shape == matrices_kspace[ifr].shape
        assert torch.norm(reconstructed_matrices_kspace[ifr] - matrices_kspace[ifr]) < tol, (ifr, torch.norm(reconstructed_matrices_kspace[ifr] - matrices_kspace[ifr]))