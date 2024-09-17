import io
import os
import sys
import warnings
from contextlib import redirect_stderr
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import ase
import hickle
import metatensor.torch as mts
import numpy as np
import torch
from ase.io import read
from metatensor.torch import Labels, TensorBlock, TensorMap
from torch.utils.data import Dataset

from mlelec.targets import ModelTargets

import copy
from pathlib import Path

import torch.utils.data as data

from mlelec.data.pyscf_calculator import _instantiate_pyscf_mol, get_scell_phase
from mlelec.utils.pbc_utils import inverse_fourier_transform, unique_Aij_block
from mlelec.utils.twocenter_utils import map_targetkeys_to_featkeys

warnings.simplefilter("always", DeprecationWarning)


class QMDataset:
    """
    Class containing information about the quantum chemistry calculation and its
    results.
    """

    def __init__(
        self,
        frames,
        # frame_slice: slice = slice(None),
        kmesh: Union[List[int], List[List[int]]] = None,
        fock_kspace: Union[List, torch.tensor, np.ndarray] = None,
        fock_realspace: Union[Dict, torch.tensor, np.ndarray] = None,
        overlap_kspace: Union[List, torch.tensor, np.ndarray] = None,
        overlap_realspace: Union[Dict, torch.tensor, np.ndarray] = None,
        # aux: List[str] = ["real_overlap"],
        # use_precomputed: bool = True,
        device="cuda",
        orbs_name: str = "sto-3g",
        orbs: List = None,
        dimension: int = 3,
        fix_p_orbital_order=False,
        apply_condon_shortley=False,
    ):
        # TODO: probably to remove soon. Keeping the warning for now to avoid silly
        # mistakes
        if fix_p_orbital_order or apply_condon_shortley:
            warnings.warn(
                (
                    "The `fix_p_orbital_order` and `apply_condon_shortley` options have"
                    " been moved to MLDataset."
                ),
                DeprecationWarning,
            )
        fix_p_orbital_order = False
        apply_condon_shortley = False

        self._device = device
        self._basis = orbs
        self._basis_name = orbs_name

        self._dimension = dimension
        self._structures = self._wrap_frames(frames)

        # TODO: is this necessary?
        # self.frame_slice = frame_slice

        self._kmesh = self._set_kmesh(kmesh)
        self._nao = self._set_nao()
        self._ncore = self._set_ncore()

        self._initialize_pyscf_objects()

        # Assign/compute Hamiltonians and Overlaps
        self._set_matrices(
            fock_realspace=fock_realspace,
            fock_kspace=fock_kspace,
            overlap_realspace=overlap_realspace,
            overlap_kspace=overlap_kspace,
        )

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if value not in ["cpu", "cuda"]:
            raise ValueError("device must be either cpu or cuda")
        self._device = value

    @property
    def basis(self):
        return self._basis

    @property
    def basis_name(self):
        return self._basis_name

    @property
    def dimension(self):
        return self._dimension

    @property
    def is_molecule(self):
        return self._dimension == 0

    @property
    def structures(self):
        return self._structures

    def _wrap_frames(self, frames):
        for f in frames:
            if self.dimension == 2:
                f.pbc = [True, True, False]
                f.wrap(center=(0, 0, 0), eps=1e-60)
                f.pbc = True
            elif self.dimension == 3:
                f.wrap(center=(0, 0, 0), eps=1e-60)
                f.pbc = True
            elif self.dimension == 0:  # Handle molecules
                f.pbc = False
            else:
                raise NotImplementedError("dimension must be 0, 2 or 3")
        return frames

    @property
    def nstructs(self):
        return len(self.structures)

    @property
    def kmesh(self):
        return self._kmesh

    def _set_kmesh(self, kmesh):
        if self.is_molecule:
            return None
        else:
            if kmesh is None:
                kmesh = [1, 1, 1]
        if isinstance(kmesh[0], list):
            assert len(kmesh) == self.nstructs, (
                "If kmesh is a list, it must have the same "
                "length as the number of structures"
            )
            _kmesh = kmesh
        else:
            _kmesh = [kmesh for _ in range(self.nstructs)]

        return _kmesh

    @property
    def nao(self):
        return self._nao

    def _set_nao(self):
        return [
            sum(len(self._basis[s]) for s in frame.numbers)
            for frame in self._structures
        ]

    @property
    def ncore(self):
        return self._ncore

    def _set_ncore(self):
        ncore = {}
        for s in self._basis:
            basis = np.array(self._basis[s])
            nmin = np.min(basis[:, 0])
            ncore[s] = 0
            for n in np.arange(nmin):
                for l in range(n):
                    ncore[s] += 2 * (2 * l + 1)
            llist = set(basis[np.argwhere(basis[:, 0] == nmin)][:, 0, 1])
            llist_nmin = set(range(max(llist) + 1))
            l_diff = llist_nmin - llist
            for l in l_diff:
                ncore[s] += 2 * (2 * l + 1)
        return ncore

    def _initialize_pyscf_objects(self):
        if self.is_molecule:
            self._mols = self._initialize_pyscf_mol()
            self._cells = None
        else:
            self._mols = None
            self._cells = self._initialize_pyscf_cell()
            self._set_kpts()

    def _initialize_pyscf_cell(self):
        cells = []

        _stderr_capture = io.StringIO()

        with redirect_stderr(_stderr_capture):
            for ifr, structure in enumerate(self._structures):
                cell, _, _ = get_scell_phase(
                    structure, self._kmesh[ifr], basis=self._basis_name
                )
                cells.append(cell)
        try:
            assert (
                _stderr_capture.getvalue()
                == """WARNING!
  Very diffused basis functions are found in the basis set. They may lead to severe
  linear dependence and numerical instability.  You can set  cell.exp_to_discard=0.1
  to remove the diffused Gaussians whose exponents are less than 0.1.\n\n"""
                * len(self)
            )
        except:
            sys.stderr.write(_stderr_capture.getvalue())

        return cells

    def _initialize_pyscf_mol(self):
        mols = []
        _stderr_capture = io.StringIO()

        with redirect_stderr(_stderr_capture):
            for structure in self._structures:
                mols.append(_instantiate_pyscf_mol(structure, basis=self._basis_name))
        try:
            assert (
                _stderr_capture.getvalue()
                == """WARNING!
  Very diffused basis functions are found in the basis set. They may lead to severe
  linear dependence and numerical instability.  You can set  cell.exp_to_discard=0.1
  to remove the diffused Gaussians whose exponents are less than 0.1.\n\n"""
                * len(self)
            )
        except:
            sys.stderr.write(_stderr_capture.getvalue())

        return mols

    @property
    def cells(self):
        if self.is_molecule:
            raise AttributeError("This system is not periodic")
        return self._cells

    @property
    def mols(self):
        if not self.is_molecule:
            raise AttributeError("This system is not a molecule")
        return self._mols

    @property
    def kpts_rel(self):
        return self._kpts_rel

    @property
    def kpts_abs(self):
        return self._kpts_abs

    def _set_kpts(self):
        self._kpts_rel = [
            c.get_scaled_kpts(c.make_kpts(k)) for c, k in zip(self.cells, self.kmesh)
        ]
        self._kpts_abs = [
            c.get_abs_kpts(kpts) for c, kpts in zip(self.cells, self.kpts_rel)
        ]

    @property
    def fock_realspace(self):
        return self._fock_realspace

    @property
    def fock_kspace(self):
        return self._fock_kspace

    @property
    def overlap_realspace(self):
        return self._overlap_realspace

    @property
    def overlap_kspace(self):
        return self._overlap_kspace

    def _set_matrices(
        self,
        fock_realspace: Optional[Union[Dict, List]] = None,
        fock_kspace: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
        overlap_realspace: Optional[Union[Dict, List]] = None,
        overlap_kspace: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
    ):
        self._fock_realspace, self._fock_kspace = self._assign_or_compute_matrices(
            fock_realspace, fock_kspace, self._set_fock_kspace
        )
        (
            self._overlap_realspace,
            self._overlap_kspace,
        ) = self._assign_or_compute_matrices(
            overlap_realspace, overlap_kspace, self._set_overlap_kspace
        )

    def _assign_or_compute_matrices(
        self,
        realspace: Optional[Union[Dict, List]],
        kspace: Optional[Union[np.ndarray, torch.Tensor, List]],
        kspace_setter: Any,
    ):
        if kspace is not None and realspace is None:
            kspace_setter(kspace)
            realspace = None
        elif kspace is None and realspace is not None:
            realspace = self._set_matrices_realspace(realspace)
            if not self.is_molecule:
                kspace = self.bloch_sum(realspace, is_tensor=True)
        elif kspace is None and realspace is None:
            warnings.warn("Matrices not provided.")
            realspace = None
            kspace = None
        elif kspace is not None and realspace is not None:
            raise NotImplementedError(
                "Check consistency between realspace and kspace matrices."
            )
        else:
            raise NotImplementedError("Unhandled condition.")

        return realspace, kspace

    def _set_matrices_realspace(
        self, matrices_realspace: Union[Dict, List[Dict]]
    ) -> List[Dict]:
        if not isinstance(matrices_realspace[0], dict):
            assert (
                self.is_molecule
            ), "matrices_realspace should be a dictionary unless it's a molecule"
            return matrices_realspace

        return [self._convert_matrix(m) for m in matrices_realspace]

    def _convert_matrix(self, matrix: Dict) -> Dict:
        return {k: self._to_tensor(v) for k, v in matrix.items()}

    def _to_tensor(self, data: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device=self.device)
        elif isinstance(data, list):
            return torch.tensor(data, device=self.device)
        else:
            raise ValueError(
                "Matrix elements should be torch.Tensor, numpy.ndarray, or list"
            )

    def _set_matrices_kspace(
        self, matrices_kspace: Union[List, np.ndarray, torch.Tensor]
    ) -> List[torch.Tensor]:
        if isinstance(matrices_kspace, list):
            return [self._to_tensor(m) for m in matrices_kspace]
        elif isinstance(matrices_kspace, (np.ndarray, torch.Tensor)):
            assert matrices_kspace.shape[0] == len(
                self.structures
            ), "Provide matrices_kspace for each structure"
            return [
                self._to_tensor(matrices_kspace[i])
                for i in range(matrices_kspace.shape[0])
            ]
        else:
            raise TypeError(
                "matrices_kspace should be a list, np.ndarray, or torch.Tensor"
            )

    def _set_fock_kspace(self, fock_kspace: Union[List, np.ndarray, torch.Tensor]):
        self._fock_kspace = self._set_matrices_kspace(fock_kspace)

    def _set_overlap_kspace(
        self, overlap_kspace: Union[List, np.ndarray, torch.Tensor]
    ):
        self._overlap_kspace = self._set_matrices_kspace(overlap_kspace)

    def compute_matrices_realspace(self, matrices_kspace: Any):
        raise NotImplementedError("This must happen when the targets are computed!")

    def bloch_sum(
        self,
        matrices_realspace: List[Dict],
        is_tensor: bool = True,
        structure_ids: Optional[List[int]] = None,
    ) -> List[Optional[torch.Tensor]]:
        matrices_kspace = []
        structure_ids = structure_ids or range(len(matrices_realspace))

        for ifr, H in zip(structure_ids, matrices_realspace):
            if H:
                H_T = self._stack_tensors(H, is_tensor)
                T_list = self._convert_keys_to_tensor(H, is_tensor)
                k = torch.from_numpy(self.kpts_rel[ifr]).to(device=self.device)
                matrices_kspace.append(
                    inverse_fourier_transform(H_T, T_list=T_list, k=k, norm=1)
                )
            else:
                matrices_kspace.append(
                    None
                )  # FIXME: not the best way to handle this situation

        return matrices_kspace

    def _stack_tensors(self, H: Dict, is_tensor: bool) -> torch.Tensor:
        if is_tensor:
            return torch.stack(list(H.values())).to(device=self.device)
        else:
            return torch.from_numpy(np.array(list(H.values()))).to(device=self.device)

    def _convert_keys_to_tensor(self, H: Dict, is_tensor: bool) -> torch.Tensor:
        if is_tensor:
            return torch.tensor(list(H.keys()), dtype=torch.float64, device=self.device)
        else:
            return torch.from_numpy(np.array(list(H.keys()), dtype=np.float64)).to(
                device=self.device
            )

    def __len__(self):
        return self.nstructs


# Tests
def check_fourier_duality(matrices_kspace, matrices_realspace, kpoints, tol=1e-10):
    from mlelec.utils.pbc_utils import inverse_fourier_transform

    reconstructed_matrices_kspace = []
    for ifr, H in enumerate(matrices_realspace):
        reconstructed_matrices_kspace.append([])
        kpts = kpoints[ifr]
        for k in kpts:
            reconstructed_matrices_kspace[ifr].append(
                inverse_fourier_transform(
                    np.array(list(H.values())), np.array(list(H.keys())), k
                )
            )
        reconstructed_matrices_kspace[ifr] = torch.from_numpy(
            np.array(reconstructed_matrices_kspace[ifr])
        )
        assert reconstructed_matrices_kspace[ifr].shape == matrices_kspace[ifr].shape
        assert (
            torch.norm(reconstructed_matrices_kspace[ifr] - matrices_kspace[ifr]) < tol
        ), (ifr, torch.norm(reconstructed_matrices_kspace[ifr] - matrices_kspace[ifr]))


####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################


mlelec_dir = Path(__file__).parents[3]


# Dataset class  - to load and pass around structures, targets and
# required auxillary data wherever necessary
class precomputed_molecules(Enum):  # RENAME to precomputed_structures?
    water_1000 = f"{mlelec_dir}/examples/data/water_1000"
    water_rotated = f"{mlelec_dir}/examples/data/water_rotated"
    ethane = f"{mlelec_dir}/examples/data/ethane"
    pbc_c2_rotated = f"{mlelec_dir}/examples/data/pbc/c2_rotated"


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
                        (
                            "No structures found at DATA path, either specify "
                            "frame_path or ensure strures present at data_path"
                        )
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
        # TODO: ensure that all keys of self.target names are filled even if they are
        # not provided in target_data
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
            # This, for each frame is the Trace(overlap @ Density matrix) = number
            # of electrons

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

    def __len__(self):
        return len(self.structures)


class MLDataset(Dataset):
    # TODO: add compatibility with PeriodicDataset
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
        self.model_type = model_type  # flag to know if we are using acdc features or
        # want to cycle hrough positons
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

    def _get_subset(self, y: torch.ScriptObject, indices: torch.tensor):
        # indices = indices.cpu().numpy()
        assert (
            isinstance(y, torch.ScriptObject) and y._type().name() == "TensorMap"
        ), "y must be a TensorMap"

        # for k, b in y.items():
        #     b = b.values.to(device=self.device)
        return mts.slice(
            y,
            axis="samples",
            labels=Labels(
                names=["structure"], values=torch.tensor(indices).reshape(-1, 1)
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
            assert (
                len(self.test_idx)
                > 0.0  # and len(self.val_idx) > 0 and len(self.train_idx) > 0
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
        # if torch.is_tensor(idx):
        #     # idx = [i.item() for i in idx]
        #     idx = idx.tolist()
        if not self.model_type == "acdc":
            return self.structures[idx], self.target.tensor[idx]
        else:
            assert (
                self.features is not None
            ), "Features not set, call _set_features() first"
            x = mts.slice(
                self.features,
                axis="samples",
                labels=Labels(names=["structure"], values=idx.reshape(-1, 1)),
            )
            if self.model_return == "blocks":
                y = mts.slice(
                    self.target.blocks,
                    axis="samples",
                    labels=Labels(names=["structure"], values=idx.reshape(-1, 1)),
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


def split_block_by_Aij(block):
    Aij, where_inv = unique_Aij_block(block)

    values = {}
    b_values = block.values
    for I, (A, i, j) in enumerate(Aij):
        idx = np.where(where_inv == I)[0]
        values[A, i, j] = b_values[idx]

    return values


def split_block_by_Aij_mts(block):
    Aij, where_inv = unique_Aij_block(block)

    new_blocks = {}
    b_values = block.values
    b_samples = block.samples
    b_components = block.components
    b_properties = block.properties
    for I, (A, i, j) in enumerate(Aij):
        idx = np.where(where_inv == I)[0]
        new_blocks[A, i, j] = TensorBlock(
            samples=Labels(
                b_samples.names, torch.tensor(b_samples.values[idx].tolist())
            ),
            components=b_components,
            properties=b_properties,
            values=b_values[idx],
        )

    return new_blocks


def split_by_Aij(tensor, features=None):
    if features is None:
        values = {}
        keys = {}
        for k, b in tensor.items():
            kl = tuple(k.values.tolist())
            value = split_block_by_Aij(b)

            for Aij in value:
                if Aij not in values:
                    values[Aij] = []
                    keys[Aij] = []
                keys[Aij].append(kl)
                values[Aij].append(value[Aij])

        outdict = {}
        for Aij in values:
            outdict[Aij] = {k: v for k, v in zip(keys[Aij], values[Aij])}

        return outdict

    else:
        target_values = {}
        feature_values = {}
        keys = {}

        for k, target in tensor.items():
            kl = tuple(k.values.tolist())
            feature = map_targetkeys_to_featkeys(features, k)

            tvalue = split_block_by_Aij(target)
            fvalue = split_block_by_Aij(feature)

            for Aij in tvalue:
                if Aij not in target_values:
                    target_values[Aij] = []
                    feature_values[Aij] = []
                    keys[Aij] = []
                keys[Aij].append(kl)
                target_values[Aij].append(tvalue[Aij])
                feature_values[Aij].append(fvalue[Aij])

        target_outdict = {}
        features_outdict = {}
        for Aij in target_values:
            target_outdict[Aij] = {k: v for k, v in zip(keys[Aij], target_values[Aij])}
            features_outdict[Aij] = {
                k: v for k, v in zip(keys[Aij], feature_values[Aij])
            }

        return features_outdict, target_outdict


def split_by_Aij_mts(tensor, features=None):
    if features is None:
        blocks = {}
        keys = {}
        for k, b in tensor.items():
            block = split_block_by_Aij_mts(b)

            for Aij in block:
                if Aij not in blocks:
                    blocks[Aij] = []
                    keys[Aij] = []
                keys[Aij].append(k.values.tolist())
                blocks[Aij].append(block[Aij])

        tmaps = {}
        for Aij in blocks:
            tmap_keys = Labels(tensor.keys.names, torch.tensor(keys[Aij]))
            tmap_blocks = blocks[Aij]
            tmaps[Aij] = TensorMap(tmap_keys, tmap_blocks)

        return tmaps

    else:
        feature_blocks = {}
        target_blocks = {}
        keys = {}
        for k, b in tensor.items():
            feature = map_targetkeys_to_featkeys(features, k)

            target_block = split_block_by_Aij_mts(b)
            feature_block = split_block_by_Aij_mts(feature)

            for Aij in target_block:
                if Aij not in target_blocks:
                    feature_blocks[Aij] = []
                    target_blocks[Aij] = []
                    keys[Aij] = []
                kval = k.values.tolist()
                keys[Aij].append(kval)
                feature_blocks[Aij].append(feature_block[Aij])
                target_blocks[Aij].append(target_block[Aij])

        tmaps_feature = {}
        tmaps_target = {}
        for Aij in feature_blocks:
            tmap_keys = Labels(tensor.keys.names, torch.tensor(keys[Aij]))
            tmaps_feature[Aij] = TensorMap(tmap_keys, feature_blocks[Aij])
            tmaps_target[Aij] = TensorMap(tmap_keys, target_blocks[Aij])

        return tmaps_feature, tmaps_target
