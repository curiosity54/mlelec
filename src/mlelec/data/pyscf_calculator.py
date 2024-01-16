# write a functionality to take input structures, calculate desired target
from typing import List, Optional, Union
from ase.io import read
import ase
import numpy as np
import pyscf  # eventually replace with pyscfad #TODO
import os
from pathlib import Path
import hickle
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import torch
from collections import defaultdict
from ase.data import atomic_numbers

# will be updated to work directly with datasets so that we have access
# to the structures, all species present and ensure basis for all
import re


def convert_str_to_nlm(x: str):
    """x : string of the form 'nlm' where n is the principal quantum number, l is the azimuthal quantum number and m is the magnetic quantum number
    example: '2px' -> [2,1,1]
    """
    orb_map = {
        "s": [0, 0],
        "px": [1, 1],
        "py": [1, -1],
        "pz": [1, 0],
        "dxy": [2, -2],
        "dyz": [2, -1],
        "dz^2": [2, 0],
        "dxz": [2, 1],
        "dx2-y2": [2, 2],
        "f-3": [3, -3],
        # TODO: For orbitals>=f, the name is lm - might be easier to extract
        "f-2": [3, -2],
        "f-1": [3, -1],
        "f+0": [3, 0],
        "f+1": [3, 1],
        "f+2": [3, 2],
        "f+3": [3, 3],
        # "f3x^2y-y^2":[3,-3],
        # "fyz^2":[3,-2],
        # "fxyz": [3, -1],
        # "fz3": [3, 0],
        # "fxz^2": [3,1],
        # "fx^2z-y^2z": [3,2],
        # "f": [3,3]
    }
    match = re.match(r"([0-9]+)(.+)", x, re.I)
    # match = re.match(r"([0-9]+)([a-z]+)", x, re.I)
    n, lm = match.groups()
    print(n, lm)
    return [int(n)] + orb_map[lm]


def _instantiate_pyscf_mol(frame, basis="sto-3g"):
    mol = pyscf.gto.Mole()
    mol.atom = pyscf_ase.ase_atoms_to_pyscf(frame)
    mol.basis = basis
    mol.build()
    return mol


class calculator:
    def __init__(
        self,
        path: str,
        structures: Optional[List[ase.Atoms]] = None,
        mol_name: str = "water",
        frame_slice=":",
        dft: bool = False,
        target: Union[str, List[str]] = "fock",
    ):  # self.kwargs:Dict[str, Any]
        self.path = path
        self.structures = structures
        self.mol_name = mol_name
        self.slice = frame_slice
        self.dft = dft
        self.load_structures()
        self.pbc = False
        if np.any(self.structures[0].cell):
            self.pbc = True
        self.nframes = len(self.structures)
        print("Number of frames: ", self.nframes)

        if isinstance(target, str):
            target = [target]
        self.target = target
        if "fock" in self.target:
            self.target.append("overlap")
        self.results = {t: [] for t in self.target}
        # self.results = {str(target): []}
        self.ao_labels = defaultdict(list)

    def load_structures(self):
        if self.structures is None:
            try:
                print("Loading")
                self.structures = read(
                    self.path + "/{}.xyz".format(self.mol_name), index=self.slice
                )
            except:
                raise FileNotFoundError("No structures found at the given path")

    def calculate(self, basis_set: str = "sto-3g", **kwargs: Optional[dict]):
        """
        kwargs -
        dft: run dft
        pbc: bool = False,
        spin: int = 0,
        charge: int = 0,
        symmetry: bool = False,
        kpts: Optional[List] = None,

        """

        self.basis = basis_set
        verbose = kwargs.get("verbose", 5)
        spin = kwargs.get("spin", 0)
        charge = kwargs.get("charge", 0)
        symmetry = kwargs.get("symmetry", False)
        self.max_cycle = kwargs.get("max_cycle", 100)
        self.diis_space = kwargs.get("diis_space", 10)
        self.conv_tol = kwargs.get("conv_tol", 1e-10)
        self.conv_tol_grad = kwargs.get("conv_tol_grad", 1e-10)
        self.dm = kwargs.get("dm", None)
        # calculation = kwargs.get('calc', 'RHF')
        # if spin!=0:
        #     #Unresticted calculation
        #     calculation = 'UHF'
        #     if self.dft:
        #         calculation = 'UKS'

        if self.pbc:
            self.kpts = kwargs.get("kpts", [0, 0, 0])
            self.mol = pyscf.pbc.gto.Cell()
            if self.dft:
                self.calc = getattr(pyscf.pbc.dft, "KRKS")
            else:
                self.calc = getattr(pyscf.pbc.scf, "KRHF")

        else:
            self.mol = pyscf.gto.Mole()
            if self.dft:
                self.calc = getattr(pyscf.dft, "RKS")
            else:
                self.calc = getattr(pyscf.scf, "RHF")

        self.mol.basis = basis_set
        self.mol.verbose = verbose
        self.mol.charge = charge
        self.mol.spin = spin
        self.mol.symmetry = symmetry

        for frame in self.structures:
            self.single_calc(
                frame,
            )

    def single_calc(self, frame):
        mol = self.mol
        mol.atom = pyscf_ase.ase_atoms_to_pyscf(frame)
        mol.build()
        if self.pbc:
            mf = self.calc(mol, kpts=self.kpts)
            mf = mf.density_fit()
        else:
            mf = self.calc(mol)

        mf.conv_tol = self.conv_tol
        mf.conv_tol_grad = self.conv_tol_grad
        mf.max_cycle = self.max_cycle
        mf.diis_space = self.diis_space
        if self.dm is None:
            mf.kernel()
        else:
            mf.kernel(self.dm)
        print(mol.ao_labels())
        for label in mol.ao_labels():
            _, elem, bas = label.split(" ")[:3]
            if bas not in self.ao_labels[atomic_numbers[elem]]:
                self.ao_labels[atomic_numbers[elem]].append(bas)

        print("converged:", mf.converged)
        if not mf.converged:
            raise ValueError("PYSCF Calculation did not converge")

        self.dm = mf.make_rdm1()
        fock = mf.get_fock()
        overlap = mf.get_ovlp()
        hcore = mf.get_hcore()
        if "fock" in self.target:
            self.results["fock"].append(fock)
            self.results["overlap"].append(overlap)
        if "energy" in self.target:
            self.results["energy"].append(mf.e_tot)
        if "density" in self.target:
            self.results["density"].append(self.dm)
        if "hcore" in self.target:
            self.results["hcore"].append(hcore)
        if "dipole_moment" in self.target:
            mo_energy, mo_coeff = mf.eig(fock, overlap)
            mo_occ = mf.get_occ(mo_energy)  # get_occ returns a numpy array
            dm1 = mf.make_rdm1(mo_coeff, mo_occ)
            self.results["dipole_moment"].append(mf.dip_moment(dm=dm1))

    # TODO: support multiple targets
    def save_results(self, path: str = None):
        if path is None:
            path = os.path.join(self.path, self.basis)
            p = Path(path).mkdir(parents=True, exist_ok=True)
        else:
            # check if path exists
            if not os.path.exists(path):
                print("Creating path", path)
                p = Path(path).mkdir(parents=True, exist_ok=True)

        for k in self.results.keys():
            assert len(self.results[k]) == self.nframes
            self.results[k] = torch.from_numpy(np.array(self.results[k]))
            hickle.dump(self.results[k], os.path.join(path, k + ".hickle"))

        ao_nlm = {i: [] for i in self.ao_labels.keys()}

        for k in self.ao_labels.keys():
            for v in self.ao_labels[k]:
                ao_nlm[k].append(convert_str_to_nlm(v))
        print(ao_nlm)

        hickle.dump(ao_nlm, os.path.join(path, "orbitals.hickle"))
        print("All done, results saved at: ", path)


class calculator_PBC:
    def __init__(self, structure) -> None:
        pass


# ------TODO: Incorporate the following into the above class -----

import pyscf.pbc.gto as pbcgto
from pyscf.pbc.tools.k2gamma import get_phase, kpts_to_kmesh, double_translation_indices
from collections import defaultdict


def translation_vectors_for_kmesh(cell, kmesh, wrap_around=False, return_rel=False):
    from pyscf import lib

    """
    Adapted from pyscf.pbc.tools.k2gamma to return relative translation vectors

    Translation vectors to construct super-cell of which the gamma point is
    identical to the k-point mesh of primitive cell
    """
    latt_vec = cell.lattice_vectors()
    R_rel_a = np.arange(kmesh[0])
    R_rel_b = np.arange(kmesh[1])
    R_rel_c = np.arange(kmesh[2])
    if wrap_around:
        R_rel_a[(kmesh[0] + 1) // 2 :] -= kmesh[0]
        R_rel_b[(kmesh[1] + 1) // 2 :] -= kmesh[1]
        R_rel_c[(kmesh[2] + 1) // 2 :] -= kmesh[2]
    R_vec_rel = lib.cartesian_prod((R_rel_a, R_rel_b, R_rel_c))
    if return_rel:
        return R_vec_rel
    R_vec_abs = np.dot(R_vec_rel, latt_vec)
    return R_vec_abs


def get_scell_phase(frame, kmesh, basis="sto-3g"):
    """
    Returns the phase factors for the supercell corresponding to the kpoint mesh
    Input:
    frame: ase atoms object
    kmesh: kpoint mesh
    basis: basis set
    """
    cell = pbcgto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(frame)
    cell.basis = basis
    cell.a = frame.cell
    cell.build()
    kpts = cell.make_kpts(kmesh)
    scell, phase = get_phase(cell, kpts)
    return cell, scell, phase


def check_translation_hermiticity(H: Union[dict, np.ndarray], NR=None):
    """
    Check if matric corresponding to H[M,:,N,:] is the same as the matrix corresponding to H[N,:,M,:].
    """
    if not isinstance(H, dict):
        assert len(H.shape) == 4, "H must be a 4D tensor of shape (NR,nao,NR,nao)"
        H = {
            (i, j): H[i, :, j, :] for i in range(H.shape[0]) for j in range(H.shape[2])
        }
    elif NR is None:
        NR = next(iter(H.values()))[0].shape[0]
    # if NR is None:
    #     try:
    #         assert len(H.shape)==4
    #         assert H.shape[0]==H.shape[3]
    #         NR = H.shape[0]
    #     except:
    #         raise ValueError("H must be a 4D tensor of shape (NR,nao,NR,nao)")

    for i in range(NR):
        for j in range(NR):
            norm = np.linalg.norm(H[(i, j)] - H[(j, i)].T)
            if norm > 1e-8:
                raise ValueError("Non hermiticity detected", norm, i, j)


def _map_transidx_to_relative_translation(
    output_tensor: Union[dict, np.ndarray],
    cell: Optional[pyscf.pbc.gto.cell] = None,
    kmesh: Optional[List] = None,
    R_rel: Optional[np.ndarray] = None,
):
    """
    Find the relative translation vector that the translation index (M,N) of the output tensor (M, :, N,:) corresponds to
    NOTE: M-N is NOT the translation vector directly. Instead M, N indices must be used to identify the correct translation vectors from where the RELATIVE translation vectors must be generated
    Args:
        output_tensor: dict of shape (M, :, N, :), could be any tensor
        cell: ase cell object
        kmesh: kpoint mesh
        R_rel: relative translation vectors
    Returns:
        maps_Ls: dict indexed by R_rel, with value corresponding to (M-N, M, N, i) cor, where R_rel is the relative translation vector
        output_tensor_
    """
    if not isinstance(output_tensor, dict):
        assert (
            len(output_tensor.shape) == 4
        ), "output tensor must be a 4D tensor of shape (M, :, N, :)"
        output_tensor = {
            (i, j): output_tensor[i, :, j, :]
            for i in range(output_tensor.shape[0])
            for j in range(output_tensor.shape[2])
        }

    if R_rel is None:
        assert (
            cell is not None and kmesh is not None
        ), " Pass cell and kmesh to generate relative displacements"
        R_rel = translation_vectors_for_kmesh(
            cell, kmesh, wrap_around=False, return_rel=True
        )

    maps_Ls = defaultdict(list)
    for i, (M, N) in enumerate(output_tensor.keys()):
        maps_Ls[tuple((R_rel[M] - R_rel[N]))].append((M - N, M, N, i))

    maps_Ls = {k: v for k, v in maps_Ls.items() if v}
    return maps_Ls


def map_gammapoint_to_relativetrans(
    output_tensor: Union[dict, np.ndarray],
    map_reltrans=None,
    phase: np.ndarray = None,
    cell: Optional[pyscf.pbc.gto.cell] = None,
    kmesh: Optional[List] = None,
):
    """For each unique translation obtained from _map_transidx_to_relative_translation, find the corresponding block of the output tensor - make sure all instances corresponding to the same rel translation are equal - track how many times each rel translation appears in WEIGHTS - track the phase difference corresponding to each rel translation"""
    if not isinstance(output_tensor, dict):
        assert (
            len(output_tensor.shape) == 4
        ), "output tensor must be a 4D tensor of shape (M, :, N, :)"
        output_tensor = {
            (i, j): output_tensor[i, :, j, :]
            for i in range(output_tensor.shape[0])
            for j in range(output_tensor.shape[2])
        }

    if map_reltrans is None:
        map_reltrans = _map_transidx_to_relative_translation(output_tensor, cell, kmesh)

    output_maps_Ls = defaultdict(list)
    output_Ls = {}
    weight_Ls = {}
    phase_diff_Ls = {}
    for i, key in enumerate(map_reltrans.keys()):
        for x in map_reltrans[key]:
            M, N = x[1], x[2]
            output_maps_Ls[key].append((output_tensor[(M, N)]))
        # Check that all the values in the output tensor corresponding to the same relative translation vector are the same
        try:
            xx = output_maps_Ls[key][0]
            for y in output_maps_Ls[key][1:]:
                # print(np.linalg.norm(xx-y), k)
                if not np.allclose(xx, y):
                    raise ValueError(
                        "All matrices for corresponding to this translation not the same!",
                        key,
                        np.linalg.norm(xx - y),
                    )
            output_Ls[
                key
            ] = xx  # assign the first value to this relative translation vector
            weight_Ls[key] = len(
                output_maps_Ls[key]
            )  # track how many times this relative translation vector appears
            phase_diff_Ls[key] = np.array(
                phase[N] / phase[M]
            )  # track the phase difference corresponding to this relative translation vector
        except:
            print(key, "skipped")

    return output_Ls, weight_Ls, phase_diff_Ls


def map_gammapoint_to_kpoint(
    output_tensor: Union[dict, np.ndarray],
    phase: np.ndarray = None,
    map_reltrans=None,
    cell: Optional[pyscf.pbc.gto.cell] = None,
    kmesh: Optional[List] = None,
    nao=None,
):
    """Combine each relative translation vector with the corresponding phase difference to obtain the kpoint matrix. H(K) = \sum_R e^{ik.R} H(R)}"""
    Nk = phase.shape[1]
    if nao is None and cell is not None:
        nao = cell.nao
    elif nao is None and isinstance(output_tensor, dict):
        nao = next(iter(output_tensor.values())).shape[1]
    elif nao is None and not isinstance(output_tensor, dict):
        nao = output_tensor.shape[1]
        print("nao = ", nao)
    if map_reltrans is None:
        map_reltrans = _map_transidx_to_relative_translation(output_tensor, cell, kmesh)
    gamma_to_trans, weight, phase_diff = map_gammapoint_to_relativetrans(
        output_tensor, map_reltrans, phase, cell, kmesh
    )

    kmatrix = np.zeros((Nk, nao, nao), dtype=np.complex128)
    for key in gamma_to_trans.keys():
        for kpt in range(Nk):
            kmatrix[kpt] += gamma_to_trans[key] * weight[key] * phase_diff[key][kpt]
    return kmatrix / Nk


def kpoint_to_gamma(kmatrix, phase):
    """
    Convert the kpoint matrix to the gamma point matrix

    kmatrix: ndarray of shape (Nk, nao, nao)
    phase: dict with the relative translation vectors R as keys. Each value is an ndarray of shape (Nk,) corresponding to phase factors for each kpoint e^{i k.R}
    """
    nao = kmatrix.shape[-1]
    Nk = next(iter(phase.values())).shape[0]
    assert (
        len(kmatrix) == Nk
    ), "Number of kpoints in the kpoint matrix must be equal to the number of kpoints in the phase matrix"
    translated_matrices = np.zeros((len(phase.keys()), nao, nao), dtype=np.complex128)

    for i, key in enumerate(phase.keys()):
        for kpt in range(Nk):
            # km[i] +=  phase_diff[key][kpt] * phase_diff[key][kpt].conj() * gamma_to_trans[key] * weight[key] # This works
            translated_matrices[i] += kmatrix[kpt] * phase[key][kpt].conj()

    return translated_matrices / Nk


if __name__ == "main":
    calc = calculator(
        path="../../../examples/data/water/",
        mol_name="water_1000",
        dft=False,
        frame_slice="0:10",
    )
    calc.calculate()
    calc.save_results()
