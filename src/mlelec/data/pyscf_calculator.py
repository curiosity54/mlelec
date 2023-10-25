# write a functionality to take input structures, calculate desired target
from typing import List, Optional
from ase.io import read
import ase
import numpy as np
import pyscf
import os
from pathlib import Path
import hickle
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import torch
from collections import defaultdict

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


class calculator:
    def __init__(
        self,
        path: str,
        structures: Optional[List[ase.Atoms]] = None,
        mol_name: str = "water",
        frame_slice=":",
        dft: bool = False,
        target: List[str] = ["fock"],
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
                self.calc = getattr(pyscf.pbc.scf, "RKS")
            else:
                self.calc = getattr(pyscf.pbc.dft, "RHF")

        else:
            self.mol = pyscf.gto.Mole()
            if self.dft:
                self.calc = getattr(pyscf.dft, "RKS")
            else:
                self.calc = getattr(pyscf.scf, "RHF")

        self.mol.basis = basis_set
        self.mol.verbose = 5
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
        else:
            mf = self.calc(mol)
        mf = mf.density_fit()
        mf.conv_tol = self.conv_tol
        mf.conv_tol_grad = self.conv_tol_grad
        mf.max_cycle = self.max_cycle
        mf.diis_space = self.diis_space
        if self.dm is None:
            mf.kernel()
        else:
            mf.kernel(self.dm)

        for label in mol.ao_labels():
            _, elem, bas = label.split(" ")[:3]
            if bas not in self.ao_labels[elem]:
                self.ao_labels[elem].append(bas)

        print("converged:", mf.converged)
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

    # TODO: support multiple targets
    def save_results(self, path: str = None):
        if path is None:
            path = os.path.join(self.path, self.basis)
            p = Path(path).mkdir(parents=True, exist_ok=True)

        for k in self.results.keys():
            assert len(self.results[k]) == self.nframes
            self.results[k] = torch.tensor(self.results[k])
            hickle.dump(self.results[k], os.path.join(path, k + ".hickle"))

        ao_nlm = {i: [] for i in self.ao_labels.keys()}
        for k in self.ao_labels.keys():
            for v in self.ao_labels[k]:
                ao_nlm[k].append(convert_str_to_nlm(v))

        hickle.dump(ao_nlm, os.path.join(path, "orbs.hickle"))
        print("All done, results saved at: ", path)


if __name__ == "main":
    calc = calculator(
        path="/Users/jigyasa/scratch/my_mlelec/examples/data/water/",
        mol_name="water_1000",
        dft=False,
        frame_slice="0:10",
    )
    calc.calculate()
    calc.save_results()
