# write a functionality to take input structures, calculate desired target
# import pyscf
from typing import List, Optional
from ase.io import read
import ase
import numpy as np
import pyscf
import os
from pathlib import Path
import hickle


class calculator:
    def __init__(
        self,
        path: str,
        structures: Optional[List[ase.Atoms]] = None,
        mol_name: str = "water",
        frame_slice=":",
        dft: bool = False,
        target: str = "fock",
    ):  # self.kwargs:Dict[str, Any]
        self.path = path
        self.structures = structures
        self.mol_name = mol_name
        self.slice = frame_slice
        self.dft = dft
        self.load_structures()
        if np.any(self.structures[0].cell):
            self.pbc = True
        self.nframes = len(self.structures)
        print("Number of frames: ", self.nframes)
        self.target = target
        self.results = []

    def load_structures(self):
        if self.structures is None:
            try:
                self.structures = read(
                    self.path + "/{}.xyz".format(self.mol_name), index=self.slice
                )
            except:
                raise FileNotFoundError("No structures found at the given path")

    def calculate(self, frame, basis_set: str = "sto-3g", **kwargs: Optional[dict]):
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
            import pyscf.pbc.tools.pyscf_ase as pyscf_ase

            self.kpts = kwargs.get("kpts", [0, 0, 0])
            self.mol = pyscf.pbc.gto.Cell()
            if self.dft:
                self.calc = getattr(pyscf.pbc.scf, "RKS")
            else:
                self.calc = getattr(pyscf.pbc.dft, "RHF")

        else:
            import pyscf.tools.pyscf_ase as pyscf_ase

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
            tar = self.single_calc(
                frame,
            )
            self.results.append(tar)

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
        if dm == None:
            mf.kernel()
        else:
            mf.kernel(dm)

        print("AO:", mol.ao_labels())
        print("converged:", mf.converged)
        dm = mf.make_rdm1()
        fock = mf.get_fock()
        overlap = mf.get_ovlp()
        hcore = mf.get_hcore()
        if self.target == "fock":
            return fock, overlap
        elif self.target == "energy":
            return mf.e_tot
        elif self.target == "density":
            return dm
        elif self.target == "hcore":
            return hcore

    def save_results(self):
        path = os.path.join(
            self.path, self.mol_name, self.basis, self.target + ".hickle"
        )
        p = Path(path).mkdir(parents=True, exist_ok=True)
        hickle.dump(self.results, path)
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
    print(calc.results)
