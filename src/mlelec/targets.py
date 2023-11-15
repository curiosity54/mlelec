import torch
from typing import Dict, Optional, List
import numpy as np
import scipy
import mlelec.utils.rank2_utils as rank2_utils
import ase


class ModelTargets:  # generic class for different targets
    def __init__(self, name: str = "hamiltonian"):
        self.target_class = globals()[name]

    def instantiate(self, tensor: torch.tensor, frames, orbitals, overlap):
        self.target = self.target_class(tensor)
        return self.target


class SecondRank:  # class for second-rank tensors
    def __init__(
        self,
        tensor: torch.tensor,
        orbitals: Dict,
        frames: Optional[List[ase.Atoms]] = None,
    ):
        assert len(tensor.shape) == 2, "Second rank tensor must be of shape (n,n)"
        self.tensor = tensor
        self.orbitals = orbitals

    def _blocks(self):
        self.blocks = rank2_utils._to_blocks(self.tensor, frames, orbitals)
        self.block_keys = self.blocks.keys()


class Hamiltonian(SecondRank):  # if there are special cases for hamiltonian
    def __init__(self, hamiltonian, orbitals, frames):
        super().__init__(hamiltonian, orbitals, frames)
        print(self.tensor)

    def orthogonalize(self, overlap: torch.tensor):
        rank2_utils.lowin_orthogonalize(self.tensor, overlap)

    def eigval(self, overlap=None, first: int = 0, last: int = -1):
        eig = Eigenvalues(self.tensor, overlap)
        return eig.eigvalues(first, last)


class Eigenvalues:  # eigval of a second rank tensor
    def __init__(self, tensor: torch.tensor, overlap: Optional[torch.tensor] = None):
        self.tensor = tensor
        self.overlap = overlap
        assert len(tensor.shape) == 2
        if overlap is not None:
            assert tensor.shape == overlap.shape
        assert tensor.shape[0] == tensor.shape[1]  # square matrix
        # TODO : handle non symmetric matrices

        self.eigvals = torch.tensor(
            scipy.linalg.eigvalsh(self.tensor, self.overlap)
        )  # FIXME this breaks the autograd chain

    def eigvalues(self, first: int = 0, last: int = -1):
        return self.eigvals[first:last]

    def eigvectors(self, first: int = 0, last: int = -1):
        return scipy.linalg.eigh[first:last][-1]


class PBCHamiltonian(SecondRank):
    def __init__(
        self,
        hamiltonian: torch.tensor,
        orbitals: Dict,
        frames: List[ase.Atoms] = None,
        k_grid: List[int] = None,
        translations: List[int] = None,
    ):
        super().__init__(hamiltonian, orbitals, frames)

    def combine_phase():
        pass


# Handle supercell or translated hamiltonians


class ElectronDensity(ModelTargets):  # Electron density must be handled another way
    def __init__():
        pass


# class DensityMatrix(SecondRank):
#     def __init__():
#         pass


# .. Third rank tensors
