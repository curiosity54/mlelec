import torch
from typing import Dict, Optional, List
import numpy as np
import scipy
import mlelec.utils.twocenter_utils as twocenter_utils
import ase
import warnings


class ModelTargets:  # generic class for different targets
    def __init__(self, name: str = "hamiltonian", device: str = None):
        self.device = device
        if name == "hamiltonian" or name == "fock":
            name = "hamiltonian"
        name = name.capitalize()
        self.target_class = globals()[name]  # find target class from string
        # print(self.target_class)

    def instantiate(self, tensor: torch.tensor, **kwargs):
        self.target = self.target_class(
            tensor, **kwargs
        )  # instantiate target class with required arguments
        return self.target


class SingleCenter:  # class for single center tensorial properties
    def __init__(
        self, tensor: torch.tensor, orbitals: Optional[Dict], frames: List[ase.Atoms]
    ):
        self.tensor = tensor
        self.orbitals = orbitals
        self.frames = frames

    def _blocks(self):
        # decompose tensorial property to different SPH blocks if required
        # for now
        self.blocks = self.tensor
        pass


class TwoCenter:  # class for second-rank tensors
    def __init__(
        self,
        tensor: torch.tensor,
        orbitals: Dict,
        frames: Optional[List[ase.Atoms]] = None,
        device=None,
    ):
        # assert (
        #     len(tensor.shape) == 3
        # ), "Second rank tensor must be of shape (N,n,n)"  # FIXME
        self.tensor = tensor
        if isinstance(tensor, np.ndarray):
            self.tensor = [torch.from_numpy(tensor[i]) for i in range(tensor.shape[0])]

        # self.tensor = self.tensor.to(device)
        self.orbitals = orbitals
        self.frames = frames
        self.device = device

    def _to_blocks(self, device="cpu"):
        self.blocks = twocenter_utils._to_blocks(
            self.tensor, self.frames, self.orbitals, device=device
        )
        return self.blocks

    def _blocks_to_tensor(self):
        if not hasattr(self, "blocks"):
            warnings.warn(
                "Blocks not found, generating - output might be different than desired"
            )
            self.blocks = self._to_blocks()
        if "L" in self.blocks.keys.names:
            warnings.warn("L found in blocks, uncoupling first")
            self.blocks = twocenter_utils._to_uncoupled_basis(
                self.blocks, orbitals=self.orbitals, device=self.device
            )

        self.reconstruct = twocenter_utils._to_matrix(
            self.blocks, self.frames, self.orbitals, device=self.device
        )
        return self.reconstruct

    def eigval(self, overlap=None, first: int = 0, last: int = -1):
        eig = Eigenvalues(self.tensor, overlap)
        return eig.eigvalues(first, last)


class Hamiltonian(TwoCenter):  # if there are special cases for hamiltonian
    def __init__(
        self,
        tensor,
        orbitals,
        frames,
        model_strategy: str = "coupled",
        device="cpu",
        **kwargs,
    ):
        # device = kwargs.get("device", "cpu")
        model_strategy = model_strategy.lower()
        assert model_strategy in ["coupled", "uncoupled"]

        # FIX orbital order for PYSCF # TODO: make rhis optional if not using pyscf
        tensor = twocenter_utils.fix_orbital_order(
            tensor, frames=frames, orbital=orbitals
        )

        super().__init__(tensor, orbitals, frames, device=device)
        # assert torch.allclose(
        #     self.tensor, self.tensor.transpose(-1, -2), atol=1e-6
        # ), "Only symmetric Hamiltonians supported for now"
        self.model_strategy = model_strategy
        self._to_blocks()
        if self.model_strategy == "coupled":
            self._couple_blocks()
            self.blocks = self.blocks_coupled
            self.block_keys = self.coupled_keys
        else:
            self.model_strategy = "uncoupled"
            self.blocks = self.blocks_uncoupled
            self.block_keys = self.uncoupled_keys
        self.device = device

    def _to_blocks(self):
        self.blocks_uncoupled = super()._to_blocks(device=self.device)
        self.uncoupled_keys = self.blocks_uncoupled.keys

    def _couple_blocks(self):
        self.blocks_coupled = twocenter_utils._to_coupled_basis(
            self.blocks, device=self.device
        )
        self.coupled_keys = self.blocks_coupled.keys

    def orthogonalize(self, overlap: torch.tensor):
        twocenter_utils.lowin_orthogonalize(self.tensor, overlap)

    def change_basis(new_basis):
        # project onto another basis
        pass

    def _set_blocks(self, blocks):
        from metatensor import TensorMap
        assert isinstance(blocks, TensorMap)
        self.blocks = blocks
        self.block_keys = blocks.keys
        
# class Eigenvalues:  # eigval of a second rank tensor
#    def __init__(self, tensor: torch.tensor, overlap: Optional[torch.tensor] = None):
#        self.tensor = tensor
#        self.overlap = overlap
#        assert len(tensor.shape) == 2
#        if overlap is not None:
#            assert tensor.shape == overlap.shape
#        assert tensor.shape[0] == tensor.shape[1]  # square matrix
#        # TODO : handle non symmetric matrices
#
#        self.eigvals = torch.tensor(
#            scipy.linalg.eigvalsh(self.tensor, self.overlap)
#        )  # FIXME this breaks the autograd chain
#
#    def eigvalues(self, first: int = 0, last: int = -1):
#        return self.eigvals[first:last]
#
#    def eigvectors(self, first: int = 0, last: int = -1):
#        return scipy.linalg.eigh[first:last][-1]
#


class PBCHamiltonian(TwoCenter):
    # Handle supercell or translated hamiltonians
    def __init__(
        self,
        hamiltonian: torch.tensor,
        orbitals: Dict,
        frames: List[ase.Atoms] = None,
        kgrid: List[int] = None,
        translations: List[int] = None,
    ):
        super().__init__(hamiltonian, orbitals, frames)
        pass

    def combine_phase():
        pass


class ElectronDensity:  # Electron density must be handled another way
    def __init__():
        pass


# class DensityMatrix(TwoCenter):
#     def __init__():
#         pass


# .. Higher center tensors
# class ThreeCenter:
#     def __init__(self):
#         pass
