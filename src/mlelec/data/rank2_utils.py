# Handles 2 center objects - coupling decoupling
# must include preprocessing and postprocessing utils
from typing import Optional
import torch
import ase


def fix_orbital_order(matrix: torch.tensor, frame: ase.Atoms, orbital: dict):
    """Fix the l=1 matrix components from [x,y,z] to [-1, 0,1]"""

    idx = []
    iorb = 0
    atoms = list(frame.numbers)
    for atype in atoms:
        cur = ()
        for ia, a in enumerate(orbital[atype]):
            n, l, _ = a
            if (n, l) != cur:
                if l == 1:
                    idx += [iorb + 1, iorb + 2, iorb]
                else:
                    idx += range(iorb, iorb + 2 * l + 1)
                iorb += 2 * l + 1
                cur = (n, l)
    return matrix[idx][:, idx]


def lowdin_orthogonalize(fock: torch.tensor, overlap: torch.tensor):
    """
    lowdin orthogonalization of a fock matrix computing the square root of the overlap matrix
    """
    eva, eve = torch.linalg.eigh(overlap)
    sm12 = eve @ torch.diag(1.0 / torch.sqrt(eva)) @ eve.T
    return sm12 @ fock @ sm12


def _components_idx(l):
    """just a mini-utility function to get the m=-l..l indices"""
    return torch.arange(-l, l + 1, dtype=int).reshape(2 * l + 1, 1)


def _components_idx_2d(li, lj):
    """indexing the entries in a 2d (l_i, l_j) block of the hamiltonian
    in the uncoupled basis"""
    return torch.tensor(
        torch.meshgrid(_components_idx(li), _components_idx(lj)), dtype=int
    ).T.reshape(-1, 2)


# TODO
# Coupling-decoupling are model specific so we should move this
def _to_blocks(matrix, frame, orbitals):
    pass


def _to_matrix(frame, orbitals):
    pass


def _to_coupled_basis(matrix, orbitals):
    pass


def _to_uncoupled_basis():
    pass
