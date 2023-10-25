# Handles 2 center objects - coupling decoupling
# must include preprocessing and postprocessing utils
from typing import Optional, List
import torch
import ase
import numpy as np


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
def _orbs_offsets(orbs):
    """offsets for the orbital subblocks within an atom block of the Hamiltonian matrix"""
    orbs_tot = {}
    orbs_offset = {}
    for k in orbs:
        ko = 0
        for n, l, m in orbs[k]:
            if m != -l:
                continue
            orbs_offset[(k, n, l)] = ko
            ko += 2 * l + 1
        orbs_tot[k] = ko
    return orbs_tot, orbs_offset


def _atom_blocks_idx(frames, orbs_tot):
    """position of the hamiltonian subblocks for each atom in each frame"""
    atom_blocks_idx = {}
    for A, f in enumerate(frames):
        ki = 0
        for i, ai in enumerate(f.numbers):
            kj = 0
            for j, aj in enumerate(f.numbers):
                atom_blocks_idx[(A, i, j)] = (ki, kj)
                kj += orbs_tot[aj]
            ki += orbs_tot[ai]
    return atom_blocks_idx


def _to_blocks(matrices: List[torch.tensor], frames: List[ase.Atoms], orbitals: dict):
    block_builder = TensorBuilder(
        ["block_type", "a_i", "n_i", "l_i", "a_j", "n_j", "l_j"],
        ["structure", "center", "neighbor"],
        [["m1"], ["m2"]],
        ["value"],
    )
    orbs_tot, _ = _orbs_offsets(orbitals)
    for A in range(len(frames)):
        frame = frames[A]
        ham = matrices[A]
        ki_base = 0
        for i, ai in enumerate(frame.numbers):
            kj_base = 0
            for j, aj in enumerate(frame.numbers):
                if i == j:
                    block_type = 0  # diagonal
                elif ai == aj:
                    if i > j:
                        kj_base += orbs_tot[aj]
                        continue
                    block_type = 1  # same-species
                else:
                    if ai > aj:  # only sorted element types
                        kj_base += orbs_tot[aj]
                        continue
                    block_type = 2  # different species
                if isinstance(ham, np.ndarray):
                    block_data = torch.from_numpy(
                        ham[
                            ki_base : ki_base + orbs_tot[ai],  # noqa: E203
                            kj_base : kj_base + orbs_tot[aj],  # noqa: E203
                        ]
                    )
                elif isinstance(ham, torch.Tensor):
                    block_data = ham[
                        ki_base : ki_base + orbs_tot[ai],  # noqa: E203
                        kj_base : kj_base + orbs_tot[aj],  # noqa: E203
                    ]
                else:
                    raise ValueError

                # print(block_data, block_data.shape)
                if block_type == 1:
                    block_data_plus = (block_data + block_data.T) * 1 / np.sqrt(2)
                    block_data_minus = (block_data - block_data.T) * 1 / np.sqrt(2)
                ki_offset = 0
                for ni, li, mi in orbs[ai]:
                    if (
                        mi != -li
                    ):  # picks the beginning of each (n,l) block and skips the other orbitals
                        continue
                    kj_offset = 0
                    for nj, lj, mj in orbs[aj]:
                        if (
                            mj != -lj
                        ):  # picks the beginning of each (n,l) block and skips the other orbitals
                            continue
                        if ai == aj and (ni > nj or (ni == nj and li > lj)):
                            kj_offset += 2 * lj + 1
                            continue
                        block_idx = (block_type, ai, ni, li, aj, nj, lj)
                        if block_idx not in block_builder.blocks:
                            block = block_builder.add_block(
                                keys=block_idx,
                                properties=np.asarray([[0]], dtype=np.int32),
                                components=[_components_idx(li), _components_idx(lj)],
                            )

                            if block_type == 1:
                                block_asym = block_builder.add_block(
                                    keys=(-1,) + block_idx[1:],
                                    properties=np.asarray([[0]], dtype=np.int32),
                                    components=[
                                        _components_idx(li),
                                        _components_idx(lj),
                                    ],
                                )
                        else:
                            block = block_builder.blocks[block_idx]
                            if block_type == 1:
                                block_asym = block_builder.blocks[(-1,) + block_idx[1:]]

                        islice = slice(ki_offset, ki_offset + 2 * li + 1)
                        jslice = slice(kj_offset, kj_offset + 2 * lj + 1)

                        if block_type == 1:
                            block.add_samples(
                                labels=[(A, i, j)],
                                data=block_data_plus[islice, jslice].reshape(
                                    (1, 2 * li + 1, 2 * lj + 1, 1)
                                ),
                            )
                            block_asym.add_samples(
                                labels=[(A, i, j)],
                                data=block_data_minus[islice, jslice].reshape(
                                    (1, 2 * li + 1, 2 * lj + 1, 1)
                                ),
                            )

                        else:
                            block.add_samples(
                                labels=[(A, i, j)],
                                data=block_data[islice, jslice].reshape(
                                    (1, 2 * li + 1, 2 * lj + 1, 1)
                                ),
                            )

                        kj_offset += 2 * lj + 1
                    ki_offset += 2 * li + 1
                kj_base += orbs_tot[aj]

            ki_base += orbs_tot[ai]
    # print(block_builder.build())
    return block_builder.build()


def _to_matrix(frame, orbitals):
    pass


def _to_coupled_basis(matrix, orbitals):
    pass


def _to_uncoupled_basis():
    pass
