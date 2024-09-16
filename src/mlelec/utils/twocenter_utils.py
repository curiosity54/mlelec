# Handles 2 center objects - coupling decoupling
# must include preprocessing and postprocessing utils
from typing import Optional, List, Union, Tuple, Dict
import warnings
from collections import defaultdict

import torch
import ase
import numpy as np

from metatensor.torch import TensorMap, TensorBlock, Labels
import metatensor.torch as mts

from mlelec.utils.metatensor_utils import TensorBuilder, labels_where
from mlelec.utils.symmetry import ClebschGordanReal

SQRT_2 = 2 ** (0.5)
ISQRT_2 = 1 / SQRT_2
dummy_property = Labels(['dummy'], torch.tensor([[0]]))


def fix_orbital_order(
    matrix: Union[torch.tensor, np.ndarray],
    frames: Union[List, ase.Atoms],
    orbital: dict,
):
    """Fix the l=1 matrix components from [x,y,z] to [-1, 0,1], handles single and multiple frames"""

    def fix_one_matrix(
        matrix: Union[torch.tensor, np.ndarray], frame: ase.Atoms, orbital: dict
    ):
        idx = []
        iorb = 0
        atoms = list(frame.numbers)
        for atom_type in atoms:
            cur = ()
            for _, a in enumerate(orbital[atom_type]):
                n, l, _ = a
                if (n, l) != cur:
                    if l == 1:
                        idx += [iorb + 1, iorb + 2, iorb]
                    else:
                        idx += range(iorb, iorb + 2 * l + 1)
                    iorb += 2 * l + 1
                    cur = (n, l)
        return matrix[idx][:, idx]

    if isinstance(frames, list):
        assert len(matrix.shape) == 3  # (nframe, nao, nao)
        fixed_matrices = []
        for i, f in enumerate(frames):
            fixed_matrices.append(fix_one_matrix(matrix[i], f, orbital))
        if isinstance(matrix, np.ndarray):
            return np.asarray(fixed_matrices)
        return torch.stack(fixed_matrices)
    else:
        return fix_one_matrix(matrix, frames, orbital)


def unfix_orbital_order(
    matrix: Union[torch.tensor, np.ndarray],
    frames: Union[List, ase.Atoms],
    orbital: dict,
):
    """Fix the l=1 matrix components from [-1,0,1] to [x,y,z], handles single and multiple frames"""

    def unfix_one_matrix(
        matrix: Union[torch.tensor, np.ndarray], frame: ase.Atoms, orbital: dict
    ):
        idx = []
        iorb = 0
        atoms = list(frame.numbers)
        for atom_type in atoms:
            cur = ()
            for _, a in enumerate(orbital[atom_type]):
                n, l, _ = a
                if (n, l) != cur:
                    if l == 1:
                        idx += [iorb + 2, iorb, iorb + 1]
                    else:
                        idx += range(iorb, iorb + 2 * l + 1)
                    iorb += 2 * l + 1
                    cur = (n, l)
        return matrix[idx][:, idx]

    if isinstance(frames, list):
        if len(frames) == 1:
            matrix = matrix.reshape(1, *matrix.shape)
        assert len(matrix.shape) == 3, matrix.shape  # (nframe, nao, nao)
        fixed_matrices = []
        for i, f in enumerate(frames):
            fixed_matrices.append(unfix_one_matrix(matrix[i], f, orbital))
        if isinstance(matrix, np.ndarray):
            return np.asarray(fixed_matrices)
        return torch.stack(fixed_matrices)
    else:
        return unfix_one_matrix(matrix, frames, orbital)


def lowdin_orthogonalize(fock: torch.tensor, overlap: torch.tensor):
    """
    lowdin orthogonalization of a fock matrix computing the square root of the overlap matrix
    """
    eva, eve = torch.linalg.eigh(overlap)
    sm12 = eve @ torch.diag(1.0 / torch.sqrt(eva)).to(eve) @ eve.T
    return sm12.conj() @ fock @ sm12


def _components_idx(l):
    """Returns the m \in {-l,...,l} indices"""
    return torch.arange(-l, l + 1, dtype = torch.int32).reshape(2 * l + 1, 1)
    # return np.arange(-l, l + 1, dtype=int).reshape(2 * l + 1, 1)


def _components_idx_2d(li, lj):
    """Returns the 2D outerproduct of m_i \in {-l_i,... , l_i} and m_j \in {-l_j,... , l_j} to index the (l_i, l_j) block of the hamiltonian
    in the uncoupled basis"""
    # return np.asarray(
    #     np.meshgrid(_components_idx(li), _components_idx(lj)), dtype=int
    # ).T.reshape(-1, 2)
    return torch.cartesian_prod(_components_idx(li).flatten(), _components_idx(lj).flatten()).to(dtype = torch.int32)


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
    if isinstance(frames, ase.Atoms):
        frames = [frames]
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


def _to_blocks(
    matrices: Union[List[torch.tensor], torch.tensor],
    frames: Union[ase.Atoms, List[ase.Atoms]],
    orbitals: dict,
    device: str = None,
    NH=False,
):
    if not isinstance(frames, list):
        assert len(matrices.shape) == 2  # should be just one matrix (nao,nao)
        frames = [frames]
        matrices = matrices.reshape(1, *matrices.shape)
    # check hermiticity:
    if isinstance(matrices, np.ndarray):
        matrices = torch.from_numpy(matrices)
    if NH:
        warnings.warn(
            "Matrix is neither hermitian nor antihermitian - attempting to use _toblocks for NH"
        )

        nh_blocks = _matrix_to_blocks_NH_translations(
            matrices, frames, orbitals, device
        )
        return nh_blocks

    else:
        assert torch.allclose(
            torch.abs(matrices), torch.abs(matrices.transpose(-1, -2))
        ), "Matrix supposed to be hermitian but is not"
        return _matrix_to_blocks(matrices, frames, orbitals, device)

def _matrix_to_blocks_NH_translations(
    matrices: dict,  # Union[List[torch.tensor], torch.tensor],
    frames: Union[ase.Atoms, List[ase.Atoms]],
    orbitals: dict,
    device: str = None,
):
    orbs_tot, _ = _orbs_offsets(orbitals)

    block_builder = TensorBuilder(
        ["block_type", "species_i", "n_i", "l_i", "species_j", "n_j", "l_j"],
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
                    if i > j:  # order i< j
                        kj_base += orbs_tot[aj]
                        continue
                    block_type = 1  # same-species
                else:
                    if (
                        ai > aj
                    ):  # only sorted element types - #TODO this doesnt hold i think

                        kj_base += orbs_tot[aj]
                        # continue
                    block_type = 2  # different species
                bdata = ham[
                    ki_base : ki_base + orbs_tot[ai],
                    kj_base : kj_base + orbs_tot[aj],
                ]
                bdata_ij = ham[
                    kj_base : kj_base + orbs_tot[aj], ki_base : ki_base + orbs_tot[ai]
                ]
                # print(i, j, slice(ki_base, ki_base + orbs_tot[ai]), slice(kj_base, kj_base + orbs_tot[aj]))

                if isinstance(ham, np.ndarray):
                    block_data = torch.from_numpy(bdata).to(device)
                    block_data_ij = torch.from_numpy(bdata_ij).to(device)
                elif isinstance(ham, torch.Tensor):
                    block_data = bdata
                    block_data_ij = bdata_ij
                else:
                    raise ValueError
                if block_type == 1:
                    block_data_plus = (block_data + block_data_ij) * ISQRT_2
                    block_data_minus = (block_data - block_data_ij) * ISQRT_2
                ki_offset = 0
                for ni, li, mi in orbitals[ai]:
                    if (
                        mi != -li
                    ):  # picks the beginning of each (n,l) block and skips the other orbitals
                        continue
                    kj_offset = 0
                    for nj, lj, mj in orbitals[aj]:
                        if (
                            mj != -lj
                        ):  # picks the beginning of each (n,l) block and skips the other orbitals
                            continue
                        # if ai == aj and (ni > nj or (ni == nj and li > lj)): # order orbitals
                        #     kj_offset += 2 * lj + 1
                        #     continue
                        block_idx = (block_type, ai, ni, li, aj, nj, lj)
                        if block_idx not in block_builder.blocks:
                            block = block_builder.add_block(
                                key=block_idx,
                                properties=np.asarray([[0]]),
                                components=[_components_idx(li), _components_idx(lj)],
                            )

                            if block_type == 1:
                                block_asym = block_builder.add_block(
                                    key=(-1,) + block_idx[1:],
                                    properties=np.asarray([[0]]),
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
                        # print(i, islice, "I")
                        # print(j, jslice, "J")
                        # print("block_type", block_type)
                        # print(ni, li, nj, lj)

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
                            # print(
                            #     i, j, block_data[islice, jslice], 2 * li + 1, 2 * lj + 1
                            # )
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
    return block_builder.build()


def _matrix_to_blocks(
    matrices: Union[List[torch.tensor], torch.tensor],
    frames: Union[ase.Atoms, List[ase.Atoms]],
    orbitals: dict,
    device: str = None,
):
    
    orbs_tot, _ = _orbs_offsets(orbitals)

    block_builder = TensorBuilder(
        ["block_type", "species_i", "n_i", "l_i", "species_j", "n_j", "l_j"],
        ["structure", "center", "neighbor"],
        [["m_i"], ["m_j"]],
        ["dummy"],
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
                bdata = ham[
                    ki_base : ki_base + orbs_tot[ai], kj_base : kj_base + orbs_tot[aj]
                ]

                if isinstance(ham, np.ndarray):
                    block_data = torch.from_numpy(bdata)
                elif isinstance(ham, torch.Tensor):
                    block_data = bdata
                else:
                    raise ValueError

                # print(block_data, block_data.shape)
                if block_type == 1:
                    # print(block_data)
                    block_data_plus = (block_data + block_data.T) * ISQRT_2
                    block_data_minus = (block_data - block_data.T) * ISQRT_2
                ki_offset = 0
                for ni, li, mi in orbitals[ai]:
                    if (
                        mi != -li
                    ):  # picks the beginning of each (n,l) block and skips the other orbitals
                        continue
                    kj_offset = 0
                    for nj, lj, mj in orbitals[aj]:
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
                                key=block_idx,
                                properties=np.asarray([[0]]),
                                components=[_components_idx(li), _components_idx(lj)],
                            )

                            if block_type == 1:
                                block_asym = block_builder.add_block(
                                    key=(-1,) + block_idx[1:],
                                    properties=np.asarray([[0]]),
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
    return block_builder.build()


def _to_matrix(
    blocks: TensorMap,
    frames: List[ase.Atoms],
    orbitals: Dict[int, List[Tuple[int, int, int]]],
    hermitian: bool = True,
    # vectorized: bool = True,
    NH=False,
    device=None,
) -> Union[np.ndarray, torch.Tensor]:

    return _blocks_to_matrix(
        blocks=blocks,
        frames=frames,
        orbitals=orbitals,
        hermitian=hermitian,
        device=device,
        NH=NH,
    )


def _blocks_to_matrix(
    blocks: TensorMap,
    frames: Union[ase.Atoms, List[ase.Atoms]],
    orbitals: Dict[int, List[Tuple[int, int, int]]],
    device=None,
    hermitian: bool = True,
    NH=False,
) -> Union[np.ndarray, torch.Tensor]:
    """from tensormap to dense representation

    Converts a TensorMap containing matrix blocks in the uncoupled basis,
    `blocks` into dense matrices.
    Needs `frames` and `orbs` to reconstruct matrices in the correct order.
    See `dense_to_blocks` to understant the different types of blocks.
    """
    if not isinstance(frames, list):
        frames = [frames]

    # total number of orbitals per atom, orbital offset per atom
    orbs_tot, orbs_offset = _orbs_offsets(orbitals)

    # indices of the block for each atom
    atom_blocks_idx = _atom_blocks_idx(frames, orbs_tot)

    if device is None:
        device = blocks.block(0).values.device

    matrices = []
    for f in frames:
        norbs = 0
        for ai in f.numbers:
            norbs += orbs_tot[ai]
        matrix = torch.zeros(norbs, norbs, device=device)
        matrices.append(matrix)

    # loops over block types
    for idx, block in blocks.items():
        # dense idx and cur_A track the frame
        dense_idx = -1
        cur_A = -1
        block_type = idx["block_type"]
        ai = idx["species_i"]
        ni = idx["n_i"]
        li = idx["l_i"]
        aj = idx["species_j"]
        nj = idx["n_j"]
        lj = idx["l_j"]

        # offset of the orbital block within the pair block in the matrix
        ki_offset = orbs_offset[(ai, ni, li)]
        kj_offset = orbs_offset[(aj, nj, lj)]
        same_koff = ki_offset == kj_offset

        # loops over samples (structure, i, j)
        for sample, block_data in zip(block.samples, block.values):
            A = sample["structure"]
            i = sample["center"]
            j = sample["neighbor"]
            # check if we have to update the frame and index
            if A != cur_A:
                cur_A = A
                dense_idx += 1

            matrix = matrices[dense_idx]

            # coordinates of the atom block in the matrix
            ki_base, kj_base = atom_blocks_idx[(dense_idx, i, j)]

            # values to assign
            values = block_data[:, :, 0].reshape(2 * li + 1, 2 * lj + 1)
            # assign values
            if NH:
                _fill_NH(
                    block_type,
                    matrix,
                    values,
                    ki_base,
                    kj_base,
                    ki_offset,
                    kj_offset,
                    same_koff,
                    li,
                    lj,
                )
            else:
                _fill(
                    block_type,
                    matrix,
                    values,
                    ki_base,
                    kj_base,
                    ki_offset,
                    kj_offset,
                    same_koff,
                    li,
                    lj,
                    hermitian=hermitian,
                )

    if len(matrices) == 1:
        return matrices[0]
    else:
        return torch.stack(matrices)


def _fill_NH(
    type: int,
    matrix: Union[np.ndarray, torch.Tensor],
    values: Union[np.ndarray, torch.Tensor],
    ki_base: int,
    kj_base: int,
    ki_offset: int,
    kj_offset: int,
    same_koff: bool,
    li: int,
    lj: int,
):
    """fill block of type <type> where type is either -1,0,1,2"""
    # TODO: check matrix device, values devide are the same
    islice = slice(ki_base + ki_offset, ki_base + ki_offset + 2 * li + 1)
    jslice = slice(kj_base + kj_offset, kj_base + kj_offset + 2 * lj + 1)
    # bdata_ij = ham[kj_base : kj_base + orbs_tot[aj], ki_base : ki_base + orbs_tot[ai]]

    if type == 0:
        matrix[islice, jslice] = values

    if type == 2:
        matrix[islice, jslice] = values

    if abs(type) == 1:  # FIXME
        values = values * ISQRT_2
        matrix[islice, jslice] += values
        # if not same_koff:
        iislice = slice(ki_base + kj_offset, ki_base + kj_offset + 2 * lj + 1)
        jjslice = slice(kj_base + ki_offset, kj_base + ki_offset + 2 * li + 1)
        if type == 1:
            matrix[jjslice, iislice] += values
        else:
            matrix[jjslice, iislice] -= values

def _fill(
    type: int,
    matrix: Union[np.ndarray, torch.Tensor],
    values: Union[np.ndarray, torch.Tensor],
    ki_base: int,
    kj_base: int,
    ki_offset: int,
    kj_offset: int,
    same_koff: bool,
    li: int,
    lj: int,
    hermitian: bool = True,
):
    """fill block of type <type> where type is either -1,0,1,2"""
    # TODO: check matrix device, values devide are the same
    islice = slice(ki_base + ki_offset, ki_base + ki_offset + 2 * li + 1)
    jslice = slice(kj_base + kj_offset, kj_base + kj_offset + 2 * lj + 1)
    if type == 0:
        matrix[islice, jslice] = values
        if not same_koff:
            if hermitian:
                matrix[jslice, islice] = values.T
            else:
                matrix[jslice, islice] = values.T * -1
    if type == 2:
        matrix[islice, jslice] = values
        if hermitian:
            matrix[jslice, islice] = values.T
        else:
            matrix[jslice, islice] = values.T * -1

    if abs(type) == 1:
        values_2norm = values * ISQRT_2
        matrix[islice, jslice] += values_2norm
        if hermitian:
            matrix[jslice, islice] += values_2norm.T
        else:
            matrix[jslice, islice] -= values_2norm.T
        if not same_koff:
            islice = slice(ki_base + kj_offset, ki_base + kj_offset + 2 * lj + 1)
            jslice = slice(kj_base + ki_offset, kj_base + ki_offset + 2 * li + 1)
            if type == 1:
                matrix[islice, jslice] += values_2norm.T
                if hermitian:
                    matrix[jslice, islice] += values_2norm
                else:
                    matrix[jslice, islice] -= values_2norm
            else:
                matrix[islice, jslice] -= values_2norm.T
                if hermitian:
                    matrix[jslice, islice] -= values_2norm
                else:
                    matrix[jslice, islice] += values_2norm


def _to_coupled_basis_OLD(
    blocks: Union[torch.tensor, TensorMap],
    orbitals: Optional[dict] = None,
    cg: Optional[ClebschGordanReal] = None,
    device: str = "cpu",
    skip_symmetry: bool = False,
    translations: bool = False,
):
    if torch.is_tensor(blocks):
        print("Converting matrix to blocks before coupling")
        assert orbitals is not None, "Need orbitals to convert matrix to blocks"
        blocks = _to_blocks(blocks, orbitals)
    if cg is None:
        lmax = max(blocks.keys["l_i"] + blocks.keys["l_j"])
        cg = ClebschGordanReal(lmax, device=device)
    if not translations:
        block_builder = TensorBuilder(
            ["block_type", "species_i", "n_i", "l_i", "species_j", "n_j", "l_j", "L"],
            ["structure", "center", "neighbor"],
            [["M"]],
            ["value"],
        )
    else:
        block_builder = TensorBuilder(
            [
                "block_type",
                "species_i",
                "n_i",
                "l_i",
                "species_j",
                "n_j",
                "l_j",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
                "L",
            ],
            ["structure", "center", "neighbor"],
            [["M"]],
            ["value"],
        )
    for idx, block in blocks.items():
        block_type = idx["block_type"]
        ai = idx["species_i"]
        ni = idx["n_i"]
        li = idx["l_i"]
        aj = idx["species_j"]
        nj = idx["n_j"]
        lj = idx["l_j"]

        # Moves the components at the end as cg.couple assumes so
        decoupled = torch.moveaxis(block.values, -1, -2).reshape(
            (len(block.samples), len(block.properties), 2 * li + 1, 2 * lj + 1)
        )

        # selects the (only) key in the coupled dictionary (l1 and l2
        # that contribute to the coupled terms L, with L going from
        # |l1 - l2| up to |l1 + l2|
        coupled = cg.couple(decoupled)[(li, lj)]
        # if ai == aj == 1 and block_type == -1:
        #     print(idx, coupled)  # decoupled)

        for L in coupled:
            block_idx = tuple(idx) + (L,)
            # skip blocks that are zero because of symmetry
            if ai == aj and ni == nj and li == lj:
                parity = (-1) ** (li + lj + L)
                if (
                    (parity == -1 and block_type in (0, 1))
                    or (parity == 1 and block_type == -1)
                ) and not skip_symmetry:
                    continue

            new_block = block_builder.add_block(
                key=block_idx,
                properties=np.asarray([[0]], dtype=np.int32),
                components=[_components_idx(L).reshape(-1, 1)],
            )

            new_block.add_samples(
                labels=np.asarray(block.samples.values).reshape(
                    block.samples.values.shape[0], -1
                ),
                data=torch.moveaxis(coupled[L], -1, -2),
            )

    return block_builder.build()


def _to_coupled_basis(
    blocks: Union[torch.tensor, TensorMap],
    orbitals: Optional[dict] = None,
    cg: Optional[ClebschGordanReal] = None,
    device: str = "cpu",
    skip_symmetry: bool = False,
    translations: bool = None,
    high_rank= False # means rank_three  for now TODO: Support a general high rank?
):
    if torch.is_tensor(blocks):
        print("Converting matrix to blocks before coupling")
        assert orbitals is not None, "Need orbitals to convert matrix to blocks"
        blocks = _to_blocks(blocks, orbitals)
    if cg is None:
        lmax = max(blocks.keys["l_i"] + blocks.keys["l_j"])
        cg = ClebschGordanReal(lmax, device=device)
   
    key_names = ["block_type", "species_i", "n_i", "l_i", "species_j", "n_j", "l_j", "L"]
    if high_rank:
        key_names = ["block_type", "species_i", "n_i", "l_i", "species_j", "n_j", "l_j", "l_3", "K", "L"]
    property_names = ["dummy"]

    if translations is None:
        block_builder = TensorBuilder(
            key_names,
            ["structure", "center", "neighbor"],
            [["M"]],
            property_names,
        )
    else:
        if translations:
            block_builder = TensorBuilder(
                key_names,
                ["structure", "center", "neighbor", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
                [["M"]],
                property_names,
            )
        else:
            block_builder = TensorBuilder(
                ["block_type", "species_i", "n_i", "l_i", "species_j", "n_j", "l_j", "L"],
                ["structure", "center", "neighbor", "kpoint"],
                [["M"]],
                property_names,
            )
        
        
    for idx, block in blocks.items():
        block_type = idx["block_type"]
        ai = idx["species_i"]
        ni = idx["n_i"]
        li = idx["l_i"]
        aj = idx["species_j"]
        nj = idx["n_j"]
        lj = idx["l_j"]
        if high_rank:
            l3 = idx["l_3"]
            assert l3==1, "Only tensors of type position supported for now"
        

        # Moves the components at the end as cg.couple assumes so
        if high_rank:
            decoupled = torch.moveaxis(block.values, (4,3), (1,2)).reshape(
            (len(block.samples), len(block.properties), 2 * l3 + 1, 2 * lj + 1, 2 * li + 1))
            # reshapes into nsamples, nprops, 2*l3+1, 2*lj+1, 2*li+1
            # decoupled = block.values.reshapoperties), 2 * li + 1, 2 * lj + 1, 2 * l3 + 1))
            assert decoupled.shape[2] == 3, "Only tensors of type position supported for now"
            coupled = cg.couple(decoupled, iterate = 1)
            
            for coupledkey in coupled:
                k = coupledkey[1]
                for L in coupled[coupledkey]:
                    block_idx = tuple(idx) + (k, L)
                    # skip blocks that are zero because of symmetry - TBD 
                    # if ai == aj and ni == nj and li == lj:
                    #     parity = (-1) ** (li + lj + L)
                    #     if ((parity == -1 and block_type in (0, 1)) or (parity == 1 and block_type == -1)) and not skip_symmetry:
                    #         continue
                    new_block = block_builder.add_block(
                        key=block_idx,
                        properties=torch.tensor([[0]], dtype=torch.int32),
                        components=[_components_idx(L).reshape(-1, 1)],
                    )
                    new_block.add_samples(
                            labels = block.samples.values.reshape(block.samples.values.shape[0], -1),
                            data = torch.moveaxis(coupled[coupledkey][L], -1, -2),
                        )

        else:
            decoupled = torch.moveaxis(block.values, -1, -2).reshape(
            (len(block.samples), len(block.properties), 2 * li + 1, 2 * lj + 1)
        )
            coupled = cg.couple(decoupled)[(li, lj)]
            # selects the (only) key in the coupled dictionary (l1 and l2
            # that contribute to the coupled terms L, with L going from
            # |l1 - l2| up to |l1 + l2|
            for L in coupled:
                block_idx = tuple(idx) + (L,)
                # skip blocks that are zero because of symmetry
                if ai == aj and ni == nj and li == lj:
                    parity = (-1) ** (li + lj + L)
                    if ((parity == -1 and block_type in (0, 1)) or (parity == 1 and block_type == -1)) and not skip_symmetry:
                        continue

                new_block = block_builder.add_block(
                    key=block_idx,
                    properties=torch.tensor([[0]], dtype=torch.int32),
                    components=[_components_idx(L).reshape(-1, 1)],
                )

                new_block.add_samples(
                    labels = block.samples.values.reshape(block.samples.values.shape[0], -1),
                    data = torch.moveaxis(coupled[L], -1, -2),
                )

    return block_builder.build()

def _to_uncoupled_basis(
    blocks: TensorMap,
    cg: Optional[ClebschGordanReal] = None,
    device: str = "cpu",
    translations: bool = False,
    high_rank= False # means rank_three  for now TODO: Support a general high rank?
):
    # ----------li, lj in properties------
    if cg is None:
        lmax = max(blocks.keys["L"])
        cg = ClebschGordanReal(lmax, device = device)
    key_names = ["block_type", "species_i", "n_i", "l_i", "species_j", "n_j", "l_j"]
    if high_rank:
        key_names = key_names +["l_3"]

    uncoupled_blocks = {}
    samples = {}
    for key, block in blocks.items():

        if block.values.numel() == 0:
            # Empty block
            continue
        values = block.values
        dtype = values.dtype

        block_type, ai, aj, L  = key.values[:4].tolist()

        for ip, (ni, li, nj, lj) in enumerate(block.properties.values[:, :4].tolist()):
            k = block_type, ai, ni, li, aj, nj, lj
            
            if k not in uncoupled_blocks:
                 uncoupled_blocks[k] = torch.zeros((values.shape[0], 2*li+1, 2*lj+1, 1), device = device, dtype = dtype)
                 samples[k] = block.samples

            uncoupled_blocks[k].add_(torch.tensordot(values[:,:,ip:ip+1], cg._cg[(li, lj, L)].to(dtype = dtype), dims=([1], [2])).permute(0, 2, 3, 1))

    new_blocks = []
    new_keys = []

    components_i = {}    
    components_j = {}    
    for k in uncoupled_blocks:
        _, _, _, li, _, _, lj = k
        try:
            _ = components_i[li]
        except:
            components_i[li] = Labels(['m_i'], torch.arange(-li, li+1).reshape(-1, 1))
        try:
            _ = components_j[lj]
        except:
            components_j[lj] = Labels(['m_j'], torch.arange(-lj, lj+1).reshape(-1, 1))
            
        new_keys.append(k)
        new_blocks.append(
            TensorBlock(
                values = uncoupled_blocks[k],
                samples = samples[k],
                properties = dummy_property,
                components = [components_i[li], components_j[lj]]
                )
        )
    return TensorMap(Labels(['block_type', 'species_i', 'n_i', 'l_i', 'species_j', 'n_j', 'l_j'], torch.tensor(new_keys)), new_blocks)

def _to_uncoupled_basis_old(
    blocks: TensorMap,
    # orbitals: Optional[dict] = None,
    cg: Optional[ClebschGordanReal] = None,
    device: str = "cpu",
    translations: bool = False,
    high_rank= False # means rank_three  for now TODO: Support a general high rank?
):
    # ------li, lj in keys------
    if "l_i" not in blocks.keys.names:
        from mlelec.utils.pbc_utils import move_orbitals_to_keys
        blocks = move_orbitals_to_keys(blocks)
    if cg is None:
        lmax = max(blocks.keys["L"])
        cg = ClebschGordanReal(lmax, device=device)
    if 'inversion_sigma' in blocks.keys.names:
        has_sigma = True
        key_names = blocks.keys.names[:-2]
        if high_rank:
            raise NotImplementedError("High rank not supported for inversion symmetry")
    else: 
        inv_sigma = None
        has_sigma = False
        if not high_rank:    
            key_names = blocks.keys.names[:-1]
        else:
            key_names = blocks.keys.names[:-2]
    component_names = [["m_i"], ["m_j"]]
    if high_rank:
        component_names += [ "m_3"]
    

    block_builder = TensorBuilder(
        # last key name is L, we remove it here
        key_names,
        # sample_names from the blocks
        # this is because, e.g. for multiple molecules, we
        # may have an additional sample name indexing the
        # molecule id
        blocks.sample_names,
        component_names,
        ["dummy"],
        device = device
    )
    for idx, block in blocks.items():

        if block.values.numel() == 0:
            # Empty block
            continue

        block_type = idx["block_type"]
        ai = idx["species_i"]
        ni = idx["n_i"]
        li = idx["l_i"]
        aj = idx["species_j"]
        nj = idx["n_j"]
        lj = idx["l_j"]
        L = idx["L"]
        block_idx = (block_type, ai, ni, li, aj, nj, lj)
        if high_rank:
            l3 = idx["l_3"]
            position_components = Labels(['m_3'], values = torch.tensor([-1,0,1]).reshape(3,-1))
            block_idx = block_idx + (l3,)
        if has_sigma:
            inv_sigma = idx["inversion_sigma"] 
       
            
        if translations:
            
            block_idx = (block_type, ai, ni, li, aj, nj, lj, idx["cell_shift_a"], idx["cell_shift_b"], idx["cell_shift_c"])
        
        
        # block_type, ai, ni, li, aj, nj, lj, L = tuple(idx)
        # if has_sigma:
        #     block_idx = blocks.keys.position(block_idx + (L,))
        # if block_idx in block_builder.blocks:
            continue
        coupled = {}
        if high_rank:
            for K in range(np.abs(li - lj), li + lj + 1):
                coupled[(l3, K, li, lj)]={}
                for L in range(np.abs(K -l3), K + l3 + 1): # l3 
                    if has_sigma:
                        bidx = blocks.keys.position(block_idx + (K, L, inv_sigma))
                    else: 
                        bidx = blocks.keys.position(block_idx + (K, L))
                        # if bidx is None:
                            # print(bidx, 'bidx', block_idx)
                    if bidx is not None:
                        coupled[(l3, K, li, lj)][L] = torch.moveaxis(blocks.block(bidx).values, -1, -2)

            decoupled = cg.decouple(coupled, iterate = 1)
            decoupled = torch.moveaxis(decoupled, (4,3,2,1), (2,1,3,4))#(1,2,3,4))
            new_block = block_builder.add_block(
                key=block_idx,
                properties=torch.tensor([[0]], dtype = torch.int32),
                components=[_components_idx(li), _components_idx(lj), _components_idx(l3)],
            )
            new_block.add_samples(
                labels = block.samples.values.reshape(
                    block.samples.values.shape[0], -1
                ),
                data=decoupled,
            )
            
        else: 
            for L in range(np.abs(li - lj), li + lj + 1):

                if has_sigma:
                    bidx = blocks.keys.position(block_idx + (L, inv_sigma))
                    
                else: 
                    bidx = blocks.keys.position(block_idx + (L,))
                # if has_sigma: 
                #     bidx = bidx + (inv_sigma,)
                if bidx is not None:
                    coupled[L] = torch.moveaxis(blocks.block(bidx).values, -1, -2)
            # if ai == aj== 6 and ni == nj == 2 and li == lj == 1 and block_type == 0:
            #     print(idx, coupled)
                decoupled = cg.decouple({(li, lj): coupled})
                new_block = block_builder.add_block(
                    key=block_idx,
                    properties=torch.tensor([[0]], dtype = torch.int32),
                    components=[_components_idx(li), _components_idx(lj)],
                )
                new_block.add_samples(
                    labels = block.samples.values.reshape(
                        block.samples.values.shape[0], -1
                    ),
                    data=torch.moveaxis(decoupled, 1, -1),
                )
    return block_builder.build()

def map_targetkeys_to_featkeys(features, key, cell_shift=None, return_key=False):
    try:
        block_type = key["block_type"]
        species_center = key["species_i"]
        species_neighbor = key["species_j"]
        L = key["L"]
        li = key["l_i"]
        lj = key["l_j"]
        # ni = key["n_i"]
        # nj = key["n_j"]
    except Exception as e:
        print(e)

        # block_type, ai, ni, li, aj, nj, lj = key
    inversion_sigma = (-1) ** (li + lj + L)
    if cell_shift is None:
        if return_key:
            return labels_where(
                features.keys,
                Labels(
                    [
                        "block_type",
                        "spherical_harmonics_l",
                        "inversion_sigma",
                        "species_center",
                        "species_neighbor",
                    ],
                    values = torch.tensor(
                        [
                            block_type,
                            L,
                            inversion_sigma,
                            species_center,
                            species_neighbor,
                        ]
                    ).reshape(1, -1),
                ),
            )
        block = features.block(
            dict(
            block_type=block_type,
            spherical_harmonics_l=L,
            inversion_sigma=inversion_sigma,
            species_center=species_center,
            species_neighbor=species_neighbor,
            )
        )
        return block
    else:
        assert isinstance(cell_shift, List)
        assert len(cell_shift) == 3
        cell_shift_a, cell_shift_b, cell_shift_c = cell_shift
        if return_key:
            return labels_where(
                features.keys,
                Labels(
                    [
                        "block_type",
                        "spherical_harmonics_l",
                        "inversion_sigma",
                        "species_center",
                        "species_neighbor",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                    ],
                    values = torch.tensor(
                        [
                            block_type,
                            L,
                            inversion_sigma,
                            species_center,
                            species_neighbor,
                            cell_shift_a,
                            cell_shift_b,
                            cell_shift_c,
                        ]
                    ).reshape(1, -1),
                ),
            )
        block = features.block(
            block_type=block_type,
            spherical_harmonics_l=L,
            inversion_sigma=inversion_sigma,
            species_center=species_center,
            species_neighbor=species_neighbor,
            cell_shift_a=cell_shift_a,
            cell_shift_b=cell_shift_b,
            cell_shift_c=cell_shift_c,
        )
        return block

def map_targetkeys_to_featkeys_integrated(features, key, cell_shift=None, return_key=False):
    try:
        block_type = key["block_type"]
        species_center = key["species_i"]
        species_neighbor = key["species_j"]
        L = key["L"]
        inversion_sigma = key["inversion_sigma"]
        # li = key["l_i"]
        # lj = key["l_j"]
        # ni = key["n_i"]
        # nj = key["n_j"]
    except Exception as e:
        print(e)

        # block_type, ai, ni, li, aj, nj, lj = key
    # inversion_sigma = (-1) ** (li + lj + L)
    if cell_shift is None:
        if return_key:
            return labels_where(
                features.keys,
                Labels(
                    [
                        "block_type",
                        "spherical_harmonics_l",
                        "inversion_sigma",
                        "species_center",
                        "species_neighbor",
                    ],
                    values = torch.tensor(
                        [
                            block_type,
                            L,
                            inversion_sigma,
                            species_center,
                            species_neighbor,
                        ]
                    ).reshape(1, -1),
                ),
            )
        block = features.block(
            dict(
            block_type=block_type,
            spherical_harmonics_l=L,
            inversion_sigma=inversion_sigma,
            species_center=species_center,
            species_neighbor=species_neighbor,
            )
        )
        return block
    else:
        assert isinstance(cell_shift, List)
        assert len(cell_shift) == 3
        cell_shift_a, cell_shift_b, cell_shift_c = cell_shift
        if return_key:
            return labels_where(
                features.keys,
                Labels(
                    [
                        "block_type",
                        "spherical_harmonics_l",
                        "inversion_sigma",
                        "species_center",
                        "species_neighbor",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                    ],
                    values = torch.tensor(
                        [
                            block_type,
                            L,
                            inversion_sigma,
                            species_center,
                            species_neighbor,
                            cell_shift_a,
                            cell_shift_b,
                            cell_shift_c,
                        ]
                    ).reshape(1, -1),
                ),
            )
        block = features.block(
            block_type=block_type,
            spherical_harmonics_l=L,
            inversion_sigma=inversion_sigma,
            species_center=species_center,
            species_neighbor=species_neighbor,
            cell_shift_a=cell_shift_a,
            cell_shift_b=cell_shift_b,
            cell_shift_c=cell_shift_c,
        )
        return block


def rotate_matrix(
    matrix, rotation_angles: Union[List, np.array, torch.tensor], frame, orbitals
):
    """Rotate a matrix by a rotation matrix"""
    from mlelec.utils.symmetry import _wigner_d_real

    coupled_blocks = _to_coupled_basis(_to_blocks(matrix, frame, orbitals))
    wd_real = {}
    for l in set(bc.keys["L"]): # FIXME: what is bc?
        wd_real[l] = (
            _wigner_d_real(l, *rotation_angles)
            .type(torch.float)
            .to(coupled_blocks[0].values.device)
        )

    rot_blocks = []
    for i, (key, block) in enumerate(coupled_blocks.items()):
        L = key["L"]
        rvalues = torch.einsum("mM, nMf -> nmf", wd_real[L], block.values)
        rot_blocks.append(
            TensorBlock(
                values=rvalues,
                components=block.components,
                samples=block.samples,
                properties=block.properties,
            )
        )
    rot_blocks_coupled = TensorMap(coupled_blocks.keys, rot_blocks)
    rot_matrix = _to_matrix(
        _to_uncoupled_basis(rot_blocks_coupled, orbitals), frame, orbitals
    )
    return rot_matrix


def discard_nonhermiticity(matrices, retain="upper"):
    """For each T, create a hermitian target with the upper triangle reflected across the diagonal
    retain = "upper" or "lower"
    target :str to specify which matrices to discard nonhermiticity from
    """
    retain = retain.lower()
    retain_upper = retain == "upper"
    fixed = np.zeros_like(matrices)
    for i, mat in enumerate(matrices):
        assert (
            len(mat.shape) == 2
        ), "matrix to discard non-hermiticity from must be a 2D matrix"
        fixed[i] = _reflect_hermitian(mat, retain_upper=retain_upper)
    return fixed
import scipy
import ase 
def compute_eigenval(fock, overlap, eV=False):
    
    eigenval=scipy.linalg.eigvals(fock,overlap)#,UPLO='U')
    eigval=sorted(eigenval)
    e_shift=np.array(eigval)
    if eV:
        from ase.units import Hartree
        return e_shift*Hartree
 
    return e_shift