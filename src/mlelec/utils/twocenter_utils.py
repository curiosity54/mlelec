# Handles 2 center objects - coupling decoupling
# must include preprocessing and postprocessing utils
from typing import Optional, List, Union, Tuple, Dict
from metatensor import TensorMap, TensorBlock
import torch
import ase
import numpy as np
from mlelec.utils.metatensor_utils import TensorBuilder, _to_tensormap
from mlelec.utils.symmetry import ClebschGordanReal


def fix_orbital_order(
    matrix: torch.tensor, frames: Union[List, ase.Atoms], orbital: dict
):
    """Fix the l=1 matrix components from [x,y,z] to [-1, 0,1], handles single and multiple frames"""

    def fix_one_matrix(matrix: torch.tensor, frame: ase.Atoms, orbital: dict):
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
        return torch.stack(fixed_matrices)
    else:
        return fix_one_matrix(matrix, frames, orbital)


def lowdin_orthogonalize(fock: torch.tensor, overlap: torch.tensor):
    """
    lowdin orthogonalization of a fock matrix computing the square root of the overlap matrix
    """
    eva, eve = torch.linalg.eigh(overlap)
    sm12 = eve @ torch.diag(1.0 / torch.sqrt(eva)) @ eve.T
    return sm12 @ fock @ sm12


def _components_idx(l):
    """Returns the m \in {-l,...,l} indices"""
    return np.arange(-l, l + 1, dtype=int).reshape(2 * l + 1, 1)


def _components_idx_2d(li, lj):
    """Returns the 2D outerproduct of m_i \in {-l_i,... , l_i} and m_j \in {-l_j,... , l_j} to index the (l_i, l_j) block of the hamiltonian
    in the uncoupled basis"""
    return np.asarray(
        np.meshgrid(_components_idx(li), _components_idx(lj)), dtype=int
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


def _to_blocks(
    matrices: Union[List[torch.tensor], torch.tensor],
    frames: Union[ase.Atoms, List[ase.Atoms]],
    orbitals: dict,
    device: str = None,
):
    if not isinstance(frames, list):
        assert len(matrices.shape) == 2  # should be just one matrix (nao,nao)
        frames = [frames]
        matrices = matrices.reshape(1, *matrices.shape)
    block_builder = TensorBuilder(
        ["block_type", "a_i", "n_i", "l_i", "a_j", "n_j", "l_j"],
        ["structure", "center", "neighbor"],
        [["m1"], ["m2"]],
        ["value"],
    )
    orbs_tot, _ = _orbs_offsets(orbitals)

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
                            ki_base : ki_base + orbs_tot[ai],
                            kj_base : kj_base + orbs_tot[aj],
                        ]
                    )
                elif isinstance(ham, torch.Tensor):
                    block_data = ham[
                        ki_base : ki_base + orbs_tot[ai],
                        kj_base : kj_base + orbs_tot[aj],
                    ]
                else:
                    raise ValueError

                # print(block_data, block_data.shape)
                if block_type == 1:
                    # print(block_data)
                    block_data_plus = (block_data + block_data.T) / 2 ** (0.5)
                    block_data_minus = (block_data - block_data.T) / 2 ** (0.5)
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
    return block_builder.build()


def blocks_to_matrix(
    blocks: TensorMap,
    frames: List[ase.Atoms],
    orbs: Dict[int, List[Tuple[int, int, int]]],
    # vectorized: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    # if vectorized:
    #     return _vectorized_blocks_to_matrix(blocks=blocks, frames=frames, orbs=orbs)
    # else:
    return _to_matrix(blocks=blocks, frames=frames, orbs=orbs)


def _to_matrix(
    blocks: TensorMap,
    frames: Union[ase.Atoms, List[ase.Atoms]],
    orbs: Dict[int, List[Tuple[int, int, int]]],
    device=None,
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
    orbs_tot, orbs_offset = _orbs_offsets(orbs)

    # indices of the block for each atom
    atom_blocks_idx = _atom_blocks_idx(frames, orbs_tot)

    if device is None:
        device = blocks.block(0).values.device
    # else:
    # assert (
    #     device == blocks.block(0).values.device
    # ), "device mismatch between blocks and device argument"

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

        block_type, ai, ni, li, aj, nj, lj = tuple(idx)

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
            )
    if len(matrices) == 1:
        return matrices[0]
    else:
        return torch.stack(matrices)


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
):
    """fill block of type <type> where type is either -1,0,1,2"""
    islice = slice(ki_base + ki_offset, ki_base + ki_offset + 2 * li + 1)
    jslice = slice(kj_base + kj_offset, kj_base + kj_offset + 2 * lj + 1)
    if type == 0:
        matrix[islice, jslice] = values
        if not same_koff:
            matrix[jslice, islice] = values.T
    if type == 2:
        matrix[islice, jslice] = values
        matrix[jslice, islice] = values.T

    if abs(type) == 1:
        values_2norm = values / (2 ** (0.5))
        matrix[islice, jslice] += values_2norm
        matrix[jslice, islice] += values_2norm.T
        if not same_koff:
            islice = slice(ki_base + kj_offset, ki_base + kj_offset + 2 * lj + 1)
            jslice = slice(kj_base + ki_offset, kj_base + ki_offset + 2 * li + 1)
            if type == 1:
                matrix[islice, jslice] += values_2norm.T
                matrix[jslice, islice] += values_2norm
            else:
                matrix[islice, jslice] -= values_2norm.T
                matrix[jslice, islice] -= values_2norm


def _to_coupled_basis(
    blocks: Union[torch.tensor, TensorMap],
    orbitals: Optional[dict] = None,
    cg: Optional[ClebschGordanReal] = None,
    device: str = None,
):
    if torch.is_tensor(blocks):
        print("Converting matrix to blocks before coupling")
        assert orbitals is not None, "Need orbitals to convert matrix to blocks"
        blocks = _to_blocks(blocks, orbitals)
    if cg is None:
        lmax = max(blocks.keys["l_i"] + blocks.keys["l_j"])
        cg = ClebschGordanReal(lmax, device=device)

    block_builder = TensorBuilder(
        ["block_type", "a_i", "n_i", "l_i", "a_j", "n_j", "l_j", "L"],
        ["structure", "center", "neighbor"],
        [["M"]],
        ["value"],
    )
    for idx, block in blocks.items():
        block_type, ai, ni, li, aj, nj, lj = tuple(idx)

        # Moves the components at the end as cg.couple assumes so
        decoupled = torch.moveaxis(block.values, -1, -2).reshape(
            (len(block.samples), len(block.properties), 2 * li + 1, 2 * lj + 1)
        )
        # selects the (only) key in the coupled dictionary (l1 and l2
        # that gave birth to the coupled terms L, with L going from
        # |l1 - l2| up to |l1 + l2|
        coupled = cg.couple(decoupled)[(li, lj)]

        for L in coupled:
            block_idx = tuple(idx) + (L,)
            # skip blocks that are zero because of symmetry
            if ai == aj and ni == nj and li == lj:
                parity = (-1) ** (li + lj + L)
                if (parity == -1 and block_type in (0, 1)) or (
                    parity == 1 and block_type == -1
                ):
                    continue

            new_block = block_builder.add_block(
                keys=block_idx,
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


def _to_uncoupled_basis(
    blocks: TensorMap,
    orbitals: Optional[dict] = None,
    cg: Optional[ClebschGordanReal] = None,
    device: str = None,
):
    if cg is None:
        lmax = max(blocks.keys["L"])
        cg = ClebschGordanReal(lmax)

    block_builder = TensorBuilder(
        # last key name is L, we remove it here
        blocks.keys.names[:-1],
        # sample_names from the blocks
        # this is because, e.g. for multiple molecules, we
        # may have an additional sample name indexing the
        # molecule id
        blocks.sample_names,
        [["m1"], ["m2"]],
        ["value"],
    )
    for idx, block in blocks.items():
        block_type, ai, ni, li, aj, nj, lj, L = tuple(idx)
        block_idx = (block_type, ai, ni, li, aj, nj, lj)
        if block_idx in block_builder.blocks:
            continue
        coupled = {}
        for L in range(np.abs(li - lj), li + lj + 1):
            bidx = blocks.keys.position(block_idx + (L,))
            if bidx is not None:
                coupled[L] = torch.moveaxis(blocks.block(bidx).values, -1, -2)
        decoupled = cg.decouple({(li, lj): coupled})

        new_block = block_builder.add_block(
            keys=block_idx,
            properties=np.asarray([[0]], dtype=np.int32),
            components=[_components_idx(li), _components_idx(lj)],
        )
        new_block.add_samples(
            labels=np.asarray(block.samples.values).reshape(
                block.samples.values.shape[0], -1
            ),
            data=torch.moveaxis(decoupled, 1, -1),
        )
    return block_builder.build()


def map_targetkeys_to_featkeys(features, key):
    try:
        block_type = key["block_type"]
        species_center = key["a_i"]
        species_neighbor = key["a_j"]
        L = key["L"]
        li = key["l_i"]
        lj = key["l_j"]
        ni = key["n_i"]
        nj = key["n_j"]
    except Exception as e:
        print(e)

        # block_type, ai, ni, li, aj, nj, lj = key
    inversion_sigma = (-1) ** (li + lj + L)
    block = features.block(
        block_type=block_type,
        spherical_harmonics_l=L,
        inversion_sigma=inversion_sigma,
        species_center=species_center,
        species_neighbor=species_neighbor,
    )
    return block


def rotate_matrix(
    matrix, rotation_angles: Union[List, np.array, torch.tensor], frame, orbitals
):
    """Rotate a matrix by a rotation matrix"""
    from mlelec.utils.symmetry import _wigner_d_real

    coupled_blocks = _to_coupled_basis(_to_blocks(matrix, frame, orbitals))
    wd_real = {}
    for l in set(bc.keys["L"]):
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
