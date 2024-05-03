import numpy as np
import warnings
from scipy.fft import fftn, ifftn
from torch.fft import fftn as torch_fftn, ifftn as torch_ifftn
from ase.units import Bohr


def scidx_from_unitcell(frame, j=0, T=[0, 0, 0], kmesh=None):
    """Find index of atom j belonging to cell with translation vector T to its index in the supercell consistent with an SCF calc with kgrid =kmesh"""
    assert j < len(frame), "j must be less than the number of atoms in the unit cell"
    assert len(T) == 3, "T must be a 3-tuple"
    assert all(
        [t >= 0 for t in T]
    ), "T must be non-negative"  # T must be positive for the logic
    if kmesh is None:
        kmesh = np.asarray([1, 1, 1])
        warnings.warn("kmesh not specified, assuming 1x1x1")
    if isinstance(kmesh, int):
        kmesh = np.asarray([kmesh, kmesh, kmesh])
    else:
        assert len(kmesh) == 3, "kmesh must be a 3-tuple"

    N1, N2, N3 = kmesh
    natoms = len(frame)
    J = j + natoms * ((N3 * N2) * T[0] + (N3 * T[1]) + T[2])
    return J


def _position_in_translation(frame, j, T):
    """Return the position of atom j in unit cell translated by vector T"""
    return frame.positions[j] + np.dot((T[0], T[1], T[2]), frame.cell)


def scidx_to_mic_translation(
    frame,
    I=0,
    J=0,
    j=0,
    kmesh=[1, 1, 1],
    epsilon=0,
    ifprint=False,
    return_distance=False,
):
    def fix_translation_sign(frame, mic_T, i, j):
        """Return the "correct mic_T"""
        cell = frame.cell.array.T

        # Vector joining atoms i and j in the unit cell
        rij_in_cell = frame.positions[j] - frame.positions[i]

        # i-j MIC distance in the supercell
        rij_T = cell @ mic_T + rij_in_cell

        # same thing, but the sign of T is reversed
        rij_mT = -cell @ mic_T + rij_in_cell

        # Check whether the two distances are the same
        equal_distance = np.isclose(np.linalg.norm(rij_T), np.linalg.norm(rij_mT))

        if not equal_distance:
            return mic_T, False
            # # Return the T that gives the shortest distance
            # if np.linalg.norm(rij_T) < np.linalg.norm(rij_mT):
            #     return mic_T
            # else:
            #     return -np.asarray(mic_T)
        else:
            # If the distances are equal, choose the T that has the first non-zero component posiviely signed
            idx = np.where(~(np.sign(mic_T) == np.sign(-np.asarray(mic_T))))

            if len(idx[0]) == 0:
                assert np.linalg.norm(mic_T) == 0, mic_T
                return mic_T, False
            else:
                idx = idx[0][0]
                if np.sign(mic_T[idx]) == 1:
                    return mic_T, False
                else:
                    return -np.asarray(mic_T), True

    """Find the minimum image convention translation vector from atom I to atom J in the supercell of size kmesh"""
    assert frame.cell is not None, "Cell must be defined"
    cell_inv = np.linalg.inv(frame.cell.array.T)

    superframe = frame.repeat(kmesh)
    if J >= len(superframe):
        # print(J)
        J = J % len(superframe)
        warnings.warn(
            "J is greater than the number of atoms in the supercell. Mapping J to J % len(superframe)"
        )
    assert I < len(frame) and J < len(
        superframe
    ), "I and J must be less than the number of atoms in the supercell"
    # this works correctly only when I<J - J should not be greater than I anyway as I always in 000 cell
    d = superframe.get_distance(I, J, mic=True, vector=True).T
    p_i = frame.positions[I]
    p_j = frame.positions[j]
    dplus = p_i + d - p_j  # from i to J to the origin of that (unit) cell
    dminus = p_j - d - p_i  # from j to I to the origin of that (unit) cell
    # print(
    # cell_inv @ dplus, cell_inv @ dminus, cell_inv @ (dplus + dminus), "mict,micmt"
    # )
    mic_T = np.round(cell_inv @ dplus + epsilon).astype(int)
    # mic_minusT = np.round(cell_inv @ dminus + epsilon).astype(int)
    # print("bef", mic_T, mic_minusT, end=" ")
    mic_T, fixed_plus = fix_translation_sign(frame, mic_T, I, j)
    mic_minusT = -1 * mic_T
    fixed_minus = fixed_plus
    # mic_minusT, fixed_minus = fix_translation_sign(frame, mic_minusT, j, I)
    # fixed_plus = fixed_minus = False
    # print("aft", mic_T, mic_minusT)
    if ifprint:
        print(cell_inv @ d)
        print(d)
    if return_distance:
        distance = superframe.get_distance(I, J, mic=True, vector=False)
        return mic_T, mic_minusT, fixed_plus, fixed_minus, distance
    else:
        return (
            mic_T,
            mic_minusT,
            fixed_plus,
            fixed_minus,
        )  # adding noise to avoid numerical issues


# np.floor(cell_inv @ d + epsilon).astype(
# int
# )  # adding noise to avoid numerical issues
# This sometimes returns [-3,-2,0] for [1 2 0] which is correect based on the distance

from mlelec.utils.metatensor_utils import TensorBuilder
from mlelec.utils.twocenter_utils import (
    _components_idx,
    ISQRT_2,
    _orbs_offsets,
    _atom_blocks_idx,
)
import torch

def inverse_bloch_sum(dataset, matrix, A, cutoff):
    dimension = dataset.dimension
    T_list = np.linalg.solve(dataset.cells[A].lattice_vectors().T, dataset.cells[A].get_lattice_Ls(rcut = cutoff/Bohr, dimension = dimension).T).T
    assert np.linalg.norm(T_list - np.round(T_list)) < 1e-9, np.linalg.norm(Ts - np.round(Ts))
    Ts = torch.from_numpy(np.round(T_list))
    k_list = torch.from_numpy(dataset.kpts_rel[A])
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)
    else:
        assert isinstance(matrix, torch.Tensor), f"Matrix must be np.ndarray or torch.Tensor, but it's {type(matrix)}"

    HT = fourier_transform(matrix, T_list = Ts, k = k_list, norm = 1/k_list.shape[0])

    T_list = np.int32(np.round(T_list))
    H_T = {}
    for T, H in zip(T_list, HT):
        assert torch.norm(H - H.real) < 1e-10, torch.norm(H - H.real).item()
        # print(torch.norm(H - H.real))
        H_T[tuple(T)] = H.real
    return H_T

def matrix_to_blocks_OLD(dataset, device=None, all_pairs = True, cutoff = None, target='fock'):
    from mlelec.utils.metatensor_utils import TensorBuilder

    if device is None:
        device = dataset.device

    key_names = [
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
    ]
    sample_names = ["structure", "center", "neighbor"]
    property_names = ["dummy"]
    property_values = np.asarray([[0]])
    component_names = [["m_i"], ["m_j"]]

    # multiplicity
    orbs_mult = {}
    for species in dataset.basis:
        _, orbidx, count = np.unique(
            np.asarray(dataset.basis[species])[:, :2],
            axis=0,
            return_counts=True,
            return_index=True,
        )
        idx = np.argsort(orbidx)
        unique_orbs = np.asarray(dataset.basis[species])[orbidx[idx]][:, :2]
        orbs_mult[species] = {tuple(k): v for k, v in zip(unique_orbs, count[idx])}

    block_builder = TensorBuilder(
        key_names,
        sample_names,
        component_names,
        property_names,
    )

    if cutoff is None:
        cutoff_was_none = True
    else:
        cutoff_was_none = False

    from itertools import product
    orbs_tot, _ = _orbs_offsets(dataset.basis)  # returns orbs_tot,
    # if target.lower() == "fock":
    #     matrices = inverse_bloch_sum(dataset.fock_kspace, cutoff)
    # elif target.lower() == "overlap":
    #     matrices = inverse_bloch_sum(dataset.overlap_kspace, cutoff)
    # else:
    #     raise ValueError("target must be either 'fock' or 'overlap'")
    # for T in dataset.desired_shifts: # Loop over translations given in input

    for A, frame in enumerate(dataset.structures):  # Loop over frames

        if cutoff_was_none:
            cutoff = dataset.cells[A].rcut * Bohr
            warnings.warn('Automatic choice of the cutoff for structure {A}. rcut = {rcut:.2f} Angstrom')

        if target.lower() == "fock":
            matrices = inverse_bloch_sum(dataset, dataset.fock_kspace[A], A, cutoff)
        elif target.lower() == "overlap":
            matrices = inverse_bloch_sum(dataset, dataset.overlap_kspace[A], A, cutoff)
        else:
            raise ValueError("target must be either 'fock' or 'overlap'")

        for T in matrices:
            mT = tuple(-t for t in T)
            assert mT in matrices, f"{mT} not in the real space matrix keys"

            matrixT = matrices[T]
            matrixmT = matrices[mT]
            if isinstance(matrixT, np.ndarray):
                matrixT = torch.from_numpy(matrixT).to(device)
                matrixmT = torch.from_numpy(matrixmT).to(device)
            else:
                matrixT = matrixT.to(device)
                matrixmT = matrixmT.to(device)
            assert np.isclose(torch.norm(matrixT - matrixmT.T).item(), 0.0), f"Failed to check H({T}) = H({mT})^\dagger"

            i_start = 0
            for i, ai in enumerate(frame.numbers):
                orbs_i = orbs_mult[ai]
                j_start = 0

                for j, aj in enumerate(frame.numbers):

                    # skip_pair = False # uncomment for MIC

                    # Handle the case only the upper triangle is learnt
                    if not all_pairs: # not all orbital pairs
                        if i > j and ai == aj: # skip block type 1 if i>j 
                            j_start += orbs_tot[aj]
                            continue
                        elif ai > aj: # keep only sorted species 
                            j_start += orbs_tot[aj]
                            continue
                       
                    # Skip the pair if their distance exceeds the cutoff
                    ij_distance = np.linalg.norm(frame.cell.array.T @ np.array(T) + frame.positions[j] - frame.positions[i])
                    if ij_distance > cutoff:
                        j_start += orbs_tot[aj]
                        continue
                    # if cutoff is not None: # uncomment for MIC
                        # for mic_T in dataset._translation_dict[A][T]: # FIXME allow for mic=False # uncomment for MIC
                        #     if dataset._translation_counter[A][mic_T][i, j]: # uncomment for MIC
                        #         ij_distance = np.linalg.norm(frame.cell.array.T @ np.array(mic_T) + frame.positions[j] - frame.positions[i]) # uncomment for MIC
                        #         if ij_distance > cutoff: # uncomment for MIC
                        #             skip_pair = True # uncomment for MIC
                        #         break # uncomment for MIC
                    # if skip_pair: # uncomment for MIC
                        # j_start += orbs_tot[aj] # uncomment for MIC
                        # continue # uncomment for MIC
                        
                    orbs_j = orbs_mult[aj]

                    # add what kind of blocks we expect in the tensormap
                    if all_pairs:
                        # n1l1n2l2 = np.concatenate(tuple(tuple(k2 + k1 for k1 in orbs_i) for k2 in orbs_j))
                        n1l1n2l2 = list(sum([tuple(k2 + k1 for k1 in orbs_i) for k2 in orbs_j], ()))
                        # print(n1l1n2l2)
                    else:
                        sorted_orbs = np.sort([(o1, o2) for o1, o2 in product(list(orbs_i.keys()), list(orbs_j.keys()))], axis=1)
                        orbs, orbital_idx = np.unique(sorted_orbs, return_index = True, axis = 0)
                        n1l1n2l2 = [tuple(o1) + tuple(o2) for o1, o2 in orbs]
                        # print(n1l1n2l2)

                    # print(i,j,slice(i_start, i_start+orbs_tot[ai]), slice(j_start, j_start+orbs_tot[aj]))
                    block_ij = matrixT[i_start:i_start + orbs_tot[ai], j_start:j_start + orbs_tot[aj]]

                    block_split = [torch.split(blocki, list(orbs_j.values()), dim = 1) for blocki in torch.split(block_ij, list(orbs_i.values()), dim=0)]
                    block_split = [y for x in block_split for y in x]  # flattening the list of lists above

                    for iorbital, (ni, li, nj, lj) in enumerate(n1l1n2l2):
                        if not all_pairs:
                            iorbital = orbital_idx[iorbital]
                        value = block_split[iorbital]

                        if i == j and np.linalg.norm(T) == 0:
                            # On-site
                            block_type = 0
                            key = (block_type, ai, ni, li, aj, nj, lj, *T)

                        elif (ai == aj) or (i == j and T != [0, 0, 0]):
                            # Same species interaction
                            block_type = 1
                            key = (block_type, ai, ni, li, aj, nj, lj, *T)
                            block_jimT = matrixmT[j_start : j_start + orbs_tot[aj], i_start : i_start + orbs_tot[ai]]
                            block_jimT_split = [torch.split(blocki, list(orbs_i.values()), dim=1) for blocki in torch.split(block_jimT, list(orbs_j.values()), dim = 0)]
                            block_jimT_split = [y for x in block_jimT_split for y in x]  # flattening the list of lists above
                            value_ji = block_jimT_split[iorbital]  # same orbital in the ji subblock
                        else:
                            # Different species interaction
                            # skip ai>aj
                            block_type = 2
                            key = (block_type, ai, ni, li, aj, nj, lj, *T)

                        if key not in block_builder.blocks:
                            # add blocks if not already present
                            block = block_builder.add_block(key=key, properties=property_values, components=[_components_idx(li), _components_idx(lj)])
                            if block_type == 1:
                                block = block_builder.add_block(
                                    key=(-1,) + key[1:],
                                    properties=property_values,
                                    components=[_components_idx(li), _components_idx(lj)],
                                )

                        # add samples to the blocks when present
                        block = block_builder.blocks[key]
                        if block_type == 1:
                            block_asym = block_builder.blocks[(-1,) + key[1:]]

                        if block_type == 1:
                            # if i > j:  # keep only (i,j) and not (j,i)
                                # continue
                            # bplus = value
                            bplus = (value + value_ji) * ISQRT_2
                            # bminus = value_ji
                            bminus = (value - value_ji) * ISQRT_2
                            # print(i,j)
                            block.add_samples(
                                labels=[(A, i, j)],
                                data=bplus.reshape(1, 2 * li + 1, 2 * lj + 1, 1),
                            )

                            block_asym.add_samples(
                                labels=[(A, i, j)],
                                data=bminus.reshape(1, 2 * li + 1, 2 * lj + 1, 1),
                            )

                        elif block_type == 0 or block_type == 2:
                            block.add_samples(
                                labels=[(A, i, j)],
                                data=value.reshape(1, 2 * li + 1, 2 * lj + 1, 1),
                            )
                        
                        else:
                            raise ValueError("Block type not implemented")
                    j_start += orbs_tot[aj]

                i_start += orbs_tot[ai]
    return block_builder.build()

def matrix_to_blocks(dataset, device=None, all_pairs = True, cutoff = None, target='fock'):
    from mlelec.utils.metatensor_utils import TensorBuilder

    if device is None:
        device = dataset.device

    key_names = [
        "block_type",
        "species_i",
        "n_i",
        "l_i",
        "species_j",
        "n_j",
        "l_j"
    ]
    sample_names = ["structure", "center", "neighbor", "cell_shift_a", "cell_shift_b", "cell_shift_c"]
    property_names = ["dummy"]
    property_values = np.asarray([[0]])
    component_names = [["m_i"], ["m_j"]]

    # multiplicity
    orbs_mult = {}
    for species in dataset.basis:
        _, orbidx, count = np.unique(
            np.asarray(dataset.basis[species])[:, :2],
            axis=0,
            return_counts=True,
            return_index=True,
        )
        idx = np.argsort(orbidx)
        unique_orbs = np.asarray(dataset.basis[species])[orbidx[idx]][:, :2]
        orbs_mult[species] = {tuple(k): v for k, v in zip(unique_orbs, count[idx])}

    block_builder = TensorBuilder(
        key_names,
        sample_names,
        component_names,
        property_names,
    )

    if cutoff is None:
        cutoff_was_none = True
    else:
        cutoff_was_none = False

    from itertools import product
    orbs_tot, _ = _orbs_offsets(dataset.basis)  # returns orbs_tot,

    for A, frame in enumerate(dataset.structures):  # Loop over frames

        if cutoff_was_none:
            cutoff = dataset.cells[A].rcut * Bohr
            warnings.warn(f'Automatic choice of the cutoff for structure {A}. rcut = {cutoff:.2f} Angstrom')

        if target.lower() == "fock":
            if dataset.fock_realspace is None:
                matrices = inverse_bloch_sum(dataset, dataset.fock_kspace[A], A, cutoff)
            else:
                matrices = dataset.fock_realspace[A]
        elif target.lower() == "overlap":
            if dataset.overlap_realspace is None:
                matrices = inverse_bloch_sum(dataset, dataset.overlap_kspace[A], A, cutoff)
            else:
                matrices = dataset.overlap_realspace[A]
        else:
            raise ValueError("target must be either 'fock' or 'overlap'")

        for iii,T in enumerate(matrices):
            mT = tuple(-t for t in T)
            assert mT in matrices, f"{mT} not in the real space matrix keys"

            matrixT = matrices[T]
            matrixmT = matrices[mT]
            if isinstance(matrixT, np.ndarray):
                matrixT = torch.from_numpy(matrixT).to(device)
                matrixmT = torch.from_numpy(matrixmT).to(device)
            else:
                matrixT = matrixT.to(device)
                matrixmT = matrixmT.to(device)
            assert np.isclose(torch.norm(matrixT - matrixmT.T).item(), 0.0), f"Failed to check H({T}) = H({mT})^\dagger"

            i_start = 0
            # Loop over the all the atoms in the structure, by atomic number
            for i, ai in enumerate(frame.numbers):
                orbs_i = orbs_mult[ai]
                j_start = 0

                # Loop over the all the atoms in the structure, by atomic number
                for j, aj in enumerate(frame.numbers):

                    # Handle the case only the upper triangle is learnt
                    if not all_pairs: # not all orbital pairs
                        if i > j and ai == aj: # skip block type 1 if i>j 
                            j_start += orbs_tot[aj]
                            continue
                        elif ai > aj: # keep only sorted species 
                            j_start += orbs_tot[aj]
                            continue
                       
                    # Skip the pair if their distance exceeds the cutoff
                    ij_distance = np.linalg.norm(frame.cell.array.T @ np.array(T) + frame.positions[j] - frame.positions[i])
                    if ij_distance > cutoff:
                        j_start += orbs_tot[aj]
                        continue
                        
                    orbs_j = orbs_mult[aj]

                    # add what kind of blocks we expect in the tensormap
                    n1l1n2l2 = list(sum([tuple(k2 + k1 for k1 in orbs_i) for k2 in orbs_j], ()))

                    # if all_pairs:
                    #     # n1l1n2l2 = np.concatenate(tuple(tuple(k2 + k1 for k1 in orbs_i) for k2 in orbs_j))
                    #     n1l1n2l2 = list(sum([tuple(k2 + k1 for k1 in orbs_i) for k2 in orbs_j], ()))
                    #     # print(n1l1n2l2)
                    # else:
                    #     sorted_orbs = np.sort([(o1, o2) for o1, o2 in product(list(orbs_i.keys()), list(orbs_j.keys()))], axis=1)
                    #     orbs, orbital_idx = np.unique(sorted_orbs, return_index = True, axis = 0)
                    #     n1l1n2l2 = [tuple(o1) + tuple(o2) for o1, o2 in orbs]
                        # print(n1l1n2l2)

                    # print(i,j,slice(i_start, i_start+orbs_tot[ai]), slice(j_start, j_start+orbs_tot[aj]))
                    block_ij = matrixT[i_start:i_start + orbs_tot[ai], j_start:j_start + orbs_tot[aj]]

                    block_split = [torch.split(blocki, list(orbs_j.values()), dim = 1) for blocki in torch.split(block_ij, list(orbs_i.values()), dim=0)]
                    block_split = [y for x in block_split for y in x]  # flattening the list of lists above

                    for iorbital, (ni, li, nj, lj) in enumerate(n1l1n2l2):
                        # if not all_pairs:
                        #     iorbital = orbital_idx[iorbital]
                        value = block_split[iorbital]

                        if i == j and np.linalg.norm(T) == 0:
                            # On-site
                            block_type = 0
                            key = (block_type, ai, ni, li, aj, nj, lj)

                        elif (ai == aj) or (i == j and T != [0, 0, 0]):
                            # Same species interaction
                            block_type = 1
                            key = (block_type, ai, ni, li, aj, nj, lj)
                            block_jimT = matrixmT[j_start : j_start + orbs_tot[aj], i_start : i_start + orbs_tot[ai]]
                            block_jimT_split = [torch.split(blocki, list(orbs_i.values()), dim=1) for blocki in torch.split(block_jimT, list(orbs_j.values()), dim = 0)]
                            block_jimT_split = [y for x in block_jimT_split for y in x]  # flattening the list of lists above
                            value_ji = block_jimT_split[iorbital]  # same orbital in the ji subblock
                        else:
                            # Different species interaction
                            # skip ai>aj
                            block_type = 2
                            key = (block_type, ai, ni, li, aj, nj, lj)

                        if key not in block_builder.blocks:
                            # add blocks if not already present
                            block = block_builder.add_block(key=key, properties=property_values, components=[_components_idx(li), _components_idx(lj)])
                            if block_type == 1:
                                block = block_builder.add_block(
                                    key=(-1,) + key[1:],
                                    properties=property_values,
                                    components=[_components_idx(li), _components_idx(lj)],
                                )

                        # add samples to the blocks when present
                        block = block_builder.blocks[key]
                        if block_type == 1:
                            block_asym = block_builder.blocks[(-1,) + key[1:]]

                        if block_type == 1:
                            # if i > j:  # keep only (i,j) and not (j,i)
                                # continue
                            # bplus = value
                            bplus = (value + value_ji) * ISQRT_2
                            # bminus = value_ji
                            bminus = (value - value_ji) * ISQRT_2
                            # print(i,j)
                            block.add_samples(
                                labels = [(A, i, j, *T)],
                                data = bplus.reshape(1, 2 * li + 1, 2 * lj + 1, 1),
                            )

                            block_asym.add_samples(
                                labels = [(A, i, j, *T)],
                                data = bminus.reshape(1, 2 * li + 1, 2 * lj + 1, 1),
                            )

                        elif block_type == 0 or block_type == 2:
                            block.add_samples(
                                labels = [(A, i, j, *T)],
                                data = value.reshape(1, 2 * li + 1, 2 * lj + 1, 1),
                            )
                        
                        else:
                            raise ValueError("Block type not implemented")
                    j_start += orbs_tot[aj]

                i_start += orbs_tot[ai]
    return block_builder.build()

def move_cell_shifts_to_keys(blocks):
    """ Move cell shifts when present in samples, to keys"""
    from metatensor import Labels, TensorBlock, TensorMap

    out_blocks = []
    out_block_keys = []

    for key, block in blocks.items():        
        translations = np.unique(block.samples.values[:, -3:], axis = 0)
        for T in translations:
            block_view = block.samples.view(["cell_shift_a", "cell_shift_b", "cell_shift_c"]).values
            idx = np.where(np.all(np.isclose(np.array(block_view),np.array([T[0], T[1], T[2]])), axis = 1))[0]

            if len(idx):
                out_block_keys.append(list(key.values)+[T[0], T[1], T[2]])
                out_blocks.append(TensorBlock(
                        samples = Labels(
                            blocks.sample_names[:-3],
                            values = np.asarray(block.samples.values[idx])[:, :-3],
                        ),
                        values = block.values[idx],
                        components = block.components,
                        properties = block.properties,
                    ))
                
    return TensorMap(Labels(blocks.keys.names + ["cell_shift_a", "cell_shift_b", "cell_shift_c"], np.asarray(out_block_keys)), out_blocks)

def NEW_blocks_to_matrix(blocks, dataset, device=None, return_negative=False, cg = None):
    if device is None:
        device = dataset.device

    if "L" in blocks.keys.names:
        from mlelec.utils.twocenter_utils import _to_uncoupled_basis
        blocks = _to_uncoupled_basis(blocks, cg = cg)

    orbs_tot, orbs_offset = _orbs_offsets(dataset.basis)
    atom_blocks_idx = _atom_blocks_idx(dataset.structures, orbs_tot)
    orbs_mult = {
        species: 
                {tuple(k): v
            for k, v in zip(
                *np.unique(
                    np.asarray(dataset.basis[species])[:, :2],
                    axis=0,
                    return_counts=True,
                )
            )
        }
        for species in dataset.basis
    }

    reconstructed_matrices_plus = []
    reconstructed_matrices_minus = []

    # Loop over frames
    for A, shifts in enumerate(dataset.realspace_translations):
        norbs = np.sum([orbs_tot[ai] for ai in dataset.structures[A].numbers])

        reconstructed_matrices_plus.append({T: torch.zeros(norbs, norbs, device = device) for T in shifts})
        reconstructed_matrices_minus.append({T: torch.zeros(norbs, norbs, device = device) for T in shifts})

    # loops over block types
    for key, block in blocks.items():
        block_type = key["block_type"]
        ai, ni, li = key["species_i"], key["n_i"], key["l_i"]
        aj, nj, lj = key["species_j"], key["n_j"], key["l_j"]
        
        # What's the multiplicity of the orbital type, ex. 2p_x, 2p_y, 2p_z makes the multiplicity 
        # of a p block = 3
        orbs_i = orbs_mult[ai]
        orbs_j = orbs_mult[aj]
        
        # The shape of the block corresponding to the orbital pair
        shapes = {
            (k1 + k2): (orbs_i[tuple(k1)], orbs_j[tuple(k2)])
            for k1 in orbs_i
            for k2 in orbs_j
        }
        # offset of the orbital (ni, li) within a block of atom i
        ioffset = orbs_offset[(ai, ni, li)] 
        # offset of the orbital (nj,lj) within a block of atom j
        joffset = orbs_offset[(aj, nj, lj)]

        i_end, j_end = shapes[(ni, li, nj, lj)]

        # loops over samples (structure, i, j)
        for sample, blockval in zip(block.samples, block.values):
            
            A = sample["structure"]
            i = sample["center"]
            j = sample["neighbor"]
            Tx, Ty, Tz = sample["cell_shift_a"], sample["cell_shift_b"], sample["cell_shift_c"]

            matrix_T_plus  = reconstructed_matrices_plus[A][Tx, Ty, Tz]

            if return_negative:
                matrix_T_minus = reconstructed_matrices_minus[A][Tx, Ty, Tz]
            # i_start, j_start = atom_blocks_idx[(A, i, j)]

            i_start, j_start = atom_blocks_idx[(A, i, j)]
            i_slice = slice(i_start + ioffset, i_start + ioffset + i_end) 
            j_slice = slice(j_start + joffset, j_start + joffset + j_end)

            # OPT (commented)
            # values = blockval[:, :, 0].clone()

            if block_type == 1:
                matrix_T_plus[
                    # i_start + ioffset : i_start + ioffset + i_end,
                    # j_start + joffset : j_start + joffset + j_end,
                    i_slice, j_slice
                ] += blockval[:, :, 0]*ISQRT_2 # OPTvalues

                if return_negative:
                    matrix_T_minus[
                        # j_start + ioffset : j_start + ioffset + i_end,
                        # i_start + joffset : i_start + joffset + j_end,
                        i_slice, j_slice
                    ] += blockval[:, :, 0]*ISQRT_2 # values # OPT
                        
            elif block_type == -1:
                    
                matrix_T_plus[
                    # i_start + ioffset : i_start + ioffset + i_end,
                    # j_start + joffset : j_start + joffset + j_end,
                    i_slice, j_slice
                ] += blockval[:, :, 0] # values # OPT

                if return_negative:
                    matrix_T_minus[
                        # j_start + ioffset : j_start + ioffset + i_end,
                        # i_start + joffset : i_start + joffset + j_end,
                        i_slice, j_slice
                    ] -= blockval[:, :, 0] # values # OPT

            # if block_type == 0 or block_type == 2:
            else: # bt = 0 or 2

                matrix_T_plus[
                    # i_start + ioffset : i_start + ioffset + i_end,
                    # j_start + joffset : j_start + joffset + j_end,
                    i_slice,
                    j_slice,
                             ] = blockval[:, :, 0] # values # OPT


    if return_negative:
        return reconstructed_matrices_plus, reconstructed_matrices_minus
    return reconstructed_matrices_plus

def blocks_to_matrix(blocks, dataset, device=None, return_negative=False, cg = None):
    if device is None:
        device = dataset.device
        
    if "cell_shift_a" not in blocks.keys.names:
        assert "cell_shift_b" not in blocks.keys.names, "Weird! keys contain 'cell_shift_b' but not 'cell_shift_a'."
        assert "cell_shift_c" not in blocks.keys.names, "Weird! keys contain 'cell_shift_c' but not 'cell_shift_a'."

        assert "cell_shift_a" in blocks.sample_names, "Cell shifts must be in samples."
        assert "cell_shift_b" in blocks.sample_names, "Cell shifts must be in samples."
        assert "cell_shift_c" in blocks.sample_names, "Cell shifts must be in samples."

        if "L" in blocks.keys.names:
            from mlelec.utils.twocenter_utils import _to_uncoupled_basis
            blocks = _to_uncoupled_basis(blocks, cg = cg, device = device)
        blocks = move_cell_shifts_to_keys(blocks)      

    orbs_tot, orbs_offset = _orbs_offsets(dataset.basis)
    atom_blocks_idx = _atom_blocks_idx(dataset.structures, orbs_tot)
    orbs_mult = {
        species: 
                {tuple(k): v
            for k, v in zip(
                *np.unique(
                    np.asarray(dataset.basis[species])[:, :2],
                    axis=0,
                    return_counts=True,
                )
            )
        }
        for species in dataset.basis
    }

    reconstructed_matrices_plus = []
    reconstructed_matrices_minus = []
    # translations = np.unique(blocks[15].samples.values[:, -3:], axis = 1) 
    # print(translations)
    #if translations != dataset.realspace_translations:
    #    warnings.warn('Using more translations than the ones in the dataset.')
    
    #else: 
    #    translations = dataset.realspace_translations
    # Loop over frames
    # for A, shifts in enumerate(dataset.realspace_translations):
    for A in range(len(dataset.structures)):
        norbs = np.sum([orbs_tot[ai] for ai in dataset.structures[A].numbers])
        reconstructed_matrices_plus.append({})
        reconstructed_matrices_minus.append({})

        # reconstructed_matrices_plus.append({T: torch.zeros(norbs, norbs, device = device) for T in shifts})
        # reconstructed_matrices_minus.append({T: torch.zeros(norbs, norbs, device = device) for T in shifts})

    # loops over block types
    for key, block in blocks.items():
        block_type = key["block_type"]
        ai, ni, li = key["species_i"], key["n_i"], key["l_i"]
        aj, nj, lj = key["species_j"], key["n_j"], key["l_j"]
        T = key["cell_shift_a"], key["cell_shift_b"], key["cell_shift_c"]
        # What's the multiplicity of the orbital type, ex. 2p_x, 2p_y, 2p_z makes the multiplicity 
        # of a p block = 3
        orbs_i = orbs_mult[ai]
        orbs_j = orbs_mult[aj]
        
        # The shape of the block corresponding to the orbital pair
        shapes = {
            (k1 + k2): (orbs_i[tuple(k1)], orbs_j[tuple(k2)])
            for k1 in orbs_i
            for k2 in orbs_j
        }
        # offset of the orbital (ni, li) within a block of atom i
        ioffset = orbs_offset[(ai, ni, li)] 
        # offset of the orbital (nj,lj) within a block of atom j
        joffset = orbs_offset[(aj, nj, lj)]

        # loops over samples (structure, i, j)
        for sample, blockval in zip(block.samples, block.values):
            
            A = sample["structure"]
            i = sample["center"]
            j = sample["neighbor"]

            if T not in reconstructed_matrices_plus[A]:
                norbs = np.sum([orbs_tot[ai] for ai in dataset.structures[A].numbers])
                reconstructed_matrices_plus[A][T] = torch.zeros(norbs, norbs, device = device)
                if return_negative:
                    reconstructed_matrices_minus[A][T] = torch.zeros(norbs, norbs, device = device)

            matrix_T_plus  = reconstructed_matrices_plus[A][T]
            if return_negative:
                matrix_T_minus = reconstructed_matrices_minus[A][T]
            # beginning of the block corresponding to the atom i-j pair
            i_start, j_start = atom_blocks_idx[(A, i, j)]
            
            i_end = shapes[(ni, li, nj, lj)][0]  # orb end
            j_end = shapes[(ni, li, nj, lj)][1]  # orb end

            values = blockval[:, :, 0].clone().reshape(2 * li + 1, 2 * lj + 1)

            # position of the orbital within this block
            if block_type == 0 or block_type == 2:

                matrix_T_plus[
                    i_start + ioffset : i_start + ioffset + i_end,
                    j_start + joffset : j_start + joffset + j_end,
                             ] = values

                # Add the corresponding hermitian part
                # if (ni, li)!=(nj, lj):
                #     matrix_T_plus[j_start + joffset : j_start + joffset + j_end, 
                #                 i_start + ioffset : i_start + ioffset + i_end
                #                 ] = values.T

            if abs(block_type) == 1:
                values *= ISQRT_2
                if block_type == 1:
                    matrix_T_plus[
                        i_start + ioffset : i_start + ioffset + i_end,
                        j_start + joffset : j_start + joffset + j_end,
                    ] += values


                    # matrix_T_plus[
                    #         j_start + ioffset : j_start + ioffset + i_end,
                    #         i_start + joffset : i_start + joffset + j_end,
                    #     ] += values
                   
                    if return_negative:
                        matrix_T_minus[
                        j_start + ioffset : j_start + ioffset + i_end,
                        i_start + joffset : i_start + joffset + j_end,
                        ] += values

                    # print([Tx,Ty,Tz],i,j, slice(i_start + ioffset, i_start + ioffset + i_end), slice(j_start + joffset, j_start + joffset + j_end))
                
                    # # Add the corresponding hermitian part
                    # if (ni, li)!= (nj, lj):

                    #     # <i \phi |H| j \psi> = <j \psi |H| i \phi>^\dagger
                    #     matrix_T_plus[
                    #     j_start + joffset : j_start + joffset + j_end,
                    #     i_start + ioffset : i_start + ioffset + i_end,
                    # ] += values.T
                        
                    #     matrix_T_plus[
                    #         i_start + joffset : i_start + joffset + j_end,
                    #         j_start + ioffset : j_start + ioffset + i_end,
                    #     ] += values.T

                else:
                    
                    matrix_T_plus[
                        i_start + ioffset : i_start + ioffset + i_end,
                        j_start + joffset : j_start + joffset + j_end,
                    ] += values

                    # matrix_T_plus[
                    #         j_start + ioffset : j_start + ioffset + i_end,
                    #         i_start + joffset : i_start + joffset + j_end,
                    #     ] -= values
                    if return_negative:
                        matrix_T_minus[
                            j_start + ioffset : j_start + ioffset + i_end,
                            i_start + joffset : i_start + joffset + j_end,
                        ] -= values

                    # Add the corresponding hermitian part
                    # if (ni, li)!= (nj, lj):
                    #     matrix_T_plus[
                    #         j_start + joffset : j_start + joffset + j_end,
                    #         i_start + ioffset : i_start + ioffset + i_end,
                    #     ] += values.T

                    #     matrix_T_plus[
                    #         i_start + joffset : i_start + joffset + j_end,
                    #         j_start + ioffset : j_start + ioffset + i_end,
                    #     ] -= values.T

    # Fill what's left from symmetries [impose H(T) = H(-T)^\dagger]
    for A, matrix in enumerate(reconstructed_matrices_plus):
        Ts = list(matrix.keys())
        for T in Ts:
            mT = tuple(-t for t in T)
            if mT not in matrix:
                reconstructed_matrices_plus[A][mT] = torch.clone(matrix[T].T)
            else:
                upper = torch.triu(matrix[T])
                lower = torch.triu(reconstructed_matrices_plus[A][mT], diagonal = 1).T
                matrix[T] = upper + lower
                reconstructed_matrices_plus[A][mT] = (upper + lower).T
            assert torch.all(torch.isclose(matrix[T] - reconstructed_matrices_plus[A][mT].T, torch.zeros_like(matrix[T])))

    if return_negative:
        return reconstructed_matrices_plus, reconstructed_matrices_minus
    return reconstructed_matrices_plus

# def fourier_transform(H_k, kpts, T):
#     '''
#     Compute the Fourier Transform
#     '''    
#     return 1/np.sqrt(np.shape(kpts)[0])*np.sum([np.exp(-2j*np.pi * np.dot(ki, T)) * H_ki.cpu() for ki, H_ki in zip(kpts, H_k)], axis = 0)

def fourier_transform(H_k, T_list = None, k = None, phase = None, norm = None):
    '''
    Compute the Fourier Transform of a k-space tensor in a (list of) T lattice vectors
    '''    
    if isinstance(H_k, np.ndarray):
        H_k = torch.from_numpy(H_k)
    
    if isinstance(H_k, torch.Tensor):
        # H_T is a torch tensor
        # also k must be a tensor on the same device as T
        # k = torch.tensor(k).to(T_list.device)
        if phase is None:
            assert T_list is not None and k is not None, "T_list and k must be provided when phase is None"
            phase = torch.exp(-2j * np.pi * torch.tensordot(k, T_list, dims = ([-1], [-1])))
            if len(phase.shape) == 1:
                phase = phase.reshape(1, -1)
        if norm is None:
            norm = 1 / np.sqrt(phase.shape[0])
        return norm*torch.tensordot(H_k.to(phase), phase, dims = ([0], [0])).permute(2, 0, 1) # FIXME: the normalization is incorrect when there are less Ts than kpoints (e.g., when there is a cutoff in real space)
    else:
        raise ValueError("H_k must be np.ndarray or torch.Tensor")
    
def inverse_fourier_transform(H_T, T_list = None, k = None, phase = None, norm = None):
    '''
    Compute the Inverse Fourier Transform of a real-space tensor in a (list of) k points
    '''    
    # print( k, '<')
    # print(H_T.shape, T_list.shape)
    if isinstance(H_T, np.ndarray):
        if norm is None:
            norm = 1/np.sqrt(np.shape(T_list)[0])
        # H_T is a numpy array
        return norm*np.sum([np.exp(2j*np.pi * np.dot(k, Ti)) * H_Ti for Ti, H_Ti in zip(T_list, H_T)], axis = 0)  
    
    elif isinstance(H_T, torch.Tensor):
        # H_T is a torch tensor
        # also k must be a tensor on the same device as T
        # k = torch.tensor(k).to(T_list.device)
        if phase is None:
            assert T_list is not None and k is not None, "T_list and k must be provided when phase is None"
            phase = torch.exp(2j * np.pi * torch.tensordot(T_list, k, dims = ([-1], [-1])))
            if len(phase.shape) == 1:
                phase = phase.reshape(1, -1)
        if norm is None:
            norm = 1 / np.sqrt(phase.shape[0])
        return norm*torch.tensordot(H_T.to(phase), phase, dims = ([0], [0])).permute(2, 0, 1) # FIXME: the normalization is incorrect when there are less Ts than kpoints (e.g., when there is a cutoff in real space)
        # return 1/np.sqrt(len(T_list))*torch.sum(torch.stack([torch.exp(2j*np.pi * torch.dot(k, Ti.type(torch.float64))) * H_Ti for Ti, H_Ti in zip(T_list, H_T)]),  dim=0)
    else:
        raise ValueError("H_T must be np.ndarray or torch.Tensor")
    
def inverse_fourier_transform_OLD(H_T, T_list, k):
    '''
    Compute the Inverse Fourier Transform
    '''    
    # print( k, '<')
    # print(H_T.shape, T_list.shape)
    if isinstance(H_T, np.ndarray):
        return 1/np.sqrt(np.shape(T_list)[0])*np.sum([np.exp(2j*np.pi * np.dot(k, Ti)) * H_Ti for Ti, H_Ti in zip(T_list, H_T)], axis = 0)  
    elif isinstance(H_T, torch.Tensor):
        k = torch.tensor(k).to(T_list.device)
        # print(k, T_list)
        return 1/np.sqrt(len(T_list))*torch.sum(torch.stack([torch.exp(2j*np.pi * torch.dot(k, Ti.type(torch.float64))) * H_Ti for Ti, H_Ti in zip(T_list, H_T)]),  dim=0)
    else:
        raise ValueError("H_T must be np.ndarray or torch.Tensor")
    
def inverse_fft(H_T, kmesh):
    '''
    Compute the Inverse Fourier Transform
    '''    
    # print( k, '<')
    # print(H_T.shape, T_list.shape)
    if isinstance(H_T, np.ndarray):
        # return 1/np.sqrt(np.shape(T_list)[0])*np.sum([np.exp(2j*np.pi * np.dot(k, Ti)) * H_Ti for Ti, H_Ti in zip(T_list, H_T)], axis = 0)  
        shape = H_T.shape[-1]
        return ifftn(H_T.reshape(*kmesh, -1), axes = (0, 1, 2), norm = 'ortho').reshape(np.prod(kmesh), shape, shape)
    elif isinstance(H_T, torch.Tensor):
        # return 1/np.sqrt(len(T_list))*torch.sum(torch.stack([torch.exp(2j*np.pi * torch.dot(k, Ti.type(torch.float64))) * H_Ti for Ti, H_Ti in zip(T_list, H_T)]),  dim=0)
        shape = H_T.shape[-1]
        return torch_ifftn(H_T.reshape(*kmesh, -1), dim = (0, 1, 2), norm = 'ortho').reshape(np.prod(kmesh), shape, shape)
    else:
        raise ValueError("H_T must be np.ndarray or torch.Tensor")

def get_T_from_pair(frame, supercell, i, j, dummy_T, kmesh):
    assert np.all(np.sign(dummy_T) >= 0) or np.all(np.sign(dummy_T) <= 0), "The translation indices must either be all positive or all negative (or zero)"
    sign = np.sum(dummy_T)
    if sign != 0:
        sign = sign/np.abs(sign)
    dummy_T = np.abs(dummy_T)
    supercell = frame.repeat(kmesh)
    I = i
    J = scidx_from_unitcell(frame, j = j, T = dummy_T, kmesh = kmesh)
    d = supercell.get_distance(I, J, mic = True, vector = True) - frame.positions[j] + frame.positions[i]
    mic_T = np.int32(np.round(np.linalg.inv(frame.cell.array).T@d))
    return I, J, np.int32(sign*mic_T)

def kmatrix_to_blocks(dataset, device=None, all_pairs = True, cutoff = None, target='fock'):
    from mlelec.utils.metatensor_utils import TensorBuilder

    if device is None:
        device = dataset.device
    key_names = [
        "block_type",
        "species_i",
        "n_i",
        "l_i",
        "species_j",
        "n_j",
        "l_j"
    ]
    sample_names = ["structure", "center", "neighbor", "kpoint"]
    property_names = ["dummy"]
    property_values = np.asarray([[0]])
    component_names = [["m_i"], ["m_j"]]

    # multiplicity
    orbs_mult = {}
    for species in dataset.basis:
        _, orbidx, count = np.unique(
            np.asarray(dataset.basis[species])[:, :2],
            axis=0,
            return_counts=True,
            return_index=True,
        )
        idx = np.argsort(orbidx)
        unique_orbs = np.asarray(dataset.basis[species])[orbidx[idx]][:, :2]
        orbs_mult[species] = {tuple(k): v for k, v in zip(unique_orbs, count[idx])}

    block_builder = TensorBuilder(
        key_names,
        sample_names,
        component_names,
        property_names,
    )

    from itertools import product
    orbs_tot, _ = _orbs_offsets(dataset.basis)  # returns orbs_tot,
    if target.lower() == "fock":
        matrices = dataset.fock_kspace
    elif target.lower() == "overlap":
        matrices = dataset.overlap_kspace
    else:
        raise ValueError("target must be either 'fock' or 'overlap'")

    for A in range(len(dataset.structures)):  # Loop over frames

        frame = dataset.structures[A]
        for ik, matrixT in enumerate(matrices[A]):  # Loop over the dataset.fock_kspace


            # Not 100% this is correct: FIXME
            # When the calculation is at Gamma you want to skip i==j samples
            is_gamma_point = dataset.kmesh[A] == [1,1,1] and ik == 0

            matrixmT = matrixT.conj()
            if isinstance(matrixT, np.ndarray):
                matrixT = torch.from_numpy(matrixT).to(device)
                matrixmT = torch.from_numpy(matrixmT).to(device)
            else:
                matrixT = matrixT.to(device)
                matrixmT = matrixmT.to(device)

            i_start = 0
            for i, ai in enumerate(frame.numbers):
                orbs_i = orbs_mult[ai]
                j_start = 0

                for j, aj in enumerate(frame.numbers):

                    # Skip the pair if their distance exceeds the cutoff
                    ij_distance = frame.get_distance(i, j, mic = False)
                    if ij_distance > cutoff:
                        j_start += orbs_tot[aj]
                        continue

                    if not all_pairs: # not all orbital pairs
                        if i > j and ai == aj: # skip block type 1 if i>j 
                            j_start += orbs_tot[aj]
                            continue
                        elif ai > aj: # keep only sorted species 
                            j_start += orbs_tot[aj]
                            continue
                        
                    orbs_j = orbs_mult[aj]
                    
                    n1l1n2l2 = list(sum([tuple(k2 + k1 for k1 in orbs_i) for k2 in orbs_j], ()))

                    block_ij = matrixT[i_start:i_start + orbs_tot[ai], j_start:j_start + orbs_tot[aj]]

                    block_split = [torch.split(blocki, list(orbs_j.values()), dim = 1) for blocki in torch.split(block_ij, list(orbs_i.values()), dim=0)]
                    block_split = [y for x in block_split for y in x]  # flattening the list of lists above

                    for iorbital, (ni, li, nj, lj) in enumerate(n1l1n2l2):
                        value = block_split[iorbital]

                        same_orbitals = ni == nj and li == lj
                        
                        if (ai == aj):
                            # Same species interaction
                            block_type = 1
                            key = (block_type, ai, ni, li, aj, nj, lj)
                            block_jimT = matrixmT[j_start : j_start + orbs_tot[aj], i_start : i_start + orbs_tot[ai]]
                            block_jimT_split = [torch.split(blocki, list(orbs_i.values()), dim=1) for blocki in torch.split(block_jimT, list(orbs_j.values()), dim = 0)]
                            block_jimT_split = [y for x in block_jimT_split for y in x]  # flattening the list of lists above
                            value_ji = block_jimT_split[iorbital]  # same orbital in the ji subblock
                        else:
                            # Different species interaction
                            # skip ai>aj
                            block_type = 2
                            key = (block_type, ai, ni, li, aj, nj, lj)

                        if key not in block_builder.blocks:
                            # add blocks if not already present
                            block = block_builder.add_block(key=key, properties=property_values, components=[_components_idx(li), _components_idx(lj)])
                            if block_type == 1:
                                block = block_builder.add_block(
                                    key=(-1,) + key[1:],
                                    properties=property_values,
                                    components=[_components_idx(li), _components_idx(lj)],
                                )

                        # add samples to the blocks when present
                        block = block_builder.blocks[key]

                        # if block_type == 1:
                        #     block_asym = block_builder.blocks[(-1,) + key[1:]]

                        if block_type == 1:
                            bplus = (value + value_ji) * ISQRT_2
                            bminus = (value - value_ji) * ISQRT_2

                            block.add_samples(
                                labels=[(A, i, j, ik)],
                                data=bplus.reshape(1, 2 * li + 1, 2 * lj + 1, 1),
                            )

                            # The same-orbital and i == j element is exactly zero, so we skip it. 
                            # Skip also the Gamma point sample when the calculation is only done at Gamma 
                            # if (not (same_orbitals and i == j)):
                            if (not (is_gamma_point and i == j)):
                                block_asym = block_builder.blocks[(-1,) + key[1:]]
                                block_asym.add_samples(
                                    labels=[(A, i, j, ik)],
                                    data = bminus.reshape(1, 2 * li + 1, 2 * lj + 1, 1),
                                )

                        elif block_type == 2:
                            block.add_samples(
                                labels=[(A, i, j, ik)],
                                data=value.reshape(1, 2 * li + 1, 2 * lj + 1, 1),
                            )
                        
                        else:
                            raise ValueError("Block type not implemented")
                    j_start += orbs_tot[aj]

                i_start += orbs_tot[ai]

    tmap = block_builder.build()
    from metatensor import sort
    tmap = sort(tmap.to(arrays='numpy')).to(arrays='torch')
    return tmap

def precompute_phase(target_blocks, dataset, cutoff = np.inf):
    phase = {}
    indices = {}
    where_inv = {}
    for k, b in target_blocks.items():
        kl = tuple(k.values.tolist())
        phase[kl] = {}
        indices[kl] = {}        
        ifrij, inv = np.unique(b.samples.values[:,:3].tolist(), axis = 0, return_inverse = True)
        where_inv[kl] = inv
        for I, (ifr, i, j) in enumerate(ifrij):
            dist = dataset.structures[ifr].get_distance(i, j, mic = False)
            if dist > cutoff:
                continue
            idx = np.where(where_inv[kl] == I)[0]
            indices[kl][ifr,i,j] = idx
            kpts = torch.from_numpy(dataset.kpts_rel[ifr])
            Ts = torch.from_numpy(b.samples.values[idx, 3:6]).to(kpts)
            phase[kl][ifr,i,j] = torch.exp(2j*np.pi*torch.einsum('ka,Ta->kT', kpts, Ts))
    return phase, indices

def TMap_bloch_sums_OLD(target_blocks, phase, indices):
    from metatensor import Labels, TensorBlock, TensorMap

    _Hk = {}
    _Hk0 = {}
    for k, b in target_blocks.items():

        # LabelValues to tuple
        kl = tuple(k.values.tolist())

        # Block type
        bt = kl[0]

        # define dummy key pointing to block type 1 when block type is zero
        if bt == 0:
            _kl = (1, *kl[1:])
            if _kl not in _Hk0:
                _Hk0[_kl] = {}
        else:
            _kl = kl

        if _kl not in _Hk:
            _Hk[_kl] = {}

        # Loop through the unique (ifr, i, j) triplets
        for I, (ifr, i, j) in enumerate(phase[kl]):
            # idx = np.where(where_inv[kl] == I)[0]
            idx = indices[kl][ifr,i,j]
        
            if bt != 0:
                _Hk[_kl][ifr, i, j] = torch.einsum('Tmnv,kT->kmnv', b.values[idx].to(phase[kl][ifr, i, j]), phase[kl][ifr, i, j])
            else:
                _Hk0[_kl][ifr, i, j] = torch.einsum('Tmnv,kT->kmnv', b.values[idx].to(phase[kl][ifr, i, j]), phase[kl][ifr, i, j])*np.sqrt(2)
                
    # Now store in a tensormap
    _k_target_blocks = []
    keys = []
    properties = Labels(['dummy'], np.array([[0]]))
    for kl in _Hk:
        
        values = []
        samples = []
        
        for I in _Hk[kl]:

            # Add bt=0 contributions
            if kl in _Hk0:
                if I in _Hk0[kl]:
                    _Hk[kl][I] += _Hk0[kl][I]

            # Fill values and samples
            values.append(_Hk[kl][I])
            samples.extend([list(I) + [ik] for ik in range(_Hk[kl][I].shape[0])])
            
        values = torch.concatenate(values)
        _, n_mi, n_mj, _ = values.shape
        samples = Labels(['structure', 'center', 'neighbor', 'kpoint'], np.array(samples))
        components = [Labels(['m_i'], np.arange(-n_mi//2+1, n_mi//2+1).reshape(-1,1)), Labels(['m_j'], np.arange(-n_mj//2+1, n_mj//2+1).reshape(-1, 1))]
        
        _k_target_blocks.append(
            TensorBlock(
                samples = samples,
                components = components,
                properties = properties,
                values = values
            )
        )
        
        keys.append(list(kl))

    _k_target_blocks = TensorMap(Labels(['block_type', 'species_i', 'n_i', 'l_i', 'species_j', 'n_j', 'l_j'], np.array(keys)), _k_target_blocks)

    return _k_target_blocks

def TMap_bloch_sums(target_blocks, phase, indices):
    from metatensor import Labels, TensorBlock, TensorMap

    _Hk = {}
    _Hk0 = {}
    for k, b in target_blocks.items():

        # LabelValues to tuple
        kl = tuple(k.values.tolist())

        # Block type
        bt = kl[0]

        # define dummy key pointing to block type 1 when block type is zero
        if bt == 0:
            _kl = (1, *kl[1:])
        else:
            _kl = kl

        if _kl not in _Hk:
            _Hk[_kl] = {}

        # Loop through the unique (ifr, i, j) triplets
        for I, (ifr, i, j) in enumerate(phase[kl]):
            idx = indices[kl][ifr,i,j]
            values = b.values[idx].to(phase[kl][ifr, i, j])

            if bt != 0:
                # block type not zero: create dictionary element
                if (ifr, i, j) in _Hk[_kl]:
                    _Hk[_kl][ifr, i, j] += torch.einsum('Tmnv,kT->kmnv', values, phase[kl][ifr, i, j])
                else:
                    _Hk[_kl][ifr, i, j] = torch.einsum('Tmnv,kT->kmnv', values, phase[kl][ifr, i, j])
            else:
                # block type zero
                if (ifr, i, j) in _Hk[_kl]:
                    # if the corresponding bt = +1 element exists, sum to it the bt=0 contribution
                    _Hk[_kl][ifr, i, j] += torch.einsum('Tmnv,kT->kmnv', values, phase[kl][ifr, i, j])*np.sqrt(2) # TODO
                else:
                    # The corresponding bt = +1 element does not exist. Create the dictionary element
                    _Hk[_kl][ifr, i, j] = torch.einsum('Tmnv,kT->kmnv', values, phase[kl][ifr, i, j])*np.sqrt(2) # TODO
                    
    # Now store in a tensormap
    _k_target_blocks = []
    keys = []
    properties = Labels(['dummy'], np.array([[0]]))
    for kl in _Hk:

        same_orbitals = kl[2] == kl[5] and kl[3] == kl[6]

        values = []
        samples = []
        
        for ifr, i, j in sorted(_Hk[kl]):
            
            # skip when same orbitals, atoms, and block type == -1
            # print(kl[0], '|', kl[2], kl[3], kl[5], kl[6],'|',ifr,i,j)
            # if not (same_orbitals and (i == j) and (kl[0] == -1)):
                # if kl[0] == -1:
                #     print(kl[2], kl[5], kl[3], kl[6],ifr,i,j)
                # Fill values and samples
                values.append(_Hk[kl][ifr, i, j])
                samples.extend([[ifr, i, j] + [ik] for ik in range(_Hk[kl][ifr, i, j].shape[0])])
            
        values = torch.concatenate(values)
        _, n_mi, n_mj, _ = values.shape
        samples = Labels(['structure', 'center', 'neighbor', 'kpoint'], np.array(samples))
        components = [Labels(['m_i'], np.arange(-n_mi//2+1, n_mi//2+1).reshape(-1,1)), Labels(['m_j'], np.arange(-n_mj//2+1, n_mj//2+1).reshape(-1, 1))]
        
        _k_target_blocks.append(
            TensorBlock(
                samples = samples,
                components = components,
                properties = properties,
                values = values
            )
        )
        
        keys.append(list(kl))

    _k_target_blocks = TensorMap(Labels(['block_type', 'species_i', 'n_i', 'l_i', 'species_j', 'n_j', 'l_j'], np.array(keys)), _k_target_blocks)

    return _k_target_blocks


def kblocks_to_matrix(k_target_blocks, dataset):
    from mlelec.utils.pbc_utils import _orbs_offsets, _atom_blocks_idx
    orbs_tot, orbs_offset = _orbs_offsets(dataset.basis)
    atom_blocks_idx = _atom_blocks_idx(dataset.structures, orbs_tot)
    orbs_mult = {
        species: 
                {tuple(k): v
            for k, v in zip(
                *np.unique(
                    np.asarray(dataset.basis[species])[:, :2],
                    axis=0,
                    return_counts=True,
                )
            )
        }
        for species in dataset.basis
    }

    recon_Hk = {}
    for k, block in k_target_blocks.items():
        bt, ai, ni, li, aj, nj, lj = k.values

        #####################################################################################
        # From blocks_to_matrix
        #####################################################################################
        orbs_i = orbs_mult[ai]
        orbs_j = orbs_mult[aj]
        
        # The shape of the block corresponding to the orbital pair
        shapes = {
            (k1 + k2): (orbs_i[tuple(k1)], orbs_j[tuple(k2)])
            for k1 in orbs_i
            for k2 in orbs_j
        }
        ioffset = orbs_offset[(ai, ni, li)] 
        joffset = orbs_offset[(aj, nj, lj)]
        #####################################################################################

        for sample, blockval_ in zip(block.samples, block.values):

            blockval = blockval_.clone()

            A = sample["structure"]
            i = sample["center"]
            j = sample["neighbor"]
            ik = sample['kpoint']
           
            norbs = np.sum([orbs_tot[ai] for ai in dataset.structures[A].numbers])
            if A not in recon_Hk:
                recon_Hk[A] = torch.zeros(dataset.kpts_rel[A].shape[0], norbs, norbs, dtype = torch.complex128)
            
            i_start, j_start = atom_blocks_idx[(A, i, j)]   
            i_end, j_end = shapes[(ni, li, nj, lj)]

            
            islice = slice(i_start + ioffset, i_start + ioffset + i_end)
            jslice = slice(j_start + joffset, j_start + joffset + j_end)
            ijslice = slice(i_start + joffset, i_start + joffset + j_end)
            jislice = slice(j_start + ioffset, j_start + ioffset + i_end)

            if bt == 0 or bt == 2:
                recon_Hk[A][ik, islice, jslice] += blockval[..., 0]
                
            if abs(bt) == 1:
                blockval /= np.sqrt(2)
                if bt == 1:
                    recon_Hk[A][ik, islice, jslice] += blockval[..., 0]
                    if i != j:
                        recon_Hk[A][ik, jislice, ijslice] += blockval[..., 0].conj()
                else:
                    recon_Hk[A][ik, islice, jslice] += blockval[..., 0]
                    if i != j:
                        recon_Hk[A][ik, jislice, ijslice] -= blockval[..., 0].conj()

    recon_Hk = list(recon_Hk.values())
    return recon_Hk