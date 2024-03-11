import numpy as np
import warnings


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


def matrix_to_blocks(dataset, negative_shift_matrices, device=None, all_pairs = True, cutoff = None):
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

    from itertools import product
    orbs_tot, _ = _orbs_offsets(dataset.basis)  # returns orbs_tot,
    matrices = dataset.fock_realspace
    # for T in dataset.desired_shifts: # Loop over translations given in input

    for A in range(len(dataset.structures)):  # Loop over frames

        frame = dataset.structures[A]

        for T in dataset.fock_realspace[A]:  # Loop over the actual translations (in MIC) which label dataset.fock_realspace

            matrixT = matrices[A][T]

            # Take the H given in input and labeled by the same T
            matrixmT = negative_shift_matrices[A][T]

            if isinstance(matrixT, np.ndarray):
                matrixT = torch.from_numpy(matrixT).to(device)
                matrixmT = torch.from_numpy(matrixmT).to(device)
            else:
                matrixT = matrixT.to(device)
                matrixmT = matrixmT.to(device)
            assert np.isclose( torch.norm(matrixT - matrixmT.T).item(), 0.0), (T, torch.norm(matrixT), torch.norm(matrixmT), torch.norm(matrixT - matrixmT.T), matrixT.shape, matrixmT.shape)

            i_start = 0
            for i, ai in enumerate(frame.numbers):
                orbs_i = orbs_mult[ai]
                j_start = 0

                for j, aj in enumerate(frame.numbers):

                    skip_pair = False

                    if not all_pairs: # not all orbital pairs
                        if i > j and ai == aj: # skip block type 1 if i>j 
                            j_start += orbs_tot[aj]
                            continue
                        elif ai > aj: # keep only sorted species 
                            j_start += orbs_tot[aj]
                            continue
                       
                    # Skip the pair if their distance exceeds the cutoff
                    if cutoff is not None:
                        for mic_T in dataset._translation_dict[A][T]:
                            if dataset._translation_counter[A][mic_T][i, j]:
                                ij_distance = np.linalg.norm(frame.cell.array.T @ np.array(mic_T) + frame.positions[j] - frame.positions[i])
                                if ij_distance > cutoff:
                                    skip_pair = True
                                break
                    if skip_pair:
                        j_start += orbs_tot[aj]
                        continue
                        
                    orbs_j = orbs_mult[aj]

                    # add what kind of blocks we expect in the tensormap
                    if all_pairs:
                        n1l1n2l2 = np.concatenate([[k2 + k1 for k1 in orbs_i] for k2 in orbs_j])
                    else:
                        sorted_orbs = np.sort([(o1, o2) for o1, o2 in product(list(orbs_i.keys()), list(orbs_j.keys()))], axis=1)
                        orbs, orbital_idx = np.unique(sorted_orbs, return_index = True, axis = 0)
                        n1l1n2l2 = [tuple(o1) + tuple(o2) for o1, o2 in orbs]

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
                            # print(block_jimT.shape)
                            # print(block_jimT.shape, block_ij.shape, iorbital, i, j, ai, aj, ni, li, nj, lj, T, matrixT.shape, matrixmT.shape)
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

def blocks_to_matrix(blocks, dataset, device=None, return_negative=False):
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
            blocks = _to_uncoupled_basis(blocks)
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
        Tx, Ty, Tz = key["cell_shift_a"], key["cell_shift_b"], key["cell_shift_c"]
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

        # j_end = joffset + shapes[(ni, li, nj, lj)][1]
        # print('jend 1')

        # loops over samples (structure, i, j)
        for sample, blockval in zip(block.samples, block.values):
            
            A = sample["structure"]
            i = sample["center"]
            j = sample["neighbor"]


            matrix_T_plus  = reconstructed_matrices_plus[A][Tx, Ty, Tz]
            matrix_T_minus = reconstructed_matrices_minus[A][Tx, Ty, Tz]
            # beginning of the block corresponding to the atom i-j pair
            i_start, j_start = atom_blocks_idx[(A, i, j)]
            
            i_end = shapes[(ni, li, nj, lj)][0]  # orb end
            j_end = shapes[(ni, li, nj, lj)][1]  # orb end

            # print()
            # print((i,j),(ni, li, nj, lj), (i_start + ioffset, i_start + ioffset + i_end), (j_start + joffset, j_start + joffset + j_end))

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



    if return_negative:
        return reconstructed_matrices_plus, reconstructed_matrices_minus
    return reconstructed_matrices_plus

def fourier_transform(H_k, kpts, T):
    '''
    Compute the Fourier Transform
    '''    
    return 1/np.sqrt(np.shape(kpts)[0])*np.sum([np.exp(-2j*np.pi * np.dot(ki, T)) * H_ki for ki, H_ki in zip(kpts, H_k)], axis = 0)
    
def inverse_fourier_transform(H_T, T_list, k):
    '''
    Compute the Inverse Fourier Transform
    '''    
    return 1/np.sqrt(np.shape(T_list)[0])*np.sum([np.exp(2j*np.pi * np.dot(k, Ti)) * H_Ti for Ti, H_Ti in zip(T_list, H_T)], axis = 0)

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