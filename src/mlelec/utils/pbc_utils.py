from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import warnings
from scipy.fft import fftn, ifftn
from torch.fft import fftn as torch_fftn, ifftn as torch_ifftn
from ase.units import Bohr
import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap

# from mlelec.data.dataset import QMDataset
from mlelec.utils.metatensor_utils import TensorBuilder
from mlelec.utils.twocenter_utils import (
    _components_idx,
    ISQRT_2,
    _orbs_offsets,
    _atom_blocks_idx,
    _to_uncoupled_basis
)

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
    T_list = np.int32(np.round(T_list))
    HT = fourier_transform(matrix, T_list = Ts, k = k_list, norm = 1/k_list.shape[0])

    frame = dataset.structures[A]
    cell = frame.cell.array.T
    rji_mat = frame.get_all_distances(mic = False, vector = True)
    natm = frame.get_global_number_of_atoms()
    lengths = dataset.structures[A].repeat(dataset.kmesh[A]).cell.lengths()
    phys_cutoff = np.min(lengths/2)
    if phys_cutoff < cutoff:
        warnings.warn(f"Structure {A} is not large enough for the selected cutoff. Real space target computed with cutoff of {phys_cutoff:.2f} Angstrom")
        cutoff = phys_cutoff
    offsets = np.cumsum([len(dataset.basis[species]) for species in frame.numbers])
    offsets -= offsets[0]

    H_T = {}
    for T, H in zip(T_list, HT):
        assert torch.norm(H - H.real) < 1e-10, torch.norm(H - H.real).item()
        H = H.real
        
        CT = cell @ T
        dist_ij = np.linalg.norm(rji_mat + CT[np.newaxis, np.newaxis, :], axis = 2)
        dist = dist_ij <= cutoff
        for i in range(natm):
            i_off = offsets[i]
            i_orbs = len(dataset.basis[frame.numbers[i]])
            for j in range(natm):
                j_off = offsets[j]
                j_orbs = len(dataset.basis[frame.numbers[j]])
                if not dist[i, j]:
                    H[i_off:i_off+i_orbs, j_off:j_off+j_orbs] = 0.0

        H_T[tuple(T)] = H
    return H_T

def matrix_to_blocks(dataset, device=None, all_pairs = False, cutoff = None, target='fock', matrix=None, sort_orbs = True):
    from mlelec.utils.metatensor_utils import TensorBuilder

    if device is None:
        device = dataset.device

    key_names = ["block_type", "species_i", "n_i", "l_i", "species_j", "n_j", "l_j"]
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
        device = device
    )

    if cutoff is None:
        cutoff_was_none = True
    else:
        cutoff_was_none = False

    from itertools import product
    orbs_tot, _ = _orbs_offsets(dataset.basis)  # returns orbs_tot,

    for A, frame in enumerate(dataset.structures):  # Loop over frames
        if not dataset._ismolecule:
            if cutoff_was_none:
                cutoff = dataset.cells[A].rcut * Bohr
                warnings.warn(f'Automatic choice of the cutoff for structure {A}. rcut = {cutoff:.2f} Angstrom')
            if matrix is None:
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
            else: 
                matrices = matrix[A]
        else: 
            matrices= {}
            if target.lower() == "fock":
                matrices[0,0,0] = dataset.fock_realspace[A]
            elif target.lower() == "overlap":
                matrices[0,0,0] = dataset.overlap_realspace[A]
            else:
                raise ValueError("target must be either 'fock' or 'overlap")
        
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
            # assert np.isclose(torch.norm(matrixT - matrixmT.T).item(), 0.0), f"Failed to check H({T}) = H({mT})^\dagger"
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
                    ij_distance = np.linalg.norm(frame.cell.array.T @ np.array(T) + frame.get_distance(i,j,mic=False,vector=True))
                    if ij_distance > cutoff:
                        j_start += orbs_tot[aj]
                        continue
                        
                    orbs_j = orbs_mult[aj]

                    # add what kind of blocks we expect in the tensormap
                    # n1l1n2l2 = list(sum([tuple(k2 + k1 for k1 in orbs_i) for k2 in orbs_j], ()))
                    n1l1n2l2 = list(sum([tuple(k2 + k1 for k1 in orbs_j) for k2 in orbs_i], ()))

                    # print(i,j,slice(i_start, i_start+orbs_tot[ai]), slice(j_start, j_start+orbs_tot[aj]))
                    block_ij = matrixT[i_start:i_start + orbs_tot[ai], j_start:j_start + orbs_tot[aj]]

                    block_split = [torch.split(blocki, list(orbs_j.values()), dim = 1) for blocki in torch.split(block_ij, list(orbs_i.values()), dim = 0)]
                    block_split = [y for x in block_split for y in x]  # flattening the list of lists above

                    for iorbital, (ni, li, nj, lj) in enumerate(n1l1n2l2):
                        value = block_split[iorbital]

                        if i == j and np.linalg.norm(T) == 0:
                            if sort_orbs:
                                if ni > nj or (ni == nj and li > lj):
                                    continue
                            # On-site
                            # we could further sort n1l1,n2l2 pairs :TODO
                            block_type = 0
                            key = (block_type, ai, ni, li, aj, nj, lj)

                        elif (ai == aj) or (i == j and T != [0, 0, 0]):
                            # Same species interaction
                           #----sorting ni,li,nj,lj---  
                            if sort_orbs:
                                if ni > nj or (ni == nj and li > lj):
                                    continue
                            #-------
                            block_type = 1
                            key = (block_type, ai, ni, li, aj, nj, lj)
                            block_jimT = matrixmT[j_start : j_start + orbs_tot[aj], i_start : i_start + orbs_tot[ai]]
                            block_jimT_split = [torch.split(blocki, list(orbs_i.values()), dim=1) for blocki in torch.split(block_jimT, list(orbs_j.values()), dim = 0)]
                            block_jimT_split = [y for x in block_jimT_split for y in x]  # flattening the list of lists above
                            # value_ji \equiv H_{ji}(-T)[\phi, \psi]
                            value_ji = block_jimT_split[iorbital]  # same orbital in the ji subblock
                            
                        else:
                            # Different species interaction
                            # skip ai>aj if not all_pairs
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

                            # block_(+1)ijT = <i \phi| H(T)|j \psi> + <j \phi| H(-T)|i \psi>
                            bplus = (value + value_ji) * ISQRT_2
                            # block_(-1)ijT = <i \phi| H(T)|j \psi> - <j \phi| H(-T)|i \psi>
                            bminus = (value - value_ji) * ISQRT_2

                            block.add_samples(     labels = [(A, i, j, *T)], data =  bplus.reshape(1, 2 * li + 1, 2 * lj + 1, 1))
                            block_asym.add_samples(labels = [(A, i, j, *T)], data = bminus.reshape(1, 2 * li + 1, 2 * lj + 1, 1))

                        elif block_type == 0 or block_type == 2:
                            block.add_samples(labels = [(A, i, j, *T)], data = value.reshape(1, 2 * li + 1, 2 * lj + 1, 1))
                        
                        else:
                            raise ValueError("Block type not implemented")
                    j_start += orbs_tot[aj]

                i_start += orbs_tot[ai]
    return block_builder.build()

def move_cell_shifts_to_keys(blocks):
    """ Move cell shifts when present in samples, to keys"""

    out_blocks = []
    out_block_keys = []

    for key, block in blocks.items():        
        translations = torch.unique(block.samples.values[:, -3:], dim = 0)
        for T in translations:
            block_view = block.samples.view(["cell_shift_a", "cell_shift_b", "cell_shift_c"]).values
            idx = torch.where(torch.all(torch.isclose(block_view, torch.tensor([T[0], T[1], T[2]])), dim = 1))[0]

            if len(idx):
                out_block_keys.append(list(key.values) + [T[0], T[1], T[2]])
                out_blocks.append(TensorBlock(
                        samples = Labels(blocks.sample_names[:-3], values = block.samples.values[idx][:, :-3]),
                        values = block.values[idx],
                        components = block.components,
                        properties = block.properties,
                    ))
                
    return TensorMap(Labels(blocks.keys.names + ["cell_shift_a", "cell_shift_b", "cell_shift_c"], torch.tensor(out_block_keys)), out_blocks)

def move_orbitals_to_keys(in_blocks, dummy_property = None):

    device = in_blocks.device

    if dummy_property is None: 
        dummy_property = Labels(["dummy"], torch.tensor([[0]], device = device))
    else:
        dummy_property.to(device = device)

    blocks = []
    keys = []
    for k,b in in_blocks.items():
        n1l1n2l2 = torch.unique(b.properties.values[:,:4], dim=0)#
        block_view = b.properties.view(['n_i', 'l_i', 'n_j', 'l_j']).values
        
        for nlinlj in n1l1n2l2:
            idx = torch.where(torch.all(torch.isclose(block_view, nlinlj), dim = 1))[0]
            
            keys.append(torch.hstack((k.values, nlinlj.clone().detach())))
            if len(idx):
                blocks.append(TensorBlock(
                            samples = b.samples,
                            values = b.values[...,idx],
                            components = b.components,
                            properties = dummy_property
                        )
                )
    keys = Labels(in_blocks.keys.names+['n_i', 'l_i', 'n_j', 'l_j'], torch.stack(keys).to(device = device))
    # keys = block_type, species_i, species_j, L, sigma, n_i, l_i, n_j, l_j
    tmap = TensorMap(keys, blocks)
    return mts.permute_dimensions(tmap, axis='keys', dimensions_indexes = [0,1,5,6,2,7,8,3,4])

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

def kmatrix_to_blocks(dataset, device=None, all_pairs = False, cutoff = None, target='fock', sort_orbs=True, matrix=None):
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
    if matrix is not None: 
        matrices = matrix 
    else:
        if target.lower() == "fock":
            matrices = dataset.fock_kspace
        elif target.lower() == "overlap":
            matrices = dataset.overlap_kspace
        else:
            raise ValueError("target must be either 'fock' or 'overlap'")

    for A in range(len(dataset.structures)):  # Loop over frames

        frame = dataset.structures[A]
        for ik, matrixT in enumerate(matrices[A]):  # Loop over the dataset.fock_kspace

            # When the calculation is at Gamma you want to skip i==j samples
            # is_gamma_point = np.linalg.norm(dataset.kpts_rel[A][ik]) < 1e-30

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

                    same_species = ai == aj
                    same_atom_in_unit_cell = i == j

                    # Skip the pair if their MIC distance exceeds the cutoff.
                    # MIC must be true because any contribution from unit cell translations contributes to H(k)
                    # If MIC=False, you miss the contributions of i,j pairs which are far away in the unit cell but close in another nonzero translation
                    ij_distance = frame.get_distance(i, j, mic = True) 
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
                    
                    n1l1n2l2 = list(sum([tuple(k2 + k1 for k1 in orbs_j) for k2 in orbs_i], ()))
                    block_ij = matrixT[i_start:i_start + orbs_tot[ai], j_start:j_start + orbs_tot[aj]]

                    block_split = [torch.split(blocki, list(orbs_j.values()), dim = 1) for blocki in torch.split(block_ij, list(orbs_i.values()), dim=0)]
                    block_split = [y for x in block_split for y in x]  # flattening the list of lists above

                    for iorbital, (ni, li, nj, lj) in enumerate(n1l1n2l2):
                        if sort_orbs:
                            if same_species and (ni > nj or (ni == nj and li > lj)):
                                continue
                        value = block_split[iorbital]                        
                        # if i == 0 and j == 1 and ik == 0 and ni == 1 and li == 0 and nj == 2 and lj == 0:
                        #     print(value)
                        if same_species:
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
                            bplus = (value + value_ji) * ISQRT_2
                            bminus = (value - value_ji) * ISQRT_2

                            block.add_samples(
                                labels=[(A, i, j, ik)],
                                data=bplus.reshape(1, 2 * li + 1, 2 * lj + 1, 1),
                            )

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
    #tmap = sort(tmap.to(arrays='numpy')).to(arrays='torch')
    return tmap


def kblocks_to_matrix(k_target_blocks, dataset, all_pairs = False, sort_orbs = True, detach = False,):
    """
    k_target_blocks: UNCOUPLED blocks of H(k)
   
    """
    from mlelec.utils.pbc_utils import _orbs_offsets, _atom_blocks_idx
    orbs_tot, orbs_offset = _orbs_offsets(dataset.basis)
    atom_blocks_idx = _atom_blocks_idx(dataset.structures, orbs_tot)
    if "L" in k_target_blocks.keys.names:
        from mlelec.utils.twocenter_utils import _to_uncoupled_basis
        k_target_blocks = _to_uncoupled_basis(k_target_blocks)
    if "l_i" not in k_target_blocks.keys.names:
        k_target_blocks = move_orbitals_to_keys(k_target_blocks)
    
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
    bt1factor = ISQRT_2 
    bt2factor = 1

    if all_pairs:
        bt1factor /= 2
        bt2factor *= 2 # because we add both <I \phi | J \psi> and <I \psi | J \phi> 

    recon_Hk = {}
    for k, block in k_target_blocks.items():
        # bt, ai, ni, li, aj, nj, lj = k.values
        bt = k["block_type"]
        ai = k["species_i"]
        ni = k["n_i"]
        li = k["l_i"]
        aj = k["species_j"]
        nj = k["n_j"]
        lj = k["l_j"]

        different_orbitals = not (ni == nj and li == lj)
        orbs_i = orbs_mult[ai]
        orbs_j = orbs_mult[aj]
        
        # The shape of the block corresponding to the orbital pair
        shapes = {
            (k1 + k2): (orbs_i[tuple(k1)], orbs_j[tuple(k2)])
            for k1 in orbs_i
            for k2 in orbs_j
        }
        phioffset = orbs_offset[(ai, ni, li)] 
        psioffset = orbs_offset[(aj, nj, lj)]

        if detach:
            blockval_ = block.values[..., 0].clone().detach()
        else:
            blockval_ = block.values[..., 0].clone()

        for sample, blockval in zip(block.samples, blockval_):

            # blockval = blockval_[...,0].clone()

            A = sample["structure"]
            i = sample["center"]
            j = sample["neighbor"]
            ik = sample['kpoint']

            same_species = ai==aj
            same_atom = i == j
            bt0factor = 1

            norbs = np.sum([orbs_tot[ai] for ai in dataset.structures[A].numbers])
            if A not in recon_Hk:
                recon_Hk[A] = torch.zeros(dataset.kpts_rel[A].shape[0], norbs, norbs, dtype = torch.complex128)
            
            i_start, j_start = atom_blocks_idx[(A, i, j)]   
            phi_end, psi_end = shapes[(ni, li, nj, lj)]
            
            iphi_jpsi = slice(i_start + phioffset, i_start + phioffset + phi_end),\
                        slice(j_start + psioffset, j_start + psioffset + psi_end)
            ipsi_jphi = slice(i_start + psioffset, i_start + psioffset + psi_end),\
                        slice(j_start + phioffset, j_start + phioffset + phi_end)
            jpsi_iphi = slice(j_start + psioffset, j_start + psioffset + psi_end),\
                        slice(i_start + phioffset, i_start + phioffset + phi_end)
            jphi_ipsi = slice(j_start + phioffset, j_start + phioffset + phi_end),\
                        slice(i_start + psioffset, i_start + psioffset + psi_end)
                                    
            if abs(bt) == 1:
                if not all_pairs: 
                    if not sort_orbs:
                        bt0factor = 0.25
                        if same_species and not same_atom:
                            bt0factor = 0.5
                            
                    else: 
                        bt0factor = 0.5
                        if same_atom and not different_orbitals:
                            bt0factor = 0.25
                        if same_species and different_orbitals and not same_atom:
                            bt0factor = 1
                else: 
                    if not sort_orbs:
                        bt0factor = 0.5
                    else: 
                        if same_species and not different_orbitals:
                            bt0factor = 0.5
                #----sorting ni,li,nj,lj---
                # if not all_pairs: 
                #     if same_atom and same_species: 
                #         bt0factor = 0.5
                #----sorting ni,li,nj,lj---

                blockval = blockval * bt1factor*bt0factor
            
                if bt == 1:
                    recon_Hk[A][ik][iphi_jpsi] += blockval
                    recon_Hk[A][ik][jphi_ipsi] += blockval.conj()
                    # if not same_species or (sort_orbs and different_orbitals):
                    # if sort_orbs and different_orbitals:
                    recon_Hk[A][ik][ipsi_jphi] += blockval.T
                    recon_Hk[A][ik][jpsi_iphi] += blockval.conj().T
                else:
                    recon_Hk[A][ik][iphi_jpsi] += blockval
                    recon_Hk[A][ik][jphi_ipsi] -= blockval.conj()
                    # if not same_species or (sort_orbs and different_orbitals):
                    # if sort_orbs and different_orbitals:
                    recon_Hk[A][ik][ipsi_jphi] -= blockval.T
                    recon_Hk[A][ik][jpsi_iphi] += blockval.conj().T

            elif bt == 2:
                recon_Hk[A][ik][iphi_jpsi] += blockval/bt2factor
                recon_Hk[A][ik][jpsi_iphi] += blockval.conj().T/bt2factor

            else:
                raise ValueError(f"bt = {bt} should not be present in kblocks_to_matrix.")
    
    for Hk in recon_Hk: 
        for ik in range(len(recon_Hk[Hk])):
            assert torch.norm(recon_Hk[Hk][ik] -  recon_Hk[Hk][ik].conj().T) < 1e-10, "Hk is not hermitian"
    recon_Hk = list(recon_Hk.values())
    return recon_Hk

def tmap_to_dict(tmap):
    temp={}
    for k,b in tmap.items():
        kl = tuple(k.values.tolist())
        temp[kl] = {}
        bsamp = np.array(b.samples.values.tolist())
        values = b.values.clone()
        ifrij = np.unique(bsamp[:,:3], axis = 0)
        for I in ifrij:
            idx = np.where(np.all(bsamp[:,:3] == I, axis = 1))[0]
            temp[kl][tuple(I.tolist())] = values[idx]
    return temp

#------------------------------------------------------
#SHOULD WE BE DOING THIS WITH TORCH?  FIXME 
def unique_Aij_block(block):
    Aij, inv = np.unique(block.samples.values[:, :3].tolist(), axis = 0, return_inverse = True)
    return Aij, inv

def unique_Aij(tensor):
    Aijs = []
    invs = []
    for b in tensor.blocks():
        Aij, inv = unique_Aij_block(b)
        Aijs.append(Aij)
        invs.append(inv)
    return Aijs, invs

def precompute_phase(target_blocks, dataset, cutoff = np.inf):
    phase = {}
    indices = {}
    where_inv = {}
    kpts_idx = []
    for k, b in target_blocks.items():
        kl = tuple(k.values.tolist())
        # bt_is_minus_1 = kl[0] == -1

        phase[kl] = {}
        indices[kl] = {}
        
        ifrij, where_inv[kl] = unique_Aij_block(b) #np.unique(b.samples.values[:,:3].tolist(), axis = 0, return_inverse = True)

        # if bt_is_minus_1:
            # where_k_is_not_Gamma = [np.where(np.linalg.norm(dataset.kpts_rel[ifr], axis = 1) > 1e-30)[0] for ifr in np.unique(b.samples.values[:,0])]
            # kpts_idx.append(where_k_is_not_Gamma)
        
        for I, (ifr, i, j) in enumerate(ifrij):
            # dist = dataset.structures[ifr].get_distance(i, j, mic = False)
            # if dist > cutoff:
            #     continue
            idx = np.where(where_inv[kl] == I)[0]
            indices[kl][ifr,i,j] = idx

            kpts = torch.from_numpy(dataset.kpts_rel[ifr])
            # if bt_is_minus_1 and i == j:
                # kpts = kpts[kpts_idx[-1][ifr]]

            Ts = b.samples.values[idx, 3:6].to(kpts)
            # Ts = torch.from_numpy(b.samples.values[idx, 3:6]).to(kpts)
            phase[kl][ifr,i,j] = torch.exp(2j*np.pi*torch.einsum('ka,Ta->kT', kpts, Ts))
    return phase, indices, kpts_idx


dummy_prop = Labels(['dummy'], torch.tensor([[0]]))
def TMap_bloch_sums(target_blocks, phase, indices=None, kpts_idx=None, return_tensormap = True, use_dummy_prop = True):

    is_coupled = False
    if 'L' in target_blocks.keys.names:
        is_coupled = True
    blockproperty = {}
    _Hk = {}
    for k, b in target_blocks.items():
        # LabelValues to tuple
        kl = tuple(k.values.tolist())

        # Block type
        bt = kl[0]
        
        # define dummy key pointing to block type 1 when block type is zero
        factor = 1
        if bt == 0:
            _kl = (1, *kl[1:])
            factor = np.sqrt(2)
        else:
            _kl = kl
        
        blockproperty[_kl] = b.properties 
        if _kl not in _Hk:
            _Hk[_kl] = {}
            
        # Loop through the unique (ifr, i, j) triplets
        b_values = b.values.to(next(iter(next(iter(phase.values())).values())))

        # If batching, we can't loop over phase keys anymore
        ifrij, where_inv = unique_Aij_block(b)

        for I, (ifr, i, j) in enumerate(ifrij): #enumerate(phase[kl]):
            if (ifr, i, j) not in phase[kl]:
                continue

            idx = np.where(where_inv == I)[0]
            # indices[kl][ifr,i,j] = idx
            # idx = indices[kl][ifr,i,j]
            values = b_values[idx]
            vshape = values.shape
            pshape = phase[kl][ifr, i, j].shape

            # equivalent to torch.einsum('Tmnv,kT->kmnv', values.to(phase[kl][ifr, i, j]), phase[kl][ifr, i, j]), but faster
            contraction = (phase[kl][ifr, i, j]@values.reshape(vshape[0], -1)).reshape(pshape[0], *vshape[1:])*factor

            # if bt == 1 or bt == 2 or (bt == -1 and i != j):

            # if bt != -1 or (bt == -1 and i != j): 

            if (ifr, i, j) in _Hk[_kl]:
                _Hk[_kl][ifr, i, j] += contraction
            else:
                _Hk[_kl][ifr, i, j] = contraction

            # elif bt == 0:
            #     # block type zero
            #     if (ifr, i, j) in _Hk[_kl]:
            #         # if the corresponding bt = +1 element exists, sum to it the bt=0 contribution
            #         _Hk[_kl][ifr, i, j] += contraction*np.sqrt(2)
            #     else:
            #         # The corresponding bt = +1 element does not exist. Create the dictionary element
            #         _Hk[_kl][ifr, i, j] = contraction*np.sqrt(2)
                    
    if return_tensormap:
        # Now store in a tensormap
        _k_target_blocks = []
        keys = []
        count = 0
        for kl in _Hk:

            # same_orbitals = kl[2] == kl[5] and kl[3] == kl[6]
            bt_is_minus_1 = kl[0] == -1

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

                    # if i == j and bt_is_minus_1:
                    #     samples.extend([[ifr, i, j] + [ik] for ik in kpts_idx[count][ifr]])
                    # else:
                    #     samples.extend([[ifr, i, j] + [ik] for ik in range(_Hk[kl][ifr, i, j].shape[0])])
                    samples.extend([[ifr, i, j] + [ik] for ik in range(_Hk[kl][ifr, i, j].shape[0])])
            
            # if bt_is_minus_1:
            #     count += 1

            if values != []:
                samples = Labels(['structure', 'center', 'neighbor', 'kpoint'], torch.tensor(samples))                
                values = torch.concatenate(values)
                
                if is_coupled:
                    n_M = values.shape[1]
                    components = [Labels(['M'], torch.arange(-n_M//2+1, n_M//2+1).reshape(-1,1))]
                else:
                    n_mi, n_mj = values.shape[1:3]
                    components = [Labels(['m_i'], torch.arange(-n_mi//2+1, n_mi//2+1).reshape(-1,1)), 
                                  Labels(['m_j'], torch.arange(-n_mj//2+1, n_mj//2+1).reshape(-1, 1))]
                
                _k_target_blocks.append(
                    TensorBlock(
                        samples = samples,
                        components = components,
                        properties = blockproperty[kl],
                        values = values
                    )
                )
            
                keys.append(list(kl))

        _k_target_blocks = TensorMap(Labels(target_blocks.keys.names, torch.tensor(keys)), _k_target_blocks)
            # ['block_type', 'species_i', 'n_i', 'l_i', 'species_j', 'n_j', 'l_j']        
        _Hk = None
        del _Hk

        return _k_target_blocks
    else:
        Hk = {}
        for k in _Hk:
            Hk[k] = {}
            for ifr, i, j in sorted(_Hk[k]):
                Hk[k][ifr, i, j] = _Hk[k][ifr, i, j]
        return Hk
#------------------------------------------------------

def TMap_bloch_sums_feat(target_blocks, phase, indices=None, kpts_idx=None, return_tensormap = False):
    # from mlelec.utils.pbc_utils import unique_Aij_block
    is_coupled = True
    _Hk = {}
    props = {}
    for k, b in target_blocks.items():
        # LabelValues to tuple
        
        kl = tuple(k.values.tolist())

        # Block type
        bt = kl[-1]
        
        # define dummy key pointing to block type 1 when block type is zero
        factor = 1
        if bt == 0:
            _kl = (*kl[:-1], 1)
            factor = np.sqrt(2)
        else:
            _kl = kl

        if _kl not in _Hk:
            _Hk[_kl] = {}
            props[_kl] = b.properties
        # Loop through the unique (ifr, i, j) triplets
        b_values = b.values.to(next(iter(next(iter(phase.values())).values())))

        # If batching, we can't loop over phase keys anymore
        ifrij, where_inv = unique_Aij_block(b)

        for I, (ifr, i, j) in enumerate(ifrij): #enumerate(phase[kl]):
            if (ifr, i, j) not in phase[kl]:
                continue

            idx = np.where(where_inv == I)[0]
            values = b_values[idx]
            vshape = values.shape
            pshape = phase[kl][ifr, i, j].shape

            # equivalent to torch.einsum('Tmnv,kT->kmnv', values.to(phase[kl][ifr, i, j]), phase[kl][ifr, i, j]), but faster
            contraction = (phase[kl][ifr, i, j]@values.reshape(vshape[0], -1)).reshape(pshape[0], *vshape[1:])*factor

            if bt != -1 or (bt == -1 and i != j): 

                if (ifr, i, j) in _Hk[_kl]:
                    _Hk[_kl][ifr, i, j] += contraction
                else:
                    _Hk[_kl][ifr, i, j] = contraction
    if return_tensormap:
        # Now store in a tensormap
        _k_target_blocks = []
        keys = []
        count = 0
        for kl in _Hk:

            # same_orbitals = kl[2] == kl[5] and kl[3] == kl[6]
            bt_is_minus_1 = kl[0] == -1

            values = []
            samples = []
           
            for ifr, i, j in sorted(_Hk[kl]):
                    values.append(_Hk[kl][ifr, i, j])
                    samples.extend([[ifr, i, j] + [ik] for ik in range(_Hk[kl][ifr, i, j].shape[0])])
            
            # if bt_is_minus_1:
            #     count += 1

            if values != []:
                samples = Labels(['structure', 'center', 'neighbor', 'kpoint'], torch.tensor(samples))                
                values = torch.concatenate(values)
                
                if is_coupled:
                    n_M = values.shape[1]
                    components = [Labels(['spherical_harmonics_m'], torch.arange(-n_M//2+1, n_M//2+1).reshape(-1,1))]
               
                
                _k_target_blocks.append(
                    TensorBlock(
                        samples = samples,
                        components = components,
                        properties = props[kl],
                        values = values
                    )
                )
            
                keys.append(list(kl))

        _k_target_blocks = TensorMap(Labels(target_blocks.keys.names, torch.tensor(keys)), _k_target_blocks)
            # ['block_type', 'species_i', 'n_i', 'l_i', 'species_j', 'n_j', 'l_j']        
        _Hk = None
        del _Hk

        return _k_target_blocks
    else:
        Hk = {}
        for k in _Hk:
            Hk[k] = {}
            for ifr, i, j in sorted(_Hk[k]):
                Hk[k][ifr, i, j] = _Hk[k][ifr, i, j]
        return Hk

######--------------- NEW/OLD - to discard or incorporate? --------------------------- ###########################

def blocks_to_matrix_working(blocks, dataset, device=None, cg = None, all_pairs = False, sort_orbs = True, detach = False, sample_id = None):

    if device is None:
        device = dataset.device
        
    if "L" in blocks.keys.names:
        from mlelec.utils.twocenter_utils import _to_uncoupled_basis
        blocks = _to_uncoupled_basis(blocks, cg = cg, device = device)

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

    reconstructed_matrices = []
    
    bt1factor = ISQRT_2
    if all_pairs:
        bt1factor /= 2

    for A in range(len(dataset.structures)):
        norbs = np.sum([orbs_tot[ai] for ai in dataset.structures[A].numbers])
        reconstructed_matrices.append({})

    # loops over block types
    for key, block in blocks.items():
        block_type = key["block_type"]
        ai, ni, li = key["species_i"], key["n_i"], key["l_i"]
        aj, nj, lj = key["species_j"], key["n_j"], key["l_j"]
        
        #----sorting ni,li,nj,lj---
        if sort_orbs:
            fac=1 # sorted orbs - we only count everything once
            if ai == aj and (ni == nj and li == lj): #except these diag blocks
                fac=2 #so we need to divide by 2 to avoic double count
        else: 
            # no sorting -->  we count everything twice
            fac=2
        #----sorting ni,li,nj,lj---
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
        # where does orbital PHI = (ni, li) start within a block of atom i
        phioffset = orbs_offset[(ai, ni, li)] 
        # where does orbital PSI = (nj,lj) start within a block of atom j
        psioffset = orbs_offset[(aj, nj, lj)]

        # loops over samples (structure, i, j)
    
        for sample, blockval in zip(block.samples.values, block.values):

            if blockval.numel() == 0:
                # Empty block
                continue        

            A, i, j, Tx, Ty, Tz = sample.tolist()
            T = Tx, Ty, Tz
            mT = tuple(-t for t in T)

            other_fac = 1
            if i == j and T != (0,0,0) and not all_pairs:
                other_fac = 0.5

            # bt 0
            if not sort_orbs:
                bt0_factor_p = 0.5
            else: 
                if not(ni==nj and li==lj):
                    bt0_factor_p = 1
                else:
                    bt0_factor_p = 0.5
            bt0_factor_m = bt0_factor_p*other_fac

            # bt 2 
            bt2_factor_p=0.5
            if not all_pairs:
                bt2_factor_p=1
            bt2_factor_m = bt2_factor_p*other_fac

            # bt 1
            bt1_fact_fin = bt1factor/fac*other_fac

            
            if T not in reconstructed_matrices[A]:
                assert mT not in reconstructed_matrices[A], "why is mT present but not T?"
                norbs = np.sum([orbs_tot[ai] for ai in dataset.structures[A].numbers])
                reconstructed_matrices[A][T] = torch.zeros(norbs, norbs, device = device)
                reconstructed_matrices[A][mT] = torch.zeros(norbs, norbs, device = device)

            matrix_T  = reconstructed_matrices[A][T]
            matrix_mT = reconstructed_matrices[A][mT]
            # beginning of the block corresponding to the atom i-j pair
            i_start, j_start = atom_blocks_idx[(A, i, j)]
            # where does orbital (ni, li) end (or how large is it)
            phi_end = shapes[(ni, li, nj, lj)][0]  # orb end
            # where does orbital (nj, lj) end (or how large is it)
            psi_end = shapes[(ni, li, nj, lj)][1]  

            iphi_jpsi_slice = slice(i_start + phioffset , i_start + phioffset + phi_end),\
                              slice(j_start + psioffset , j_start + psioffset + psi_end)
            ipsi_jphi_slice = slice(i_start + psioffset , i_start + psioffset + psi_end),\
                              slice(j_start + phioffset , j_start + phioffset + phi_end),
                            
            jphi_ipsi_slice = slice(j_start + phioffset , j_start + phioffset + phi_end),\
                              slice(i_start + psioffset , i_start + psioffset + psi_end)
            
            jpsi_iphi_slice = slice(j_start + psioffset , j_start + psioffset + psi_end),\
                              slice(i_start + phioffset , i_start + phioffset + phi_end)

            if detach:
                bv = blockval[:, :, 0].detach().clone()
            else:
                bv = blockval[:, :, 0]
            # position of the orbital within this block
            if block_type == 0:
                # <i \phi| H(T)|j \psi> = # <i \phi| H(-T)|j \psi>^T 
                # if not sort_orbs:
                #     ff = 0.5
                # else: 
                #     # ff = 0.5
                #     if not(ni==nj and li==lj):
                #         ff = 1
                #     else:
                #         ff = 0.5

                matrix_T[iphi_jpsi_slice] += bv*bt0_factor_p
                matrix_mT[jpsi_iphi_slice] += bv.T*bt0_factor_m
                
            elif block_type == 2:
                
                # ff=0.5
                # if not all_pairs:
                #     ff=1
                
                matrix_T[iphi_jpsi_slice] += bv*bt2_factor_p
                matrix_mT[jpsi_iphi_slice] += bv.T*bt2_factor_m
                
            elif abs(block_type) == 1:
                # Eq (1) <i \phi| H(T)|j \psi> = # block_(+1)ijT + block_(-1)ijT 
                # Eq (2) <j \phi| H(-T)|i \psi> = # block_(+1)ijT - block_(-1)ijT 
                # Eq (3) <j \psi| H(-T)|i \phi> = # block_(+1)ijT^\dagger + block_(-1)ijT^\dagger (Transpose of Eq1) 
                # Eq (4) <i \psi| H(T)|j \phi> = # block_(+1)ijT^\dagger - block_(-1)ijT^\dagger (Transpose of Eq2)
                bv = bv*bt1_fact_fin

                if block_type == 1:
                    # first half of Eq (1) 
                    matrix_T[iphi_jpsi_slice] += bv
                    # first half of Eq (2)
                    matrix_mT[jphi_ipsi_slice] += bv
                    # first half of Eq (3)
                    matrix_mT[jpsi_iphi_slice] += bv.T
                    # first half of Eq (4)
                    matrix_T[ ipsi_jphi_slice] += bv.T
        
                else:
                    # second half of Eq (1)
                    matrix_T[iphi_jpsi_slice] += bv
                    # second half of Eq (2)
                    matrix_mT[jphi_ipsi_slice] -= bv
                    # second half of Eq (3)
                    matrix_mT[jpsi_iphi_slice] += bv.T
                    # second half of Eq (4)
                    matrix_T[ipsi_jphi_slice ] -= bv.T
         
    for A, matrix in enumerate(reconstructed_matrices):
        Ts = list(matrix.keys())
        for T in Ts:
            mT = tuple(-t for t in T)
         
            assert torch.all(torch.isclose(matrix[T] - reconstructed_matrices[A][mT].T, torch.zeros_like(matrix[T]))), torch.norm(matrix[T] - reconstructed_matrices[A][mT].T).item()

    return reconstructed_matrices

def blocks_to_matrix_try(blocks, dataset, device=None, cg = None, all_pairs = False, sort_orbs = True, detach = False, sample_id = None):

    if device is None:
        device = dataset.device
        
    if "L" in blocks.keys.names:
        from mlelec.utils.twocenter_utils import _to_uncoupled_basis
        blocks = _to_uncoupled_basis(blocks, cg = cg, device = device)

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

    reconstructed_matrices = []
    
    # bt1
    bt1factor = ISQRT_2
    if all_pairs:
        bt1factor /= 2

    # bt 2 
    bt2_factor_p=0.5
    if not all_pairs:
        bt2_factor_p=1

    for A in range(len(dataset.structures)):
        norbs = np.sum([orbs_tot[ai] for ai in dataset.structures[A].numbers])
        reconstructed_matrices.append({})

    # loops over block types
    for key, block in blocks.items():
        block_type = key["block_type"]
        ai, ni, li = key["species_i"], key["n_i"], key["l_i"]
        aj, nj, lj = key["species_j"], key["n_j"], key["l_j"]
        
        #----sorting ni,li,nj,lj---
        if sort_orbs:
            fac=1 # sorted orbs - we only count everything once
            if ai == aj and (ni == nj and li == lj): #except these diag blocks
                fac=2 #so we need to divide by 2 to avoic double count
        else: 
            # no sorting -->  we count everything twice
            fac=2
        #----sorting ni,li,nj,lj---
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
        # where does orbital PHI = (ni, li) start within a block of atom i
        phioffset = orbs_offset[(ai, ni, li)] 
        # where does orbital PSI = (nj,lj) start within a block of atom j
        psioffset = orbs_offset[(aj, nj, lj)]

        # loops over samples (structure, i, j)

        samples = block.samples.values.tolist()
        
        blockvalues = block.values
        if detach:
            blockvalues = blockvalues.detach() #.clone()

        for sample, blockval in zip(samples, blockvalues[:,:,:,0]):
    
        # for sample, blockval in zip(block.samples.values, block.values):

            if blockval.numel() == 0:
                # Empty block
                continue        

            A, i, j, Tx, Ty, Tz = sample #.tolist()
            T = Tx, Ty, Tz
            mT = -Tx, -Ty, -Tz

            other_fac = 1
            if i == j and T != (0,0,0) and not all_pairs:
                other_fac = 0.5

            # bt 0
            if not sort_orbs:
                bt0_factor_p = 0.5
            else: 
                if not(ni==nj and li==lj):
                    bt0_factor_p = 1
                else:
                    bt0_factor_p = 0.5
            bt0_factor_m = bt0_factor_p*other_fac

            # bt 2 
            # bt2_factor_p=0.5
            # if not all_pairs:
            #     bt2_factor_p=1
            bt2_factor_m = bt2_factor_p*other_fac

            # bt 1
            bt1_fact_fin = bt1factor/fac*other_fac

            
            if T not in reconstructed_matrices[A]:
                assert mT not in reconstructed_matrices[A], "why is mT present but not T?"
                norbs = np.sum([orbs_tot[ai] for ai in dataset.structures[A].numbers])
                reconstructed_matrices[A][T] = torch.zeros(norbs, norbs, device = device)
                reconstructed_matrices[A][mT] = torch.zeros(norbs, norbs, device = device)

            matrix_T  = reconstructed_matrices[A][T]
            matrix_mT = reconstructed_matrices[A][mT]
            # beginning of the block corresponding to the atom i-j pair
            i_start, j_start = atom_blocks_idx[(A, i, j)]
            # where does orbital (ni, li) end (or how large is it)
            phi_end = shapes[(ni, li, nj, lj)][0]  # orb end
            # where does orbital (nj, lj) end (or how large is it)
            psi_end = shapes[(ni, li, nj, lj)][1]  

            iphi_jpsi_slice = slice(i_start + phioffset , i_start + phioffset + phi_end),\
                              slice(j_start + psioffset , j_start + psioffset + psi_end)
            ipsi_jphi_slice = slice(i_start + psioffset , i_start + psioffset + psi_end),\
                              slice(j_start + phioffset , j_start + phioffset + phi_end),
                            
            jphi_ipsi_slice = slice(j_start + phioffset , j_start + phioffset + phi_end),\
                              slice(i_start + psioffset , i_start + psioffset + psi_end)
            
            jpsi_iphi_slice = slice(j_start + psioffset , j_start + psioffset + psi_end),\
                              slice(i_start + phioffset , i_start + phioffset + phi_end)

            # if detach:
            #     bv = blockval[:, :, 0].detach().clone()
            # else:
                # bv = blockval[:, :, 0]
            bv = blockval #[:, :, 0]
            # position of the orbital within this block
            if block_type == 0:
                # <i \phi| H(T)|j \psi> = # <i \phi| H(-T)|j \psi>^T 
                # if not sort_orbs:
                #     ff = 0.5
                # else: 
                #     # ff = 0.5
                #     if not(ni==nj and li==lj):
                #         ff = 1
                #     else:
                #         ff = 0.5

                matrix_T[iphi_jpsi_slice] += bv*bt0_factor_p
                matrix_mT[jpsi_iphi_slice] += bv.T*bt0_factor_m
                
            elif block_type == 2:
                
                # ff=0.5
                # if not all_pairs:
                #     ff=1
                
                matrix_T[iphi_jpsi_slice] += bv*bt2_factor_p
                matrix_mT[jpsi_iphi_slice] += bv.T*bt2_factor_m
                
            elif abs(block_type) == 1:
                # Eq (1) <i \phi| H(T)|j \psi> = # block_(+1)ijT + block_(-1)ijT 
                # Eq (2) <j \phi| H(-T)|i \psi> = # block_(+1)ijT - block_(-1)ijT 
                # Eq (3) <j \psi| H(-T)|i \phi> = # block_(+1)ijT^\dagger + block_(-1)ijT^\dagger (Transpose of Eq1) 
                # Eq (4) <i \psi| H(T)|j \phi> = # block_(+1)ijT^\dagger - block_(-1)ijT^\dagger (Transpose of Eq2)
                bv = bv*bt1_fact_fin

                if block_type == 1:
                    # first half of Eq (1) 
                    matrix_T[iphi_jpsi_slice] += bv
                    # first half of Eq (2)
                    matrix_mT[jphi_ipsi_slice] += bv
                    # first half of Eq (3)
                    matrix_mT[jpsi_iphi_slice] += bv.T
                    # first half of Eq (4)
                    matrix_T[ ipsi_jphi_slice] += bv.T
        
                else:
                    # second half of Eq (1)
                    matrix_T[iphi_jpsi_slice] += bv
                    # second half of Eq (2)
                    matrix_mT[jphi_ipsi_slice] -= bv
                    # second half of Eq (3)
                    matrix_mT[jpsi_iphi_slice] += bv.T
                    # second half of Eq (4)
                    matrix_T[ipsi_jphi_slice ] -= bv.T
         
    for A, matrix in enumerate(reconstructed_matrices):
        Ts = list(matrix.keys())
        for T in Ts:
            mT = tuple(-t for t in T)
         
            assert torch.all(torch.isclose(matrix[T] - reconstructed_matrices[A][mT].T, torch.zeros_like(matrix[T]))), torch.norm(matrix[T] - reconstructed_matrices[A][mT].T).item()

    return reconstructed_matrices

def blocks_to_matrix_OLD(blocks, dataset, device=None, cg = None, all_pairs = False, sort_orbs = True, detach = False, sample_id = None):

    if device is None:
        device = dataset.device
        
    if "L" in blocks.keys.names:
        from mlelec.utils.twocenter_utils import _to_uncoupled_basis
        blocks = _to_uncoupled_basis(blocks, cg = cg, device = device)

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

    reconstructed_matrices = []
    
    # bt 1
    bt1factor = ISQRT_2
    if all_pairs:
        bt1factor /= 2

    # bt 2 
    bt2_factor_p=0.5
    if not all_pairs:
        bt2_factor_p=1

    for A in range(len(dataset.structures)):
        norbs = np.sum([orbs_tot[ai] for ai in dataset.structures[A].numbers])
        reconstructed_matrices.append({})

    # loops over block types
    for key, block in blocks.items():

        block_type = key["block_type"]
        ai, ni, li = key["species_i"], key["n_i"], key["l_i"]
        aj, nj, lj = key["species_j"], key["n_j"], key["l_j"]
        
        #----sorting ni,li,nj,lj---
        if sort_orbs:
            fac=1 # sorted orbs - we only count everything once
            if ai == aj and (ni == nj and li == lj): #except these diag blocks
                fac=2 #so we need to divide by 2 to avoic double count
        else: 
            # no sorting -->  we count everything twice
            fac=2
        #----sorting ni,li,nj,lj---
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
        # where does orbital PHI = (ni, li) start within a block of atom i
        phioffset = orbs_offset[(ai, ni, li)] 
        # where does orbital PSI = (nj,lj) start within a block of atom j
        psioffset = orbs_offset[(aj, nj, lj)]

        # loops over samples (structure, i, j)

        # bt 0
        if not sort_orbs:
            bt0_factor_p = 0.5
        else: 
            if not(ni==nj and li==lj):
                bt0_factor_p = 1
            else:
                bt0_factor_p = 0.5
    
        samples = block.samples.values.tolist()
        
        blockvalues = block.values
        if detach:
            blockvalues = blockvalues.detach() #.clone()

        for sample, blockval in zip(samples, blockvalues[:,:,:,0]):

            if blockval.numel() == 0:
                # Empty block
                continue        

            A, i, j, Tx, Ty, Tz = sample
            T = Tx, Ty, Tz
            mT = -Tx, -Ty, -Tz

            other_fac = 1
            if i == j and T != (0,0,0) and not all_pairs:
                other_fac = 0.5

            # bt 0
            bt0_factor_m = bt0_factor_p*other_fac

            # bt 2 
            bt2_factor_m = bt2_factor_p*other_fac

            # bt 1
            bt1_fact_fin = bt1factor/fac*other_fac
            
            if T not in reconstructed_matrices[A]:
                assert mT not in reconstructed_matrices[A], "why is mT present but not T?"
                norbs = np.sum([orbs_tot[ai] for ai in dataset.structures[A].numbers])
                reconstructed_matrices[A][T] = torch.zeros(norbs, norbs, device = device)
                reconstructed_matrices[A][mT] = torch.zeros(norbs, norbs, device = device)

            matrix_T  = reconstructed_matrices[A][T]
            matrix_mT = reconstructed_matrices[A][mT]

            phi_end, psi_end = shapes[(ni, li, nj, lj)]

            # beginning of the block corresponding to the atom i-j pair
            i_start, j_start = atom_blocks_idx[(A, i, j)]

            iphi_jpsi_slice = slice(i_start + phioffset , i_start + phioffset + phi_end),\
                            slice(j_start + psioffset , j_start + psioffset + psi_end)
            ipsi_jphi_slice = slice(i_start + psioffset , i_start + psioffset + psi_end),\
                            slice(j_start + phioffset , j_start + phioffset + phi_end),
                            
            jphi_ipsi_slice = slice(j_start + phioffset , j_start + phioffset + phi_end),\
                            slice(i_start + psioffset , i_start + psioffset + psi_end)
            
            jpsi_iphi_slice = slice(j_start + psioffset , j_start + psioffset + psi_end),\
                            slice(i_start + phioffset , i_start + phioffset + phi_end)

            
            # bv = blockval #[:, :, 0]

            # position of the orbital within this block
            if block_type == 0:
                # <i \phi| H(T)|j \psi> = # <i \phi| H(-T)|j \psi>^T 
                # if not sort_orbs:
                #     ff = 0.5
                # else: 
                #     # ff = 0.5
                #     if not(ni==nj and li==lj):
                #         ff = 1
                #     else:
                #         ff = 0.5

                matrix_T[iphi_jpsi_slice] += blockval*bt0_factor_p
                matrix_mT[jpsi_iphi_slice] += blockval.T*bt0_factor_m
                
            elif block_type == 2:
                
                # ff=0.5
                # if not all_pairs:
                #     ff=1
                
                matrix_T[iphi_jpsi_slice] += blockval*bt2_factor_p
                matrix_mT[jpsi_iphi_slice] += blockval.T*bt2_factor_m
                
            elif abs(block_type) == 1:
                # Eq (1) <i \phi| H(T)|j \psi> = # block_(+1)ijT + block_(-1)ijT 
                # Eq (2) <j \phi| H(-T)|i \psi> = # block_(+1)ijT - block_(-1)ijT 
                # Eq (3) <j \psi| H(-T)|i \phi> = # block_(+1)ijT^\dagger + block_(-1)ijT^\dagger (Transpose of Eq1) 
                # Eq (4) <i \psi| H(T)|j \phi> = # block_(+1)ijT^\dagger - block_(-1)ijT^\dagger (Transpose of Eq2)
                # bv = bv*bt1_fact_fin

                if block_type == 1:
                    # first half of Eq (1) 
                    matrix_T[iphi_jpsi_slice] += blockval*bt1_fact_fin
                    # first half of Eq (2)
                    matrix_mT[jphi_ipsi_slice] += blockval*bt1_fact_fin
                    # first half of Eq (3)
                    matrix_mT[jpsi_iphi_slice] += blockval.T*bt1_fact_fin
                    # first half of Eq (4)
                    matrix_T[ ipsi_jphi_slice] += blockval.T*bt1_fact_fin
        
                else:
                    # second half of Eq (1)
                    matrix_T[iphi_jpsi_slice] += blockval*bt1_fact_fin
                    # second half of Eq (2)
                    matrix_mT[jphi_ipsi_slice] -= blockval*bt1_fact_fin
                    # second half of Eq (3)
                    matrix_mT[jpsi_iphi_slice] += blockval.T*bt1_fact_fin
                    # second half of Eq (4)
                    matrix_T[ipsi_jphi_slice ] -= blockval.T*bt1_fact_fin
         
    for A, matrix in enumerate(reconstructed_matrices):
        Ts = list(matrix.keys())
        for T in Ts:
            mT = tuple(-t for t in T)
         
            assert torch.all(torch.isclose(matrix[T] - reconstructed_matrices[A][mT].T, torch.zeros_like(matrix[T]))), torch.norm(matrix[T] - reconstructed_matrices[A][mT].T).item()

    return reconstructed_matrices

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

# #############

# @torch.jit.script
# def _process_block(block_type: int, blockval: torch.Tensor, matrix_T: torch.Tensor, matrix_mT: torch.Tensor, 
#                    i_start: int, j_start: int, phioffset: int, psioffset: int, phi_end: int, psi_end: int,
#                    bt0_factor_p: float, bt0_factor_m: float, bt2_factor_p: float, bt2_factor_m: float, bt1_fact_fin: float):
#     iphi_jpsi_slice_0 = slice(i_start + phioffset, i_start + phioffset + phi_end)
#     iphi_jpsi_slice_1 = slice(j_start + psioffset, j_start + psioffset + psi_end)
#     jpsi_iphi_slice_0 = slice(j_start + psioffset, j_start + psioffset + psi_end)
#     jpsi_iphi_slice_1 = slice(i_start + phioffset, i_start + phioffset + phi_end)
#     jphi_ipsi_slice_0 = slice(j_start + phioffset, j_start + phioffset + phi_end)
#     jphi_ipsi_slice_1 = slice(i_start + psioffset, i_start + psioffset + psi_end)
#     ipsi_jphi_slice_0 = slice(i_start + psioffset, i_start + psioffset + psi_end)
#     ipsi_jphi_slice_1 = slice(j_start + phioffset, j_start + phioffset + phi_end)

#     if block_type == 0:
#         matrix_T[iphi_jpsi_slice_0, iphi_jpsi_slice_1] += blockval * bt0_factor_p
#         matrix_mT[jpsi_iphi_slice_0, jpsi_iphi_slice_1] += blockval.T * bt0_factor_m
#     elif block_type == 2:
#         matrix_T[iphi_jpsi_slice_0, iphi_jpsi_slice_1] += blockval * bt2_factor_p
#         matrix_mT[jpsi_iphi_slice_0, jpsi_iphi_slice_1] += blockval.T * bt2_factor_m
#     elif abs(block_type) == 1:
#         if block_type == 1:
#             matrix_T[iphi_jpsi_slice_0, iphi_jpsi_slice_1] += blockval * bt1_fact_fin
#             matrix_mT[jphi_ipsi_slice_0, jphi_ipsi_slice_1] += blockval * bt1_fact_fin
#             matrix_mT[jpsi_iphi_slice_0, jpsi_iphi_slice_1] += blockval.T * bt1_fact_fin
#             matrix_T[ipsi_jphi_slice_0, ipsi_jphi_slice_1] += blockval.T * bt1_fact_fin
#         else:
#             matrix_T[iphi_jpsi_slice_0, iphi_jpsi_slice_1] += blockval * bt1_fact_fin
#             matrix_mT[jphi_ipsi_slice_0, jphi_ipsi_slice_1] -= blockval * bt1_fact_fin
#             matrix_mT[jpsi_iphi_slice_0, jpsi_iphi_slice_1] += blockval.T * bt1_fact_fin
#             matrix_T[ipsi_jphi_slice_0, ipsi_jphi_slice_1] -= blockval.T * bt1_fact_fin

# def blocks_to_matrix(blocks, dataset, device=None, cg=None, all_pairs=False, sort_orbs=True, detach=False, sample_id=None):
#     if device is None:
#         device = dataset.device
        
#     if "L" in blocks.keys.names:
#         from mlelec.utils.twocenter_utils import _to_uncoupled_basis
#         blocks = _to_uncoupled_basis(blocks, cg=cg, device=device)

#     orbs_tot, orbs_offset = _orbs_offsets(dataset.basis)
#     atom_blocks_idx = _atom_blocks_idx(dataset.structures, orbs_tot)
#     orbs_mult = {
#         species: {tuple(k): v for k, v in zip(*np.unique(np.asarray(dataset.basis[species])[:, :2], axis=0, return_counts=True))}
#         for species in dataset.basis
#     }

#     reconstructed_matrices = []
    
#     bt1factor = 1/np.sqrt(2)
#     if all_pairs:
#         bt1factor *= 0.5

#     bt2_factor_p = 0.5 if all_pairs else 1

#     for A in range(len(dataset.structures)):
#         norbs = sum(orbs_tot[ai] for ai in dataset.structures[A].numbers)
#         reconstructed_matrices.append({})

#     for key, block in blocks.items():
#         block_type = key["block_type"]
#         ai, ni, li = key["species_i"], key["n_i"], key["l_i"]
#         aj, nj, lj = key["species_j"], key["n_j"], key["l_j"]
        
#         if sort_orbs:
#             fac = 1 if not (ai == aj and ni == nj and li == lj) else 2
#         else:
#             fac = 2

#         orbs_i = orbs_mult[ai]
#         orbs_j = orbs_mult[aj]
        
#         shapes = {(k1 + k2): (orbs_i[tuple(k1)], orbs_j[tuple(k2)])
#                   for k1 in orbs_i for k2 in orbs_j}
#         phioffset = orbs_offset[(ai, ni, li)]
#         psioffset = orbs_offset[(aj, nj, lj)]

#         bt0_factor_p = 0.5 if not sort_orbs else (1 if not(ni == nj and li == lj) else 0.5)
    
#         samples = block.samples.values.tolist()
#         blockvalues = block.values.detach() if detach else block.values

#         for sample, blockval in zip(samples, blockvalues[:,:,0]):
#             if blockval.numel() == 0:
#                 continue        

#             A, i, j, Tx, Ty, Tz = sample
#             T = (Tx, Ty, Tz)
#             mT = (-Tx, -Ty, -Tz)

#             other_fac = 0.5 if (i == j and T != (0,0,0) and not all_pairs) else 1

#             bt0_factor_m = bt0_factor_p * other_fac
#             bt2_factor_m = bt2_factor_p * other_fac
#             bt1_fact_fin = bt1factor / fac * other_fac
            
#             if T not in reconstructed_matrices[A]:
#                 assert mT not in reconstructed_matrices[A], "why is mT present but not T?"
#                 norbs = sum(orbs_tot[ai] for ai in dataset.structures[A].numbers)
#                 reconstructed_matrices[A][T] = torch.zeros(norbs, norbs, device=device)
#                 reconstructed_matrices[A][mT] = torch.zeros(norbs, norbs, device=device)

#             matrix_T = reconstructed_matrices[A][T]
#             matrix_mT = reconstructed_matrices[A][mT]

#             i_start, j_start = atom_blocks_idx[(A, i, j)]
#             phi_end, psi_end = shapes[(ni, li, nj, lj)]

#             _process_block(block_type, blockval, matrix_T, matrix_mT, 
#                            i_start, j_start, phioffset, psioffset, phi_end, psi_end,
#                            bt0_factor_p, bt0_factor_m, bt2_factor_p, bt2_factor_m, bt1_fact_fin)
         
#     for A, matrix in enumerate(reconstructed_matrices):
#         Ts = list(matrix.keys())
#         for T in Ts:
#             mT = tuple(-t for t in T)
#             assert torch.all(torch.isclose(matrix[T] - reconstructed_matrices[A][mT].T, torch.zeros_like(matrix[T]))), \
#                    torch.norm(matrix[T] - reconstructed_matrices[A][mT].T).item()

#     return reconstructed_matrices


#################################################################################################################

def _compute_orbs_mult(basis: Dict[str, List[List[int]]]) -> Dict[str, Dict[Tuple[int, int], int]]:
    orbs_mult = {}
    for species, vals in basis.items():
        unique_vals, counts = torch.unique(torch.tensor(vals)[:, :2], dim=0, return_counts=True)
        orbs_mult[species] = {(int(k[0]), int(k[1])): int(v) for k, v in zip(unique_vals.tolist(), counts.tolist())}
    return orbs_mult

def _compute_shapes(orbs_i: Dict[Tuple[int, int], int], orbs_j: Dict[Tuple[int, int], int]) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, int]]:
    return {(k1, k2): (orbs_i[k1], orbs_j[k2]) for k1 in orbs_i for k2 in orbs_j}

def _get_or_create_matrices(
    reconstructed_matrices: List[Dict[Tuple[int, int, int], torch.Tensor]],
    A: int,
    T: Tuple[int, int, int],
    mT: Tuple[int, int, int],
    norbs: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    if T not in reconstructed_matrices[A]:
        assert mT not in reconstructed_matrices[A], "mT present but not T"
        reconstructed_matrices[A][T] = torch.zeros(norbs, norbs, device=device)
        reconstructed_matrices[A][mT] = torch.zeros(norbs, norbs, device=device)
    return reconstructed_matrices[A][T], reconstructed_matrices[A][mT]

def _collect_slice_info(blocks, dataset, orbs_tot, orbs_offset, atom_blocks_idx, orbs_mult, all_pairs, sort_orbs, detach):
    slice_info = defaultdict(lambda: defaultdict(list))
    
    for key, block in blocks.items():
        block_type = key["block_type"]
        ai, ni, li = key["species_i"], key["n_i"], key["l_i"]
        aj, nj, lj = key["species_j"], key["n_j"], key["l_j"]
        
        fac = 1 if sort_orbs else 2
        if sort_orbs and ai == aj and (ni == nj and li == lj):
            fac = 2

        orbs_i, orbs_j = orbs_mult[ai], orbs_mult[aj]
        shapes = _compute_shapes(orbs_i, orbs_j)
        phioffset, psioffset = orbs_offset[(ai, ni, li)], orbs_offset[(aj, nj, lj)]

        samples = block.samples.values.detach() if detach else block.samples.values
        blockvalues = block.values.detach() if detach else block.values

        for sample, blockval in zip(samples, blockvalues[:,:,:,0]):
            if blockval.numel() == 0:
                continue

            A, i, j, Tx, Ty, Tz = sample.tolist()
            T = (Tx, Ty, Tz)
            mT = (-Tx, -Ty, -Tz)

            other_fac = 0.5 if i == j and T != (0,0,0) and not all_pairs else 1
            bt0_factor_p = 0.5 if not sort_orbs or (ni == nj and li == lj) else 1
            bt0_factor_m = bt0_factor_p * other_fac
            bt2_factor_p = 0.5 if all_pairs else 1
            bt2_factor_m = bt2_factor_p * other_fac
            bt1_fact_fin = (ISQRT_2 / 2 if all_pairs else ISQRT_2) / fac * other_fac

            i_start, j_start = atom_blocks_idx[(A, i, j)]
            phi_end, psi_end = shapes[(ni, li), (nj, lj)]
            slices = (
                slice(i_start + phioffset, i_start + phioffset + phi_end),
                slice(j_start + psioffset, j_start + psioffset + psi_end),
                slice(i_start + psioffset, i_start + psioffset + psi_end),
                slice(j_start + phioffset, j_start + phioffset + phi_end),
                slice(j_start + phioffset, j_start + phioffset + phi_end),
                slice(i_start + psioffset, i_start + psioffset + psi_end),
                slice(j_start + psioffset, j_start + psioffset + psi_end),
                slice(i_start + phioffset, i_start + phioffset + phi_end)
            )

            slice_info[A][T].append({
                'block_type': block_type,
                'blockval': blockval,
                'bt0_factor_p': bt0_factor_p,
                'bt0_factor_m': bt0_factor_m,
                'bt2_factor_p': bt2_factor_p,
                'bt2_factor_m': bt2_factor_m,
                'bt1_fact_fin': bt1_fact_fin,
                'slices': slices,
                'mT': mT
            })
    
    return slice_info

def _update_matrices_v(slice_info, device, orbs_tot, dataset):
    reconstructed_matrices = [{} for _ in range(len(dataset.structures))]
    
    for A, A_info in slice_info.items():
        norbs = sum(orbs_tot[ai] for ai in dataset.structures[A].numbers)
        
        for T, T_info in A_info.items():
            matrix_T, matrix_mT = _get_or_create_matrices(reconstructed_matrices, A, T, T_info[0]['mT'], norbs, device)
            
            for info in T_info:
                block_type = info['block_type']
                blockval = info['blockval']
                slices = info['slices']
                
                if block_type == 0:
                    matrix_T[slices[0], slices[1]] += blockval * info['bt0_factor_p']
                    matrix_mT[slices[6], slices[7]] += blockval.T * info['bt0_factor_m']
                elif block_type == 2:
                    matrix_T[slices[0], slices[1]] += blockval * info['bt2_factor_p']
                    matrix_mT[slices[6], slices[7]] += blockval.T * info['bt2_factor_m']
                elif abs(block_type) == 1:
                    bv = blockval * info['bt1_fact_fin']
                    if block_type == 1:
                        matrix_T[slices[0], slices[1]] += bv
                        matrix_mT[slices[4], slices[5]] += bv
                        matrix_mT[slices[6], slices[7]] += bv.T
                        matrix_T[slices[2], slices[3]] += bv.T
                    else:
                        matrix_T[slices[0], slices[1]] += bv
                        matrix_mT[slices[4], slices[5]] -= bv
                        matrix_mT[slices[6], slices[7]] += bv.T
                        matrix_T[slices[2], slices[3]] -= bv.T
    
    return reconstructed_matrices

def blocks_to_matrix(
    blocks: torch.ScriptObject,
    dataset: Any,
    device: Optional[torch.device] = None,
    cg: Any = None,
    all_pairs: bool = False,
    sort_orbs: bool = True,
    detach: bool = False,
    sample_id: Optional[int] = None
) -> List[Dict[str, torch.Tensor]]:
    if device is None:
        device = dataset.device
    
    if "L" in blocks.keys.names:
        blocks = _to_uncoupled_basis(blocks, cg=cg, device=device)

    orbs_tot, orbs_offset = _orbs_offsets(dataset.basis)
    atom_blocks_idx = _atom_blocks_idx(dataset.structures, orbs_tot)
    orbs_mult = _compute_orbs_mult(dataset.basis)

    slice_info = _collect_slice_info(blocks, dataset, orbs_tot, orbs_offset, atom_blocks_idx, orbs_mult, all_pairs, sort_orbs, detach)
    reconstructed_matrices = _update_matrices_v(slice_info, device, orbs_tot, dataset)

    _validate_matrices(reconstructed_matrices)

    return reconstructed_matrices

def _validate_matrices(reconstructed_matrices: List[Dict[str, torch.Tensor]]):
    for A, matrix in enumerate(reconstructed_matrices):
        for T in list(matrix.keys()):
            mT = tuple(-t for t in T)
            # mT = tuple_to_string(tuple(-t for t in string_to_tuple(T)))
            assert torch.allclose(matrix[T], matrix[mT].T), f"Matrix mismatch for structure {A}, T={T}"

# # @torch.jit.script
# def tuple_to_string(t: Tuple[int, int, int]) -> str:
#     return f"{t[0]}_{t[1]}_{t[2]}"

# # @torch.jit.script
# def string_to_tuple(s: str) -> Tuple[int, int, int]:
#     return tuple(map(int, s.split('_')))

# def blocks_to_matrix(
#     blocks: torch.ScriptObject,
#     dataset: Any,
#     device: Optional[torch.device] = None,
#     cg: Any = None,
#     all_pairs: bool = False,
#     sort_orbs: bool = True,
#     detach: bool = False,
#     sample_id: Optional[int] = None
# ) -> List[Dict[str, torch.Tensor]]:
#     # Set device if not provided
#     if device is None:
#         device = dataset.device
    
#     # Convert to uncoupled basis if necessary
#     if "L" in blocks.keys.names:
#         blocks = _to_uncoupled_basis(blocks, cg=cg, device=device)

#     # Precompute necessary data
#     orbs_tot, orbs_offset = _orbs_offsets(dataset.basis)
#     atom_blocks_idx = _atom_blocks_idx(dataset.structures, orbs_tot)
#     orbs_mult = _compute_orbs_mult(dataset.basis)

#     # Initialize reconstructed matrices
#     reconstructed_matrices = [{} for _ in range(len(dataset.structures))]
    
#     # Compute factors
#     bt1factor = ISQRT_2 / 2 if all_pairs else ISQRT_2
#     bt2_factor_p = 0.5 if all_pairs else 1

#     # Main loop over blocks
#     for key, block in blocks.items():
#         block_type = key["block_type"]
#         ai, ni, li = key["species_i"], key["n_i"], key["l_i"]
#         aj, nj, lj = key["species_j"], key["n_j"], key["l_j"]
        
#         # Compute factors based on sorting
#         fac = 1 if sort_orbs else 2
#         if sort_orbs and ai == aj and (ni == nj and li == lj):
#             fac = 2

#         orbs_i, orbs_j = orbs_mult[ai], orbs_mult[aj]
#         shapes = _compute_shapes(orbs_i, orbs_j)
#         phioffset, psioffset = orbs_offset[(ai, ni, li)], orbs_offset[(aj, nj, lj)]

#         samples = block.samples.values.tolist()
#         blockvalues = block.values.detach() if detach else block.values

#         # Loop over samples
#         for sample, blockval in zip(samples, blockvalues[:,:,:,0]):
#             if blockval.numel() == 0:
#                 continue

#             A, i, j, Tx, Ty, Tz = sample
#             T = Tx, Ty, Tz #tuple_to_string((Tx, Ty, Tz))
#             mT = -Tx, -Ty, -Tz #tuple_to_string((-Tx, -Ty, -Tz))

#             # Compute factors
#             other_fac = 0.5 if i == j and (Tx, Ty, Tz) != (0,0,0) and not all_pairs else 1
#             bt0_factor_p = 0.5 if not sort_orbs or (ni == nj and li == lj) else 1
#             bt0_factor_m = bt0_factor_p * other_fac
#             bt2_factor_m = bt2_factor_p * other_fac
#             bt1_fact_fin = bt1factor / fac * other_fac

#             # Get or create matrices
#             matrix_T, matrix_mT = _get_or_create_matrices(reconstructed_matrices, A, T, mT, dataset, device, orbs_tot)

#             # Compute slices
#             i_start, j_start = atom_blocks_idx[(A, i, j)]
#             phi_end, psi_end = shapes[(ni, li), (nj, lj)]
#             slices = _compute_slices(i_start, j_start, phioffset, psioffset, phi_end, psi_end)

#             # Update matrices
#             _update_matrices(matrix_T, matrix_mT, blockval, block_type, bt0_factor_p, bt0_factor_m,
#                              bt2_factor_p, bt2_factor_m, bt1_fact_fin, slices)

#     # Validate matrices
#     _validate_matrices(reconstructed_matrices)

#     return reconstructed_matrices

# # Helper functions

# # @torch.jit.script
# def _compute_orbs_mult(basis: Dict[str, List[List[int]]]) -> Dict[str, Dict[str, int]]:
#     # return {
#     #     species: {
#     #         f"{k[0]}_{k[1]}": int(v)
#     #         for k, v in zip(
#     #             *np.unique(
#     #                 np.asarray(basis[species])[:, :2],
#     #                 axis=0,
#     #                 return_counts=True,
#     #             )
#     #         )
#     #     }
#     #     for species in basis
#     # }
#     return {
#         species: {
#             (k[0], k[1]): int(v)
#             for k, v in zip(
#                 *np.unique(
#                     np.asarray(basis[species])[:, :2],
#                     axis=0,
#                     return_counts=True,
#                 )
#             )
#         }
#         for species in basis
#     }

# # @torch.jit.script
# def _compute_shapes(orbs_i: Dict[str, int], orbs_j: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
#     # return {
#     #     f"{k1}_{k2}": (orbs_i[k1], orbs_j[k2])
#     #     for k1 in orbs_i
#     #     for k2 in orbs_j
#     # }
#     return {
#         (k1, k2): (orbs_i[k1], orbs_j[k2])
#         for k1 in orbs_i
#         for k2 in orbs_j
#     }

# # @torch.jit.script
# def _get_or_create_matrices(
#     reconstructed_matrices: List[Dict[str, torch.Tensor]],
#     A: int,
#     T: Tuple[int],
#     mT: Tuple[int],
#     dataset: Any,
#     device: torch.device,
#     orbs_tot: Dict[int, int]
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     if T not in reconstructed_matrices[A]:
#         assert mT not in reconstructed_matrices[A], "mT present but not T"
#         norbs = sum(orbs_tot[ai] for ai in dataset.structures[A].numbers)
#         reconstructed_matrices[A][T] = torch.zeros(norbs, norbs, device=device)
#         reconstructed_matrices[A][mT] = torch.zeros(norbs, norbs, device=device)
#     return reconstructed_matrices[A][T], reconstructed_matrices[A][mT]

# # @torch.jit.script
# def _compute_slices(
#     i_start: int, j_start: int, phioffset: int, psioffset: int, phi_end: int, psi_end: int
# ) -> Tuple[slice, slice, slice, slice, slice, slice, slice, slice]:
#     iphi_jpsi = (slice(i_start + phioffset, i_start + phioffset + phi_end),
#                  slice(j_start + psioffset, j_start + psioffset + psi_end))
#     ipsi_jphi = (slice(i_start + psioffset, i_start + psioffset + psi_end),
#                  slice(j_start + phioffset, j_start + phioffset + phi_end))
#     jphi_ipsi = (slice(j_start + phioffset, j_start + phioffset + phi_end),
#                  slice(i_start + psioffset, i_start + psioffset + psi_end))
#     jpsi_iphi = (slice(j_start + psioffset, j_start + psioffset + psi_end),
#                  slice(i_start + phioffset, i_start + phioffset + phi_end))
#     return iphi_jpsi[0], iphi_jpsi[1], ipsi_jphi[0], ipsi_jphi[1], jphi_ipsi[0], jphi_ipsi[1], jpsi_iphi[0], jpsi_iphi[1]


# # @torch.jit.script
# def _update_matrices(
#     matrix_T: torch.Tensor,
#     matrix_mT: torch.Tensor,
#     blockval: torch.Tensor,
#     block_type: int,
#     bt0_factor_p: float,
#     bt0_factor_m: float,
#     bt2_factor_p: float,
#     bt2_factor_m: float,
#     bt1_fact_fin: float,
#     slices: Tuple[slice, slice, slice, slice, slice, slice, slice, slice]
# ):
#     iphi_jpsi_0, iphi_jpsi_1, ipsi_jphi_0, ipsi_jphi_1, jphi_ipsi_0, jphi_ipsi_1, jpsi_iphi_0, jpsi_iphi_1 = slices

#     if block_type == 0:
#         matrix_T[iphi_jpsi_0, iphi_jpsi_1] += blockval * bt0_factor_p
#         matrix_mT[jpsi_iphi_0, jpsi_iphi_1] += blockval.T * bt0_factor_m
#     elif block_type == 2:
#         matrix_T[iphi_jpsi_0, iphi_jpsi_1] += blockval * bt2_factor_p
#         matrix_mT[jpsi_iphi_0, jpsi_iphi_1] += blockval.T * bt2_factor_m
#     elif abs(block_type) == 1:
#         bv = blockval * bt1_fact_fin
#         if block_type == 1:
#             matrix_T[iphi_jpsi_0, iphi_jpsi_1] += bv
#             matrix_mT[jphi_ipsi_0, jphi_ipsi_1] += bv
#             matrix_mT[jpsi_iphi_0, jpsi_iphi_1] += bv.T
#             matrix_T[ipsi_jphi_0, ipsi_jphi_1] += bv.T
#         else:
#             matrix_T[iphi_jpsi_0, iphi_jpsi_1] += bv
#             matrix_mT[jphi_ipsi_0, jphi_ipsi_1] -= bv
#             matrix_mT[jpsi_iphi_0, jpsi_iphi_1] += bv.T
#             matrix_T[ipsi_jphi_0, ipsi_jphi_1] -= bv.T


# # @torch.jit.script
# def _validate_matrices(reconstructed_matrices: List[Dict[str, torch.Tensor]]):
#     for A, matrix in enumerate(reconstructed_matrices):
#         for T in list(matrix.keys()):
#             mT = tuple(-t for t in T)
#             # mT = tuple_to_string(tuple(-t for t in string_to_tuple(T)))
#             assert torch.allclose(matrix[T], matrix[mT].T), f"Matrix mismatch for structure {A}, T={T}"



# def _collect_slice_info(blocks, dataset, orbs_tot, orbs_offset, atom_blocks_idx, orbs_mult, all_pairs, sort_orbs, detach):
#     slice_info = defaultdict(lambda: defaultdict(list))
#     for key, block in blocks.items():
#         block_type = key["block_type"]
#         ai, ni, li = key["species_i"], key["n_i"], key["l_i"]
#         aj, nj, lj = key["species_j"], key["n_j"], key["l_j"]
        
#         fac = 1 if sort_orbs else 2
#         if sort_orbs and ai == aj and (ni == nj and li == lj):
#             fac = 2

#         orbs_i, orbs_j = orbs_mult[ai], orbs_mult[aj]
#         shapes = _compute_shapes(orbs_i, orbs_j)
#         phioffset, psioffset = orbs_offset[(ai, ni, li)], orbs_offset[(aj, nj, lj)]

#         samples = block.samples.values.tolist()
#         blockvalues = block.values.detach() if detach else block.values

#         for sample, blockval in zip(samples, blockvalues[:,:,:,0]):
#             if blockval.numel() == 0:
#                 continue

#             A, i, j, Tx, Ty, Tz = sample
#             T = (Tx, Ty, Tz)
#             mT = (-Tx, -Ty, -Tz)

#             other_fac = 0.5 if i == j and T != (0,0,0) and not all_pairs else 1
#             bt0_factor_p = 0.5 if not sort_orbs or (ni == nj and li == lj) else 1
#             bt0_factor_m = bt0_factor_p * other_fac
#             bt2_factor_p = 0.5 if all_pairs else 1
#             bt2_factor_m = bt2_factor_p * other_fac
#             bt1_fact_fin = (ISQRT_2 / 2 if all_pairs else ISQRT_2) / fac * other_fac

#             i_start, j_start = atom_blocks_idx[(A, i, j)]
#             phi_end, psi_end = shapes[(ni, li), (nj, lj)]
#             slices = _compute_slices(i_start, j_start, phioffset, psioffset, phi_end, psi_end)

#             slice_info[A][T].append({
#                 'block_type': block_type,
#                 'blockval': blockval,
#                 'bt0_factor_p': bt0_factor_p,
#                 'bt0_factor_m': bt0_factor_m,
#                 'bt2_factor_p': bt2_factor_p,
#                 'bt2_factor_m': bt2_factor_m,
#                 'bt1_fact_fin': bt1_fact_fin,
#                 'slices': slices,
#                 'mT': mT
#             })
    
#     return slice_info

# def _update_matrices_v(slice_info, device, orbs_tot, dataset):
#     reconstructed_matrices = [{} for _ in range(len(dataset.structures))]
    
#     for A, A_info in slice_info.items():
#         norbs = sum(orbs_tot[ai] for ai in dataset.structures[A].numbers)
        
#         for T, T_info in A_info.items():
#             if T not in reconstructed_matrices[A]:
#                 reconstructed_matrices[A][T] = torch.zeros((norbs, norbs), device=device)
#             if T != (0, 0, 0) and T_info[0]['mT'] not in reconstructed_matrices[A]:
#                 reconstructed_matrices[A][T_info[0]['mT']] = torch.zeros((norbs, norbs), device=device)
            
#             for info in T_info:
#                 block_type = info['block_type']
#                 blockval = info['blockval']
#                 slices = info['slices']
                
#                 if block_type == 0:
#                     reconstructed_matrices[A][T][slices[0], slices[1]] += blockval * info['bt0_factor_p']
#                     reconstructed_matrices[A][info['mT']][slices[6], slices[7]] += blockval.T * info['bt0_factor_m']
#                 elif block_type == 2:
#                     reconstructed_matrices[A][T][slices[0], slices[1]] += blockval * info['bt2_factor_p']
#                     reconstructed_matrices[A][info['mT']][slices[6], slices[7]] += blockval.T * info['bt2_factor_m']
#                 elif block_type == 1:
#                     bv = blockval * info['bt1_fact_fin']
#                     reconstructed_matrices[A][T][slices[0], slices[1]] += bv
#                     reconstructed_matrices[A][info['mT']][slices[4], slices[5]] += bv
#                     reconstructed_matrices[A][info['mT']][slices[6], slices[7]] += bv.T
#                     reconstructed_matrices[A][T][slices[2], slices[3]] += bv.T
#                 else:  # block_type == -1
#                     bv = blockval * info['bt1_fact_fin']
#                     reconstructed_matrices[A][T][slices[0], slices[1]] += bv
#                     reconstructed_matrices[A][info['mT']][slices[4], slices[5]] -= bv
#                     reconstructed_matrices[A][info['mT']][slices[6], slices[7]] += bv.T
#                     reconstructed_matrices[A][T][slices[2], slices[3]] -= bv.T
    
#     return reconstructed_matrices

# def blocks_to_matrix_v(
#     blocks: torch.ScriptObject,
#     dataset: Any,
#     device: Optional[torch.device] = None,
#     cg: Any = None,
#     all_pairs: bool = False,
#     sort_orbs: bool = True,
#     detach: bool = False,
#     sample_id: Optional[int] = None
# ) -> List[Dict[str, torch.Tensor]]:
#     if device is None:
#         device = dataset.device
    
#     if "L" in blocks.keys.names:
#         blocks = _to_uncoupled_basis(blocks, cg=cg, device=device)

#     orbs_tot, orbs_offset = _orbs_offsets(dataset.basis)
#     atom_blocks_idx = _atom_blocks_idx(dataset.structures, orbs_tot)
#     orbs_mult = _compute_orbs_mult(dataset.basis)

#     slice_info = _collect_slice_info(blocks, dataset, orbs_tot, orbs_offset, atom_blocks_idx, orbs_mult, all_pairs, sort_orbs, detach)
#     reconstructed_matrices = _update_matrices_v(slice_info, device, orbs_tot, dataset)

#     # print(slice_info)
#     # print(reconstructed_matrices)
#     _validate_matrices(reconstructed_matrices)

#     return reconstructed_matrices



