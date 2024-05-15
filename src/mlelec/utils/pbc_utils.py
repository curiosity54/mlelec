import numpy as np
import warnings
from scipy.fft import fftn, ifftn
from torch.fft import fftn as torch_fftn, ifftn as torch_ifftn
from ase.units import Bohr
from metatensor import Labels, TensorBlock, TensorMap
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



def matrix_to_blocks(dataset, device=None, all_pairs = True, cutoff = None, target='fock', matrix=None, sort_orbs = False):
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
                    ij_distance = np.linalg.norm(frame.cell.array.T @ np.array(T) + frame.positions[j] - frame.positions[i])
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



def blocks_to_matrix(blocks, dataset, device=None, cg = None, all_pairs = False, sort_orbs = False):
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

    reconstructed_matrices = []
    
    bt1factor = ISQRT_2 
    bt2factor =2 # because all_pairs=False in matrix_to_blocks still returns all species pairs (not ordered)
    if all_pairs:
        bt1factor/=2
        bt2factor = 2

    for A in range(len(dataset.structures)):
        norbs = np.sum([orbs_tot[ai] for ai in dataset.structures[A].numbers])
        reconstructed_matrices.append({})

    # loops over block types
    for key, block in blocks.items():
        block_type = key["block_type"]
        ai, ni, li = key["species_i"], key["n_i"], key["l_i"]
        aj, nj, lj = key["species_j"], key["n_j"], key["l_j"]
        T = key["cell_shift_a"], key["cell_shift_b"], key["cell_shift_c"]
        
        #----sorting ni,li,nj,lj---
        if sort_orbs:
            fac=1 # sorted orbs - we only count everything once
            if ai == aj and (ni ==nj and li == lj): #except these diag blocks
                fac=2 #so we need to divide by 2 to avoic double count
        else: 
            # no sorting -->  we count everything twice
            fac=2
        #----sorting ni,li,nj,lj---
        #TODO: make consistent in kmatrix_to_blocs, kblocks_to_matrix
        mT = tuple(-t for t in T)
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
    
        for sample, blockval in zip(block.samples, block.values):
            
            A = sample["structure"]
            i = sample["center"]
            j = sample["neighbor"]
            
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
            values = blockval[:, :, 0].clone()
            # position of the orbital within this block
            if block_type == 0 or block_type == 2:
                # <i \phi| H(T)|j \psi> = # <i \phi| H(-T)|j \psi>^T 
                matrix_T[
                    i_start + phioffset : i_start + phioffset + phi_end,
                    j_start + psioffset : j_start + psioffset + psi_end,
                             ] = values
                matrix_mT[
                    j_start + psioffset : j_start + psioffset + psi_end,
                    i_start + phioffset : i_start + phioffset + phi_end,
                             ] = values.T
            
            elif abs(block_type) == 1:
                values *= bt1factor/fac
                
                iphi_jpsi_slice = slice(i_start + phioffset , i_start + phioffset + phi_end),\
                                  slice(j_start + psioffset , j_start + psioffset + psi_end)
                ipsi_jphi_slice = slice(i_start + psioffset , i_start + psioffset + psi_end),\
                                slice(j_start + phioffset , j_start + phioffset + phi_end),
                                
                jphi_ipsi_slice = slice(j_start + phioffset , j_start + phioffset + phi_end),\
                                 slice(i_start + psioffset , i_start + psioffset + psi_end)
                
                jpsi_iphi_slice = slice(j_start + psioffset , j_start + psioffset + psi_end),\
                        slice(i_start + phioffset , i_start + phioffset + phi_end)
                                  
                                  
                # Eq (1) <i \phi| H(T)|j \psi> = # block_(+1)ijT + block_(-1)ijT 
                # Eq (2) <j \phi| H(-T)|i \psi> = # block_(+1)ijT - block_(-1)ijT 
                # Eq (3) <j \psi| H(-T)|i \phi> = # block_(+1)ijT^\dagger + block_(-1)ijT^\dagger (Transpose of Eq1) 
                # Eq (4) <i \psi| H(T)|j \phi> = # block_(+1)ijT^\dagger - block_(-1)ijT^\dagger (Transpose of Eq2)
                if block_type == 1:
                    # first half of Eq (1) 
                    matrix_T[iphi_jpsi_slice] += values
                    # first half of Eq (2)
                    matrix_mT[jphi_ipsi_slice] += values
                    # first half of Eq (3)
                    matrix_mT[jpsi_iphi_slice] += values.T
                    # first half of Eq (4)
                    matrix_T[ ipsi_jphi_slice] += values.T
        
                else:
                    # second half of Eq (1)
                    matrix_T[iphi_jpsi_slice] += values
                    # second half of Eq (2)
                    matrix_mT[jphi_ipsi_slice] -= values
                    # second half of Eq (3)
                    matrix_mT[jpsi_iphi_slice] += values.T
                    # second half of Eq (4)
                    matrix_T[ipsi_jphi_slice ] -= values.T
         
    for A, matrix in enumerate(reconstructed_matrices):
        Ts = list(matrix.keys())
        for T in Ts:
            mT = tuple(-t for t in T)
         
            assert torch.all(torch.isclose(matrix[T] - reconstructed_matrices[A][mT].T, torch.zeros_like(matrix[T]))), torch.norm(matrix[T] - reconstructed_matrices[A][mT].T).item()

    return reconstructed_matrices

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

def kmatrix_to_blocks(dataset, device=None, all_pairs = True, cutoff = None, target='fock', sort_orbs=False):
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


            # When the calculation is at Gamma you want to skip i==j samples
            # is_gamma_point = dataset.kmesh[A] == [1,1,1] and ik == 0
            is_gamma_point = np.linalg.norm(dataset.kpts_rel[A][ik]) < 1e-30

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

                        # if block_type == 1:
                        #     block_asym = block_builder.blocks[(-1,) + key[1:]]

                        if block_type == 1:
                            bplus = (value + value_ji) * ISQRT_2
                            bminus = (value - value_ji) * ISQRT_2

                            block.add_samples(
                                labels=[(A, i, j, ik)],
                                data=bplus.reshape(1, 2 * li + 1, 2 * lj + 1, 1),
                            )

                            # Skip the Gamma point, bt=-1, i==j sample because 
                            # it's zero [H(Gamma)_{i,i,phi,psi,-1}=0]
                            if (not (is_gamma_point and i == j)):
                                block_asym = block_builder.blocks[(-1,) + key[1:]]
                                block_asym.add_samples(
                                    labels=[(A, i, j, ik)],
                                    data = bminus.reshape(1, 2 * li + 1, 2 * lj + 1, 1),
                                )

                        elif block_type == 2:
                            # if i == 0 and j == 1 and ik == 0 and ni == 1 and li == 0 and nj == 2 and lj == 0:

                            #     print(A, i, j, ik, value)
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


def kblocks_to_matrix(k_target_blocks, dataset, all_pairs = False, sort_orbs = False):
    """
    k_target_blocks: UNCOUPLED blocks of H(k)
   
    """
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
    bt1factor = ISQRT_2 
    bt2factor = 1

    if all_pairs:
        bt1factor /= 2
        bt2factor *= 2 # because we add both <I \phi | J \psi> and <I \psi | J \phi> 

    recon_Hk = {}
    for k, block in k_target_blocks.items():
        bt, ai, ni, li, aj, nj, lj = k.values
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

        for sample, blockval_ in zip(block.samples, block.values):

            blockval = blockval_[...,0].clone()

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

                blockval *= bt1factor*bt0factor
            
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


dummy_prop = Labels(['dummy'], np.array([[0]]))
def TMap_bloch_sums(target_blocks, phase, indices, return_tensormap = False):

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
        b_values = b.values.to(next(iter(next(iter(phase.values())).values())))
        for I, (ifr, i, j) in enumerate(phase[kl]):
            
            idx = indices[kl][ifr,i,j]
            values = b_values[idx]
            vshape = values.shape
            pshape = phase[kl][ifr, i, j].shape

            # equivalent to torch.einsum('Tmnv,kT->kmnv', values.to(phase[kl][ifr, i, j]), phase[kl][ifr, i, j]), but faster
            contraction = (phase[kl][ifr, i, j]@values.reshape(vshape[0], -1)).reshape(pshape[0], *vshape[1:])

            if bt != 0:
                # block type not zero: create dictionary element
                if (ifr, i, j) in _Hk[_kl]:
                    _Hk[_kl][ifr, i, j] += contraction
                else:
                    _Hk[_kl][ifr, i, j] = contraction
            else:
                # block type zero
                if (ifr, i, j) in _Hk[_kl]:
                    # if the corresponding bt = +1 element exists, sum to it the bt=0 contribution
                    _Hk[_kl][ifr, i, j] += contraction*np.sqrt(2)
                else:
                    # The corresponding bt = +1 element does not exist. Create the dictionary element
                    _Hk[_kl][ifr, i, j] = contraction*np.sqrt(2)
                    
    if return_tensormap:
        # Now store in a tensormap
        _k_target_blocks = []
        keys = []
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
                    properties = dummy_prop,
                    values = values
                )
            )
            
            keys.append(list(kl))

        _k_target_blocks = TensorMap(Labels(['block_type', 'species_i', 'n_i', 'l_i', 'species_j', 'n_j', 'l_j'], np.array(keys)), _k_target_blocks)

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


######--------------- NEW/OLD - to discard or incorporate? --------------------------- ###########################
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



def TMap_bloch_sums_OLD(target_blocks, phase, indices):

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