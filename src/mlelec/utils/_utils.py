# import torch
# import numpy as np
# import ase
# from mlelec.utils.twocenter_utils import _to_uncoupled_basis

# ISQRT_2 = 1 / np.sqrt(2)

# def _orbs_offsets(orbs):
#     orbs_tot = {}
#     orbs_offset = {}

#     for k, orb_list in orbs.items():
#         filtered_orbs = [(n, l) for n, l, m in orb_list if m == -l]
#         l_values = np.array([l for _, l in filtered_orbs])
#         offsets = np.cumsum(2 * l_values + 1) - (2 * l_values + 1)
        
#         orbs_offset.update({(k, n, l): offset for (n, l), offset in zip(filtered_orbs, offsets)})
#         orbs_tot[k] = offsets[-1] + (2 * l_values[-1] + 1) if offsets.size > 0 else 0

#     return orbs_tot, orbs_offset

# def _atom_blocks_idx(frames, orbs_tot):
#     if isinstance(frames, ase.Atoms):
#         frames = [frames]
    
#     atom_blocks_idx = {}
#     for A, f in enumerate(frames):
#         numbers = np.array(f.numbers)
#         ki = np.cumsum([0] + [orbs_tot[ai] for ai in numbers[:-1]])
#         kj = np.cumsum([0] + [orbs_tot[aj] for aj in numbers[:-1]])
        
#         ki_matrix, kj_matrix = np.meshgrid(ki, kj, indexing='ij')
#         for i, _ in enumerate(numbers):
#             for j, _ in enumerate(numbers):
#                 atom_blocks_idx[(A, i, j)] = (ki_matrix[i, j], kj_matrix[i, j])
    
#     return atom_blocks_idx

# def _initialize_matrices(qmdata, device):
#     nTs = None # TODO
#     return [torch.zeros((nT, nao, nao), dtype = torch.float64, device = device) for nT, nao in zip(nTs, qmdata.nao)]

# def _collect_data(blocks, atom_blocks_idx, sort_orbs, all_pairs, orbs_offset, orbs_mult):
#     '''
#     Returns a dictionary labeled by block keys containing advanced indexing info to update the Hamiltonians
#     '''
    
#     collected_data = {}

#     for key, block in blocks.items():
    
#         block_type, ai, ni, li, aj, nj, lj = kl = key.values.tolist()

#         bt1_factor = ISQRT_2 / (1 if (sort_orbs and ai == aj and ni == nj and li == lj) else 2) / (2 if all_pairs else 1)

#         orbs_i = orbs_mult[ai]
#         orbs_j = orbs_mult[aj]

#         shapes = {(k1 + k2): (orbs_i[tuple(k1)], orbs_j[tuple(k2)]) for k1 in orbs_i for k2 in orbs_j}
#         phioffset = orbs_offset[(ai, ni, li)]
#         psioffset = orbs_offset[(aj, nj, lj)]

#         samples = block.samples.values.tolist()
#         blockvalues = block.values

#         for sample, blockval in zip(samples, blockvalues[:,:,:,0]):
#             if blockval.numel() == 0:
#                 continue

#             A, i, j, Tx, Ty, Tz = sample
#             T = (Tx, Ty, Tz)
#             mT = (-Tx, -Ty, -Tz)

#             offsite_perm_fac = (0.5 if (i == j and T != (0, 0, 0) and not all_pairs) else 1)

#             bt0_factor_p = 0.5 if not sort_orbs else (1 if not (ni == nj and li == lj) else 0.5)
#             bt0_factor_m = bt0_factor_p * offsite_perm_fac
#             bt2_factor_p = 0.5 if not all_pairs else 1
#             bt2_factor_m = bt2_factor_p * offsite_perm_fac
#             bt1_fact_fin = bt1_factor * offsite_perm_fac 

#             i_start, j_start = atom_blocks_idx[(A, i, j)]
#             phi_end = shapes[(ni, li, nj, lj)][0]
#             psi_end = shapes[(ni, li, nj, lj)][1]

#             iphi_jpsi_slice = (slice(i_start + phioffset, i_start + phioffset + phi_end),
#                             slice(j_start + psioffset, j_start + psioffset + psi_end))
#             ipsi_jphi_slice = (slice(i_start + psioffset, i_start + psioffset + psi_end),
#                             slice(j_start + phioffset, j_start + phioffset + phi_end))
#             jphi_ipsi_slice = (slice(j_start + phioffset, j_start + phioffset + phi_end),
#                             slice(i_start + psioffset, i_start + psioffset + psi_end))
#             jpsi_iphi_slice = (slice(j_start + psioffset, j_start + psioffset + psi_end),
#                             slice(i_start + phioffset, i_start + phioffset + phi_end))

#             if A not in collected_data:
#                 collected_data[A] = {"T_values": [], "matrices": [], "slices": []}
            
#             collected_data[A]["T_values"].append(T)
#             collected_data[A]["matrices"].append((T, mT, blockval, block_type, bt0_factor_p, bt0_factor_m, bt2_factor_p, bt2_factor_m, bt1_fact_fin))
#             collected_data[A]["slices"].append((iphi_jpsi_slice, ipsi_jphi_slice, jphi_ipsi_slice, jpsi_iphi_slice))

#     return collected_data

# def _update_matrices(reconstructed_matrices, collected_data):
#     for A, (T_values, matrices) in enumerate(reconstructed_matrices):
#         if A not in collected_data:
#             continue

#         data = collected_data[A]
#         for idx, (T, mT, blockval, block_type, bt0_factor_p, bt0_factor_m, bt2_factor_p, bt2_factor_m, bt1_fact_fin) in enumerate(data["matrices"]):
#             iphi_jpsi_slice, ipsi_jphi_slice, jphi_ipsi_slice, jpsi_iphi_slice = data["slices"][idx]

#             if block_type == 0:
#                 matrices[idx, iphi_jpsi_slice[0], iphi_jpsi_slice[1]].add_(blockval * bt0_factor_p)
#                 matrices[idx, jpsi_iphi_slice[0], jpsi_iphi_slice[1]].add_(blockval.T * bt0_factor_m)

#             elif block_type == 2:
#                 matrices[idx, iphi_jpsi_slice[0], iphi_jpsi_slice[1]].add_(blockval * bt2_factor_p)
#                 matrices[idx, jpsi_iphi_slice[0], jpsi_iphi_slice[1]].add_(blockval.T * bt2_factor_m)

#             elif abs(block_type) == 1:
#                 blockval.mul_(bt1_fact_fin)
#                 if block_type == 1:
#                     matrices[idx, iphi_jpsi_slice[0], iphi_jpsi_slice[1]].add_(blockval)
#                     matrices[idx, jphi_ipsi_slice[0], jphi_ipsi_slice[1]].add_(blockval)
#                     matrices[idx, jpsi_iphi_slice[0], jpsi_iphi_slice[1]].add_(blockval.T)
#                     matrices[idx, ipsi_jphi_slice[0], ipsi_jphi_slice[1]].add_(blockval.T)
#                 else:
#                     matrices[idx, iphi_jpsi_slice[0], iphi_jpsi_slice[1]].add_(blockval)
#                     matrices[idx, jphi_ipsi_slice[0], jphi_ipsi_slice[1]].sub_(blockval)
#                     matrices[idx, jpsi_iphi_slice[0], jpsi_iphi_slice[1]].add_(blockval.T)
#                     matrices[idx, ipsi_jphi_slice[0], ipsi_jphi_slice[1]].sub_(blockval.T)

# def blocks_to_matrix_opt(blocks, dataset, device=None, cg=None, all_pairs=False, sort_orbs=True, detach=False, sample_id=None):
#     if device is None:
#         device = dataset.device

#     if "L" in blocks.keys.names:
#         blocks = _to_uncoupled_basis(blocks, cg=cg, device=device)

#     orbs_tot, orbs_offset = _orbs_offsets(dataset.basis)
#     atom_blocks_idx = _atom_blocks_idx(dataset.structures, orbs_tot)

#     orbs_mult = {
#         species: 
#             {tuple(k): v
#             for k, v in zip(
#                 *np.unique(
#                     np.asarray(dataset.basis[species])[:, :2],
#                     axis=0,
#                     return_counts=True,
#                 )
#             )
#         }
#         for species in dataset.basis
#     }

#     collected_data = _collect_data(blocks, atom_blocks_idx, sort_orbs, all_pairs, orbs_offset, orbs_mult)
#     reconstructed_matrices = _initialize_matrices(collected_data, orbs_tot, dataset, device)
#     _update_matrices(reconstructed_matrices, collected_data)

#     for A, (T_values, matrices) in enumerate(reconstructed_matrices):
#         for idx, T in enumerate(T_values):
#             mT = tuple(-t for t in T)
#             assert torch.all(torch.isclose(matrices[idx] - matrices[idx].T, torch.zeros_like(matrices[idx]))), torch.norm(matrices[idx] - matrices[idx].T).item()

#     return reconstructed_matrices


import torch
import numpy as np
import ase
from mlelec.utils.twocenter_utils import _to_uncoupled_basis

ISQRT_2 = 1 / np.sqrt(2)

def _orbs_offsets(orbs):
    orbs_tot = {}
    orbs_offset = {}

    for k, orb_list in orbs.items():
        filtered_orbs = [(n, l) for n, l, m in orb_list if m == -l]
        l_values = np.array([l for _, l in filtered_orbs])
        offsets = np.cumsum(2 * l_values + 1) - (2 * l_values + 1)
        
        orbs_offset.update({(k, n, l): offset for (n, l), offset in zip(filtered_orbs, offsets)})
        orbs_tot[k] = offsets[-1] + (2 * l_values[-1] + 1) if offsets.size > 0 else 0

    return orbs_tot, orbs_offset

def _atom_blocks_idx(frames, orbs_tot):
    if isinstance(frames, ase.Atoms):
        frames = [frames]
    
    atom_blocks_idx = {}
    for A, f in enumerate(frames):
        numbers = np.array(f.numbers)
        ki = np.cumsum([0] + [orbs_tot[ai] for ai in numbers[:-1]])
        kj = np.cumsum([0] + [orbs_tot[aj] for aj in numbers[:-1]])
        
        ki_matrix, kj_matrix = np.meshgrid(ki, kj, indexing='ij')
        for i, _ in enumerate(numbers):
            for j, _ in enumerate(numbers):
                atom_blocks_idx[(A, i, j)] = (ki_matrix[i, j], kj_matrix[i, j])
    
    return atom_blocks_idx

def _collect_data(blocks, atom_blocks_idx, sort_orbs, all_pairs, orbs_offset, orbs_mult):
    collected_data = {}

    for key, block in blocks.items():
        block_type = key["block_type"]
        ai, ni, li = key["species_i"], key["n_i"], key["l_i"]
        aj, nj, lj = key["species_j"], key["n_j"], key["l_j"]

        fac = 1 if (sort_orbs and ai == aj and ni == nj and li == lj) else 2

        orbs_i = orbs_mult[ai]
        orbs_j = orbs_mult[aj]

        shapes = {(k1 + k2): (orbs_i[tuple(k1)], orbs_j[tuple(k2)]) for k1 in orbs_i for k2 in orbs_j}
        phioffset = orbs_offset[(ai, ni, li)]
        psioffset = orbs_offset[(aj, nj, lj)]

        samples = block.samples.values.tolist()
        blockvalues = block.values

        for sample, blockval in zip(samples, blockvalues[:,:,:,0]):
            if blockval.numel() == 0:
                continue

            A, i, j, Tx, Ty, Tz = sample
            T = (Tx, Ty, Tz)
            mT = (-Tx, -Ty, -Tz)

            other_fac = 0.5 if (i == j and T != (0, 0, 0) and not all_pairs) else 1

            bt0_factor_p = 0.5 if not sort_orbs else (1 if not (ni == nj and li == lj) else 0.5)
            bt0_factor_m = bt0_factor_p * other_fac
            bt2_factor_p = 0.5 if not all_pairs else 1
            bt2_factor_m = bt2_factor_p * other_fac
            bt1_fact_fin = ISQRT_2 / fac * other_fac / (2 if all_pairs else 1)

            i_start, j_start = atom_blocks_idx[(A, i, j)]
            phi_end = shapes[(ni, li, nj, lj)][0]
            psi_end = shapes[(ni, li, nj, lj)][1]

            iphi_jpsi_slice = (slice(i_start + phioffset, i_start + phioffset + phi_end),
                               slice(j_start + psioffset, j_start + psioffset + psi_end))
            ipsi_jphi_slice = (slice(i_start + psioffset, i_start + psioffset + psi_end),
                               slice(j_start + phioffset, j_start + phioffset + phi_end))
            jphi_ipsi_slice = (slice(j_start + phioffset, j_start + phioffset + phi_end),
                               slice(i_start + psioffset, i_start + psioffset + psi_end))
            jpsi_iphi_slice = (slice(j_start + psioffset, j_start + psioffset + psi_end),
                               slice(i_start + phioffset, i_start + phioffset + phi_end))

            if A not in collected_data:
                collected_data[A] = {"T_values": [], "matrices": [], "slices": []}
            
            collected_data[A]["T_values"].append(T)
            collected_data[A]["matrices"].append((T, mT, blockval, block_type, bt0_factor_p, bt0_factor_m, bt2_factor_p, bt2_factor_m, bt1_fact_fin))
            collected_data[A]["slices"].append((iphi_jpsi_slice, ipsi_jphi_slice, jphi_ipsi_slice, jpsi_iphi_slice))

    return collected_data

def _initialize_matrices(collected_data, orbs_tot, dataset, device):
    reconstructed_matrices = []
    for A in range(len(dataset.structures)):
        if A not in collected_data:
            reconstructed_matrices.append(([], torch.zeros(0, 0, device=device)))
            continue

        norbs = np.sum([orbs_tot[ai] for ai in dataset.structures[A].numbers])
        T_values = collected_data[A]["T_values"]
        matrices = torch.zeros(len(T_values), norbs, norbs, device=device)
        reconstructed_matrices.append((T_values, matrices))
    
    return reconstructed_matrices

def _update_matrices(reconstructed_matrices, collected_data):
    for A, (T_values, matrices) in enumerate(reconstructed_matrices):
        if A not in collected_data:
            continue

        data = collected_data[A]
        for idx, (T, mT, blockval, block_type, bt0_factor_p, bt0_factor_m, bt2_factor_p, bt2_factor_m, bt1_fact_fin) in enumerate(data["matrices"]):
            iphi_jpsi_slice, ipsi_jphi_slice, jphi_ipsi_slice, jpsi_iphi_slice = data["slices"][idx]

            if block_type == 0:
                matrices[idx, iphi_jpsi_slice[0], iphi_jpsi_slice[1]].add_(blockval * bt0_factor_p)
                matrices[idx, jpsi_iphi_slice[0], jpsi_iphi_slice[1]].add_(blockval.T * bt0_factor_m)

            elif block_type == 2:
                matrices[idx, iphi_jpsi_slice[0], iphi_jpsi_slice[1]].add_(blockval * bt2_factor_p)
                matrices[idx, jpsi_iphi_slice[0], jpsi_iphi_slice[1]].add_(blockval.T * bt2_factor_m)

            elif abs(block_type) == 1:
                blockval.mul_(bt1_fact_fin)
                if block_type == 1:
                    matrices[idx, iphi_jpsi_slice[0], iphi_jpsi_slice[1]].add_(blockval)
                    matrices[idx, jphi_ipsi_slice[0], jphi_ipsi_slice[1]].add_(blockval)
                    matrices[idx, jpsi_iphi_slice[0], jpsi_iphi_slice[1]].add_(blockval.T)
                    matrices[idx, ipsi_jphi_slice[0], ipsi_jphi_slice[1]].add_(blockval.T)
                else:
                    matrices[idx, iphi_jpsi_slice[0], iphi_jpsi_slice[1]].add_(blockval)
                    matrices[idx, jphi_ipsi_slice[0], jphi_ipsi_slice[1]].sub_(blockval)
                    matrices[idx, jpsi_iphi_slice[0], jpsi_iphi_slice[1]].add_(blockval.T)
                    matrices[idx, ipsi_jphi_slice[0], ipsi_jphi_slice[1]].sub_(blockval.T)

def blocks_to_matrix_opt(blocks, dataset, device=None, cg=None, all_pairs=False, sort_orbs=True, detach=False, sample_id=None):
    if device is None:
        device = dataset.device

    if "L" in blocks.keys.names:
        blocks = _to_uncoupled_basis(blocks, cg=cg, device=device)

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

    collected_data = _collect_data(blocks, atom_blocks_idx, sort_orbs, all_pairs, orbs_offset, orbs_mult)
    reconstructed_matrices = _initialize_matrices(collected_data, orbs_tot, dataset, device)
    _update_matrices(reconstructed_matrices, collected_data)

    for A, (T_values, matrices) in enumerate(reconstructed_matrices):
        for idx, T in enumerate(T_values):
            mT = tuple(-t for t in T)
            assert torch.all(torch.isclose(matrices[idx] - matrices[idx].T, torch.zeros_like(matrices[idx]))), torch.norm(matrices[idx] - matrices[idx].T).item()

    return reconstructed_matrices
