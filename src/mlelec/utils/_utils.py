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
        for i, ai in enumerate(numbers):
            for j, aj in enumerate(numbers):
                atom_blocks_idx[(A, i, j)] = (ki_matrix[i, j], kj_matrix[i, j])
    
    return atom_blocks_idx

def _precompute_block_info(blocks, atom_blocks_idx, sort_orbs, all_pairs, orbs_offset, orbs_mult, detach):

    slices_info = {}

    for key, block in blocks.items():
        
        block_type = key["block_type"]
        ai, ni, li = key["species_i"], key["n_i"], key["l_i"]
        aj, nj, lj = key["species_j"], key["n_j"], key["l_j"]

        fac = 0.5 if (sort_orbs and ai == aj and ni == nj and li == lj) else 1
        # fac = 2

        orbs_i = orbs_mult[ai]
        orbs_j = orbs_mult[aj]

        shapes = {(k1 + k2): (orbs_i[tuple(k1)], orbs_j[tuple(k2)]) for k1 in orbs_i for k2 in orbs_j}
        phioffset = orbs_offset[(ai, ni, li)]
        psioffset = orbs_offset[(aj, nj, lj)]

        samples = block.samples.values.tolist()
        blockvalues = block.values[:,:,:,0].detach().clone() if detach else block.values[:,:,:,0]

        for sample, blockval in zip(samples, blockvalues):
            
            if blockval.numel() == 0:
                continue

            A, i, j, Tx, Ty, Tz = sample #[s.item() for s in sample]

            if A not in slices_info:
                slices_info[A] = []

            T = (Tx, Ty, Tz)
            mT = (-Tx, -Ty, -Tz)

            other_fac = 0.5 if (i == j and T != (0, 0, 0) and not all_pairs) else 1

            bt0_factor_p = 0.5 if not sort_orbs else (1 if not (ni == nj and li == lj) else 0.5)
            bt0_factor_m = bt0_factor_p * other_fac
            bt2_factor_p = 0.5 if not all_pairs else 1
            bt2_factor_m = bt2_factor_p * other_fac
            bt1_fact_fin = ISQRT_2 * fac * other_fac / (2 if all_pairs else 1)

            i_start, j_start = atom_blocks_idx[(A, i, j)]
            phi_end = shapes[(ni, li, nj, lj)][0]
            psi_end = shapes[(ni, li, nj, lj)][1]

            iphi = i_start + phioffset
            ipsi = i_start + psioffset
            jpsi = j_start + psioffset
            jphi = j_start + phioffset

            slice_iphi = slice(iphi, iphi + phi_end)
            slice_jpsi = slice(jpsi, jpsi + psi_end)
            slice_ipsi = slice(ipsi, ipsi + psi_end)
            slice_jphi = slice(jphi, jphi + phi_end)

            iphi_jpsi_slice = (slice_iphi, slice_jpsi)
            ipsi_jphi_slice = (slice_ipsi, slice_jphi)
            jphi_ipsi_slice = (slice_jphi, slice_ipsi)
            jpsi_iphi_slice = (slice_jpsi, slice_iphi)

            slices_info[A].append({"T": T, "mT": mT, "block_type": block_type, "blockval": blockval,
                                   "iphi_jpsi_slice": iphi_jpsi_slice, "ipsi_jphi_slice": ipsi_jphi_slice,
                                   "jphi_ipsi_slice": jphi_ipsi_slice, "jpsi_iphi_slice": jpsi_iphi_slice,
                                   "bt0_factor_p": bt0_factor_p, "bt0_factor_m": bt0_factor_m,
                                   "bt2_factor_p": bt2_factor_p, "bt2_factor_m": bt2_factor_m,
                                   "bt1_fact_fin": bt1_fact_fin})
            # slices_info.append({
            #     "A": A, "T": T, "mT": mT, "block_type": block_type, "blockval": blockval,
            #     "iphi_jpsi_slice": iphi_jpsi_slice, "ipsi_jphi_slice": ipsi_jphi_slice,
            #     "jphi_ipsi_slice": jphi_ipsi_slice, "jpsi_iphi_slice": jpsi_iphi_slice,
            #     "bt0_factor_p": bt0_factor_p, "bt0_factor_m": bt0_factor_m,
            #     "bt2_factor_p": bt2_factor_p, "bt2_factor_m": bt2_factor_m,
            #     "bt1_fact_fin": bt1_fact_fin
            # })

    return slices_info

def _initialize_matrices(reconstructed_matrices, slices_info, orbs_tot, dataset, device, structure_ids):
    initialized = set()
    for iA, A in enumerate(structure_ids):
        for info in slices_info[A]:
            T = info["T"]
            mT = info["mT"]
            if (A, T) not in initialized:
                norbs = np.sum([orbs_tot[ai] for ai in dataset.structures[A].numbers])
                reconstructed_matrices[iA][T] = torch.zeros(norbs, norbs, device=device)
                reconstructed_matrices[iA][mT] = torch.zeros(norbs, norbs, device=device)
                initialized.add((A, T))
                initialized.add((A, mT))

def _update_matrices(reconstructed_matrices, slices_info, structure_ids):
    for iA, A in enumerate(structure_ids):
        for info in slices_info[A]:

            T = info["T"]
            mT = info["mT"]
            block_type = info["block_type"]
            blockval = info["blockval"]
            iphi_jpsi_slice = info["iphi_jpsi_slice"]
            ipsi_jphi_slice = info["ipsi_jphi_slice"]
            jphi_ipsi_slice = info["jphi_ipsi_slice"]
            jpsi_iphi_slice = info["jpsi_iphi_slice"]
            bt0_factor_p = info["bt0_factor_p"]
            bt0_factor_m = info["bt0_factor_m"]
            bt2_factor_p = info["bt2_factor_p"]
            bt2_factor_m = info["bt2_factor_m"]
            bt1_fact_fin = info["bt1_fact_fin"]

            matrix_T = reconstructed_matrices[iA][T]
            matrix_mT = reconstructed_matrices[iA][mT]

            if block_type == 0:
                matrix_T[iphi_jpsi_slice].add_(blockval * bt0_factor_p)
                # matrix_mT[jpsi_iphi_slice].add_(blockval.T * bt0_factor_m)

            elif block_type == 2:
                matrix_T[iphi_jpsi_slice].add_(blockval * bt2_factor_p)
                # matrix_mT[jpsi_iphi_slice].add_(blockval.T * bt2_factor_m)

            elif abs(block_type) == 1:
                blockval.mul_(bt1_fact_fin)

                if block_type == 1:
                    matrix_T[iphi_jpsi_slice].add_(blockval)
                    # matrix_mT[jpsi_iphi_slice].add_(blockval.T)
                    matrix_T[ipsi_jphi_slice].add_(blockval.T)
                    # matrix_mT[jphi_ipsi_slice].add_(blockval)


                else:
                    matrix_T[iphi_jpsi_slice].add_(blockval)
                    # matrix_mT[jpsi_iphi_slice].add_(blockval.T)
                    matrix_T[ipsi_jphi_slice].sub_(blockval.T)
                    # matrix_mT[jphi_ipsi_slice].sub_(blockval)


def blocks_to_matrix_opt(blocks, 
                         dataset, 
                         structure_ids = None, 
                         device=None, 
                         cg=None, 
                         all_pairs=False, 
                         sort_orbs=True, 
                         detach=False, 
                        #  check_hermiticity=True,
                         slices_info = None):

    if device is None:
        device = dataset.device

    if structure_ids is None:
        structure_ids = range(len(dataset))

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

    reconstructed_matrices = [dict() for _ in structure_ids]

    if slices_info is None:
        slices_info = _precompute_block_info(blocks, atom_blocks_idx, sort_orbs, all_pairs, orbs_offset, orbs_mult, detach)
    _initialize_matrices(reconstructed_matrices, slices_info, orbs_tot, dataset, device, structure_ids)
    _update_matrices(reconstructed_matrices, slices_info, structure_ids)

    for A, matrix in enumerate(reconstructed_matrices):
        Ts = list(matrix.keys())
        done = []
        for T in Ts:
            if T in done:
                continue
            mT = tuple(-t for t in T)
            done.append(mT)
            reconstructed_matrices[A][T] = reconstructed_matrices[A][T] + reconstructed_matrices[A][mT].T
            reconstructed_matrices[A][mT] = reconstructed_matrices[A][T].T

    # if check_hermiticity:
    #     for A, matrix in enumerate(reconstructed_matrices):
    #         Ts = list(matrix.keys())
    #         for T in Ts:
    #             mT = tuple(-t for t in T)
    #             assert torch.all(torch.isclose(matrix[T] - reconstructed_matrices[A][mT].T, torch.zeros_like(matrix[T]))), torch.norm(matrix[T] - reconstructed_matrices[A][mT].T).item()

    return reconstructed_matrices
