import numpy as np
import torch
import xitorch
from xitorch.linalg import symeig


def compute_eigenvalues(As, Ms, return_eigenvectors=False):
    eigenvalues_list = []
    eigenvectors_list = []

    for ifr, (A, M) in enumerate(zip(As, Ms)):
        # shape = A.shape
        # leading_shape = shape[:-2]
        # indices = itertools.product(*[range(dim) for dim in leading_shape])

        # eigenvalues = torch.empty(leading_shape + (shape[-1],), dtype=torch.float64)
        # eigenvectors = torch.empty(leading_shape + (shape[-1], shape[-1]), dtype=torch.complex128) if return_eigenvectors else None

        Ax = xitorch.LinearOperator.m(A)
        Mx = xitorch.LinearOperator.m(M)

        eigenvalues, eigenvectors = symeig(Ax, M=Mx)

        # for index in indices:
        #     print(index)
        #     Ax = xitorch.LinearOperator.m(A[index])
        #     Mx = xitorch.LinearOperator.m(M[index]) if M is not None else None
        #     eigvals, eigvecs = symeig(Ax, M=Mx)
        #     eigenvalues[index] = eigvals
        #     if return_eigenvectors:
        #         eigenvectors[index] = eigvecs

        eigenvalues_list.append(eigenvalues)
        if return_eigenvectors:
            eigenvectors_list.append(eigenvectors)

    return (
        (eigenvalues_list, eigenvectors_list)
        if return_eigenvectors
        else eigenvalues_list
    )


def compute_atom_resolved_density(eigenvectors, frames, basis, ncore, overlaps=None):
    ard = []
    rhos = []

    use_S = overlaps is not None

    for i, (C, frame) in enumerate(zip(eigenvectors, frames)):
        ncore_val = sum(ncore[s] for s in frame.numbers)
        nelec = sum(frame.numbers) - ncore_val

        split_idx = [len(basis[s]) for s in frame.numbers]
        needed = len(np.unique(split_idx)) > 1
        max_dim = np.max(split_idx)

        occ = torch.tensor([2 if i < nelec // 2 else 0 for i in range(C.shape[-1])]).to(
            dtype=C.dtype
        )
        if use_S:
            S = overlaps[i]
            rho = torch.einsum("n,...in,...jn,...jk->ik...", occ, C, C.conj(), S)
        else:
            rho = torch.einsum("n,...in,...jn->ij...", occ, C, C.conj())

        slices = torch.split(rho, split_idx, dim=0)
        blocks = [torch.split(slice_, split_idx, dim=1) for slice_ in slices]
        blocks_flat = [block for sublist in blocks for block in sublist]

        if needed:
            squared_blocks = []
            for block in blocks_flat:
                pad_size = (0, max_dim - block.size(1), 0, max_dim - block.size(0))
                squared_block = torch.nn.functional.pad(block, pad_size, "constant", 0)
                squared_blocks.append(squared_block)
            blocks_flat = squared_blocks

        ard.append(
            torch.einsum("i...->...i", (torch.stack(blocks_flat) ** 2).sum(dim=(1, 2)))
        )
        # ard.append(torch.einsum('i...->...i', torch.stack(blocks_flat).norm(dim=(1,2))))
        rhos.append(torch.einsum("ij...->...ij", rho))

    return ard, rhos


import os

os.environ["PYSCFAD_BACKEND"] = "torch"
from pyscfad import ops
from pyscfad.ml.scf import hf as hf_ad

from mlelec.data.pyscf_calculator import _instantiate_pyscf_mol
from mlelec.utils.twocenter_utils import unfix_orbital_order


def compute_dipoles(
    focks,
    overlaps,
    mols=None,
    frames=None,
    basis=None,
    basis_name=None,
    requires_grad=True,
    unfix=True,
):
    if mols is not None:
        l = len(mols)
    else:
        assert (
            frames is not None and basis_name is not None
        ), "frames and basis_name must be provided when mols is None"
        l = len(frames)
    assert (
        l == len(focks) == len(overlaps)
    ), "Length of frames/mols, fock_predictions, and overlaps must be the same"

    dipoles = []
    mols = (
        [_instantiate_pyscf_mol(frame, basis=basis_name) for frame in frames]
        if mols is None
        else mols
    )
    device = focks[0].device

    if unfix:
        assert frames is not None, "frames are required when unfixing orbital order"
        assert basis is not None, "basis is required when unfixing orbital order"
        focks = unfix_orbital_order(focks, frames, basis)

    for H, S, mol in zip(focks, overlaps, mols):
        mf = hf_ad.SCF(mol)
        mo_energy, mo_coeff = mf.eig(H, S)
        # mo_energy, mo_coeff = symeig(xitorch.LinearOperator.m(H), M=xitorch.LinearOperator.m(S))
        # print(requires_grad)
        # print('mo_energy', mo_energy)
        # print('mo_coeff', mo_coeff)
        if not requires_grad:
            mo_energy = mo_energy.detach()
            mo_coeff = mo_coeff.detach()

        mo_energy = mo_energy.to(device=device)
        mo_coeff = mo_coeff.to(device=device)
        mo_occ = mf.get_occ(mo_energy)
        mo_occ = ops.convert_to_tensor(mo_occ).to(device=device)

        dm1 = mf.make_rdm1(mo_coeff, mo_occ).to(device=device)
        dip = mf.dip_moment(dm=dm1, verbose=0).to(device=device)

        if requires_grad:
            dipoles.append(dip)
        else:
            dipoles.append(dip.detach())

    try:
        return torch.stack(dipoles)
    except:
        return dipoles
