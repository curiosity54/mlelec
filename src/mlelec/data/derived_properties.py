import numpy as np
import torch
import itertools
import xitorch
from xitorch.linalg import symeig

def compute_eigenvalues(As, Ms, return_eigenvectors=False):
    eigenvalues_list = []
    eigenvectors_list = []

    for ifr, (A, M) in enumerate(zip(As, Ms)):
        shape = A.shape
        leading_shape = shape[:-2]
        indices = itertools.product(*[range(dim) for dim in leading_shape])

        eigenvalues = torch.empty(leading_shape + (shape[-1],), dtype=torch.float64)
        eigenvectors = torch.empty(leading_shape + (shape[-1], shape[-1]), dtype=torch.complex128) if return_eigenvectors else None

        for index in indices:
            Ax = xitorch.LinearOperator.m(A[index])
            Mx = xitorch.LinearOperator.m(M[index]) if M is not None else None
            eigvals, eigvecs = symeig(Ax, M=Mx)
            eigenvalues[index] = eigvals
            if return_eigenvectors:
                eigenvectors[index] = eigvecs

        eigenvalues_list.append(eigenvalues)
        if return_eigenvectors:
            eigenvectors_list.append(eigenvectors)

    return (eigenvalues_list, eigenvectors_list) if return_eigenvectors else eigenvalues_list

def compute_atom_resolved_density(eigenvectors, frames, basis, ncore):
    ard = []
    rhos = []

    for C, frame in zip(eigenvectors, frames):
        ncore_val = sum(ncore[s] for s in frame.numbers)
        nelec = sum(frame.numbers) - ncore_val

        split_idx = [len(basis[s]) for s in frame.numbers]
        needed = True if len(np.unique(split_idx)) > 1 else False
        max_dim = np.max(split_idx)

        occ = torch.tensor([2.0 + 0.0j if i < nelec // 2 else 0.0 + 0.0j for i in range(C.shape[-1])], dtype=torch.complex128)
        rho = torch.einsum('n,...in,...jn->ij...', occ, C, C.conj())

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

        ard.append(torch.einsum('i...->...i', torch.stack(blocks_flat).norm(dim=(1,2))))
        rhos.append(torch.einsum('ij...->...ij', rho))

    return ard, rhos
