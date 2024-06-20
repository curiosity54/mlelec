import warnings

import ase
import torch
from pyscfad import numpy as pynp
from pyscfad import ops
from pyscfad.ml.scf import hf
from torch.autograd.functional import jacobian

from mlelec.data.dataset import MLDataset
from mlelec.data.pyscf_calculator import _instantiate_pyscf_mol
from mlelec.utils.twocenter_utils import (
    _lowdin_orthogonalize,
    isqrtm,
    isqrtp,
    unfix_orbital_order,
)


def compute_eigvals(ml_data, focks, batch_indices, orthogonal=True):
    """
    Compute the eigenvalues of the fock matrix. If orthogonal is True,
    the fock matrix is assumed to be orthogonal, else the fock matrix
    is orthogonalized using the overlap matrix.

    Parameters:
    -----------
    ml_data: MLDataset
        the dataset object
    focks: torch.tensor
        the fock matrices
    batch_indices: list
        the indices of the batch
    orthogonal: bool
        whether the fock matrices are orthogonal or not

    Returns:
    --------
    list of eigenvalues for each molecule in the batch.
    """
    eva = []
    if orthogonal:
        for i in range(len(focks)):
            eva.append(torch.linalg.eigvalsh(focks[i]))
    else:
        batch_frames = [ml_data.structures[i] for i in batch_indices]
        batch_fock = unfix_orbital_order(
            focks, batch_frames, ml_data.molecule_data.aux_data["orbitals"]
        )
        batch_overlap = ml_data.molecule_data.aux_data["overlap"][batch_indices]
        ortho_focks = [_lowdin_orthogonalize(f, torch.from_numpy(o))
                       for f, o in zip(batch_fock, batch_overlap)]
        for i in range(len(ortho_focks)):
            eva.append(torch.linalg.eigvalsh(ortho_focks[i]))
    return eva


def compute_dipole_moment(frames, focks, overlaps=None, orthogonal=True):
    """
    Compute the dipole moment for different frames given the fock matrices.
    If the fock matrices are not orthogonal, the overlap matrices must be provided.
    If the fock matrices are orthogonal, the MO coefficients should be converted
    back to the non-orthogonal basis before computing the density matrix to get the
    correct dipole moments.

    Parameters:
    -----------
    frames: list
        list of ase.Atoms objects
    focks: list
        array of fock matrices
    overlaps: list
        array of overlap matrices
    orthogonal: bool
        whether the fock matrices are orthogonal or not

    Returns:
    --------
    An array of dipole moments for each molecule in the batch.
    """
    if overlaps is not None:
        assert (
            len(frames) == len(focks) == len(overlaps)
        ), "Length of frames, fock_predictions, and overlaps must be the same"

    dipoles = []
    for i, frame in enumerate(frames):
        mol = _instantiate_pyscf_mol(frame)
        mf = hf.SCF(mol)
        focks[i] = torch.autograd.Variable(focks[i].type(torch.float64),
                                           requires_grad=True)
        if overlaps is not None:           
            mo_energy, mo_coeff = mf.eig(focks[i], overlaps[i])
        else:
            mo_energy, mo_coeff = mf.eig(focks[i], overlaps)
        mo_occ = mf.get_occ(mo_energy)  
        mo_occ = ops.convert_to_tensor(mo_occ)
        if orthogonal:
            ovlp = ops.convert_to_tensor(mol.intor("int1e_ovlp"))
            mo_coeff = isqrtm(ovlp) @ mo_coeff
        dm1 = mf.make_rdm1(mo_coeff, mo_occ)
        dip = mf.dip_moment(dm=dm1, unit="A.U.")

        dipoles.append(dip)
    return torch.stack(dipoles)


def compute_polarisability(frames, fock_predictions, overlaps=None, orthogonal=True):
    """
    Compute the polarisability for different frames given the fock matrices.
    
    """
    polarisability = []
    for i, frame in enumerate(frames):
        mol = _instantiate_pyscf_mol(frame)
        ao_dip = mol.intor("int1e_r", comp=3)
        ao_dip = ops.convert_to_tensor(ao_dip)
        mf = hf.SCF(mol)
        fock = fock_predictions[i]
        if overlaps is None:
            ovlp = torch.from_numpy(mol.intor("int1e_ovlp"))
            fock = torch.einsum("ij,jk,kl->il", isqrtp(ovlp),
                                fock, isqrtp(ovlp))
        else:
            ovlp = overlaps[i]

        def apply_perturb(E):
            p_fock = fock + pynp.einsum("x,xij->ij", E, ao_dip)
            mo_energy, mo_coeff = mf.eig(p_fock, ovlp)
            mo_occ = mf.get_occ(mo_energy)
            mo_occ = ops.convert_to_tensor(mo_occ)
            dm1 = mf.make_rdm1(mo_coeff, mo_occ)
            dip = mf.dip_moment(dm=dm1, unit="A.U.")
            return dip

        E = torch.zeros((3,), dtype=float)
        pol = jacobian(apply_perturb, E)
        polarisability.append(pol)

    return torch.stack(polarisability)


def instantiate_mf(ml_data: MLDataset, fock_predictions=None, batch_indices=None):
    if fock_predictions is not None and len(batch_indices) != len(fock_predictions):
        warnings.warn("Converting shapes")
        fock_predictions = fock_predictions.reshape(1, *fock_predictions.shape)
    if fock_predictions is None:
        fock_predictions = [
            torch.zeros_like(ml_data.target.tensor[i]) for i in batch_indices
        ]

    mfs = []
    fockvar = []
    for i, idx in enumerate(batch_indices):
        mol = _instantiate_pyscf_mol(ml_data.structures[idx])
        mf = hf.SCF(mol)
        fock = torch.autograd.Variable(
            fock_predictions[i].type(torch.float64), requires_grad=True
        )
        mfs.append(mf)
        fockvar.append(fock)
    return mfs, fockvar


def compute_dipole_moment_from_mf(mfs, fock_vars, overlaps=None, orthogonal=True):
    # compute dipole moment for each molecule in batch
    dipoles = []
    eigenvalues = []

    for i in range(len(mfs)):
        mf = mfs[i]
        fock = fock_vars[i]
        if overlaps is not None:
            ovlp = overlaps[i]          
            mo_energy, mo_coeff = mf.eig(fock, ovlp)
        else:
            mo_energy, mo_coeff = mf.eig(fock, overlaps)
        mo_occ = mf.get_occ(mo_energy)  
        mo_occ = ops.convert_to_tensor(mo_occ)
        if orthogonal:
            ovlp = ops.convert_to_tensor(mf.mol.intor("int1e_ovlp"))
            mo_coeff = isqrtm(ovlp) @ mo_coeff
        dm1 = mf.make_rdm1(mo_coeff, mo_occ)
        dip = mf.dip_moment(dm=dm1, unit="A.U.")

        dipoles.append(dip)
        eigenvalues.append(mo_energy)
    return torch.stack(dipoles), eigenvalues


def compute_batch_dipole_moment(ml_data: MLDataset, batch_fockvars, batch_indices, mfs):
    # Convert fock predictions back to pyscf order
    # Compute dipole moment for each molecule in batch
    batch_frames = [ml_data.structures[i] for i in batch_indices]
    batch_fock = unfix_orbital_order(
        batch_fockvars, batch_frames, ml_data.molecule_data.aux_data["orbitals"]
    )
    batch_overlap = [
        torch.from_numpy(ml_data.molecule_data.aux_data["overlap"][i])
        for i in batch_indices
    ]
    batch_mfs = [mfs[i] for i in batch_indices]
    dipoles, eigenvalues = compute_dipole_moment_from_mf(batch_mfs,
                                                         batch_fock, batch_overlap)
    return dipoles, eigenvalues


def compute_polarisability_from_mf(mfs, fock_vars, overlaps=None, orthogonal=True):
    # compute polarisability for each molecule in batch
    polarisability = []
    eigenvalues = []

    for i in range(len(mfs)):
        mf = mfs[i]
        ao_dip = mf.mol.intor("int1e_r", comp=3)
        ao_dip = ops.convert_to_tensor(ao_dip)
        fock = fock_vars[i]
        if overlaps is None:
            ovlp = torch.from_numpy(mf.mol.intor("int1e_ovlp"))
            fock = torch.einsum("ij,jk,kl->il", isqrtp(ovlp),
                                fock, isqrtp(ovlp))
        else:
            ovlp = overlaps[i]

        def apply_perturb(E):
            p_fock = fock + pynp.einsum("x,xij->ij", E, ao_dip)
            mo_energy, mo_coeff = mf.eig(p_fock, ovlp)
            mo_occ = mf.get_occ(mo_energy)
            mo_occ = ops.convert_to_tensor(mo_occ)
            dm1 = mf.make_rdm1(mo_coeff, mo_occ)
            dip = mf.dip_moment(dm=dm1, unit="A.U.")
            return dip

        eva, _ = mf.eig(fock, ovlp)
        E = torch.zeros((3,), dtype=float)
        pol = jacobian(apply_perturb, E, create_graph=True)
        polarisability.append(pol)
        eigenvalues.append(eva)

    return torch.stack(polarisability), eigenvalues


def compute_batch_polarisability(ml_data, batch_fockvars, batch_indices, mfs):

    batch_frames = [ml_data.structures[i] for i in batch_indices]
    batch_fock = unfix_orbital_order(
        batch_fockvars, batch_frames, ml_data.molecule_data.aux_data["orbitals"]
    )
    batch_overlap = [
        torch.from_numpy(ml_data.molecule_data.aux_data["overlap"][i])
        for i in batch_indices
    ]
    batch_mfs = [mfs[i] for i in batch_indices]
    polars, eigenvalues = compute_polarisability_from_mf(batch_mfs, batch_fock, batch_overlap)
    return polars, eigenvalues
