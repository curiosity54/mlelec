import os

import ase


os.environ["PYSCFAD_BACKEND"] = "torch"
import warnings

import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import torch
from pyscf import gto
from pyscfad import numpy as pynp
from pyscfad import ops
from pyscfad.ml.scf import hf
from torch.autograd.functional import jacobian

from mlelec.data.dataset import MLDataset
from mlelec.data.pyscf_calculator import _instantiate_pyscf_mol
from mlelec.utils.twocenter_utils import unfix_orbital_order

from mlelec.utils.twocenter_utils import _lowdin_orthogonalize

def compute_eigvals(ml_data, focks, batch_indices):
    batch_frames = [ml_data.structures[i] for i in batch_indices]
    batch_fock = unfix_orbital_order(
        focks, batch_frames, ml_data.molecule_data.aux_data["orbitals"]
    )
    batch_overlap = ml_data.molecule_data.aux_data["overlap"][batch_indices]
    ortho_focks = [_lowdin_orthogonalize(f, torch.from_numpy(o)) for f, o in zip(batch_fock, batch_overlap)]
    eva = []
    for i in range(len(focks)):
        eva.append(torch.linalg.eigvalsh(ortho_focks[i]))
    return eva


def dip_moment(
    fock: torch.tensor,
    overlap: torch.tensor,
    frame: ase.Atoms,
    E: torch.tensor = None,
    polarisability: bool = True,
):
    """
    compute the perturbed dipole moment of a molecule given the fock matrix
    and the electric field for a single frame.

    parameters:
    -----------
    fock: torch.tensor
        the fock matrix of the molecule
    E: torch.tensor
        the electric field vector

    returns:
    --------
    torch.tensor
        the perturbed dipole moment of the molecule
    """

    mol = _instantiate_pyscf_mol(frame)
    ao_dip = mol.intor("int1e_r", comp=3)
    ao_dip = ops.convert_to_tensor(ao_dip)
    mf = hf.SCF(mol)
    if polarisability:
        fock = fock + pynp.einsum("x,xij->ij", E, ao_dip)
    fock = torch.autograd.Variable(fock.type(torch.float64), requires_grad=True)
    mo_energy, mo_coeff = mf.eig(fock, overlap)
    mo_occ = mf.get_occ(mo_energy)  # get_occ returns a numpy array
    mo_occ = ops.convert_to_tensor(mo_occ)
    dm1 = mf.make_rdm1(mo_coeff, mo_occ)
    dip = mf.dip_moment(dm=dm1, unit="A.U.")
    return dip


def compute_dipole_moment(frames, fock_predictions, overlaps):

    assert (
        len(frames) == len(fock_predictions) == len(overlaps)
    ), "Length of frames, fock_predictions, and overlaps must be the same"
    dipoles = []
    for i, frame in enumerate(frames):
        dip = dip_moment(fock_predictions[i], overlaps[i], frame, polarisability=False)
        dipoles.append(dip)
    return torch.stack(dipoles)


def compute_polarisability(frames, fock_predictions, overlaps):
    polarisability = []
    for i in range(len(frames)):
        mol = _instantiate_pyscf_mol(frames[i])
        ao_dip = mol.intor("int1e_r", comp=3)
        ao_dip = ops.convert_to_tensor(ao_dip)
        mf = hf.SCF(mol)
        fock = fock_predictions[i]
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


def compute_dipole_moment_from_mf(mfs, fock_vars, overlaps):
    # compute dipole moment for each molecule in batch
    dipoles = []
    eigenvalues = []

    for i in range(len(mfs)):
        mf = mfs[i]
        fock = fock_vars[i]
        ovlp = overlaps[i]  # .to(fock)
        mo_energy, mo_coeff = mf.eig(fock, ovlp)
        mo_occ = mf.get_occ(mo_energy)  # get_occ returns a numpy array
        mo_occ = ops.convert_to_tensor(mo_occ)
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
    dipoles, eigenvalues = compute_dipole_moment_from_mf(batch_mfs, batch_fock, batch_overlap)
    return dipoles, eigenvalues


def compute_polarisability_from_mf(mfs, fock_vars, overlaps):
    # compute polarisability for each molecule in batch
    polarisability = []
    eigenvalues = []

    for i in range(len(mfs)):
        mf = mfs[i]
        fock = fock_vars[i]
        ovlp = overlaps[i]  # .to(fock)
        ao_dip = mf.mol.intor("int1e_r", comp=3)
        ao_dip = ops.convert_to_tensor(ao_dip)
        eva, _ = mf.eig(fock, ovlp)

        def apply_perturb(E):
            p_fock = fock + pynp.einsum("x,xij->ij", E, ao_dip)
            mo_energy, mo_coeff = mf.eig(p_fock, ovlp)
            mo_occ = mf.get_occ(mo_energy)
            mo_occ = ops.convert_to_tensor(mo_occ)
            dm1 = mf.make_rdm1(mo_coeff, mo_occ)
            dip = mf.dip_moment(dm=dm1, unit="A.U.")
            return dip

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









# import os

# import ase


# os.environ["PYSCFAD_BACKEND"] = "torch"
# import warnings

# import pyscf.pbc.tools.pyscf_ase as pyscf_ase
# import torch
# from pyscf import gto
# from pyscfad import numpy as pynp
# from pyscfad import ops
# from pyscfad.ml.scf import hf
# from torch.autograd.functional import jacobian

# from mlelec.data.dataset import MLDataset
# from mlelec.data.pyscf_calculator import _instantiate_pyscf_mol
# from mlelec.utils.twocenter_utils import unfix_orbital_order


# def dip_moment(
#     fock: torch.tensor,
#     overlap: torch.tensor,
#     frame: ase.Atoms,
#     E: torch.tensor = None,
#     polarisability: bool = True,
# ):
#     """
#     compute the perturbed dipole moment of a molecule given the fock matrix
#     and the electric field for a single frame.

#     parameters:
#     -----------
#     fock: torch.tensor
#         the fock matrix of the molecule
#     E: torch.tensor
#         the electric field vector

#     returns:
#     --------
#     torch.tensor
#         the perturbed dipole moment of the molecule
#     """

#     mol = _instantiate_pyscf_mol(frame)
#     ao_dip = mol.intor("int1e_r", comp=3)
#     ao_dip = ops.convert_to_tensor(ao_dip)
#     mf = hf.SCF(mol)
#     if polarisability:
#         fock = fock + pynp.einsum("x,xij->ij", E, ao_dip)
#     fock = torch.autograd.Variable(fock.type(torch.float64), requires_grad=True)
#     mo_energy, mo_coeff = mf.eig(fock, overlap)
#     mo_occ = mf.get_occ(mo_energy)  # get_occ returns a numpy array
#     mo_occ = ops.convert_to_tensor(mo_occ)
#     dm1 = mf.make_rdm1(mo_coeff, mo_occ)
#     dip = mf.dip_moment(dm=dm1, unit="A.U.")
#     return dip


# def compute_dipole_moment(frames, fock_predictions, overlaps):

#     assert (
#         len(frames) == len(fock_predictions) == len(overlaps)
#     ), "Length of frames, fock_predictions, and overlaps must be the same"
#     dipoles = []
#     for i, frame in enumerate(frames):
#         dip = dip_moment(fock_predictions[i], overlaps[i], frame, polarisability=False)
#         dipoles.append(dip)
#     return torch.stack(dipoles)


# def compute_polarisability(frames, fock_predictions, overlaps, E):

#     assert (
#         len(frames) == len(fock_predictions) == len(overlaps)
#     ), "Length of frames, fock_predictions, and overlaps must be the same"

#     polar = []
#     for i, frame in enumerate(frames):
#         _, pol = jacobian(dip_moment, (frame, fock_predictions[i], overlaps[i], E))
#         polar.append(pol)
#     return torch.stack(polar)


# def instantiate_mf(ml_data: MLDataset, fock_predictions=None, batch_indices=None):
#     if fock_predictions is not None and len(batch_indices) != len(fock_predictions):
#         warnings.warn("Converting shapes")
#         fock_predictions = fock_predictions.reshape(1, *fock_predictions.shape)
#     if fock_predictions is None:
#         fock_predictions = [
#             torch.zeros_like(ml_data.target.tensor[i]) for i in batch_indices
#         ]

#     mfs = []
#     fockvar = []
#     for i, idx in enumerate(batch_indices):
#         mol = _instantiate_pyscf_mol(ml_data.structures[idx])
#         mf = hf.SCF(mol)
#         fock = torch.autograd.Variable(
#             fock_predictions[i].type(torch.float64), requires_grad=True
#         )
#         mfs.append(mf)
#         fockvar.append(fock)
#     return mfs, fockvar


# def compute_dipole_moment_from_mf(mfs, fock_vars, overlaps):
#     # compute dipole moment for each molecule in batch
#     dipoles = []
#     eigenvalues = []
#     for i in range(len(mfs)):
#         mf = mfs[i]
#         fock = fock_vars[i]
#         ovlp = overlaps[i]  # .to(fock)
#         mo_energy, mo_coeff = mf.eig(fock, ovlp)
#         mo_occ = mf.get_occ(mo_energy)  # get_occ returns a numpy array
#         mo_occ = ops.convert_to_tensor(mo_occ)
#         dm1 = mf.make_rdm1(mo_coeff, mo_occ)
#         dip = mf.dip_moment(dm=dm1, unit="A.U.")
#         eigenvalues.append(mo_energy)
#         dipoles.append(dip)
#     return torch.stack(dipoles), eigenvalues


# def compute_batch_dipole_moment(ml_data: MLDataset, batch_fockvars, batch_indices, mfs):
#     # Convert fock predictions back to pyscf order
#     # Compute dipole moment for each molecule in batch
#     batch_frames = [ml_data.structures[i] for i in batch_indices]
#     batch_fock = unfix_orbital_order(
#         batch_fockvars, batch_frames, ml_data.molecule_data.aux_data["orbitals"]
#     )
#     batch_overlap = [torch.from_numpy(ml_data.molecule_data.aux_data["overlap"][i]) for i in batch_indices]
#     batch_mfs = [mfs[i] for i in batch_indices]
#     dipoles = compute_dipole_moment_from_mf(batch_mfs, batch_fock, batch_overlap)
#     return dipoles

# def compute_eva_from_mf(mfs, fock_vars, overlaps):
#     eigenvalues = []
#     for i in range(len(mfs)):
#         mf = mfs[i]
#         fock = fock_vars[i]
#         ovlp = overlaps[i]  # .to(fock)
#         mo_energy, mo_coeff = mf.eig(fock, ovlp)
#         eigenvalues.append(mo_energy)
#     return eigenvalues

# def compute_batch_eva(ml_data: MLDataset, batch_fockvars, batch_indices, mfs):
#     batch_frames = [ml_data.structures[i] for i in batch_indices]
#     batch_fock = unfix_orbital_order(
#         batch_fockvars, batch_frames, ml_data.molecule_data.aux_data["orbitals"]
#     )
#     batch_overlap = [torch.from_numpy(ml_data.molecule_data.aux_data["overlap"][i]) for i in batch_indices]
#     batch_mfs = [mfs[i] for i in batch_indices]
#     eva = compute_eva_from_mf(batch_mfs, batch_fock, batch_overlap)
#     return eva
    
# # def compute_polarisability(frames, E0):
# #     polarisabilities = []
# #     for i in range(len(frames)):
# #         mol = _instantiate_pyscf_mol(frames[i])
# #         mf = hf.SCF(mol)
# #         h1 = mf.get_hcore()
# #         ao_dip = mol.intor_symmetric("int1e_r", comp=3)

# #         mf.get_hcore = lambda *args, **kwargs: h1 + pynp.einsum('x,xij->ij', E0, ao_dip)
# #         polar =
# #         fock = fock_vars[i]
# #         overlaps[i] = overlaps[i].to(fock)
# #         mo_energy, mo_coeff = mf.eig(fock, overlaps[i])
# #         mo_occ = mf.get_occ(mo_energy)  # get_occ returns a numpy array
# #         mo_occ = ops.convert_to_tensor(mo_occ)
# #         dm1 = mf.make_rdm1(mo_coeff, mo_occ)
# #         polar = mf.polarisability(dm=dm1)
# #         polarisabilities.append(polar)
# #     return torch.stack(polarisabilities)
