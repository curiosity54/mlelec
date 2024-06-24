import torch
from mlelec.data.pyscf_calculator import _instantiate_pyscf_mol
from pyscfad import ops
from mlelec.utils.twocenter_utils import unfix_orbital_order
from mlelec.data.dataset import MLDataset
from pyscfad.ml.scf import hf

def instantiate_mf(ml_data: MLDataset, fock_predictions=None, batch_indices=None):
    if fock_predictions is not None and len(batch_indices) != len(fock_predictions):
        warnings.warn("Converting shapes")
        fock_predictions = fock_predictions.reshape(1, *fock_predictions.shape)
    if fock_predictions is None:
        fock_predictions = torch.zeros_like(ml_data.target.tensor[batch_indices])

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
    return mfs, torch.stack(fockvar)

def compute_dipole_moment_from_mf(mfs, fock_vars, overlaps):
    '''compute dipole moment for each molecule in batch'''
    dipoles = []
    for i in range(len(mfs)):
        mf = mfs[i]
        fock = fock_vars[i]
        overlaps[i] = overlaps[i].to(fock)
        mo_energy, mo_coeff = mf.eig(fock, overlaps[i])
        mo_occ = mf.get_occ(mo_energy)  # get_occ returns a numpy array
        mo_occ = ops.convert_to_tensor(mo_occ)
        dm1 = mf.make_rdm1(mo_coeff, mo_occ)
        dip = mf.dip_moment(dm=dm1)
        dipoles.append(dip)
    return torch.stack(dipoles)

def compute_batch_dipole_moment(ml_data: MLDataset, batch_fockvars, batch_indices, mfs):
    '''Convert fock predictions back to pyscf order
    Compute dipole moment for each molecule in batch'''
    
    batch_frames = [ml_data.structures[i] for i in batch_indices]
    batch_fock = unfix_orbital_order(
        batch_fockvars, batch_frames, ml_data.molecule_data.aux_data["orbitals"]
    )
    batch_overlap = ml_data.molecule_data.aux_data["overlap"][batch_indices].to(
        batch_fock
    )
    batch_mfs = [mfs[i] for i in batch_indices]
    dipoles = compute_dipole_moment_from_mf(batch_mfs, batch_fock, batch_overlap)
    return dipoles

def compute_dipole_moment(frames, fock_predictions, overlaps):
    assert (
        len(frames) == len(fock_predictions) == len(overlaps)
    ), "Length of frames, fock_predictions, and overlaps must be the same"
    dipoles = []
    for i, frame in enumerate(frames):
        mol = _instantiate_pyscf_mol(frame)
        mf = hf.SCF(mol)
        fock = torch.autograd.Variable(
            fock_predictions[i].type(torch.float64), requires_grad=True
        )

        mo_energy, mo_coeff = mf.eig(fock, overlaps[i])
        mo_occ = mf.get_occ(mo_energy)  # get_occ returns a numpy array
        mo_occ = ops.convert_to_tensor(mo_occ)
        dm1 = mf.make_rdm1(mo_coeff, mo_occ)
        dip = mf.dip_moment(dm=dm1)
        dipoles.append(dip)
    return torch.stack(dipoles)
