import torch
import numpy as np
import math
from scipy.spatial.transform import Rotation

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def _wigner_d(l, alpha, beta, gamma):
    """Computes a Wigner D matrix
     D^l_{mm'}(alpha, beta, gamma)
    from sympy and converts it to numerical values.
    (alpha, beta, gamma) are Euler angles (radians, ZYZ convention) and l the irrep.
    """
    try:
        from sympy.physics.wigner import wigner_d
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Calculation of Wigner D matrices requires a sympy installation"
        )
    return torch.tensor(wigner_d(l, alpha, beta, gamma), dtype=torch.complex128)


def _r2c(sp):
    """Real to complex SPH. Assumes a block with 2l+1 reals corresponding
    to real SPH with m indices from -l to +l"""

    l = (len(sp) - 1) // 2  # infers l from the vector size
    rc = torch.zeros(len(sp), dtype=np.complex128)
    rc[l] = sp[l]
    for m in range(1, l + 1):
        rc[l + m] = (sp[l + m] + 1j * sp[l - m]) * I_SQRT_2 * (-1) ** m
        rc[l - m] = (sp[l + m] - 1j * sp[l - m]) * I_SQRT_2
    return rc


def _wigner_d_real(l, alpha, beta, gamma):
    r2c_mat = torch.hstack(
        [_r2c(np.eye(2 * l + 1)[i])[:, np.newaxis] for i in range(2 * l + 1)]
    )
    c2r_mat = np.conjugate(r2c_mat).T
    wig = _wigner_d(L, alpha, beta, gamma)
    return torch.real(c2r_mat @ np.conjugate(wig) @ r2c_mat)


def _cg(l1, l2, L):
    """Computes CG coefficients from sympy
    <l1 m1; l2 m2| L M>
    and converts them to numerical values.
    Returns a full (2 * l1 + 1, 2 * l2 + 1, 2 * L + 1) array, which
    is mostly zeros.
    """
    try:
        from sympy.physics.quantum.cg import CG
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Calculation of Clebsch-Gordan coefficients requires a sympy installation"
        )

    rcg = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    if np.abs(l1 - l2) > L or np.abs(l1 + l2) < L:
        return rcg
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            if np.abs(m1 + m2) > L:
                continue
            rcg[l1 + m1, l2 + m2, L + (m1 + m2)] += np.double(
                CG(l1, m1, l2, m2, L, m1 + m2).doit()
            )
    return rcg


I_SQRT_2 = 1.0 / np.sqrt(2)
SQRT_2 = np.sqrt(2)


def random_rotation(alpha, beta, gamma):
    """A Cartesian rotation matrix in the appropriate convention
    (ZYZ, implicit rotations) to be consistent with the common Wigner D definition.
    (alpha, beta, gamma) are Euler angles (radians)."""
    return torch.from_numpy(
        Rotation.from_euler("ZYZ", [alpha, beta, gamma]).as_matrix()
    )


def rotate_frame(frame, _rotation):
    """
    Utility function to also rotate a structure, given as an Atoms frame.
    NB: it will rotate positions and cell, and no other array.

    frame: ase.Atoms
        An atomic structure in ASE format, that will be modified in place

    Returns:
    -------
    The rotated frame.
    """
    frame = frame.copy()
    frame.positions = frame.positions @ _rotation.detach().numpy().T
    frame.cell = frame.cell @ _rotation.T
    return frame


def check_rotation_equivariance(frame, property_calculator, l: int = None):
    # should also implmenet when feature is passed instead of frame
    device = property_calculator.device
    alpha, beta, gamma = np.random.rand(3)
    _rotation = random_rotation(alpha, beta, gamma)
    rot_frame = rotate_frame(frame, _rotation)
    pred = property_calculator(frame.to(device))
    rot_pred = property_calculator(rot_frame.to(device))
    rotated_pred = _wigner_d_real(l, alpha, beta, gamma) @ pred
    assert torch.allclose(
        rotated_pred, rot_pred, atol=1e-6, rtol=1e-6
    ), "Rotation equivariance test failed"


def check_inversion_equivariance(x: torch.tensor, property_calculator):
    """
    x: torch.tensor, could be a frame or a feature
    """
    inv_x = -1 * x
    pred = property_calculator(x)
    inv_pred = property_calculator(inv_x)
    assert torch.allclose(
        pred, -1 * inv_pred, atol=1e-6, rtol=1e-6
    ), "Inversion equivariance test failed"


def xyz_to_spherical(data, axes=()):
    """
    Converts a vector (or a list of outer products of vectors) from
    Cartesian to l=1 spherical form. Given the definition of real
    spherical harmonics, this is just mapping (y, z, x) -> (-1,0,1)

    Automatically detects which directions should be converted

    data: array
        An array containing the data that must be converted

    axes: array_like
        A list of the dimensions that should be converted. If
        empty, selects all dimensions with size 3. For instance,
        a list of polarizabilities (ntrain, 3, 3) will convert
        dimensions 1 and 2.

    Returns:
        The array in spherical (l=1) form
    """
    shape = data.shape
    rdata = data
    # automatically detect the xyz dimensions
    if len(axes) == 0:
        axes = torch.where(torch.tensor(shape) == 3)[0]
    return torch.roll(data, -1, dims=axes)


def spherical_to_xyz(data, axes=()):
    """
    The inverse operation of xyz_to_spherical. Arguments have the
    same meaning, only it goes from l=1 to (x,y,z).
    """
    shape = data.shape
    rdata = data
    # automatically detect the l=1 dimensions
    if len(axes) == 0:
        axes = torch.where(torch.tensor(shape) == 3)[0]
    return torch.roll(data, 1, dims=axes)
