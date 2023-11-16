import torch
import numpy as np
import math
from scipy.spatial.transform import Rotation
import wigners
from typing import Dict

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

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
    frame.cell = frame.cell @ _rotation.detach().numpy().T
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


def _wigner_d(L, alpha, beta, gamma):
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
    # print(wigner_d(L, alpha, beta, gamma))
    return torch.tensor(
        wigner_d(L, alpha, beta, gamma), dtype=torch.complex128
    ).reshape((2 * L + 1, 2 * L + 1))


def _r2c(sp):
    """Real to complex SPH. Assumes a block with 2l+1 reals corresponding
    to real SPH with m indices from -l to +l"""

    l = (len(sp) - 1) // 2  # infers l from the vector size
    rc = torch.zeros(len(sp), dtype=torch.complex128)
    rc[l] = sp[l]
    for m in range(1, l + 1):
        rc[l + m] = (sp[l + m] + 1j * sp[l - m]) * I_SQRT_2 * (-1) ** m
        rc[l - m] = (sp[l + m] - 1j * sp[l - m]) * I_SQRT_2
    return rc


def _real2complex(L: int):
    """
    Computes the transformation matrix that goes from a set
    of real spherical harmonics, ordered as:

        (l, -l), (l, -l + 1), ..., (l, l - 1), (l, l)
    to complex-valued ones(following the Condon-Shortley phase convention)

    _complex2real is obtained simply by taking the conjugate transpose of the matrix.
    """
    dim = 2 * L + 1
    mat = torch.zeros((dim, dim), dtype=torch.complex128)
    # m = 0
    mat[L, L] = 1.0

    if L == 0:
        return mat
    for m in range(1, L + 1):
        # m > 0
        mat[L + m, L + m] = I_SQRT_2 * (-1) ** m
        mat[L + m, L - m] = I_SQRT_2 * 1j * (-1) ** m
        # m < 0
        mat[L - m, L + m] = I_SQRT_2
        mat[L - m, L - m] = -I_SQRT_2 * 1j

    return mat


def _wigner_d_real(L, alpha, beta, gamma):
    r2c_mat = torch.hstack(
        [_r2c(torch.eye(2 * L + 1)[i])[:, None] for i in range(2 * L + 1)]
    )
    c2r_mat = torch.conj(r2c_mat).T
    wig = _wigner_d(L, alpha, beta, gamma)
    # print(c2r_mat.shape, wig.shape, r2c_mat.shape)
    return torch.real(c2r_mat @ torch.conj(wig) @ r2c_mat)


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


def _complex_clebsch_gordan_matrix(l1: int, l2: int, L: int, device):
    """
    Computes the Clebsch-Gordan (CG) matrix for
    transforming complex-valued spherical harmonics.
    The CG matrix is computed as a 3D array of elements
    < l1 m1 l2 m2 | L M >
    where the first axis loops over m1, the second loops over m2,
    and the third one loops over M. The matrix is real.

    For example, using the relation:

        | l1 l2 L M > = \sum_{m1, m2} <l1 m1 l2 m2 | L M > | l1 m1 > | l2 m2 >

    Args:
        l1: Order of the first set of spherical harmonics
        l2: Order of the second set of spherical harmonics
        L: Order of the coupled spherical harmonics
    Returns:
        real_cg: CG matrix for transforming complex-valued spherical harmonics
    """
    if abs(l1 - l2) > L or (l1 + l2) < L:
        return torch.zeros(
            (2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=torch.double, device=device
        )
    else:
        return torch.from_numpy(wigners.clebsch_gordan_array(l1, l2, L))


def _real_clebsch_gordan_matrix(l1: int, l2: int, L: int):
    """
    Compute the Clebsch Gordan (CG) matrix for *real* valued spherical harmonics,
    constructed by contracting the CG matrix for complex-valued
    spherical harmonics with the matrices that transform between
    real-valued and complex-valued spherical harmonics.

    Args:
        l1: Order of the first set of spherical harmonics
        l2: Order of the second set of spherical harmonics
        L: Order of the coupled spherical harmonics
    Returns:
        real_cg: CG matrix for transforming real-valued spherical harmonics
    """
    r2c = {}
    c2r = {}
    for L in range(abs(l1 - l2), l1 + l2 + 1):
        r2c[L] = _real2complex(L).to(self.device)
        c2r[L] = torch.conjugate(r2c[L]).T

    complex_cg = _complex_clebsch_gordan_matrix(l1, l2, L)
    real_cg = torch.einsum("ijk,il,jm,nk->lmn", complex_cg, r2c[l1], r2c[l2], c2r[L])

    if (l1 + l2 + L) % 2 == 0:
        return torch.real(real_cg)
    else:
        return torch.imag(real_cg)


class ClebschGordanReal:
    def __init__(self, l_max: int, device: str = None):
        self._l_max = l_max
        self._cg = {}
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # real-to-complex and complex-to-real transformations as matrices
        for l1 in range(self._l_max + 1):
            for l2 in range(self._l_max + 1):
                for L in range(
                    max(l1, l2) - min(l1, l2), min(self._l_max, (l1 + l2)) + 1
                ):
                    rcg = _real_clebsch_gordan_matrix(l1, l2, L)

                    # sparsify: take only the non-zero entries (indices
                    # of m1 and m2 components) for each M
                    new_cg = []
                    for M in range(2 * L + 1):
                        cg_nonzero = torch.where(abs(rcg[:, :, M]) > 1e-15)
                        cg_M = torch.zeros(
                            len(cg_nonzero[0]),
                            dtype=[("m1", ">i4"), ("m2", ">i4"), ("cg", ">f8")],
                            device=self.device,
                        )
                        cg_M["m1"] = cg_nonzero[0]
                        cg_M["m2"] = cg_nonzero[1]
                        cg_M["cg"] = rcg[cg_nonzero[0], cg_nonzero[1], M]
                        new_cg.append(cg_M)

                    self._cg[(l1, l2, L)] = new_cg

    def combine(
        self, y1: torch.tensor, y2: torch.tensor, L: int, combination_string: str
    ) -> torch.tensor:
        """
        Combines two spherical tensors y1, y2 into a single one.
        |Y; LM> = \sum_{m1,m2} <l1 m1 l2 m2|LM> |y1; l1 m1> |y2; l2 m2>
        """
        # automatically infer l1 and l2 from the size of the coefficients vectors
        l1 = (y1.shape[1] - 1) // 2
        l2 = (y2.shape[1] - 1) // 2
        if L > self._l_max or l1 > self._l_max or l2 > self._l_max:
            raise ValueError(
                "Requested CG entry ", (l1, l2, L), " has not been precomputed"
            )

        n_items = y1.shape[0]
        if y1.shape[0] != y2.shape[0]:
            raise IndexError(
                "Cannot combine feature blocks with different number of items"
            )

        # infers the shape of the output using the einsum internals
        ycombine = torch.einsum(combination_string, y1[:, 0, ...], y2[:, 0, ...]).shape
        Y = torch.zeros((n_items, 2 * L + 1) + ycombine[1:], device=self.device)

        if (l1, l2, L) in self._cg:
            for M in range(2 * L + 1):
                for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                    Y[:, M, ...] += torch.einsum(
                        combination_string, y1[:, m1, ...], y2[:, m2, ...] * cg
                    )

        return Y

    def couple(self, decoupled: Dict, iterate: int = 0) -> Dict:
        """
        Goes from an uncoupled product basis to a coupled basis.
        A (2l1+1)x(2l2+1) matrix transforming like the outer product of
        Y^m1_l1 Y^m2_l2 can be rewritten as a list of coupled vectors,
        each transforming like a Y^M_L.
        This transformation is accomplished through the following relation:

        |L M> = |l1 l2 L M> = \sum_{m1 m2} <l1 m1 l2 m2|L M> |l1 m1> |l2 m2>

        The process can be iterated: a D dimensional array that is the product
        of D Y^m_l can be turned into a set of multiple terms transforming as
        a single Y^M_L.

        Args:
            decoupled: (...)x(2l1+1)x(2l2+1) array containing coefficients that
                       transform like products of Y^l1 and Y^l2 harmonics.
                       Can also be called on a array of higher dimensionality,
                       in which case the result will contain matrices of entries.
                       If the further index also correspond to spherical harmonics,
                       the process can be iterated, and couple() can be called onto
                       its output, in which case the coupling is applied to each
                       entry.

            iterate: calls couple iteratively the given number of times.
                     Equivalent to:

                         couple(couple(... couple(decoupled)))

        Returns:
            coupled: A dictionary tracking the nature of the coupled objects.
                     When called one time, it returns a dictionary containing (l1, l2)
                     [the coefficients of the parent Ylm] which in turns is a
                     dictionary of coupled terms, in the form

                        L:(...)x(2L+1)x(...)

                    When called multiple times, it applies the coupling to each
                    term, and keeps track of the additional l terms, so that,
                    e.g., when called with iterate=1 the return dictionary contains
                    terms of the form

                        (l3,l4,l1,l2) : { L: array }

                    Note that this coupling scheme is different from the
                    NICE-coupling where angular momenta are coupled from
                    left to right as (((l1 l2) l3) l4)... )

                    Thus results may differ when combining more than two angular
                    channels.
        """

        coupled = {}

        # when called on a matrix, turns it into a dict form to which we can
        # apply the generic algorithm
        if not isinstance(decoupled, dict):
            l2 = (decoupled.shape[-1] - 1) // 2
            decoupled = {(): {l2: decoupled}}

        # runs over the tuple of (partly) decoupled terms
        for ltuple, lcomponents in decoupled.items():
            # each is a list of L terms
            for lc in lcomponents.keys():
                # this is the actual matrix-valued coupled term,
                # of shape (..., 2l1+1, 2l2+1), transforming as Y^m1_l1 Y^m2_l2
                dec_term = lcomponents[lc]
                l1 = (dec_term.shape[-2] - 1) // 2
                l2 = (dec_term.shape[-1] - 1) // 2

                # there is a certain redundance: the L value is also the last entry
                # in ltuple
                if lc != l2:
                    raise ValueError(
                        "Inconsistent shape for coupled angular momentum block."
                    )

                # in the new coupled term, prepend (l1,l2) to the existing label
                coupled[(l1, l2) + ltuple] = {}
                for L in range(
                    max(l1, l2) - min(l1, l2), min(self._l_max, (l1 + l2)) + 1
                ):
                    # ensure that Lterm is created on the same device as the dec_term
                    device = dec_term.device
                    Lterm = torch.zeros(
                        size=dec_term.shape[:-2] + (2 * L + 1,), device=device
                    )
                    for M in range(2 * L + 1):
                        for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                            Lterm[..., M] += dec_term[..., m1, m2] * cg
                    coupled[(l1, l2) + ltuple][L] = Lterm

        # repeat if required
        if iterate > 0:
            coupled = self.couple(coupled, iterate - 1)
        return coupled

    def decouple(self, coupled: Dict, iterate: int = 0) -> Dict:
        """
        Transform from coupled to uncoupled basis

        |l1 m1> |l2 m2> = \sum_{L M} <L M |l1 m1 l2 m2> |l1 l2 L M>
        """
        coupled.to(self.device)
        decoupled = {}
        # applies the decoupling to each entry in the dictionary
        for ltuple, lcomponents in coupled.items():
            # the initial pair in the key indicates the decoupled terms that generated
            # the L entries
            l1, l2 = ltuple[:2]

            # shape of the coupled matrix (last entry is the 2L+1 M terms)
            shape = next(iter(lcomponents.values())).shape[:-1]

            dec_term = torch.zeros(
                shape
                + (  # noqa
                    2 * l1 + 1,
                    2 * l2 + 1,
                ),
                device=self.device,
            )
            for L in range(max(l1, l2) - min(l1, l2), min(self._l_max, (l1 + l2)) + 1):
                # supports missing L components, e.g. if they are zero because of symmetry
                if L not in lcomponents:
                    continue
                for M in range(2 * L + 1):
                    for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                        dec_term[..., m1, m2] += cg * lcomponents[L][..., M]
            # stores the result with a key that drops the l's we have just decoupled
            if not ltuple[2:] in decoupled:
                decoupled[ltuple[2:]] = {}
            decoupled[ltuple[2:]][l2] = dec_term

        # rinse, repeat
        if iterate > 0:
            decoupled = self.decouple(decoupled, iterate - 1)

        # if we got a fully decoupled state, just return an array
        if ltuple[2:] == ():
            decoupled = next(iter(decoupled[()].values()))
        return decoupled
