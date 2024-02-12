import numpy as np
import warnings


def scidx_from_unitcell(frame, j=0, T=[0, 0, 0], kmesh=None):
    """Find index of atom j belonging to cell with translation vector T to its index in the supercell consistent with an SCF calc with kgrid =kmesh"""
    assert j < len(frame), "j must be less than the number of atoms in the unit cell"
    assert len(T) == 3, "T must be a 3-tuple"
    assert all(
        [t >= 0 for t in T]
    ), "T must be non-negative"  # T must be positive for the logic
    if kmesh is None:
        kmesh = np.asarray([1, 1, 1])
        warnings.warn("kmesh not specified, assuming 1x1x1")
    if isinstance(kmesh, int):
        kmesh = np.asarray([kmesh, kmesh, kmesh])
    else:
        assert len(kmesh) == 3, "kmesh must be a 3-tuple"

    N1, N2, N3 = kmesh
    natoms = len(frame)
    J = j + natoms * ((N3 * N2) * T[0] + (N3 * T[1]) + T[2])
    return J


def _position_in_translation(frame, j, T):
    """Return the position of atom j in unit cell translated by vector T"""
    return frame.positions[j] + np.dot((T[0], T[1], T[2]), frame.cell)


def scidx_to_mic_translation(frame, I=0, J=0, kmesh=[1, 1, 1], epsilon=1e-10):
    """Find the minimum image convention translation vector from atom I to atom J in the supercell of size kmesh"""
    assert frame.cell is not None, "Cell must be defined"
    cell_inv = np.linalg.inv(frame.cell.array.T)
    superframe = frame.repeat(kmesh)
    if J >= len(superframe):
        # print(J)
        J = J % len(superframe)
        warnings.warn(
            "J is greater than the number of atoms in the supercell. Mapping J to J % len(superframe)"
        )
    assert I < len(superframe) and J < len(
        superframe
    ), "I and J must be less than the number of atoms in the supercell"
    # this works correctly only when I<J - J should not be greater than I anyway as I always in 000 cell
    d = superframe.get_distance(I, J, mic=True, vector=True).T
    return np.floor(cell_inv @ d + epsilon).astype(
        int
    )  # adding noise to avoid numerical issues
    # This sometimes returns [-3,-2,0] for [1 2 0] which is correect based on the distance
