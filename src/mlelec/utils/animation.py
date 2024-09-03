import numpy as np
import torch
from mlelec.utils.symmetry import _rotation_matrix_from_angles, rotate_frame, _wigner_d_real
from mlelec.utils.pbc_utils import blocks_to_matrix
import metatensor.torch as mts
from ase.visualize.plot import plot_atoms
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation

def lissajous_euler(t, a=1.0, b=1.0, delta=np.pi/2, frequency=1.0, tilt_amplitude=np.pi/8, tilt_frequency=0.1):
    """
    Generate Euler angles to rotate a 3D vector along a smooth, periodic Lissajous-like trajectory.

    Parameters:
    - t: Time or parameter (float).
    - a: Amplitude of the Lissajous curve in the X direction (float).
    - b: Amplitude of the Lissajous curve in the Y direction (float).
    - delta: Phase difference between X and Y (float).
    - frequency: Base frequency of the Lissajous motion (float).
    - tilt_amplitude: The maximum tilt angle of the trajectory plane (float).
    - tilt_frequency: How fast the tilt axis precesses over time (float).

    Returns:
    - Euler angles (theta_z, theta_y, theta_x) in radians (tuple of floats).
    """

    # Parametric equations for a Lissajous curve
    x = a * np.sin(frequency * t + delta)
    y = b * np.sin(2 * frequency * t)
    z = np.sin(frequency * t)
    
    # Compute the tilt in the plane over time
    tilt_angle = tilt_amplitude * np.sin(tilt_frequency * t)
    
    # Apply the tilt to the trajectory
    x_tilted = x * np.cos(tilt_angle) - z * np.sin(tilt_angle)
    z_tilted = x * np.sin(tilt_angle) + z * np.cos(tilt_angle)
    
    # Tangent vector of the tilted trajectory
    dx = a * frequency * np.cos(frequency * t + delta)
    dy = 2 * b * frequency * np.cos(2 * frequency * t)
    dz = frequency * np.cos(frequency * t)
    
    dx_tilted = dx * np.cos(tilt_angle) - dz * np.sin(tilt_angle)
    dz_tilted = dx * np.sin(tilt_angle) + dz * np.cos(tilt_angle)
    
    # Calculate the Euler angles
    theta_z = np.arctan2(dy, dx_tilted)  # Rotation around Z axis
    theta_y = np.arctan2(dz_tilted, np.sqrt(dx_tilted**2 + dy**2))  # Rotation around Y axis
    theta_x = 0  # Rotation around X axis (can be modified if needed)
    
    return theta_z, theta_y, theta_x

def rotated_matrices(init_frame, blocks, idx, orbitals, func, params=np.linspace(0, 2 * np.pi, 200)):
    ang = [func(param) for param in params]
    rotations = [_rotation_matrix_from_angles(*a) for a in ang]
    rot_structures = [rotate_frame(init_frame, r) for r in rotations]

    rotated_tmaps = []
    for i, rot in enumerate(ang):
        r_blocks = []
        for k, b in blocks.items():
            L = k['L']
            b0 = mts.slice_block(b, axis='samples', labels=mts.Labels(["structure"], torch.tensor([idx]).reshape(1,1)))
            wd = _wigner_d_real(L, *rot)    
            block_values = torch.einsum('ab,sbp->sap', wd, b0.values)
            r_block = mts.TensorBlock(
                values = block_values,
                samples = b0.samples, 
                properties = b0.properties, 
                components = b0.components,
            )
            r_blocks.append(r_block)
        rotated_tmaps.append(
            mts.TensorMap(
                blocks.keys,
                r_blocks
            )
        )

    rot_H = []
    for i in range(len(rotated_tmaps)):
        rot_H.append(blocks_to_matrix(rotated_tmaps[i], orbitals, {0: rot_structures[i]}, detach=True, check_hermiticity=True))

    return rot_structures, rot_H

def animate_matrices(fig, axes, structures, matrices):
    def plot_frame(i):
        # Clear the axes
        for ax in axes:
            ax.clear()
            ax.set_aspect('equal')
            ax.set_axis_off()

        # Plot the first subplot
        ax = axes[0]
        plot_atoms(structures[i], ax=ax)  # Assuming plot_atoms is a user-defined function
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)

        # Plot the second subplot
        ax = axes[1]
        ax.matshow(np.abs(matrices[i]), norm=LogNorm(vmin=1e-5, vmax=10))

        # Adjust the layout
        fig.tight_layout()

    anim = FuncAnimation(fig, plot_frame, frames=len(structures), interval=200)
    return anim