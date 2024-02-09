import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['figure.dpi'] = 400


def plot_atoms(structure, ax=None):
    pass


def plot_hamiltonian(
    matrix,
    plot_abs=False,
    write_values=True,
    plot_structure=False,
    structure=None,
    tick_labels=None,
):
    # TODO: Support passing ticklabels or generate tick labels given basis set and structure

    if plot_structure:
        fig, (ax, ax_struc) = plt.subplots(
            ncols=2,
            figsize=(
                plt.rcParams["figure.figsize"][0] * 2,
                plt.rcParams["figure.figsize"][1],
            ),
        )
    else:
        fig, ax = plt.subplots()

    if plot_abs:
        m = np.abs(matrix)
    else:
        m = matrix

    mappable = ax.matshow(m)

    if write_values:
        ind_array = np.arange(m.shape[0])
        x, y = np.meshgrid(ind_array, ind_array)
        for i, j in zip(x.flatten(), y.flatten()):
            c = matrix[j, i]
            if np.abs(c) <= 1e-50:
                c = 0
            ax.text(i, j, "{:.2e}".format(c), va="center", ha="center", fontsize=4)

    ax.set_xticks(range(m.shape[0]))
    ax.set_yticks(range(m.shape[1]))
    if tick_labels is not None:
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
    else:
        ax.set_xticklabels(
            ["1s", "2s", "2py", "2pz", "2px", "1s", "2s", "2py", "2pz", "2px"]
        )
        ax.set_yticklabels(
            ["1s", "2s", "2py", "2pz", "2px", "1s", "2s", "2py", "2pz", "2px"]
        )

    if plot_structure and structure is not None:
        plot_atoms(structure, ax=ax_struc)
        ax_struc.set_axis_off()
    elif plot_structure and structure is None:
        raise ValueError(
            "An ASE structure must be provided when plot_structure == True."
        )

    return fig, ax, mappable
