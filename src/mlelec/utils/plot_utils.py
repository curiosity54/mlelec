import matplotlib.pyplot as plt
import numpy as np
import torch

# plt.rcParams['figure.dpi'] = 400
import scipy


def plot_atoms(structure, ax=None):
    pass

def print_matrix(matrix, fmt='{:15.10f}'):
    for row in matrix:
        print(" ".join(fmt.format(num) for num in row))

def block_matrix_norm(block, dataset):
    rij = {}
    Hij = {}
    for k, b in block.items():
        rij[*k.values.tolist()] = []
        Hij[*k.values.tolist()] = []
        frame_idx, I, J = b.samples.values[:, :3].T
        Ts = b.samples.values[:, -3:]
        for ifr, i, j, T, h in zip(frame_idx, I, J, Ts, b.values):
            rij[*k.values.tolist()].append(np.linalg.norm(dataset.structures[ifr].cell.array.T.dot(T) + dataset.structures[ifr].get_distance(i, j, mic = False, vector = True)))
            Hij[*k.values.tolist()].append(torch.norm(h))
    return rij, Hij

def matrix_norm(T, matrix, frame, nao):
    from skimage.util import view_as_blocks
    blocks = view_as_blocks(matrix.numpy(), (nao, nao))
    dist = []
    norms = []
    natm = frame.get_global_number_of_atoms()
    for i in range(natm):
        for j in range(natm):
            rij = np.linalg.norm(frame.cell.array.T.dot(T) + frame.get_distance(i, j, mic = False, vector = True)) #np.linalg.norm(frame.cell.T.dot(T) + frame.positions[j] - frame.positions[i])
            Hij = np.linalg.norm(blocks[i, j])
            dist.append(rij)
            norms.append(Hij)
    return np.array(dist).reshape(natm, natm), np.array(norms).reshape(natm, natm)

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
            ax.text(i, j, "{:.2e}".format(c), va="center", ha="center", fontsize=8)

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


def plot_block_errors(target_blocks, pred_blocks, plot_loss=False, ax=None):
    try:
        # coupled block
        x = [
            ",".join([str(lbl[i]) for i in [0, 2, 3, 5, 6, 7]])
            for lbl in target_blocks.keys.values.tolist()
        ]
    except:
        # uncoupled block
        x = [
            ",".join([str(lbl[i]) for i in [0, 2, 3, 5, 6]])
            for lbl in target_blocks.keys.values.tolist()
        ]
    fs = plt.rcParams["figure.figsize"]

    return_ax = False
    if ax is None:
        return_ax = True
        fig, ax = plt.subplots(figsize=(fs[0] * 3, fs[1]))

    ax_loss = ax.twinx()
    # s = (0,0,0)
    prediction_ = np.array([torch.linalg.norm(b.values).item() for b in pred_blocks])
    target_ = np.array([torch.linalg.norm(b.values).item() for b in target_blocks])
    loss_ = np.array(
        [
            (torch.linalg.norm(b.values - b1.values)).item()
            # (torch.linalg.norm(b.values - b1.values) ** 2).item()
            for b, b1 in zip(target_blocks, pred_blocks)
        ]
    )
    x_ = 3.5 * np.arange(len(prediction_))

    labels = []
    handles = []
    pl = ax.bar(x_, prediction_, label="pred", width=1, color="tab:blue")
    handles.append(pl)
    labels.append("Prediction")
    pl = ax.bar(x_ + 1, target_, alpha=1, label="target", width=1, color="tab:orange")
    handles.append(pl)
    labels.append("Target")
    if plot_loss:
        pl = ax_loss.bar(
            x_ + 2, loss_, alpha=1, label="target", width=1, color="tab:red"
        )
        handles.append(pl)
        labels.append("Loss")
        ax_loss.set_ylabel(r"$|H-\tilde{H}|^2$  (a.u.)")
        ax_loss.set_yscale("log")
    ax.set_ylim(1e-7, 1000)
    ax.set_xticks(3.5 * np.arange(len(prediction_)) + 3.5 / 3 - 0.5)
    ax.set_xticklabels(x, rotation=90)
    ax.legend(handles, labels, loc="best")
    ax.set_ylabel("|H| (a.u.)")

    ax_loss.set_ylim(1e-10)
    ax.set_yscale("log")

    if return_ax:
        return fig, ax, ax_loss
    
def parity_plot(target, prediction):

    pass


from mlelec.data.pyscf_calculator import translation_vectors_for_kmesh
from ase.dft.kpoints import sc_special_points as special_points  # , get_bandpath
from ase.units import Hartree
from typing import List, Optional
import scipy


def plot_bands_frame(
    dataset,
    ifr,
    realfock,
    realover,
    special_symm=None,
    bandpath_str=None,
    kpath=None,
    npoints=30,
    y_min=-2 * Hartree,
    y_max=2 * Hartree,
    ax=None,
    color="blue",
    ls="-",
    marker=None,
):
    from mlelec.utils.pbc_utils import inverse_fourier_transform

    """bloch"""
    # print(npoints, bandpath_str, pbc, special_points[special_symm])
    frame = dataset.structures[ifr]
    pyscf_cell = dataset.cells[ifr]
    kmesh = dataset.kmesh[ifr]
    if bandpath_str is not None:
        kpath = frame.cell.bandpath(
            path=bandpath_str,
            npoints=npoints,
            pbc=frame.pbc,
            special_points=special_points[special_symm],
        )
    else:
        assert kpath is not None
    kpts = kpath.kpts  # units of icell
    kpts_pyscf = pyscf_cell.get_abs_kpts(kpts)

    xcoords, special_xcoords, labels = kpath.get_linear_kpoint_axis()
    special_xcoords = np.array(special_xcoords) / xcoords[-1]
    xcoords = np.array(xcoords) / xcoords[-1]
    Nk = np.prod(kmesh)

    if isinstance(next(iter(realfock.values())), np.ndarray):

        Hk = []
        Sk = []
        for k in kpts:
            Hk.append(
                inverse_fourier_transform(
                    np.array(list(realfock.values())),
                    np.array(list(realfock.keys())),
                    k,
                )
            )
            Sk.append(
                inverse_fourier_transform(
                    np.array(list(realover.values())),
                    np.array(list(realover.keys())),
                    k,
                )
            )

    elif isinstance(next(iter(realfock.values())), torch.Tensor):

        Hk = []
        Sk = []
        for k in kpts:
            Hk.append(
                inverse_fourier_transform(
                    torch.stack(list(realfock.values())).detach(),
                    torch.from_numpy(np.array(list(realfock.keys()))),
                    k,
                )
            )
            Sk.append(
                inverse_fourier_transform(
                    torch.stack(list(realover.values())).detach(),
                    torch.from_numpy(np.array(list(realover.keys()))),
                    k,
                )
            )

    e_nk = []
    for n in range(len(kpts)):
        e_nk.append(scipy.linalg.eigvalsh(Hk[n], Sk[n]))

    vbmax = -99
    for en in e_nk:
        vb_k = en[pyscf_cell.nelectron // 2 - 1]
        if vb_k > vbmax:
            vbmax = vb_k
    e_nk = [en - vbmax for en in e_nk]
    emin = y_min
    emax = y_max
    fs = plt.rcParams["figure.figsize"]

    if ax is None:
        ax_was_none = True
        fig, ax = plt.subplots(figsize=(fs[0] * 0.8, fs[1] * 1.2))
    else:
        ax_was_none = False
    nbands = pyscf_cell.nao_nr()

    for n in range(nbands):
        ax.plot(
            xcoords, [e[n] * Hartree for e in e_nk], color=color, ls=ls, marker=marker
        )

    for p in special_xcoords:
        ax.plot([p, p], [emin, emax], "k-")
    # plt.plot([0, sp_points[-1]], [0, 0], 'k-')
    # plt.xticks(x, labels)
    ax.set_xticks(special_xcoords, labels)
    ax.axis(xmin=0, xmax=special_xcoords[-1], ymin=emin, ymax=emax)
    ax.set_xlabel(r"$\mathbf{k}$")

    ax.set_ylim(y_min, y_max)

    if ax_was_none:
        return fig, ax, [[e[n] * Hartree for e in e_nk] for n in range(nbands)]
    else:
        return ax, [[e[n] * Hartree for e in e_nk] for n in range(nbands)]


# def plot_bands_frame_(
#     frame,
#     realfock,
#     realover,
#     pyscf_cell,
#     kmesh,
#     R_vec_rel_in = None,
#     special_symm=None,
#     bandpath_str=None,
#     kpath=None,
#     npoints=30,
#     pbc=[True, True, False],
#     y_min=-2 * Hartree,
#     y_max=2 * Hartree,
#     ax=None,
#     color="blue",
#     ls = "-",
#     lw = 1,
#     marker=None,
#     factor = 1,
#     ):
#     """fourier"""

#     R_vec_rel = translation_vectors_for_kmesh(pyscf_cell, kmesh, R_vec_rel=R_vec_rel_in, return_rel=True, wrap_around=True)
#     R_vec_abs = translation_vectors_for_kmesh(pyscf_cell, kmesh, R_vec_rel=R_vec_rel_in, wrap_around=True)


#     mask_x = np.where(R_vec_rel[:, 0] == -kmesh[0] // 2)[0]
#     R_vec_rel[:, 0][mask_x] = kmesh[0] // 2
#     R_vec_abs[:, 0][mask_x] = -1 * R_vec_abs[:, 0][mask_x]

#     mask_y = np.where(R_vec_rel[:, 1] == -kmesh[1] // 2)[0]
#     R_vec_rel[:, 1][mask_y] = kmesh[1] // 2
#     R_vec_abs[:, 1][mask_y] = -1 * R_vec_abs[:, 1][mask_y]

#     # print(npoints, bandpath_str, pbc, special_points[special_symm])
#     if bandpath_str is not None:
#         kpath = frame.cell.bandpath(
#             path=bandpath_str,
#             npoints=npoints,
#             pbc=pbc,
#             special_points=special_points[special_symm],
#         )
#     else:
#         assert kpath is not None
#     kpts = kpath.kpts  # units of icell
#     # kpts_pyscf = pyscf_cell.get_abs_kpts(kpts)

#     xcoords, special_xcoords, labels = kpath.get_linear_kpoint_axis()
#     special_xcoords = np.array(special_xcoords) / xcoords[-1]
#     xcoords = np.array(xcoords) / xcoords[-1]
#     Nk = np.prod(kmesh)

#     # phase = np.exp(1j * np.dot(R_vec_abs, kpts_pyscf.T))
#     phase = np.exp(2j * np.pi * np.dot(R_vec_rel, kpts.T))
#     Hk = np.einsum("tk, tij ->kij", phase, realfock)
#     Sk = np.einsum("tk, tij ->kij", phase, realover)

#     e_nk = []
#     for n in range(len(kpts)):
#         e_nk.append(scipy.linalg.eigvalsh(Hk[n], Sk[n]))

#     vbmax = -99
#     for en in e_nk:
#         vb_k = en[pyscf_cell.nelectron // 2 - 1]
#         if vb_k > vbmax:
#             vbmax = vb_k
#     print(vbmax)
#     e_nk = [en - vbmax for en in e_nk]
#     emin = y_min
#     emax = y_max
#     fs = plt.rcParams["figure.figsize"]

#     if ax is None:
#         ax_was_none = True
#         fig, ax = plt.subplots(figsize=(fs[0] * 0.8, fs[1] * 1.2))
#     else:
#         ax_was_none = False
#     nbands = pyscf_cell.nao_nr()

#     bandplot = []
#     for n in range(nbands):
#         pl, = ax.plot(
#             xcoords, [factor * e[n] * Hartree for e in e_nk], color=color, ls=ls, marker=marker, lw = lw,
#         )
#         bandplot.append(pl)

#     for p in special_xcoords:
#         ax.plot([p, p], [emin, emax], "k-")
#     # plt.plot([0, sp_points[-1]], [0, 0], 'k-')
#     # plt.xticks(x, labels)
#     ax.set_xticks(special_xcoords, labels)
#     ax.axis(xmin=0, xmax=special_xcoords[-1], ymin=emin, ymax=emax)
#     ax.set_xlabel(r"$\mathbf{k}$")

#     ax.set_ylim(y_min, y_max)

#     if ax_was_none:
#         return fig, ax, [[e[n] * Hartree for e in e_nk] for n in range(nbands)], bandplot
#     else:
#         return ax, [[e[n] * Hartree for e in e_nk] for n in range(nbands)], bandplot

def plot_bands_frame_(
    frame,
    realfock,
    realover,
    pyscf_cell,
    kmesh,
    R_vec_rel,
    special_symm=None,
    bandpath_str=None,
    kpath=None,
    npoints=30,
    pbc=[True, True, False],
    y_min=-2 * Hartree,
    y_max=2 * Hartree,
    ax=None,
    color="blue",
    ls = "-",
    lw = 1,
    marker=None,
    factor = 1,
    ):
    """fourier"""

    # R_vec_rel = translation_vectors_for_kmesh(pyscf_cell, kmesh, R_vec_rel=R_vec_rel_in, return_rel=True, wrap_around=True)
    # R_vec_abs = translation_vectors_for_kmesh(pyscf_cell, kmesh, R_vec_rel=R_vec_rel_in, wrap_around=True)


    # mask_x = np.where(R_vec_rel[:, 0] == -kmesh[0] // 2)[0]
    # R_vec_rel[:, 0][mask_x] = kmesh[0] // 2
    # R_vec_abs[:, 0][mask_x] = -1 * R_vec_abs[:, 0][mask_x]

    # mask_y = np.where(R_vec_rel[:, 1] == -kmesh[1] // 2)[0]
    # R_vec_rel[:, 1][mask_y] = kmesh[1] // 2
    # R_vec_abs[:, 1][mask_y] = -1 * R_vec_abs[:, 1][mask_y]

    # print(npoints, bandpath_str, pbc, special_points[special_symm])
    if bandpath_str is not None:
        kpath = frame.cell.bandpath(
            path=bandpath_str,
            npoints=npoints,
            pbc=pbc,
            special_points=special_points[special_symm],
        )
    else:
        assert kpath is not None
    kpts = kpath.kpts  # units of icell
    # kpts_pyscf = pyscf_cell.get_abs_kpts(kpts)

    xcoords, special_xcoords, labels = kpath.get_linear_kpoint_axis()
    special_xcoords = np.array(special_xcoords) / xcoords[-1]
    xcoords = np.array(xcoords) / xcoords[-1]
    Nk = np.prod(kmesh)

    # phase = np.exp(1j * np.dot(R_vec_abs, kpts_pyscf.T))
    print(kpts.shape)
    phase = np.exp(2j * np.pi * np.einsum('ta,ka->tk', R_vec_rel, kpts))
    Hk = np.einsum("tk, tij ->kij", phase, realfock)
    Sk = np.einsum("tk, tij ->kij", phase, realover)

    e_nk = []
    for n in range(len(kpts)):
        e_nk.append(scipy.linalg.eigvalsh(Hk[n], Sk[n]))

    vbmax = -99
    for en in e_nk:
        vb_k = en[pyscf_cell.nelectron // 2 - 1]
        if vb_k > vbmax:
            vbmax = vb_k
    print(vbmax)
    e_nk = [en - vbmax for en in e_nk]
    emin = y_min
    emax = y_max
    fs = plt.rcParams["figure.figsize"]

    if ax is None:
        ax_was_none = True
        fig, ax = plt.subplots(figsize=(fs[0] * 0.8, fs[1] * 1.2))
    else:
        ax_was_none = False
    nbands = pyscf_cell.nao_nr()

    bandplot = []
    for n in range(nbands):
        pl, = ax.plot(
            xcoords, [factor * e[n] * Hartree for e in e_nk], color=color, ls=ls, marker=marker, lw = lw,
        )
        bandplot.append(pl)

    for p in special_xcoords:
        ax.plot([p, p], [emin, emax], "k-")
    # plt.plot([0, sp_points[-1]], [0, 0], 'k-')
    # plt.xticks(x, labels)
    ax.set_xticks(special_xcoords, labels)
    ax.axis(xmin=0, xmax=special_xcoords[-1], ymin=emin, ymax=emax)
    ax.set_xlabel(r"$\mathbf{k}$")

    ax.set_ylim(y_min, y_max)

    if ax_was_none:
        return fig, ax, [[e[n] * Hartree for e in e_nk] for n in range(nbands)], bandplot
    else:
        return ax, [[e[n] * Hartree for e in e_nk] for n in range(nbands)], bandplot


def plot_bands(
    frames,
    realfock,
    realover,
    pyscf_cell,
    kmesh: List,
    special_symm: Optional[List] = "hexagonal",
    bandpath_str="GMKG",
    npoints=30,
    pbc=[True, True, False],
    y_min=-2 * Hartree,
    y_max=2 * Hartree,
):
    # makes sense to plot multiple bamds with the same bandpath
    if isinstance(frames, List):
        if not isinstance(kmesh[0], List):
            kmesh = [kmesh for _ in range(len(frames))]
        assert len(realover) == len(realfock) == len(frames) == len(pyscf_cell)
    axes = []
    # fig, ax_ = plt.subplots()
    for ifr, frame in enumerate(frames):
        fig, ax = plot_bands_frame(
            frame,
            realfock[ifr],
            realover[ifr],
            pyscf_cell[ifr],
            kmesh[ifr],
            special_symm=special_symm,
            bandpath_str=bandpath_str,
            npoints=npoints,
            pbc=pbc,
            y_min=y_min,
            y_max=y_max,
        )
        axes.append(ax)
        # ax_.plot(ax.get_xticks(), ax.get_yticks(), "k-")

    return axes


## Trying to use Hanna's code - Not working as expected
# from mlelec.utils.twocenter_utils import compute_eigenval
# from mlelec.data.pyscf_calculator import translation_vectors_for_kmesh
# def translated_to_k_space(translated_matrices, kpoint, pyscf_cell, kmesh):
#     # must return one kpoint matrix corresponding to the kpoint
#     # each entry of fock_trans corresponds to the folllowing Ls
#     Ls = translation_vectors_for_kmesh(
#             pyscf_cell, kmesh, wrap_around=False, return_rel=False
#             )
#     kpoint = np.reshape(kpoint, (1, -1))
#     kL = np.dot(kpoint, Ls.T)
#     Nk = np.prod(kmesh)
#     expkL = 1/np.sqrt(Nk) * np.exp(1j * kL)
#     kmatrix = np.einsum("np,pab->nab",expkL, translated_matrices)[0]
#     # for kpt in range(len(kpoints)):
#     #     for i, t in enumerate(translated_matrices.keys()):
#     #         # for i in range(len(translated_matrices)):
#     #         kmatrix[kpt] += (
#     #             translated_matrices[t] * expkL[kpt][i]
#     #         )

#     return kmatrix
# def get_bandenergies(fock_trans,over_trans, kmesh, pyscf_cell, klist, npoints, eV=False):
#      '''Function that returns the bandenergies of the provided realspace fock, reals
#  oints per band.
#      '''
#      rev=[]
#      for band in range(int(len(klist)/npoints)):
#          evlist=[]
#          for kpt in range(npoints):
#             s_k= translated_to_k_space(over_trans, klist[kpt+band*npoints],  pyscf_cell, kmesh)
#             h_k= translated_to_k_space(fock_trans, klist[kpt+band*npoints],  pyscf_cell, kmesh)
#             print('hermiticity', np.linalg.norm(h_k - h_k.T.conj()))
#             ev = compute_eigenval(h_k, s_k, eV=eV)
#             evlist.append(ev)
#          rev.extend(evlist)
#      return rev


# def plot_bands(frame,fock,over, kmesh, pyscf_cell, symmpoints, symmpoint_names,npoints=50,energyshift=0, ymin=-30, ymax=30, to_eV=True, real_space=True):
#     ''' Function that plots the desired band structure given a structure (frame, its cell vector is used), Hamiltonian matrix (Fock), Overlap matrix (ovl), the symmetry points between which the bandstructure should be plotted, and the names of the symmetry_points. to_eV indicates the resulting eignenvals should be returned in eV.
#     real_space indicates the Hamiltonian and overlap matrices are in real space, and should be converted to k-space.
#     '''
#     def _get_xposition_of_symmpoints(frame, symmpoints):
#         '''Function that locates the symmetry points on the x-axis for plotting, needs the cell to compute the reciprocal lattice vectors whic
#     h are used to compute the desired band lengths on the plot
#         '''
#         reciprocal_latvec= frame.cell.reciprocal()*2*np.pi
#         lengths=[np.linalg.norm(np.dot(reciprocal_latvec,symmpoints[i+1]) - np.dot(reciprocal_latvec,symmpoints[i])) for i in range(len(symmpoints)-1)]
#         symmpoints_x=np.cumsum([0]+lengths)
#         return symmpoints_x

#     def _interpolate_kpoints(symmpoints, npoints):
#         '''Function that interpolates the kpoints between the provided symmetry points. Computes npoints between the kpoints
#         '''
#         klist=[]

#         for n in range(len(symmpoints)-1):
#             k=np.linspace(symmpoints[n],symmpoints[n+1], npoints)
#             for i in range(npoints):
#                 #klist.append(list(k[i]))
#                 klist.append(k[i])
#         return klist
#     symmpoints_x=_get_xposition_of_symmpoints(frame, symmpoints)

#     klist=_interpolate_kpoints(symmpoints, npoints)
#     xlist=_interpolate_kpoints(symmpoints_x, npoints)
#     rev=get_bandenergies(fock,over,kmesh, pyscf_cell,klist,npoints, eV=to_eV)

#     plt.rcParams['lines.linewidth'] = 1
#     ax_bands = plt.subplot(1,1,1)

#     ax_bands.plot(xlist,np.array(rev)+energyshift, '-b')
#    #ax_bands.plot(xlist,np.array(rev)+9, 'rx-')
#     #print(rev)

#     labels=[(symmpoints_x[i],symmpoint_names[i]) for i in range(len(symmpoints_x))]

#     tickx = []
#     tickl = []
#     for xpos,l in labels:
#         ax_bands.axvline(xpos,color='k',linestyle=":")
#         tickx += [ xpos ]
#         if len(l)>1:
#             if l=="Gamma":
#                l = "$\\"+l+"$"
#         tickl += [ l ]
#     for x, l in zip(tickx, tickl):
#         print("| %8.3f %s" % (x, repr(l)))

#     ax_bands.set_xlim(labels[0][0],labels[-1][0])
#     ax_bands.set_xticks(tickx)
#     ax_bands.set_xticklabels(tickl)
#     ax_bands.set_ylim(ymin,ymax)
