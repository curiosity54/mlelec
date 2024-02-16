# ACDC style 1,2 centered features from rascaline
# depending on the target decide what kind of features must be computed

from rascaline import SphericalExpansion
from rascaline import SphericalExpansionByPair as PairExpansion
import ase

from metatensor import TensorMap, TensorBlock, Labels
import metatensor.operations as operations
import torch
import numpy as np
import warnings
from mlelec.utils.metatensor_utils import labels_where
from mlelec.features.acdc_utils import (
    acdc_standardize_keys,
    cg_increment,
    cg_combine,
    _pca,
    relabel_keys,
    fix_gij,
    drop_blocks_L,
    block_to_mic_translation,
    _standardize,
)
from typing import List, Optional, Union, Tuple
import tqdm

# TODO: use rascaline.clebsch_gordan.combine_single_center_to_nu when support for multiple centers is added

use_native = True  # True for rascaline


def single_center_features(
    frames, hypers, order_nu, lcut=None, cg=None, device=None, **kwargs
):
    calculator = SphericalExpansion(**hypers)
    rhoi = calculator.compute(frames, use_native_system=use_native)
    rhoi = rhoi.keys_to_properties(["species_neighbor"])
    # print(rhoi[0].samples)
    rho1i = acdc_standardize_keys(rhoi)
    # rho1i = _standardize(rho1i)
    if order_nu == 1:
        # return upto lcut? # TODO: change this maybe
        return drop_blocks_L(rho1i, lcut)
        # return rho1i
    if lcut is None:
        lcut = 10
    if cg is None:
        from mlelec.utils.symmetry import ClebschGordanReal

        L = max(lcut, hypers["max_angular"])
        cg = ClebschGordanReal(lmax=L, device=device)
    rho_prev = rho1i
    # compute nu order feature recursively
    for _ in range(order_nu - 2):
        rho_x = cg_combine(
            rho_prev,
            rho1i,
            clebsch_gordan=cg,
            lcut=lcut,
            other_keys_match=["species_center"],
            device=device,
        )
        rho_prev = _pca(
            rho_x, kwargs.get("npca", None), kwargs.get("slice_samples", None)
        )

    rho_x = cg_increment(
        rho_prev,
        rho1i,
        clebsch_gordan=cg,
        lcut=lcut,
        other_keys_match=["species_center"],
        feature_names=kwargs.get("feature_names", None),
    )
    if kwargs.get("pca_final", False):
        warnings.warn("PCA final features")
        rho_x = _pca(rho_x, kwargs.get("npca", None), kwargs.get("slice_samples", None))
    return rho_x


def pair_features(
    frames: List[ase.Atoms],
    hypers: dict,
    hypers_pair: dict = None,
    cg=None,
    rhonu_i: TensorMap = None,
    order_nu: Union[
        List[int], int
    ] = None,  # List currently not supported - useful when combining nu on i and nu' on j
    all_pairs: bool = False,
    both_centers: bool = False,
    lcut: int = 3,
    max_shift=None,
    device=None,
    kmesh=None,
    desired_shifts=None,
    mic=False,
    **kwargs,
):
    """
    hypers: dictionary of hyperparameters for the pair expansion as in Rascaline
    cg: object of utils:symmetry:ClebschGordanReal
    rhonu_i: TensorMap of single center features
    order_nu: int or list of int, order of the spherical expansion
    """
    if not isinstance(frames, list):
        frames = [frames]
    if lcut is None:
        lcut = 10
    if cg is None:
        from mlelec.utils.symmetry import ClebschGordanReal

        L = max(lcut, hypers["max_angular"])
        cg = ClebschGordanReal(lmax=L, device=device)
        # cg = ClebschGordanReal(lmax=lcut)
    if hypers_pair is None:
        hypers_pair = hypers
    calculator = PairExpansion(**hypers_pair)
    rho0_ij = calculator.compute(frames, use_native_system=use_native)
    factor = 1
    if all_pairs:
        if mic:
            factor = 1  # we only need half the cell - so we use half the cutoff
        hypers_allpairs = hypers_pair.copy()
        if max_shift is None and hypers["cutoff"] < np.max(
            [np.max(f.get_all_distances(mic=True)) / factor for f in frames]
        ):
            hypers_allpairs["cutoff"] = np.ceil(
                np.max([np.max(f.get_all_distances(mic=True)) / factor for f in frames])
            )
            # nmax = int(
            #     hypers_allpairs["max_radial"]
            #     / hypers["cutoff"]
            #     * hypers_allpairs["cutoff"]
            # )
            # hypers_allpairs["max_radial"] = nmax
        elif max_shift is not None:
            repframes = [f.repeat(max_shift) for f in frames]
            hypers_allpairs["cutoff"] = np.ceil(
                np.max(
                    [np.max(f.get_all_distances(mic=True)) / factor for f in repframes]
                )
            )
            warnings.warn(
                f"Using cutoff {hypers_allpairs['cutoff']} for all pairs feature"
            )
        else:
            warnings.warn(f"Using unchanged hypers for all pairs feature")
        print("hypers_pair", hypers_allpairs)
        calculator_allpairs = PairExpansion(**hypers_allpairs)

        rho0_ij = calculator_allpairs.compute(frames, use_native_system=use_native)

    # rho0_ij = acdc_standardize_keys(rho0_ij)
    rho0_ij = fix_gij(rho0_ij)
    rho0_ij = acdc_standardize_keys(rho0_ij)
    # ----- MIC mapping ------
    if mic:
        from mlelec.utils.pbc_utils import scidx_from_unitcell, scidx_to_mic_translation

        # generate all the translations and assign values based on mic map
        assert kmesh is not None, "kmesh must be specified for MIC mapping"
        # we should add the missing samples
        warnings.warn(f"Using kmesh {kmesh} for MIC mapping")
        blocks = []

        for key, block in rho0_ij.items():
            all_frIJ = np.unique(block.samples.values[:, :3], axis=0)
            value_indices = []
            fixed_sample = []
            all_samples = [
                (*a, x, y, z)
                for a in all_frIJ
                for x in range(kmesh[0])
                for y in range(kmesh[1])
                for z in range(kmesh[2])
            ]
            for s in all_samples:
                ifr, i, j, x, y, z = s

                # if [x, y, z] == [0, 0, 0]:
                # continue

                # print('------')
                # J = scidx_from_unitcell(frame, j=j, T=[x, y, z], kmesh=kmesh)
                # print(i, j, J,[x,y,z])
                mic_x, mic_y, mic_z = scidx_to_mic_translation(
                    frames[ifr],
                    I=i,
                    J=scidx_from_unitcell(
                        frames[ifr], j=j, T=[x, y, z], kmesh=kmesh
                    ),  # TODO - kmesh[ifr] - nonunofrm kmesh across structrues
                    j=j,
                    kmesh=kmesh,
                )
                mic_label = Labels(
                    [
                        "structure",
                        "center",
                        "neighbor",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                    ],
                    values=np.asarray(
                        [
                            ifr,
                            i,
                            j,
                            mic_x,
                            mic_y,
                            mic_z,
                        ]
                    ).reshape(1, -1),
                )[0]
                mappedidx = block.samples.position(mic_label)
                # _, mappedidx = labels_where(
                #         block.samples,
                #         Labels(
                #             [
                #                 "structure",
                #                 "center",
                #                 "neighbor",
                #                 "cell_shift_a",
                #                 "cell_shift_b",
                #                 "cell_shift_c",
                #             ],
                #             values=np.asarray(
                #                 [
                #                     ifr,
                #                     i,
                #                     j,
                #                     mic_x,
                #                     mic_y,
                #                     mic_z,
                #                 ]
                #             ).reshape(1, -1),
                #         ),
                #         return_idx=True,
                #     )
                assert isinstance(mappedidx, int), (mappedidx, mic_label)
                value_indices.append(mappedidx)
                fixed_sample.append(
                    [
                        ifr,
                        i,
                        j,
                        x,
                        y,
                        z,
                        mic_x,
                        mic_y,
                        mic_z,
                    ]
                )

                if [x, y, z] != [0, 0, 0]:
                    mic_label = Labels(
                        [
                            "structure",
                            "center",
                            "neighbor",
                            "cell_shift_a",
                            "cell_shift_b",
                            "cell_shift_c",
                        ],
                        values=np.asarray(
                            [
                                ifr,
                                j,
                                i,
                                -mic_x,
                                -mic_y,
                                -mic_z,
                            ]
                        ).reshape(1, -1),
                    )[0]
                    mappedidx = block.samples.position(mic_label)

                    assert isinstance(mappedidx, int), (mappedidx, mic_label)
                    value_indices.append(mappedidx)
                    fixed_sample.append(
                        [
                            ifr,
                            j,
                            i,
                            -x,
                            -y,
                            -z,
                            -mic_x,
                            -mic_y,
                            -mic_z,
                        ]
                    )

            blocks.append(
                TensorBlock(
                    values=block.values[value_indices],
                    samples=Labels(
                        block.samples.names
                        + ["cell_shift_a_MIC", "cell_shift_b_MIC", "cell_shift_c_MIC"],
                        np.asarray(fixed_sample),
                    ),
                    components=block.components,
                    properties=block.properties,
                )
            )

        rho0_ij = TensorMap(keys=rho0_ij.keys, blocks=blocks)
        # rho0_ij = _standardize(rho0_ij)
        # rho0_ij =  _pca(rho0_ij)
        # ----OLD-----
        # blocks = []
        # for key, block in rho0_ij.items():
        #     samples, retained_idx, validx = block_to_mic_translation(
        #         frames[0], block, kmesh  # <<<< ASSUMES UNIFORM KMESH ACROSS STRUCTURES
        #     )

        #     blocks.append(
        #         TensorBlock(
        #             values=block.values[validx],
        #             components=block.components,
        #             properties=block.properties,
        #             samples=samples,
        #         ),
        #     )

        # rho0_ij = TensorMap(keys=rho0_ij.keys, blocks=blocks)
        # ----OLD ----
    # keep only the desired translations
    if desired_shifts is not None:

        blocks = []

        for i, (k, b) in enumerate(rho0_ij.items()):

            slab, sidx = labels_where(
                b.samples,
                selection=Labels(
                    names=["cell_shift_a", "cell_shift_b", "cell_shift_c"],
                    values=np.array(desired_shifts[:]).reshape(-1, 3),
                ),
                return_idx=True,
            )  # only retain tranlsations that we want - DONT SKIP
            # slab, sidx = labels_where(b.samples, selection=Labels(names=["cell_shift_a", "cell_shift_b", "cell_shift_c"], values=np.array(desired_shifts1[:]).reshape(-1,3)), return_idx=True) # only retain tranlsations that we want - DONT SKIP

            blocks.append(
                TensorBlock(
                    values=b.values[sidx],
                    components=b.components,
                    samples=Labels(
                        names=b.samples.names, values=np.asarray(b.samples.values[sidx])
                    ),
                    properties=b.properties,
                )
            )
        rho0_ij = TensorMap(keys=rho0_ij.keys, blocks=blocks)
    # ------------------------

    if isinstance(order_nu, list):
        assert (
            len(order_nu) == 2
        ), "specify order_nu as [nu_i, nu_j] for correlation orders for i and j respectively"
        order_nu_i, order_nu_j = order_nu
    else:
        assert isinstance(order_nu, int), "specify order_nu as int or list of 2 ints"
        order_nu_i = order_nu

    if not (frames[0].pbc.any()):
        for _ in ["cell_shift_a", "cell_shift_b", "cell_shift_c"]:
            rho0_ij = operations.remove_dimension(rho0_ij, axis="samples", name=_)

    # must compute rhoi as sum of rho_0ij
    if rhonu_i is None:
        rhonu_i = single_center_features(
            frames, order_nu=order_nu_i, hypers=hypers, lcut=lcut, cg=cg, kwargs=kwargs
        )
        # rhonu_i = _standardize(rhonu_i)
    if not both_centers:
        rhonu_ij = cg_combine(
            rhonu_i,
            rho0_ij,
            clebsch_gordan=cg,
            other_keys_match=["species_center"],
            lcut=lcut,
            feature_names=kwargs.get("feature_names", None),
        )
        return rhonu_ij

    else:
        # # build the feature with atom-centered density on both centers
        # # rho_ij = rho_i x gij x rho_j
        if "order_nu_j" not in locals():
            warnings.warn("nu_j not defined, using nu_i for nu_j as well")
            order_nu_j = order_nu_i
        if order_nu_j != order_nu_i:
            # compire
            rhonup_j = single_center_features(
                frames,
                order_nu=order_nu_j,
                hypers=hypers,
                lcut=lcut,
                cg=cg,
                kwargs=kwargs,
            )
        else:
            rhonup_j = rhonu_i.copy()

        rhoj = relabel_keys(rhonup_j, "species_neighbor")
        # build rhoj x gij
        rhonuij = cg_combine(
            rhoj,
            rho0_ij,
            lcut=lcut,
            other_keys_match=["species_neighbor"],
            clebsch_gordan=cg,
            mp=True,  # for combining with neighbor
        )
        # combine with rhoi
        rhonu_nupij = cg_combine(
            rhonu_i,
            rhonuij,
            lcut=lcut,
            other_keys_match=["species_center"],
            clebsch_gordan=cg,
            feature_names=kwargs.get("feature_names", None),
        )

        return rhonu_nupij


# TODO: MP feature
# elements = np.unique(frames[0].numbers)#np.unique(np.hstack([f.numbers for f in frames]))
# rhoMPi = contract_rho_ij(rhonu_nuijp, elements, rho(NU=nu+nu'+1)i.property_names)
# print("MPi computed")

# rhoMPij = cg_increment(rhoMPi, rho0_ij, lcut=lcut, other_keys_match=["species_center"], clebsch_gordan=cg)


def twocenter_hermitian_features(
    single_center: TensorMap,
    pair: TensorMap,
) -> TensorMap:
    # keep this function only for molecules - hanldle 000 shift using hermitian PBC - MIC mapping #FIXME
    # actually special class of features for Hermitian (rank2 tensor)
    keys = []
    blocks = []
    for k, b in single_center.items():
        keys.append(
            tuple(k)
            + (
                k["species_center"],
                0,
            )
        )
        # `Try to handle the case of no computed features
        if len(list(b.samples.values)) == 0:
            samples_array = b.samples
        else:
            samples_array = np.asarray(b.samples.values)
            samples_array = np.hstack([samples_array, samples_array[:, -1:]])
        blocks.append(
            TensorBlock(
                samples=Labels(
                    names=b.samples.names + ["neighbor"],
                    values=samples_array,
                ),
                components=b.components,
                properties=b.properties,
                values=b.values,
            )
        )

    for k, b in pair.items():
        if k["species_center"] == k["species_neighbor"]:
            # off-site, same species
            idx_up = np.where(b.samples["center"] < b.samples["neighbor"])[0]
            if len(idx_up) == 0:
                continue
            idx_lo = np.where(b.samples["center"] > b.samples["neighbor"])[0]

            # we need to find the "ji" position that matches each "ij" sample.
            # we exploit the fact that the samples are sorted by structure to do a "local" rearrangement
            smp_up, smp_lo = 0, 0
            for smp_up in range(len(idx_up)):
                # ij = b.samples[idx_up[smp_up]][["center", "neighbor"]]
                ij = b.samples.view(["center", "neighbor"]).values[idx_up[smp_up]]
                for smp_lo in range(smp_up, len(idx_lo)):
                    ij_lo = b.samples.view(["neighbor", "center"]).values[
                        idx_lo[smp_lo]
                    ]
                    # ij_lo = b.samples[idx_lo[smp_lo]][["neighbor", "center"]]
                    if (
                        b.samples["structure"][idx_up[smp_up]]
                        != b.samples["structure"][idx_lo[smp_lo]]
                    ):
                        raise ValueError(
                            f"Could not find matching ji term for sample {b.samples[idx_up[smp_up]]}"
                        )
                    if tuple(ij) == tuple(ij_lo):
                        idx_lo[smp_up], idx_lo[smp_lo] = idx_lo[smp_lo], idx_lo[smp_up]
                        break

            keys.append(tuple(k) + (1,))
            keys.append(tuple(k) + (-1,))

            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names,
                        values=np.asarray(b.samples.values[idx_up]),
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=(b.values[idx_up] + b.values[idx_lo]) / np.sqrt(2),
                )
            )
            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names,
                        values=np.asarray(b.samples.values[idx_up]),
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=(b.values[idx_up] - b.values[idx_lo]) / np.sqrt(2),
                )
            )
        elif k["species_center"] < k["species_neighbor"]:
            # off-site, different species
            keys.append(tuple(k) + (2,))
            blocks.append(b.copy())
    keys = np.pad(keys, ((0, 0), (0, 6)))
    return TensorMap(
        keys=Labels(
            names=pair.keys.names
            + ["block_type"]
            + ["cell_shift_a", "cell_shift_b", "cell_shift_c"]
            + ["cell_shift_a_MIC", "cell_shift_b_MIC", "cell_shift_c_MIC"],
            values=np.asarray(keys, dtype=np.int32),
        ),
        blocks=blocks,
    )


def twocenter_features_periodic_NH(
    single_center: TensorMap, pair: TensorMap, shift=None
) -> TensorMap:
    # no hermitian symmetry
    if shift == [0, 0, 0]:
        return twocenter_hermitian_features(single_center, pair)

    keys = []
    blocks = []

    for k, b in single_center.items():
        keys.append(
            tuple(k)
            + (
                k["species_center"],
                0,
            )
        )
        # `Try to handle the case of no computed features
        if len(list(b.samples.values)) == 0:
            samples_array = b.samples
        else:
            samples_array = np.asarray(b.samples.values)
            samples_array = np.hstack([samples_array, samples_array[:, -1:]])
        blocks.append(
            TensorBlock(
                samples=Labels(
                    names=b.samples.names
                    + [
                        "neighbor",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                        "cell_shift_a_MIC",
                        "cell_shift_b_MIC",
                        "cell_shift_c_MIC",
                    ],
                    values=np.pad(samples_array, ((0, 0), (0, 6))),
                ),
                components=b.components,
                properties=b.properties,
                values=b.values,
            )
        )

    # PAIRS SHOULD NOT CONTRIBUTE to BLOCK TYPE 0
    for k, b in pair.items():
        if k["species_center"] == k["species_neighbor"]:  # self translared pairs
            # idx = []
            idx = np.where(
                (b.samples["center"] == b.samples["neighbor"])
                & (b.samples["cell_shift_a"] != 0)
                & (b.samples["cell_shift_b"] != 0)
                & (b.samples["cell_shift_c"] != 0)
            )[0]
            # idx = [
            #     i
            #     for i in idxij
            #     if b.samples["cell_shift_a"][i] != 0
            #     and b.samples["cell_shift_b"][i] != 0
            #     and b.samples["cell_shift_c"][i] != 0
            # ]

            if len(idx) == 0:
                # SHOULD BE ZERO
                print(k.values, "skipped for btype0")
                continue
            keys.append(tuple(k) + (0,))
            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names,
                        values=np.asarray(b.samples.values[idx]),
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=b.values[idx],
                )
            )
        else:
            raise NotImplementedError  # Handle periodic case for different species

    for k, b in pair.items():
        if k["species_center"] == k["species_neighbor"]:
            # off-site, same species
            idx_up = np.where(b.samples["center"] <= b.samples["neighbor"])[0]
            if len(idx_up) == 0:
                continue
            idx_lo = np.where(b.samples["center"] > b.samples["neighbor"])[0]
            if len(idx_lo) == 0:
                print(
                    "Corresponding swapped pair not found",
                    np.array(b.samples.values)[idx_up],
                )

            # else:
            # print(np.array(b.samples.values)[idx_up], "corresponf to", np.array(b.samples.values)[idx_lo])
            # we need to find the "ji" position that matches each "ij" sample.
            # we exploit the fact that the samples are sorted by structure to do a "local" rearrangement
            smp_up, smp_lo = 0, 0
            idx_ji = []
            samplecopy = np.array(
                b.samples.view(
                    [
                        "structure",
                        "center",
                        "neighbor",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                        "cell_shift_a_MIC",
                        "cell_shift_b_MIC",
                        "cell_shift_c_MIC",
                    ]
                ).values
            )

            for smp_up in range(len(idx_up)):
                # ij = b.samples[idx_up[smp_up]][["center", "neighbor"]]
                structure, i, j, Tx, Ty, Tz, Mx, My, Mz = b.samples.view(
                    [
                        "structure",
                        "center",
                        "neighbor",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                        "cell_shift_a_MIC",
                        "cell_shift_b_MIC",
                        "cell_shift_c_MIC",
                    ]
                ).values[idx_up[smp_up]]
                # ji_entry = Labels(b.samples.names[:-3], values = np.array([[structure, j, i, Tx, Ty, Tz]]))
                ji_entry = np.array([structure, j, i, -Tx, -Ty, -Tz, -Mx, -My, -Mz])
                where_ji = np.argwhere(np.all(samplecopy == ji_entry, axis=1))
                assert where_ji.shape == (1, 1), where_ji.shape
                where_ji = where_ji[0, 0]
                idx_ji.append(where_ji)

                # for smp_lo in range(smp_up, len(idx_lo)):
                #     ij_lo = b.samples.view(["neighbor", "center"]).values[
                #         idx_lo[smp_lo]
                #     ]
                #     # ij_lo = b.samples[idx_lo[smp_lo]][["neighbor", "center"]]
                #     if (
                #         (
                #             b.samples["structure"][idx_up[smp_up]]
                #             != b.samples["structure"][idx_lo[smp_lo]]
                #         )
                #         or (
                #             b.samples["cell_shift_a"][idx_up[smp_up]]
                #             != b.samples["cell_shift_a"][idx_lo[smp_lo]]
                #         )
                #         or (
                #             b.samples["cell_shift_b"][idx_up[smp_up]]
                #             != b.samples["cell_shift_b"][idx_lo[smp_lo]]
                #         )
                #         or (b.samples["cell_shift_c"][idx_up[smp_up]] != b.samples["cell_shift_c"][idx_lo[smp_lo]])
                #     ):  # Must also add checks for the translation here TODO
                #         print(np.array(b.samples.values)[idx_up])
                #         print("corresponf to")
                #         print(np.array(b.samples.values)[idx_lo])

                #         # print(b.samples["structure"][idx_up[smp_up]])
                #         # print("correspond to")
                #         # print(b.samples["structure"][idx_lo[smp_lo]])
                #         raise ValueError(
                #             f"Could not find matching ji term for sample {b.samples[idx_up[smp_up]]}"
                #         )
                #     if tuple(ij) == tuple(ij_lo):
                #         idx_lo[smp_up], idx_lo[smp_lo] = idx_lo[smp_lo], idx_lo[smp_up]
                #         break

            keys.append(tuple(k) + (1,))
            keys.append(tuple(k) + (-1,))
            # print(k.values, b.values.shape, idx_up.shape, idx_lo.shape)

            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names,
                        values=np.asarray(b.samples.values[idx_up]),
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=(b.values[idx_up] + b.values[idx_ji]) / 2,  # idx_lo
                )
            )
            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names,
                        values=np.asarray(b.samples.values[idx_up]),
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=(b.values[idx_up] - b.values[idx_ji]) / 2,  # idx_lo
                )
            )

        elif k["species_center"] < k["species_neighbor"]:
            # off-site, different species
            keys.append(tuple(k) + (2,))
            blocks.append(b.copy())

    return TensorMap(
        keys=Labels(
            names=pair.keys.names + ["block_type"],
            values=np.asarray(keys),
        ),
        blocks=blocks,
    )


def twocenter_hermitian_features_periodic(
    single_center: TensorMap,
    pair: TensorMap,
    shift: Optional[Tuple[int, int, int]] = None,
    antisymmetric: bool = False,
):
    if shift == [0, 0, 0]:
        return twocenter_hermitian_features(single_center, pair)

    keys = []
    blocks = []

    for k, b in pair.items():
        if k["species_center"] == k["species_neighbor"]:  # self translared pairs
            idx = np.where(b.samples["center"] == b.samples["neighbor"])[0]
            if len(idx) == 0:
                continue
            keys.append(tuple(k) + (0,))
            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names,
                        values=np.asarray(b.samples.values[idx]),
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=b.values[idx],
                )
            )
        else:
            raise NotImplementedError  # Handle periodic case for different species

    for k, b in pair.items():
        if k["species_center"] == k["species_neighbor"]:
            # off-site, same species
            idx_up = np.where(b.samples["center"] < b.samples["neighbor"])[0]
            if len(idx_up) == 0:
                continue
            idx_lo = np.where(b.samples["center"] > b.samples["neighbor"])[0]
            if len(idx_lo) == 0:
                print(
                    "Corresponding swapped pair not found",
                    np.array(b.samples.values)[idx_up],
                )
            # else:
            # print(np.array(b.samples.values)[idx_up], "corresponf to", np.array(b.samples.values)[idx_lo])
            # we need to find the "ji" position that matches each "ij" sample.
            # we exploit the fact that the samples are sorted by structure to do a "local" rearrangement
            smp_up, smp_lo = 0, 0
            for smp_up in range(len(idx_up)):
                # ij = b.samples[idx_up[smp_up]][["center", "neighbor"]]
                ij = b.samples.view(["center", "neighbor"]).values[idx_up[smp_up]]
                for smp_lo in range(smp_up, len(idx_lo)):
                    ij_lo = b.samples.view(["neighbor", "center"]).values[
                        idx_lo[smp_lo]
                    ]
                    # ij_lo = b.samples[idx_lo[smp_lo]][["neighbor", "center"]]
                    if (
                        b.samples["structure"][idx_up[smp_up]]
                        != b.samples["structure"][idx_lo[smp_lo]]
                    ):
                        raise ValueError(
                            f"Could not find matching ji term for sample {b.samples[idx_up[smp_up]]}"
                        )
                    if tuple(ij) == tuple(ij_lo):
                        idx_lo[smp_up], idx_lo[smp_lo] = idx_lo[smp_lo], idx_lo[smp_up]
                        break

            keys.append(tuple(k) + (1,))
            keys.append(tuple(k) + (-1,))
            # print(k.values, b.values.shape, idx_up.shape, idx_lo.shape)
            if not antisymmetric:
                blocks.append(
                    TensorBlock(
                        samples=Labels(
                            names=b.samples.names,
                            values=np.asarray(b.samples.values[idx_up]),
                        ),
                        components=b.components,
                        properties=b.properties,
                        values=(b.values[idx_up] + b.values[idx_lo]) / np.sqrt(2),
                    )
                )
                blocks.append(
                    TensorBlock(
                        samples=Labels(
                            names=b.samples.names,
                            values=np.asarray(b.samples.values[idx_up]),
                        ),
                        components=b.components,
                        properties=b.properties,
                        values=(b.values[idx_up] - b.values[idx_lo]) / np.sqrt(2),
                    )
                )
            else:
                blocks.append(
                    TensorBlock(
                        samples=Labels(
                            names=b.samples.names,
                            values=np.asarray(b.samples.values[idx_up]),
                        ),
                        components=b.components,
                        properties=b.properties,
                        values=(b.values[idx_up] - b.values[idx_lo]) / np.sqrt(2),
                    )
                )
                blocks.append(
                    TensorBlock(
                        samples=Labels(
                            names=b.samples.names,
                            values=np.asarray(b.samples.values[idx_up]),
                        ),
                        components=b.components,
                        properties=b.properties,
                        values=(b.values[idx_up] + b.values[idx_lo]) / np.sqrt(2),
                    )
                )
        elif k["species_center"] < k["species_neighbor"]:
            # off-site, different species
            keys.append(tuple(k) + (2,))
            blocks.append(b.copy())

    # kkeys = [list(k) for k in keys]
    # print([len(k) for k in keys])
    # print(keys[2], keys[3], )
    # print(np.asarray(kkeys).shape   )
    # print(Labels(
    #         names=pair.keys.names + ["block_type"],
    #         values=np.asarray(keys),
    #     ),)
    return TensorMap(
        keys=Labels(
            names=pair.keys.names + ["block_type"],
            values=np.asarray(keys),
        ),
        blocks=blocks,
    )


from mlelec.targets import SingleCenter, TwoCenter
from mlelec.data.dataset import MLDataset


def compute_features_for_target(dataset: MLDataset, **kwargs):
    hypers = kwargs.get("hypers", None)
    # if dataset.molecule_data.pbc:
    if hypers is None:
        print("Computing features with default hypers")
        hypers = {
            "cutoff": 4.0,
            "max_radial": 6,
            "max_angular": 3,
            "atomic_gaussian_width": 0.3,
            "center_atom_weight": 1,
            "radial_basis": {"Gto": {}},
            "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
        }
    single = single_center_features(
        dataset.structures, hypers, order_nu=2, lcut=hypers["max_angular"]
    )
    if isinstance(dataset.target, SingleCenter):
        features = single
    elif isinstance(dataset.target, TwoCenter):
        pairs = pair_features(
            dataset.structures,
            hypers,
            order_nu=1,
            lcut=hypers["max_angular"],
            feature_names=single[0].properties.names,
        )
        features = twocenter_hermitian_features(single, pairs)
    else:
        raise ValueError(f"Target type {type(dataset.target)} not supported")
    return features
