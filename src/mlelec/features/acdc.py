import warnings
from typing import List, Optional, Tuple, Union

import ase
import metatensor.operations as operations
import numpy as np
import torch
import tqdm
from metatensor import Labels, TensorBlock, TensorMap
from rascaline import SphericalExpansion
from rascaline import SphericalExpansionByPair as PairExpansion

from mlelec.features.acdc_utils import (
    _pca,
    _standardize,
    acdc_standardize_keys,
    block_to_mic_translation,
    cg_combine,
    cg_increment,
    drop_blocks_L,
    fix_gij,
    relabel_keys,
)
from mlelec.utils.metatensor_utils import labels_where

use_native = True  # True for rascaline


def single_center_features(
    frames, hypers, order_nu, lcut=None, cg=None, device="cpu", **kwargs
):
    """
    computes the atom-centred features for all the frames in the dataset
    using the `SphericalExpansion` calculator from rascaline. The spherical
    expansion coefficients are calculated based on the given hyperparameters.
    Clebsch-Gordan iterations are then performed to get the desired body-order
    expansion `order_nu`, using the `cg_increment` function.

    Args:
        frames: List of ase.Atoms objects.
        hypers: dictionary of hyperparameters used to compute the spherical expansion.
        order_nu: desired body_order.
        lcut: angular_cutoff for the Clebsch-Gordan iterations. Defaults to None.
        cg: `ClebshGordanReal` object. Defaults to None.
        device: device to use. Defaults to "cpu".

    Returns:
        TensorMap: TensorMap containing the atom-centred features.
    """
    calculator = SphericalExpansion(**hypers)
    rhoi = calculator.compute(frames, use_native_system=use_native)
    rhoi = rhoi.keys_to_properties(["species_neighbor"])
    rho1i = acdc_standardize_keys(rhoi)
    if order_nu == 1:
        return drop_blocks_L(rho1i, lcut)
    if lcut is None:
        lcut = 10
    if cg is None:
        from mlelec.utils.symmetry import ClebschGordanReal

        L = max(lcut, hypers["max_angular"])
        cg = ClebschGordanReal(lmax=L, device=device)
    rho_prev = rho1i

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
    order_nu: Union[List[int], int] = None,
    all_pairs: bool = False,
    both_centers: bool = False,
    lcut: int = 3,
    max_shift=None,
    device="cpu",
    kmesh=None,
    desired_shifts=None,
    mic=False,
    return_rho0ij=False,
    **kwargs,
):
    """
    computes the two-centred features for all the frames in the dataset.
    The two-centred features are computed using the `PairExpansion` calculator
    from rascaline. One can either use the same hyperparameters as that of the
    spherical expansion or specify new hyperparameters in `hypers_pair` to
    compute the pair expansion coefficients. Like `single_centre_features`
    here too Clebsch-Gordan iterations are performed to get the desired body
    -order expansion `order_nu`,using the `cg_increment` function. When working
    with periodic systems, the `mic` flag should be set to True and the `kmesh`
    parameter should specify the k-grid one wants to use. The `desired_shifts`
    parameter can be used to specify the desired translations one wants to retain
    the features for.

    Args:
        frames: A list of ase.Atoms objects.
        hypers: dictionary of hyperparameters used to compute the spherical expansion.
        hypers_pair: hyperparams for pair expansion. Defaults to None.
        cg: `ClebshGordanReal` object. Defaults to None.
        rhonu_i: spherical expansion coefficients. Defaults to None.
        order_nu: desired body order. Defaults to None.
        all_pairs: #TODO. Defaults to False.
        both_centers: #TODO. Defaults to False.
        lcut: angular_cutoff for the Clebsch-Gordan iterations. Defaults to 3.
        max_shift: #TODO. Defaults to None.
        device: device to use. Defaults to "cpu".
        kmesh: k-grid. Defaults to None.
        desired_shifts: translations to be retained. Defaults to None.
        mic: set to `True` if working with periodic systems. Defaults to False.
        return_rho0ij: #TODO. Defaults to False.

    Returns:
        TensorMap: TensorMap containing the two-centred features.
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
                np.max([np.max(f.get_all_distances(mic=mic)) / factor for f in frames])
            )

        elif max_shift is not None:
            repframes = [f.repeat(max_shift) for f in frames]
            hypers_allpairs["cutoff"] = np.ceil(
                np.max(
                    [np.max(f.get_all_distances(mic=mic)) / factor for f in repframes]
                )
            )
            # FIXME - miic = mic
            warnings.warn(
                f"Using cutoff {hypers_allpairs['cutoff']} for all pairs feature"
            )
        else:
            warnings.warn(f"Using unchanged hypers for all pairs feature")
        print("hypers_pair", hypers_allpairs)
        calculator_allpairs = PairExpansion(**hypers_allpairs)

        rho0_ij = calculator_allpairs.compute(frames, use_native_system=use_native)

    rho0_ij = fix_gij(rho0_ij)
    rho0_ij = acdc_standardize_keys(rho0_ij)

    if mic:
        from mlelec.utils.pbc_utils import scidx_from_unitcell, scidx_to_mic_translation

        # generate all the translations and assign values based on mic map
        assert kmesh is not None, "kmesh must be specified for MIC mapping"
        # we should add the missing samples
        warnings.warn(f"Using kmesh {kmesh} for MIC mapping")
        blocks = []

        for key, block in rho0_ij.items():
            if key["spherical_harmonics_l"] % 2 == 1:
                mic_phase = -1
            else:
                mic_phase = 1
            all_frIJ = np.unique(block.samples.values[:, :3], axis=0)
            value_indices = []
            fixed_sample = []
            fixed_mic = []
            all_samples = [
                (*a, x, y, z)
                for a in all_frIJ
                for x in range(kmesh[0])
                for y in range(kmesh[1])
                for z in range(kmesh[2])
            ]
            for s in all_samples:
                ifr, i, j, x, y, z = s

                if i == j and [x, y, z] == [0, 0, 0]:
                    continue

                (
                    (mic_x, mic_y, mic_z),
                    (mic_mx, mic_my, mic_mz),
                    fixed_plus,
                    fixed_minus,
                ) = scidx_to_mic_translation(
                    frames[ifr],
                    I=i,
                    J=scidx_from_unitcell(
                        frames[ifr], j=j, T=[x, y, z], kmesh=kmesh
                    ),  # TODO - kmesh[ifr] - nonunofrm kmesh across structrues
                    j=j,
                    kmesh=kmesh,
                )
                print(
                    ifr,
                    i,
                    j,
                    "mic",
                    mic_x,
                    mic_y,
                    mic_z,
                    "m_mic",
                    mic_mx,
                    mic_my,
                    mic_mz,
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

                assert isinstance(mappedidx, int), (mappedidx, mic_label)
                fixed_mic.append(mic_phase**fixed_plus)
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
                                mic_mx,
                                mic_my,
                                mic_mz,
                            ]
                        ).reshape(1, -1),
                    )[0]
                    mappedidx = block.samples.position(mic_label)

                    assert isinstance(mappedidx, int), (mappedidx, mic_label)
                    fixed_mic.append(mic_phase**fixed_minus)
                    value_indices.append(mappedidx)
                    fixed_sample.append(
                        [
                            ifr,
                            j,
                            i,
                            -x,
                            -y,
                            -z,
                            mic_mx,
                            mic_my,
                            mic_mz,
                        ]
                    )

            blocks.append(
                TensorBlock(
                    values=torch.einsum(
                        "scp,s-> scp",
                        block.values[value_indices],
                        torch.tensor(fixed_mic),
                    ),
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
            )

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
        if return_rho0ij:
            return rho0_ij
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

    if rhonu_i is None:
        rhonu_i = single_center_features(
            frames, order_nu=order_nu_i, hypers=hypers, lcut=lcut, cg=cg, kwargs=kwargs
        )

    rhonu_ij = cg_combine(
        rhonu_i,
        rho0_ij,
        clebsch_gordan=cg,
        other_keys_match=["species_center"],
        lcut=lcut,
        feature_names=kwargs.get("feature_names", None),
    )
    if not both_centers:
        return rhonu_ij

    else:
        if "order_nu_j" not in locals():
            warnings.warn("nu_j not defined, using nu_i for nu_j as well")
            order_nu_j = order_nu_i
        if order_nu_j != order_nu_i:
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
        rhonu_nupij = cg_combine(
            rhoj,
            rhonu_ij,
            # rhoj,
            lcut=lcut,
            other_keys_match=["species_neighbor"],
            clebsch_gordan=cg,
            mp=True,  # for combining with neighbor
            feature_names=kwargs.get("feature_names", None),
        )

        return rhonu_nupij


def twocenter_hermitian_features(
    single_center: TensorMap,
    pair: TensorMap,
) -> TensorMap:
    """
    Combines the atom-centred and pair-centred features to form features 
    for Hamiltonian learning. The on-site elements of the Hamiltonian are
    learned using the atom-centred features and the off-site elements are
    learned using the pair-centred features. 
    

    Args:
        single_center (TensorMap): single-centred features.
        pair (TensorMap): pair-centred features.

    Returns:
        TensorMap: TensorMap containing the combined features.
    """
    
    keys = []
    blocks = []
    if single_center is not None:

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
            if len(idx) != 0:
                # SHOULD BE ZERO
                raise ValueError("btype0 should be zero for pair")
        else:
            raise NotImplementedError  # Handle periodic case for different species

    for k, b in pair.items():
        positive_shifts_idx = []
        if k["species_center"] == k["species_neighbor"]:
            # off-site, same species
            idx_up = np.where(
                (b.samples["center"] <= b.samples["neighbor"])
                & (b.samples["cell_shift_a"] >= 0)
                & (b.samples["cell_shift_b"] >= 0)
                & (b.samples["cell_shift_c"] >= 0)
            )[0]
            # zeroshift_idx = np.argwhere(np.all(np.array(b.samples.values.view(["cell_shift_a", "cell_shift_b", "cell_shift_c"])==[0,0,0]), axis=1))
            # if b.samples["center"]
            print(len(idx_up))
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
            samplecopy = np.array(b.samples.values[:, :6])

            for smp_up in range(len(idx_up)):
                # ij = b.samples[idx_up[smp_up]][["center", "neighbor"]]

                structure, i, j, Tx, Ty, Tz, Mx, My, Mz = b.samples.values[
                    idx_up[smp_up]
                ]

                ji_entry = np.array(
                    [structure, j, i, -Tx, -Ty, -Tz]
                )  # , -Mx, -My, -Mz])
                where_ji = np.argwhere(np.all(samplecopy == ji_entry, axis=1))
                assert where_ji.shape == (1, 1), where_ji.shape
                where_ji = where_ji[0, 0]
                idx_ji.append(where_ji)
            keys.append(tuple(k) + (1,))
            keys.append(tuple(k) + (-1,))
            # print(k.values, b.values.shape, idx_up.shape, idx_lo.shape)

            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names,
                        values=np.asarray(
                            b.samples.values[idx_up]  # [positive_shifts_idx]
                        ),
                    ),
                    components=b.components,
                    properties=b.properties,
                    # values=b.values[idx_up],
                    values=(b.values[idx_up] + b.values[idx_ji]),
                    # / 2,  # idx_lo
                )
            )
            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names,
                        values=np.asarray(
                            b.samples.values[idx_up]  # [positive_shifts_idx]
                        ),
                    ),
                    components=b.components,
                    properties=b.properties,
                    # values=b.values[idx_ji],  ##[idx_ji],
                    values=(b.values[idx_up] - b.values[idx_ji]),
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


# retain only positive shifts in the end


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


from mlelec.data.dataset import MLDataset
from mlelec.targets import SingleCenter, TwoCenter


def compute_features_for_target(dataset: MLDataset, device=None, **kwargs):
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
        dataset.structures,
        hypers,
        order_nu=kwargs.get("order_nu", 2),
        lcut=hypers["max_angular"],
        device=device,
    )
    if isinstance(dataset.target, SingleCenter):
        features = single
    elif isinstance(dataset.target, TwoCenter) or issubclass(dataset.target, TwoCenter):
        pairs = pair_features(
            dataset.structures,
            hypers,
            order_nu=kwargs.get("order_nu_pair", 1),
            lcut=hypers["max_angular"],
            feature_names=single[0].properties.names,
            device=device,
            both_centers=kwargs.get("both_centers", False),
        )
        features = twocenter_hermitian_features(single, pairs)
    else:
        raise ValueError(f"Target type {type(dataset.target)} not supported")
    return features
